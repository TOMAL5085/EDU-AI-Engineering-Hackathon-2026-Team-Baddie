"""Train, serialize, and use the local horizon-specific risk models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import DATA_PATH, FEATURE_COLUMNS, build_feature_frame, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "dengue_risk_model.joblib"
HORIZONS = (2, 3, 4)
MODEL_FORMAT_VERSION = 3
INNER_FOLD_FRACTIONS = (0.55, 0.70, 0.85)


class FeatureImportanceVotingRegressor(VotingRegressor):
    """Voting ensemble exposing normalized feature_importances_ averaged across its estimators."""

    @property
    def feature_importances_(self) -> np.ndarray:
        importances = [
            getattr(est, "feature_importances_", None)
            for est in self.estimators_
        ]
        valid = [imp for imp in importances if imp is not None]
        if not valid:
            raise AttributeError("None of the underlying estimators expose feature_importances_")
        mean_imp = np.mean(valid, axis=0)
        total = np.sum(mean_imp)
        return mean_imp / total if total > 0 else mean_imp


class HistGradientBoostingWithImportance(HistGradientBoostingRegressor):
    """HistGradientBoostingRegressor exposing estimated feature importances."""

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        n_features = X.shape[1] if hasattr(X, "shape") else len(X.columns)
        counts = np.zeros(n_features, dtype=float)
        for predictor in getattr(self, "_predictors", []):
            for tree in predictor:
                for node in getattr(tree, "nodes", []):
                    if not node["is_leaf"]:
                        counts[node["feature_idx"]] += 1.0
        total = np.sum(counts)
        self.feature_importances_ = counts / total if total > 0 else np.ones(n_features) / n_features
        return self


MODEL_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "random_forest_baseline",
        "estimator": RandomForestRegressor,
        "parameters": {"n_estimators": 96, "max_depth": 9, "min_samples_leaf": 4, "max_features": 0.9},
    },
    {
        "name": "random_forest_regularized",
        "estimator": RandomForestRegressor,
        "parameters": {"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.85},
    },
    {
        "name": "random_forest_balanced",
        "estimator": RandomForestRegressor,
        "parameters": {"n_estimators": 192, "max_depth": 14, "min_samples_leaf": 2, "max_features": 0.8},
    },
    {
        "name": "extra_trees_regularized",
        "estimator": ExtraTreesRegressor,
        "parameters": {"n_estimators": 256, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.9},
    },
    {
        "name": "extra_trees_tuned_deep",
        "estimator": ExtraTreesRegressor,
        "parameters": {"n_estimators": 300, "max_depth": 11, "min_samples_leaf": 3, "max_features": 0.8},
    },
    {
        "name": "extra_trees_tuned_compact",
        "estimator": ExtraTreesRegressor,
        "parameters": {"n_estimators": 300, "max_depth": 9, "min_samples_leaf": 4, "max_features": 0.85},
    },
    {
        "name": "extra_trees_balanced",
        "estimator": ExtraTreesRegressor,
        "parameters": {"n_estimators": 192, "max_depth": 14, "min_samples_leaf": 2, "max_features": 0.8},
    },
    {
        "name": "gradient_boosting_robust",
        "estimator": GradientBoostingRegressor,
        "parameters": {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.85},
    },
    {
        "name": "gradient_boosting_conservative",
        "estimator": GradientBoostingRegressor,
        "parameters": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8},
    },
    {
        "name": "gradient_boosting_huber",
        "estimator": GradientBoostingRegressor,
        "parameters": {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3, "loss": "huber", "subsample": 0.85},
    },
    {
        "name": "hist_gradient_boosting",
        "estimator": HistGradientBoostingWithImportance,
        "parameters": {"max_iter": 120, "learning_rate": 0.04, "max_depth": 4, "min_samples_leaf": 15},
    },
    {
        "name": "voting_et_gb",
        "estimator": FeatureImportanceVotingRegressor,
        "parameters": {
            "estimators": [
                ("et", ExtraTreesRegressor, {"n_estimators": 256, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.9}),
                ("gb", GradientBoostingRegressor, {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.85}),
            ]
        },
    },
    {
        "name": "voting_et_rf_gb",
        "estimator": FeatureImportanceVotingRegressor,
        "parameters": {
            "estimators": [
                ("et", ExtraTreesRegressor, {"n_estimators": 256, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.9}),
                ("rf", RandomForestRegressor, {"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.85}),
                ("gb", GradientBoostingRegressor, {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.85}),
            ]
        },
    },
    {
        "name": "voting_et_rf",
        "estimator": FeatureImportanceVotingRegressor,
        "parameters": {
            "estimators": [
                ("et", ExtraTreesRegressor, {"n_estimators": 256, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.9}),
                ("rf", RandomForestRegressor, {"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 4, "max_features": 0.85}),
            ]
        },
    },
)


def risk_tier(predicted_cases: float, cutoffs: dict[str, float]) -> str:
    """Map predicted weekly case count to a tier using training quantiles."""
    value = max(0.0, float(predicted_cases))
    if value < cutoffs["medium"]:
        return "Low"
    if value < cutoffs["high"]:
        return "Medium"
    if value < cutoffs["critical"]:
        return "High"
    return "Critical"


def _build_estimator(candidate: dict[str, Any]) -> Any:
    """Create one deterministic, CPU-parallel estimator from a candidate specification."""
    estimator_cls = candidate["estimator"]
    params = dict(candidate["parameters"])

    if estimator_cls is FeatureImportanceVotingRegressor:
        sub_specs = params.get("estimators", [])
        estimators_list = []
        for name, sub_cls, sub_params in sub_specs:
            if sub_cls in (RandomForestRegressor, ExtraTreesRegressor):
                est = sub_cls(**sub_params, random_state=42, n_jobs=-1)
            else:
                est = sub_cls(**sub_params, random_state=42)
            estimators_list.append((name, est))
        return FeatureImportanceVotingRegressor(estimators=estimators_list, n_jobs=-1)

    if estimator_cls in (RandomForestRegressor, ExtraTreesRegressor):
        return estimator_cls(**params, random_state=42, n_jobs=-1)
    return estimator_cls(**params, random_state=42)


def _inner_fold_ranges(development_dates: list[pd.Timestamp], final_holdout_start: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return expanding validation windows contained entirely before final holdout."""
    positions = sorted({int(len(development_dates) * fraction) for fraction in INNER_FOLD_FRACTIONS})
    if len(positions) != len(INNER_FOLD_FRACTIONS) or positions[0] < 1:
        raise ValueError("Not enough development dates for chronological candidate selection")
    starts = [development_dates[position] for position in positions]
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else final_holdout_start)
        for index, start in enumerate(starts)
    ]


def _candidate_result(
    candidate: dict[str, Any],
    usable: pd.DataFrame,
    target: str,
    target_date: str,
    fold_ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    """Score one candidate in expanding, target-purged chronological folds."""
    folds: list[dict[str, Any]] = []
    for fold_number, (validation_start, validation_end) in enumerate(fold_ranges, start=1):
        train = usable.loc[(usable["date"] < validation_start) & (usable[target_date] < validation_start)]
        validation = usable.loc[
            (usable["date"] >= validation_start)
            & (usable["date"] < validation_end)
            & (usable[target_date] < validation_end)
        ]
        if train.empty or validation.empty:
            raise ValueError(f"Could not create nonempty candidate-validation rows for fold {fold_number}")

        estimator = _build_estimator(candidate)
        estimator.fit(train.loc[:, FEATURE_COLUMNS], train[target])
        predictions = np.maximum(0.0, estimator.predict(validation.loc[:, FEATURE_COLUMNS]))
        actual = validation[target].to_numpy(dtype=float)
        folds.append(
            {
                "fold": fold_number,
                "validation_start": validation_start.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "mae": float(mean_absolute_error(actual, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            }
        )

    params = {}
    for k, v in candidate["parameters"].items():
        if k == "estimators":
            params[k] = [f"{name}:{sub_cls.__name__}" for name, sub_cls, _ in v]
        else:
            params[k] = v

    return {
        "name": candidate["name"],
        "estimator": candidate["estimator"].__name__,
        "parameters": params,
        "mean_mae": float(np.mean([fold["mae"] for fold in folds])),
        "mean_rmse": float(np.mean([fold["rmse"] for fold in folds])),
        "folds": folds,
    }


def train_model_bundle(
    data: pd.DataFrame | None = None,
    output_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Select and fit three horizon-specific tree ensembles without date leakage."""
    raw = load_dataset(DATA_PATH) if data is None else data.copy()
    frame = build_feature_frame(raw)
    valid_dates = sorted(frame["date"].dropna().unique())
    if len(valid_dates) < 20:
        raise ValueError("At least 20 weekly dates are required for chronological training")

    split_index = int(len(valid_dates) * 0.8)
    split_index = min(max(split_index, 1), len(valid_dates) - 1)
    holdout_start = pd.Timestamp(valid_dates[split_index])
    models: dict[int, Any] = {}
    metrics: dict[int, dict[str, Any]] = {}
    thresholds: dict[int, dict[str, float]] = {}
    selection: dict[int, dict[str, Any]] = {}

    for horizon in HORIZONS:
        target = f"target_dengue_cases_h{horizon}"
        target_date = f"target_date_h{horizon}"
        usable = frame.dropna(subset=[*FEATURE_COLUMNS, target, target_date]).copy()
        train_mask = (usable["date"] < holdout_start) & (usable[target_date] < holdout_start)
        test_mask = usable["date"] >= holdout_start
        train = usable.loc[train_mask]
        test = usable.loc[test_mask]
        if train.empty or test.empty:
            raise ValueError(f"Could not create nonempty chronological train and holdout rows for week {horizon}")

        development_dates = sorted(pd.Timestamp(date) for date in train["date"].unique())
        fold_ranges = _inner_fold_ranges(development_dates, holdout_start)
        candidate_results = sorted(
            (
                _candidate_result(candidate, train, target, target_date, fold_ranges)
                for candidate in MODEL_CANDIDATES
            ),
            key=lambda result: (result["mean_rmse"], result["mean_mae"], result["name"]),
        )
        selected = candidate_results[0]
        selected_candidate = next(candidate for candidate in MODEL_CANDIDATES if candidate["name"] == selected["name"])
        estimator = _build_estimator(selected_candidate)
        estimator.fit(train.loc[:, FEATURE_COLUMNS], train[target])
        predictions = np.maximum(0.0, estimator.predict(test.loc[:, FEATURE_COLUMNS]))
        actual = test[target].to_numpy(dtype=float)
        metrics[horizon] = {
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "train_rows": int(len(train)),
            "holdout_rows": int(len(test)),
            "holdout_start": holdout_start.date().isoformat(),
            "train_target_end": train[target_date].max().date().isoformat(),
        }
        selection[horizon] = {
            "ranking_metric": "mean_rmse",
            "selected": selected,
            "candidates": candidate_results,
        }

        train_targets = train[target].clip(lower=0)
        p75, p90, p97 = train_targets.quantile([0.75, 0.90, 0.97]).tolist()
        thresholds[horizon] = {
            "medium": float(p75),
            "high": float(p90),
            "critical": float(p97),
        }

        # Refit the selected candidate on all eligible rows for the shipped model.
        shipped_estimator = _build_estimator(selected_candidate)
        shipped_estimator.fit(usable.loc[:, FEATURE_COLUMNS], usable[target])
        models[horizon] = shipped_estimator

    bundle: dict[str, Any] = {
        "format_version": MODEL_FORMAT_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "horizons": list(HORIZONS),
        "models": models,
        "metrics": metrics,
        "thresholds": thresholds,
        "selection": selection,
        "data_summary": {
            "rows": int(len(raw)),
            "wards": int(raw["ward_id"].nunique()),
            "cities": sorted(raw["city"].astype(str).unique().tolist()),
            "first_date": raw["date"].min().date().isoformat(),
            "latest_date": raw["date"].max().date().isoformat(),
            "holdout_start": holdout_start.date().isoformat(),
        },
        "model_config": {
            "selection": "expanding chronological validation",
            "candidate_count": len(MODEL_CANDIDATES),
            "ranking_metric": "mean_rmse",
            "random_state": 42,
        },
        "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__},
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return bundle


def load_model_bundle(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load the prebuilt local model artifact."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {source}. Run `python scripts/train_model.py` first."
        )
    bundle = joblib.load(source)
    if bundle.get("format_version") not in (1, 2, MODEL_FORMAT_VERSION):
        raise ValueError("Unsupported model bundle format; rebuild with `python scripts/train_model.py`.")
    return bundle


def predict_row(bundle: dict[str, Any], row: pd.Series | dict[str, Any], horizon: int) -> dict[str, Any]:
    """Return expected weekly cases and a four-level tier."""
    if horizon not in bundle["models"]:
        raise ValueError(f"Unsupported forecast horizon: {horizon}")
    record = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    feature_values = pd.DataFrame([{name: float(record[name]) for name in bundle["feature_columns"]}])
    model = bundle["models"][horizon]
    predicted_cases = max(0.0, float(model.predict(feature_values)[0]))
    cutoffs = bundle["thresholds"][horizon]
    return {
        "horizon_weeks": horizon,
        "predicted_cases": predicted_cases,
        "expected_cases": predicted_cases,
        "tier": risk_tier(predicted_cases, cutoffs),
        "cutoffs": cutoffs,
    }

