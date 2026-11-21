"""Train, serialize, and use the local horizon-specific risk models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import DATA_PATH, FEATURE_COLUMNS, build_feature_frame, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "dengue_risk_model.joblib"
HORIZONS = (2, 3, 4)


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


def train_model_bundle(
    data: pd.DataFrame | None = None,
    output_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Fit three RandomForest regressors with a purged chronological holdout."""
    raw = load_dataset(DATA_PATH) if data is None else data.copy()
    frame = build_feature_frame(raw)
    valid_dates = sorted(frame["date"].dropna().unique())
    if len(valid_dates) < 20:
        raise ValueError("At least 20 weekly dates are required for chronological training")

    split_index = int(len(valid_dates) * 0.8)
    split_index = min(max(split_index, 1), len(valid_dates) - 1)
    holdout_start = pd.Timestamp(valid_dates[split_index])
    models: dict[int, RandomForestRegressor] = {}
    metrics: dict[int, dict[str, Any]] = {}
    thresholds: dict[int, dict[str, float]] = {}

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

        estimator = RandomForestRegressor(
            n_estimators=96,
            max_depth=9,
            min_samples_leaf=4,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )
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

        train_targets = train[target].clip(lower=0)
        p75, p90, p97 = train_targets.quantile([0.75, 0.90, 0.97]).tolist()
        thresholds[horizon] = {
            "medium": float(p75),
            "high": float(p90),
            "critical": float(p97),
        }

        # Refit on all rows with available lags and outcomes for the shipped model.
        estimator.fit(usable.loc[:, FEATURE_COLUMNS], usable[target])
        models[horizon] = estimator

    bundle: dict[str, Any] = {
        "format_version": 1,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "horizons": list(HORIZONS),
        "models": models,
        "metrics": metrics,
        "thresholds": thresholds,
        "data_summary": {
            "rows": int(len(raw)),
            "wards": int(raw["ward_id"].nunique()),
            "cities": sorted(raw["city"].astype(str).unique().tolist()),
            "first_date": raw["date"].min().date().isoformat(),
            "latest_date": raw["date"].max().date().isoformat(),
            "holdout_start": holdout_start.date().isoformat(),
        },
        "model_config": {
            "estimator": "RandomForestRegressor",
            "n_estimators": 96,
            "max_depth": 9,
            "min_samples_leaf": 4,
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
    if bundle.get("format_version") != 1:
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

