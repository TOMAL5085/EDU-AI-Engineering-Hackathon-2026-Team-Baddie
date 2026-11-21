"""Focused tests for the WardWatch supervised forecasting pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dengue.data import FEATURE_COLUMNS, build_feature_frame, load_dataset
from dengue.model import (
    HORIZONS,
    MODEL_CANDIDATES,
    _build_estimator,
    _inner_fold_ranges,
    load_model_bundle,
    predict_row,
    risk_tier,
    train_model_bundle,
)


class ForecastPipelineTestCase(unittest.TestCase):
    """Test data alignment, leakage protections, and model contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_dataset()
        cls.features = build_feature_frame(cls.raw)

    def test_ward_boundary_isolation(self) -> None:
        """Rolling windows and forward targets must not cross ward boundaries."""
        wards = self.features["ward_id"].unique()
        self.assertGreater(len(wards), 1)

        for ward in wards:
            ward_slice = self.features.loc[self.features["ward_id"] == ward]
            # Window of 2 requires 2 periods; first row of each ward must be NaN
            first_row = ward_slice.iloc[0]
            self.assertTrue(pd.isna(first_row["rainfall_2w_mm"]))
            self.assertTrue(pd.isna(first_row["avg_temp_2w_c"]))

            # Window of 4 requires 4 periods; first 3 rows must be NaN
            self.assertTrue(ward_slice.iloc[:3]["rainfall_4w_mm"].isna().all())

            # Immediate 1-week rainfall is available from the first week
            self.assertFalse(pd.isna(first_row["rainfall_1w_mm"]))
            self.assertEqual(first_row["rainfall_1w_mm"], first_row["rainfall_mm"])

            # Forward target shift must produce NaN at the end of each ward
            for h in (2, 3, 4):
                self.assertTrue(ward_slice.iloc[-h:][f"target_dengue_cases_h{h}"].isna().all())

    def test_leakage_safe_chronological_splits(self) -> None:
        """Target dates must never cross validation or holdout boundaries."""
        valid_dates = sorted(self.features["date"].dropna().unique())
        split_index = int(len(valid_dates) * 0.8)
        holdout_start = pd.Timestamp(valid_dates[split_index])

        for horizon in (2, 3, 4):
            target = f"target_dengue_cases_h{horizon}"
            target_date = f"target_date_h{horizon}"
            usable = self.features.dropna(subset=[*FEATURE_COLUMNS, target, target_date]).copy()

            # Final holdout split
            train = usable.loc[(usable["date"] < holdout_start) & (usable[target_date] < holdout_start)]
            test = usable.loc[usable["date"] >= holdout_start]

            self.assertFalse(train.empty)
            self.assertFalse(test.empty)
            self.assertLess(train[target_date].max(), holdout_start)
            self.assertGreaterEqual(test["date"].min(), holdout_start)

            # Inner validation folds
            dev_dates = sorted(pd.Timestamp(d) for d in train["date"].unique())
            fold_ranges = _inner_fold_ranges(dev_dates, holdout_start)
            for v_start, v_end in fold_ranges:
                f_train = train.loc[(train["date"] < v_start) & (train[target_date] < v_start)]
                f_val = train.loc[
                    (train["date"] >= v_start)
                    & (train["date"] < v_end)
                    & (train[target_date] < v_end)
                ]
                self.assertFalse(f_train.empty)
                self.assertFalse(f_val.empty)
                self.assertLess(f_train[target_date].max(), v_start)
                self.assertLess(f_val[target_date].max(), v_end)
                self.assertGreaterEqual(f_val["date"].min(), v_start)

    def test_all_candidates_expose_feature_importances(self) -> None:
        """Every registered candidate estimator must expose valid feature_importances_."""
        n_features = len(FEATURE_COLUMNS)
        X = np.abs(np.random.RandomState(42).randn(40, n_features))
        y = np.abs(np.random.RandomState(42).randn(40))

        for candidate in MODEL_CANDIDATES:
            with self.subTest(candidate=candidate["name"]):
                est = _build_estimator(candidate)
                est.fit(X, y)
                self.assertTrue(
                    hasattr(est, "feature_importances_"),
                    f"{candidate['name']} lacks feature_importances_",
                )
                importances = est.feature_importances_
                self.assertEqual(len(importances), n_features)
                self.assertTrue(np.all(importances >= 0.0))
                self.assertAlmostEqual(float(np.sum(importances)), 1.0, places=4)

    def test_risk_tier_mapping(self) -> None:
        """Predicted counts must map deterministically to cutoffs."""
        cutoffs = {"medium": 5.0, "high": 15.0, "critical": 30.0}
        self.assertEqual(risk_tier(0.0, cutoffs), "Low")
        self.assertEqual(risk_tier(4.9, cutoffs), "Low")
        self.assertEqual(risk_tier(5.0, cutoffs), "Medium")
        self.assertEqual(risk_tier(14.9, cutoffs), "Medium")
        self.assertEqual(risk_tier(15.0, cutoffs), "High")
        self.assertEqual(risk_tier(29.9, cutoffs), "High")
        self.assertEqual(risk_tier(30.0, cutoffs), "Critical")
        self.assertEqual(risk_tier(100.0, cutoffs), "Critical")
        self.assertEqual(risk_tier(-5.0, cutoffs), "Low")

    def test_predict_row_interface_and_bounds(self) -> None:
        """predict_row must adhere strictly to the expected interface."""
        # Create a small mock bundle
        mock_models = {}
        for h in HORIZONS:
            est = _build_estimator(MODEL_CANDIDATES[0])
            est.fit(np.zeros((10, len(FEATURE_COLUMNS))), np.zeros(10))
            mock_models[h] = est

        bundle = {
            "format_version": 3,
            "feature_columns": list(FEATURE_COLUMNS),
            "models": mock_models,
            "thresholds": {h: {"medium": 2.0, "high": 5.0, "critical": 10.0} for h in HORIZONS},
        }

        sample_row = {col: 1.0 for col in FEATURE_COLUMNS}
        res = predict_row(bundle, sample_row, 2)
        self.assertIn("horizon_weeks", res)
        self.assertIn("predicted_cases", res)
        self.assertIn("expected_cases", res)
        self.assertIn("tier", res)
        self.assertIn("cutoffs", res)
        self.assertEqual(res["horizon_weeks"], 2)
        self.assertGreaterEqual(res["predicted_cases"], 0.0)
        self.assertIn(res["tier"], ("Low", "Medium", "High", "Critical"))


if __name__ == "__main__":
    unittest.main()
