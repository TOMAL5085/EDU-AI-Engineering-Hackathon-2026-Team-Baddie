"""Load and align the weekly ward-level dengue dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dengue_weekly.csv"

REQUIRED_COLUMNS = (
    "date",
    "city",
    "ward_id",
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
)
NUMERIC_COLUMNS = (
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
)
CITY_LEVELS = ("Chattogram", "Gazipur", "Narayanganj")
FEATURE_COLUMNS = (
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "rainfall_1w_mm",
    "rainfall_2w_mm",
    "rainfall_4w_mm",
    "avg_temp_2w_c",
    "avg_temp_4w_c",
    "relative_humidity_2w_pct",
    "relative_humidity_4w_pct",
    "dengue_cases_4w",
    "week_sin",
    "week_cos",
    "city_Chattogram",
    "city_Gazipur",
    "city_Narayanganj",
)


class DatasetError(ValueError):
    """Raised when source data cannot safely be used for inference."""


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the bundled CSV and validate its schema and keys."""
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"Dataset not found: {source}")

    try:
        frame = pd.read_csv(source)
    except Exception as exc:  # pandas includes useful parser details
        raise DatasetError(f"Could not read dataset at {source}: {exc}") from exc

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {', '.join(missing)}")

    unexpected = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if unexpected:
        raise DatasetError(f"Dataset has unexpected columns: {', '.join(unexpected)}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        bad_rows = (frame.index[frame["date"].isna()] + 2).tolist()[:5]
        raise DatasetError(f"Invalid date values near CSV row(s): {bad_rows}")

    for column in NUMERIC_COLUMNS:
        original = frame[column]
        frame[column] = pd.to_numeric(original, errors="coerce")
        invalid = frame[column].isna()
        if invalid.any():
            bad_rows = (frame.index[invalid] + 2).tolist()[:5]
            raise DatasetError(f"Nonnumeric or missing values in {column} near CSV row(s): {bad_rows}")

    for column in ("city", "ward_id"):
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise DatasetError(f"Dataset contains an empty {column} value")

    unknown_cities = sorted(set(frame["city"].dropna()) - set(CITY_LEVELS))
    if unknown_cities:
        raise DatasetError(
            f"Dataset contains unsupported city value(s): {', '.join(unknown_cities)}"
        )

    if (frame["population_density_sqkm"] <= 0).any():
        raise DatasetError("population_density_sqkm must be greater than zero")
    if (frame["dengue_cases"] < 0).any():
        raise DatasetError("dengue_cases cannot be negative")

    duplicate_mask = frame.duplicated(["date", "ward_id"], keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, ["date", "ward_id"]].head(5)
        examples = ", ".join(f"{row.date.date()} / {row.ward_id}" for row in duplicates.itertuples())
        raise DatasetError(f"Duplicate (date, ward_id) keys: {examples}")

    return frame.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Create lagged weather/vulnerability features and forward case targets.

    All rolling windows and shifts are grouped by ward so no history crosses
    from one neighborhood into another. Target columns are kept on source rows;
    rows without enough history or future observations remain present with NaNs.
    """
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ward_id", "date"]).reset_index(drop=True)

    rolling_specs = (
        ("rainfall_mm", 1, "rainfall_1w_mm", "sum"),
        ("rainfall_mm", 2, "rainfall_2w_mm", "sum"),
        ("rainfall_mm", 4, "rainfall_4w_mm", "sum"),
        ("avg_temp_c", 2, "avg_temp_2w_c", "mean"),
        ("avg_temp_c", 4, "avg_temp_4w_c", "mean"),
        ("relative_humidity_pct", 2, "relative_humidity_2w_pct", "mean"),
        ("relative_humidity_pct", 4, "relative_humidity_4w_pct", "mean"),
        ("dengue_cases", 4, "dengue_cases_4w", "sum"),
    )
    for source, window, output, operation in rolling_specs:
        rolling = frame.groupby("ward_id", sort=False)[source].transform(
            lambda values: getattr(values.rolling(window=window, min_periods=window), operation)()
        )
        frame[output] = rolling

    if hasattr(frame["date"].dt, "day_of_year"):
        day_of_year = frame["date"].dt.day_of_year
    else:
        day_of_year = frame["date"].dt.dayofyear
    week_fraction = day_of_year / 365.25
    frame["week_sin"] = np.sin(2 * np.pi * week_fraction)
    frame["week_cos"] = np.cos(2 * np.pi * week_fraction)
    for city in CITY_LEVELS:
        frame[f"city_{city}"] = frame["city"].eq(city).astype(float)

    grouped = frame.groupby("ward_id", sort=False)
    for horizon in (2, 3, 4):
        frame[f"target_dengue_cases_h{horizon}"] = grouped["dengue_cases"].shift(-horizon)
        frame[f"target_date_h{horizon}"] = grouped["date"].shift(-horizon)

    return frame

