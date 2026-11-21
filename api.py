"""FastAPI backend for WardWatch Dengue Risk Prediction Engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dengue.data import FEATURE_COLUMNS, build_feature_frame, load_dataset
from dengue.model import HORIZONS, MODEL_PATH, load_model_bundle, predict_row, risk_tier

ROOT = Path(__file__).resolve().parent

FEATURE_HUMAN_LABELS = {
    "population_density_sqkm": "Population density",
    "water_logging_risk_index": "Water-logging risk",
    "sanitation_score": "Sanitation score",
    "rainfall_1w_mm": "Rainfall · 1 week",
    "rainfall_2w_mm": "Rainfall · 2 weeks",
    "rainfall_4w_mm": "Rainfall · 4 weeks",
    "avg_temp_2w_c": "Average temperature · 2 weeks",
    "avg_temp_4w_c": "Average temperature · 4 weeks",
    "relative_humidity_2w_pct": "Humidity · 2 weeks",
    "relative_humidity_4w_pct": "Humidity · 4 weeks",
    "dengue_cases_4w": "Cases · prior 4 weeks",
    "week_sin": "Seasonality · annual cycle sin",
    "week_cos": "Seasonality · annual cycle cos",
    "city_Chattogram": "City · Chattogram",
    "city_Gazipur": "City · Gazipur",
    "city_Narayanganj": "City · Narayanganj",
}


class ModelStore:
    """In-memory cache for data and ML model bundle to guarantee zero per-request latency."""

    raw: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    bundle: dict[str, Any] | None = None
    cities: list[str] = []
    wards_by_city: dict[str, list[str]] = {}
    valid_dates: list[str] = []


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and cache dataset and serialized models on application startup."""
    store.raw = load_dataset()
    store.features = build_feature_frame(store.raw)
    store.bundle = load_model_bundle(MODEL_PATH)
    store.cities = sorted(store.raw["city"].astype(str).unique().tolist())
    store.wards_by_city = {
        city: sorted(store.raw.loc[store.raw["city"] == city, "ward_id"].unique().tolist())
        for city in store.cities
    }
    dates = store.features.dropna(subset=list(FEATURE_COLUMNS))["date"].dt.strftime("%Y-%m-%d").unique()
    store.valid_dates = sorted(dates.tolist())
    yield


app = FastAPI(
    title="WardWatch Dengue Early Warning API",
    version="2.0.0",
    description="High-performance local inference API for ward-level dengue risk forecasting in Bangladesh.",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WeatherScenario(BaseModel):
    rainfall_1w_mm: float = Field(..., ge=0.0, le=500.0, description="Observed/scenario 1-week rainfall (mm)")
    rainfall_2w_mm: float = Field(..., ge=0.0, le=800.0, description="Observed/scenario 2-week rolling rainfall (mm)")
    rainfall_4w_mm: float = Field(..., ge=0.0, le=1500.0, description="Observed/scenario 4-week rolling rainfall (mm)")
    avg_temp_2w_c: float = Field(..., ge=10.0, le=45.0, description="2-week rolling average temperature (°C)")
    avg_temp_4w_c: float = Field(..., ge=10.0, le=45.0, description="4-week rolling average temperature (°C)")
    relative_humidity_2w_pct: float = Field(..., ge=20.0, le=100.0, description="2-week average relative humidity (%)")
    relative_humidity_4w_pct: float = Field(..., ge=20.0, le=100.0, description="4-week average relative humidity (%)")


class PredictRequest(BaseModel):
    city: str
    ward_id: str
    reference_date: str
    horizon: int = Field(2, ge=2, le=4)
    weather_scenario: WeatherScenario | None = None


class CityOverviewRequest(BaseModel):
    city: str
    reference_date: str
    horizon: int = Field(2, ge=2, le=4)
    selected_ward: str | None = None
    weather_scenario: WeatherScenario | None = None


@app.get("/api/health")
def health_check():
    """Health status and data bounds check."""
    if store.bundle is None or store.features is None:
        raise HTTPException(status_code=503, detail="Models or dataset not initialized")
    return {
        "status": "healthy",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_version": store.bundle.get("format_version"),
        "total_records": len(store.raw),
        "total_wards": len(store.raw["ward_id"].unique()),
        "cities": store.cities,
        "date_range": {
            "start": store.valid_dates[0] if store.valid_dates else None,
            "end": store.valid_dates[-1] if store.valid_dates else None,
        },
    }


@app.get("/api/metadata")
def get_metadata():
    """Return available cities, wards, date bounds, and model holdout metrics."""
    if store.bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded")

    return {
        "cities": store.cities,
        "wards_by_city": store.wards_by_city,
        "available_dates": {
            "min": store.valid_dates[0],
            "max": store.valid_dates[-1],
        },
        "horizons": list(HORIZONS),
        "metrics": store.bundle["metrics"],
        "selection": store.bundle.get("selection", {}),
    }


@app.get("/api/wards")
def list_wards(city: str | None = None):
    """Return static ward vulnerabilities and latest observation."""
    if store.features is None:
        raise HTTPException(status_code=503, detail="Features not loaded")

    df = store.features
    if city and city in store.cities:
        df = df[df["city"] == city]

    latest_rows = df.groupby("ward_id").tail(1)
    results = []
    for _, r in latest_rows.iterrows():
        results.append({
            "ward_id": r["ward_id"],
            "city": r["city"],
            "population_density_sqkm": float(r["population_density_sqkm"]),
            "water_logging_risk_index": float(r["water_logging_risk_index"]),
            "sanitation_score": float(r["sanitation_score"]),
            "latest_observed_cases": float(r["dengue_cases"]),
            "latest_rainfall_1w_mm": float(r["rainfall_1w_mm"]),
            "latest_rainfall_2w_mm": float(r["rainfall_2w_mm"]),
            "latest_rainfall_4w_mm": float(r["rainfall_4w_mm"]),
            "latest_avg_temp_2w_c": float(r["avg_temp_2w_c"]),
            "latest_relative_humidity_2w_pct": float(r["relative_humidity_2w_pct"]),
        })
    return results


@app.get("/api/ward-history")
def get_ward_history(ward_id: str = Query(...), weeks: int = Query(52, ge=12, le=104)):
    """Return past observed cases and multi-week rainfall history for chart plotting."""
    if store.features is None:
        raise HTTPException(status_code=503, detail="Features not loaded")

    ward_data = store.features.loc[store.features["ward_id"] == ward_id]
    if ward_data.empty:
        raise HTTPException(status_code=404, detail=f"Ward {ward_id} not found")

    recent = ward_data.tail(weeks)
    return [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "dengue_cases": float(row["dengue_cases"]),
            "rainfall_1w_mm": float(row["rainfall_1w_mm"]) if pd.notna(row["rainfall_1w_mm"]) else 0.0,
            "rainfall_2w_mm": float(row["rainfall_2w_mm"]) if pd.notna(row["rainfall_2w_mm"]) else 0.0,
            "rainfall_4w_mm": float(row["rainfall_4w_mm"]) if pd.notna(row["rainfall_4w_mm"]) else 0.0,
            "avg_temp_2w_c": float(row["avg_temp_2w_c"]) if pd.notna(row["avg_temp_2w_c"]) else 27.0,
            "relative_humidity_2w_pct": float(row["relative_humidity_2w_pct"]) if pd.notna(row["relative_humidity_2w_pct"]) else 70.0,
        }
        for _, row in recent.iterrows()
    ]


@app.post("/api/predict")
def predict_risk(payload: PredictRequest):
    """Run dynamic ML inference for a ward and reference date with scenario support."""
    if store.bundle is None or store.features is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    horizon = payload.horizon
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail=f"Invalid horizon {horizon}. Must be 2, 3, or 4.")

    ref_dt = pd.to_datetime(payload.reference_date)
    candidates = store.features.loc[
        (store.features["ward_id"] == payload.ward_id) & (store.features["date"] <= ref_dt)
    ].dropna(subset=list(FEATURE_COLUMNS))

    if candidates.empty:
        # Fall back to earliest or latest valid row for this ward
        candidates = store.features.loc[store.features["ward_id"] == payload.ward_id].dropna(subset=list(FEATURE_COLUMNS))
        if candidates.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ward {payload.ward_id}")

    base_row = candidates.iloc[-1].copy()
    observed_date = base_row["date"]
    forecast_date = observed_date + pd.Timedelta(weeks=horizon)

    # Apply weather scenario overrides if supplied
    eval_row = base_row.copy()
    if payload.weather_scenario:
        scenario_dict = payload.weather_scenario.dict()
        for k, v in scenario_dict.items():
            if k in eval_row:
                eval_row[k] = v

    prediction = predict_row(store.bundle, eval_row, horizon)
    expected_cases = float(prediction["expected_cases"])

    # Compute empirical 95% confidence interval using holdout RMSE
    holdout_rmse = float(store.bundle["metrics"][horizon]["rmse"])
    ci_margin = 1.96 * holdout_rmse
    ci_lower = max(0.0, round(expected_cases - ci_margin, 2))
    ci_upper = round(expected_cases + ci_margin, 2)

    # Extract model feature importances
    model = store.bundle["models"][horizon]
    raw_importances = getattr(model, "feature_importances_", None)
    drivers = []
    if raw_importances is not None:
        for feat_name, imp in zip(store.bundle["feature_columns"], raw_importances):
            drivers.append({
                "feature": feat_name,
                "label": FEATURE_HUMAN_LABELS.get(feat_name, feat_name),
                "importance": float(imp),
                "value": float(eval_row[feat_name]) if feat_name in eval_row else 0.0,
            })
        drivers.sort(key=lambda d: d["importance"], reverse=True)

    selected_info = store.bundle.get("selection", {}).get(horizon, {})

    return {
        "city": payload.city,
        "ward_id": payload.ward_id,
        "horizon_weeks": horizon,
        "reference_date": observed_date.strftime("%Y-%m-%d"),
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "expected_cases": round(expected_cases, 2),
        "predicted_cases": round(expected_cases, 2),
        "tier": prediction["tier"],
        "confidence_interval": {
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "cutoffs": prediction["cutoffs"],
        "model_name": selected_info.get("selected", {}).get("name", "Tree Ensemble"),
        "validation_rmse": float(selected_info.get("selected", {}).get("mean_rmse", 0.0)),
        "holdout_rmse": float(store.bundle["metrics"][horizon]["rmse"]),
        "holdout_mae": float(store.bundle["metrics"][horizon]["mae"]),
        "drivers": drivers[:8],
        "defaults": {
            "rainfall_1w_mm": float(base_row["rainfall_1w_mm"]),
            "rainfall_2w_mm": float(base_row["rainfall_2w_mm"]),
            "rainfall_4w_mm": float(base_row["rainfall_4w_mm"]),
            "avg_temp_2w_c": float(base_row["avg_temp_2w_c"]),
            "avg_temp_4w_c": float(base_row["avg_temp_4w_c"]),
            "relative_humidity_2w_pct": float(base_row["relative_humidity_2w_pct"]),
            "relative_humidity_4w_pct": float(base_row["relative_humidity_4w_pct"]),
        },
    }


@app.post("/api/city-overview")
def city_overview(payload: CityOverviewRequest):
    """Return ward-by-horizon risk grid for the interactive city risk heatmap."""
    if store.bundle is None or store.features is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    city_data = store.features.loc[store.features["city"] == payload.city]
    if city_data.empty:
        raise HTTPException(status_code=404, detail=f"City {payload.city} not found")

    ref_dt = pd.to_datetime(payload.reference_date)
    available = city_data.loc[city_data["date"] <= ref_dt].dropna(subset=list(FEATURE_COLUMNS))
    if available.empty:
        available = city_data.dropna(subset=list(FEATURE_COLUMNS))

    latest_per_ward = available.groupby("ward_id", sort=True).tail(1)
    grid = []

    scenario_dict = payload.weather_scenario.dict() if payload.weather_scenario else {}

    for _, base in latest_per_ward.iterrows():
        row = base.copy()
        if payload.selected_ward and row["ward_id"] == payload.selected_ward and scenario_dict:
            for k, v in scenario_dict.items():
                if k in row:
                    row[k] = v

        for h in HORIZONS:
            pred = predict_row(store.bundle, row, h)
            grid.append({
                "ward_id": row["ward_id"],
                "horizon": h,
                "tier": pred["tier"],
                "expected_cases": round(float(pred["expected_cases"]), 2),
            })

    return {
        "city": payload.city,
        "reference_date": payload.reference_date,
        "grid": grid,
    }


@app.get("/api/model-info")
def model_info():
    """Return exhaustive model validation architecture and candidate ranking data."""
    if store.bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded")

    return {
        "format_version": store.bundle.get("format_version"),
        "trained_at_utc": store.bundle.get("trained_at_utc"),
        "horizons": store.bundle.get("horizons"),
        "feature_columns": store.bundle.get("feature_columns"),
        "metrics": store.bundle.get("metrics"),
        "selection": store.bundle.get("selection"),
        "model_config": store.bundle.get("model_config"),
        "data_summary": store.bundle.get("data_summary"),
    }
