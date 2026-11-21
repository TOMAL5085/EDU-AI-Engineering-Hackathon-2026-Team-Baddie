# WardWatch | Dengue Early-Warning System

WardWatch is a local, reproducible dengue early-warning and epidemiological risk-forecasting system for municipal wards in Bangladesh (Chattogram, Gazipur, Narayanganj). It combines multi-week weather lags, static ward vulnerability indices, and historical case trajectories to forecast dengue counts and risk tiers (+2, +3, and +4 weeks ahead) without future data leakage.

It features a modern **Next.js 16** web dashboard backed by a **FastAPI** high-performance local inference engine, along with a fallback Streamlit dashboard.

---

## Quick Start (Under 3 Minutes)

Requires **Python 3.11+**, **Node.js 18+**, and **Git**.

### Option A: Modern Next.js Dashboard + FastAPI Backend (Recommended)

#### 1. Start the Python FastAPI Backend (Terminal 1)

**macOS / Linux:**
```bash
git clone https://github.com/TOMAL5085/EDU-AI-Engineering-Hackathon-2026-Team-Baddie.git
cd EDU-AI-Engineering-Hackathon-2026-Team-Baddie
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Windows PowerShell:**
```powershell
git clone https://github.com/TOMAL5085/EDU-AI-Engineering-Hackathon-2026-Team-Baddie.git
cd EDU-AI-Engineering-Hackathon-2026-Team-Baddie
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```
*Backend runs at `http://localhost:8000` with interactive API docs at `http://localhost:8000/docs`.*

#### 2. Start the Next.js Frontend (Terminal 2)

**macOS / Linux / Windows:**
```bash
cd EDU-AI-Engineering-Hackathon-2026-Team-Baddie/frontend
npm install
npm run dev
```
*Open `http://localhost:3000` in your browser.*

---

### Option B: Standalone Streamlit App

If you prefer running a single-process dashboard:

**macOS / Linux:**
```bash
cd EDU-AI-Engineering-Hackathon-2026-Team-Baddie
./.venv/bin/python -m streamlit run app.py
```

**Windows PowerShell:**
```powershell
cd EDU-AI-Engineering-Hackathon-2026-Team-Baddie
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*Opens automatically at `http://localhost:8501`.*

---

## Automated Tests & Verification

Run the test suite verifying ward boundary isolation, chronological target purging, risk tier mapping, and inference contracts:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

To retrain and re-evaluate all models across expanding chronological folds:

```bash
python scripts/train_model.py
```

---

## What the Dashboard Shows

1. **Target Selection**:
   - City (Chattogram, Gazipur, Narayanganj).
   - Ward selector (20 wards total).
   - Reference week and forecast horizon (+2, +3, or +4 weeks).
2. **Prominent Risk Tier & Case Forecast**:
   - Four-level risk tier badge: **Low**, **Medium**, **High**, **Critical** (calculated dynamically from training count quantiles).
   - Expected weekly case count and empirical 95% Confidence Interval ($CI = \hat{y} \pm 1.96 \times \text{holdout\_rmse}$).
3. **Ward Vulnerability Metrics**:
   - Population density / km², water-logging hazard index, sanitation score.
4. **Interactive Ward-by-Horizon Risk Heatmap**:
   - Real-time city-wide grid comparing all wards across lead horizons. Clicking any ward updates inference immediately.
5. **Observed Case & Rainfall Trajectory**:
   - Past 52-week timeline tracking historical outbreaks against multi-week rainfall accumulations.
6. **Feature Importance & Drivers Breakdown**:
   - Relative model-wide feature importance weights explaining prediction signals (precipitation lags, temperatures, humidity, vulnerability).
7. **Interactive Weather Scenario Sliders**:
   - Sliders for 1w, 2w, 4w rainfall, temperature, and humidity. Adjustments recompute predictions in **<10ms** without model retraining.

---

## Data & Machine Learning Methodology

### Data Scope
- Bundled dataset: `data/dengue_weekly.csv` (10,440 weekly rows across 20 municipal wards from 2016-01-04 to 2025-12-29).
- Covers Chattogram (CCC), Gazipur (GCC), and Narayanganj (NCC). No Dhaka rows or spatial boundary geometry exist in the source data.
- The target is weekly `dengue_cases`. It is not converted to a per-capita rate because ward population totals are unavailable in the source data.

### Nested Chronological Validation
- **Quarantined Final 20% Holdout**: Dates from **2024-01-01 to 2025-12-29** (105 weeks) were strictly untouched until final evaluation.
- **Expanding Development Folds**: Three expanding chronological windows within the first 80% of dates (2016 to 2023) scored candidate models.
- **Strict Target-Date Purging**: Any row whose forward target date reached into a validation or holdout window was purged from training sets to guarantee zero future data leakage.

### Model Evaluation & Holdout Results

| Lead Week | Selected Model | Validation RMSE | Final Holdout MAE | Final Holdout RMSE | Baseline vs New Status |
|:---:|---|:---:|:---:|:---:|---|
| **+2 weeks** | `voting_et_gb` (Extra Trees + Gradient Boosting) | **2.43** | 1.13 | 2.20 | Validation RMSE improved (2.46 → 2.43); Holdout equivalent |
| **+3 weeks** | `extra_trees_tuned_deep` (300 trees, depth 11, leaf 3) | **2.32** | **1.09** | **2.09** | **Improved across both validation and final holdout** |
| **+4 weeks** | `extra_trees_regularized` (256 trees, depth 10, leaf 4) | **2.74** | **1.24** | **2.35** | **Substantially improved across both validation and final holdout** |

*All metrics are in dengue cases per ward-week. Lower error is better; these describe historical regression error, not accuracy percentages.*

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow diagrams, system diagrams, candidate models evaluated, and compute cost controls.

---

## AI and External-Service Disclosure

- **AI Assistance**: OpenAI Codex and DeepMind Antigravity assisted with architecture, code implementation, test suites, and documentation.
- **Application APIs**: None. WardWatch makes **zero** live weather, geocoding, health-data, LLM, or telemetry API calls.
- **Compute Efficiency**: Runs 100% locally on CPU with zero cloud costs ($0.00). All models are serialized via Joblib and loaded into in-memory caches on startup for sub-10ms inference.
