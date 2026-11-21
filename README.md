# WardWatch

WardWatch is a local dengue early-warning dashboard for comparing ward-level signals two to four weeks ahead. It combines recent weather, ward vulnerability, and reported case history, then shows a predicted tier, city-wide ward comparison, and the model's validation results.

## Run locally

Requires Python 3.11 or newer and Git.

### Windows PowerShell

Copy and paste these commands into PowerShell:

```powershell
git clone https://github.com/TOMAL5085/EDU-AI-Engineering-Hackathon-2026-Team-Baddie.git
cd EDU-AI-Engineering-Hackathon-2026-Team-Baddie
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS Terminal

Copy and paste these commands into Terminal:

```bash
git clone https://github.com/TOMAL5085/EDU-AI-Engineering-Hackathon-2026-Team-Baddie.git
cd EDU-AI-Engineering-Hackathon-2026-Team-Baddie
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py
```

These commands use the virtual environment's Python directly, so you do not need to activate it. The repository includes the prebuilt model and dataset, so a normal launch does not train a model or call a weather service. The local app opens at the address printed by Streamlit, usually `http://localhost:8501`.

To retrain after changing the CSV:

```bash
python scripts/train_model.py
```

## What the dashboard shows

- City, ward, reference week, and a week +2, +3, or +4 forecast horizon.
- Editable two- and four-week rainfall, temperature, and humidity scenario inputs.
- Expected weekly case count, mapped to Low, Medium, High, or Critical tiers.
- Ward-by-horizon risk heatmap, recent case/rainfall history, model-wide feature importance, and holdout MAE/RMSE.

Scenario controls change inference inputs immediately; they do not retrain the model. Feature importance describes the fitted model across all wards and should not be read as a causal or ward-specific explanation.

## Data and model

`data/dengue_weekly.csv` was extracted from the supplied `hackathon_preli_dataset - bangladesh_dengue_large_dataset.pdf` Google Sheets export. The bundled file has 10,440 weekly rows for 20 wards in Chattogram, Gazipur, and Narayanganj, dated 2016-01-04 through 2025-12-29. The supplied data has no Dhaka rows or ward boundary geometry.

The app creates grouped two- and four-week weather averages/totals, a four-week case-history feature, static ward vulnerability features, and annual seasonality. For each forecast horizon, it compares compact Random Forest and Extra Trees configurations across three earlier expanding chronological validation windows, then chooses the lowest validation RMSE. The final chronological holdout begins on 2024-01-01 and is not used to select a model; training rows whose target date reached the holdout are purged. Population density remains a predictor; it is not used as a substitute for ward population.

| Lead week | Selected model | Validation RMSE | Final holdout MAE | Final holdout RMSE |
|---:|---|---:|---:|---:|
| +2 | Extra Trees · regularized | 2.46 | 1.11 | 2.18 |
| +3 | Extra Trees · regularized | 2.36 | 1.14 | 2.18 |
| +4 | Random Forest · baseline | 2.85 | 1.30 | 2.49 |

All values are cases per ward-week. Lower MAE and RMSE values are better; they are error measurements, not accuracy percentages. These metrics describe historical periods, not a guarantee of future performance. Use the dashboard as a planning prototype; it does not issue official health alerts.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow, model selection, validation split, cost controls, and system limits.

## AI and external-service disclosure

- **AI assistant:** OpenAI Codex assisted with the architecture, implementation, and documentation.
- **Application APIs:** None. WardWatch makes no live weather, geocoding, health-data, LLM, or telemetry API calls.
- **Development services:** Python dependencies are installed from PyPI using `pip`. The dataset was supplied as a PDF export; the app uses its bundled CSV copy at runtime.

