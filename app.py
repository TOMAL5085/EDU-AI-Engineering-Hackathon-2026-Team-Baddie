"""WardWatch: a local dengue early-warning dashboard."""

from __future__ import annotations

from html import escape as html_escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dengue.data import FEATURE_COLUMNS, build_feature_frame, load_dataset
from dengue.model import HORIZONS, MODEL_PATH, load_model_bundle, predict_row

TIER_COLORS = {
    "Low": "#77bf8b",
    "Medium": "#b8e63f",
    "High": "#eda64b",
    "Critical": "#ea625e",
}
TIER_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
FEATURE_LABELS = {
    "population_density_sqkm": "Population density",
    "water_logging_risk_index": "Water-logging risk",
    "sanitation_score": "Sanitation score",
    "rainfall_2w_mm": "Rainfall · 2 weeks",
    "rainfall_4w_mm": "Rainfall · 4 weeks",
    "avg_temp_2w_c": "Average temperature · 2 weeks",
    "avg_temp_4w_c": "Average temperature · 4 weeks",
    "relative_humidity_2w_pct": "Humidity · 2 weeks",
    "relative_humidity_4w_pct": "Humidity · 4 weeks",
    "dengue_cases_4w": "Cases · prior 4 weeks",
    "week_sin": "Seasonality · annual cycle",
    "week_cos": "Seasonality · annual cycle",
    "city_Chattogram": "City · Chattogram",
    "city_Gazipur": "City · Gazipur",
    "city_Narayanganj": "City · Narayanganj",
}


st.set_page_config(
    page_title="WardWatch | Dengue Early Warning",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #122014; --muted: #65705f; --deep: #0d1a0e; --lime: #c1f23a; --line: #dfe7d6; }
      .stApp { background: #f2f5ee; color: var(--ink); }
      [data-testid="stHeader"] { background: rgba(242,245,238,.92); }
      [data-testid="stSidebar"] { background: #0d1a0e; border-right: 1px solid #283b26; }
      [data-testid="stSidebar"] * { color: #eff5e7; }
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] input { background: #172719; border-color: #3a4c36; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #cbd6c2; }
      .block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3.5rem; }
      .hero { background: radial-gradient(ellipse at 82% 0%, #294d1b 0, transparent 40%), linear-gradient(120deg,#0a150b,#142611 72%,#203817); border-radius: 24px; padding: 2rem 2.2rem 1.8rem; color: #f5f8ef; margin-bottom: 1.4rem; border: 1px solid #31472a; box-shadow: 0 18px 48px rgba(15,32,13,.14); }
      .hero-eyebrow { color: var(--lime); font-size: .75rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }
      .hero h1 { color: #fff; font-size: clamp(2rem,4vw,3.3rem); line-height: 1.06; margin: .6rem 0 .5rem; letter-spacing: -.04em; }
      .hero p { color: #d4dfc9; font-size: 1rem; max-width: 780px; margin: 0; }
      .hero-chip { display: inline-block; margin-top: 1.2rem; background: var(--lime); color: #14200b; border-radius: 999px; padding: .4rem .8rem; font-size: .78rem; font-weight: 800; }
      .section-kicker { color: #6b8d32; font-size: .72rem; font-weight: 850; letter-spacing: .14em; text-transform: uppercase; margin: .1rem 0 .2rem; }
      .section-title { color: #172317; font-weight: 780; letter-spacing: -.025em; margin: 0 0 .9rem; }
      .risk-panel { border-radius: 18px; padding: 1.25rem 1.45rem; color: #f5f8ef; background: #101d11; border: 1px solid #263c23; min-height: 180px; }
      .risk-panel .label { font-size: .72rem; color: #c0ceba; text-transform: uppercase; letter-spacing: .14em; font-weight: 800; }
      .risk-panel .tier { font-size: 2.55rem; line-height: 1.05; font-weight: 850; margin: .45rem 0 .25rem; }
      .risk-panel .summary { color: #d6dfce; font-size: .94rem; }
      .risk-panel .dot { display: inline-block; width: .62rem; height: .62rem; border-radius: 50%; margin-right: .45rem; vertical-align: .02rem; }
      .soft-card { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.2rem; }
      .soft-card h3 { margin: 0 0 .45rem; font-size: 1rem; color: #182318; }
      .soft-card p { margin: 0; color: var(--muted); font-size: .88rem; }
      .stMetric { background: #fff; border: 1px solid var(--line); border-radius: 15px; padding: .9rem 1rem; box-shadow: 0 4px 16px rgba(28,47,23,.035); }
      [data-testid="stMetricLabel"] p { color: #687364; font-weight: 650; }
      [data-testid="stMetricValue"] { color: #142216; }
      .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
      .stTabs [data-baseweb="tab"] { background: #e7eddf; border-radius: 999px; padding: .35rem .9rem; }
      .stTabs [aria-selected="true"] { background: #c1f23a !important; color: #17220d !important; }
      div[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: .35rem; }
      .stAlert { border-radius: 14px; }
      .footnote { color: #6d7768; font-size: .8rem; }
      @media (max-width: 740px) { .hero { padding: 1.4rem; } .block-container { padding-top: .8rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading weekly ward data…")
def cached_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner="Building weather lag features…")
def cached_feature_frame() -> pd.DataFrame:
    return build_feature_frame(cached_dataset())


@st.cache_resource(show_spinner="Loading local prediction models…")
def cached_model_bundle(model_path: str, modified_ns: int) -> dict:
    return load_model_bundle(model_path)


def as_float(value: object, default: float = 0.0) -> float:
    result = float(value) if pd.notna(value) else default
    return result


def capped_slider_default(value: float, upper: float) -> float:
    return round(min(max(float(value), 0.0), upper), 1)


def selected_row(features: pd.DataFrame, ward_id: str, reference_date: pd.Timestamp) -> pd.Series:
    candidates = features.loc[(features["ward_id"] == ward_id) & (features["date"] <= reference_date)]
    candidates = candidates.dropna(subset=list(FEATURE_COLUMNS))
    if candidates.empty:
        raise ValueError("This ward has no complete weather history by the selected reference week.")
    return candidates.iloc[-1].copy()


def make_risk_heatmap(grid: pd.DataFrame, city: str) -> go.Figure:
    wards = list(dict.fromkeys(grid["ward_id"].tolist()))
    horizons = list(HORIZONS)
    lookup = {(row.ward_id, row.horizon): row for row in grid.itertuples()}
    values = [[TIER_RANK[lookup[(ward, horizon)].tier] for horizon in horizons] for ward in wards]
    labels = [[lookup[(ward, horizon)].tier for horizon in horizons] for ward in wards]
    hover = [
        [
            f"{ward}<br>Week +{horizon}<br>{lookup[(ward, horizon)].tier}<br>"
            f"{lookup[(ward, horizon)].expected_cases:.1f} cases / week"
            for horizon in horizons
        ]
        for ward in wards
    ]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=[f"Week +{horizon}" for horizon in horizons],
            y=wards,
            text=labels,
            customdata=hover,
            colorscale=[
                [0.00, TIER_COLORS["Low"]],
                [0.249, TIER_COLORS["Low"]],
                [0.25, TIER_COLORS["Medium"]],
                [0.499, TIER_COLORS["Medium"]],
                [0.50, TIER_COLORS["High"]],
                [0.749, TIER_COLORS["High"]],
                [0.75, TIER_COLORS["Critical"]],
                [1.00, TIER_COLORS["Critical"]],
            ],
            zmin=0,
            zmax=3,
            showscale=False,
            texttemplate="%{text}",
            hovertemplate="%{customdata}<extra></extra>",
            xgap=5,
            ygap=4,
        )
    )
    fig.update_layout(
        title=f"Ward outlook · {city}",
        height=max(300, 46 * len(wards) + 100),
        margin=dict(l=12, r=12, t=55, b=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#203020", family="Inter, Arial, sans-serif"),
        xaxis=dict(side="top", ticks="", showgrid=False),
        yaxis=dict(autorange="reversed", ticks="", showgrid=False),
    )
    return fig


def make_history_chart(history: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=history["date"],
            y=history["dengue_cases"],
            name="Reported cases",
            marker_color="#608b43",
            opacity=0.82,
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["rainfall_4w_mm"],
            name="Rainfall · rolling 4 weeks",
            mode="lines",
            line=dict(color="#a2c936", width=3),
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title="Observed cases and recent rainfall",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(color="#203020", family="Inter, Arial, sans-serif"),
        hovermode="x unified",
    )
    figure.update_yaxes(title_text="Cases / week", secondary_y=False, rangemode="tozero", gridcolor="#edf1e9")
    figure.update_yaxes(title_text="Rainfall / 4 weeks (mm)", secondary_y=True, rangemode="tozero", showgrid=False)
    return figure


def make_importance_chart(bundle: dict, horizon: int) -> go.Figure:
    model = bundle["models"][horizon]
    importance = pd.DataFrame(
        {"feature": bundle["feature_columns"], "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).head(7)
    importance["label"] = importance["feature"].map(FEATURE_LABELS).fillna(importance["feature"])
    figure = go.Figure(
        go.Bar(
            x=importance["importance"],
            y=importance["label"],
            orientation="h",
            marker_color="#6b973f",
            hovertemplate="%{y}<br>Relative model importance: %{x:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Model-wide feature importance",
        height=310,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#203020", family="Inter, Arial, sans-serif"),
        xaxis=dict(title="Relative importance", gridcolor="#edf1e9"),
        yaxis=dict(autorange="reversed"),
    )
    return figure


def build_city_grid(
    city_features: pd.DataFrame,
    city: str,
    reference_date: pd.Timestamp,
    bundle: dict,
    selected_ward: str,
    scenario: dict[str, float],
) -> pd.DataFrame:
    city_data = city_features.loc[city_features["city"] == city]
    available = city_data.loc[city_data["date"] <= reference_date]
    available = available.dropna(subset=list(FEATURE_COLUMNS))
    if available.empty:
        raise ValueError(f"No data for {city} by {reference_date.date()}.")
    latest = available.groupby("ward_id", sort=True).tail(1)
    results: list[dict[str, object]] = []
    for _, base in latest.iterrows():
        row = base.copy()
        if row["ward_id"] == selected_ward:
            for name, value in scenario.items():
                row[name] = value
        for horizon in HORIZONS:
            prediction = predict_row(bundle, row, horizon)
            results.append(
                {
                    "ward_id": row["ward_id"],
                    "horizon": horizon,
                    "tier": prediction["tier"],
                    "expected_cases": prediction["expected_cases"],
                }
            )
    return pd.DataFrame(results)


def main() -> None:
    try:
        data = cached_dataset()
        features = cached_feature_frame()
        bundle = cached_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"WardWatch could not load its local data or model: {exc}")
        st.stop()

    latest_date = pd.Timestamp(data["date"].max())
    valid_reference_dates = features.dropna(subset=list(FEATURE_COLUMNS))["date"]
    min_date = pd.Timestamp(valid_reference_dates.min()).date()
    cities = sorted(data["city"].astype(str).unique().tolist())

    st.sidebar.markdown("## WardWatch")
    st.sidebar.caption("DENGUE EARLY WARNING · LOCAL MODEL")
    city = st.sidebar.selectbox("City", cities, index=cities.index("Chattogram") if "Chattogram" in cities else 0)
    city_wards = sorted(data.loc[data["city"] == city, "ward_id"].unique().tolist())
    ward_id = st.sidebar.selectbox("Ward", city_wards)
    reference_date = st.sidebar.date_input(
        "Reference week",
        value=latest_date.date(),
        min_value=min_date,
        help="Select a historical week. Future dates use the latest source row as a scenario baseline.",
    )
    horizon = st.sidebar.radio("Forecast horizon", HORIZONS, horizontal=True, format_func=lambda weeks: f"+{weeks} weeks")

    reference_ts = pd.Timestamp(reference_date)
    selected = selected_row(features, ward_id, reference_ts)
    observed_week = pd.Timestamp(selected["date"])
    scenario_defaults = {
        "rainfall_2w_mm": as_float(selected["rainfall_2w_mm"]),
        "rainfall_4w_mm": as_float(selected["rainfall_4w_mm"]),
        "avg_temp_2w_c": as_float(selected["avg_temp_2w_c"], 27.0),
        "avg_temp_4w_c": as_float(selected["avg_temp_4w_c"], 27.0),
        "relative_humidity_2w_pct": as_float(selected["relative_humidity_2w_pct"], 70.0),
        "relative_humidity_4w_pct": as_float(selected["relative_humidity_4w_pct"], 70.0),
    }
    st.sidebar.markdown("### Weather scenario")
    st.sidebar.caption("Defaults use the selected ward's observed rolling values.")
    rainfall_2w = st.sidebar.slider(
        "Rainfall · 2 weeks (mm)", 0.0, 600.0, capped_slider_default(scenario_defaults["rainfall_2w_mm"], 600), 5.0
    )
    rainfall_4w_default = min(max(scenario_defaults["rainfall_4w_mm"], rainfall_2w), 1_200.0)
    rainfall_4w = st.sidebar.slider(
        "Rainfall · 4 weeks (mm)", float(rainfall_2w), 1_200.0, rainfall_4w_default, 10.0
    )
    scenario = {
        "rainfall_2w_mm": rainfall_2w,
        "rainfall_4w_mm": rainfall_4w,
        "avg_temp_2w_c": st.sidebar.slider(
            "Average temperature · 2 weeks (°C)", 15.0, 40.0, capped_slider_default(scenario_defaults["avg_temp_2w_c"], 40), 0.5
        ),
        "avg_temp_4w_c": st.sidebar.slider(
            "Average temperature · 4 weeks (°C)", 15.0, 40.0, capped_slider_default(scenario_defaults["avg_temp_4w_c"], 40), 0.5
        ),
        "relative_humidity_2w_pct": st.sidebar.slider(
            "Humidity · 2 weeks (%)", 35.0, 100.0, capped_slider_default(scenario_defaults["relative_humidity_2w_pct"], 100), 1.0
        ),
        "relative_humidity_4w_pct": st.sidebar.slider(
            "Humidity · 4 weeks (%)", 35.0, 100.0, capped_slider_default(scenario_defaults["relative_humidity_4w_pct"], 100), 1.0
        ),
    }

    scenario_row = selected.copy()
    for name, value in scenario.items():
        scenario_row[name] = value
    prediction = predict_row(bundle, scenario_row, horizon)
    forecast_origin = reference_ts if reference_ts.date() > latest_date.date() else observed_week
    prediction_week = forecast_origin + pd.Timedelta(weeks=horizon)
    tier = prediction["tier"]
    tier_color = TIER_COLORS[tier]

    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-eyebrow">Bangladesh · ward-level outlook</div>
          <h1>See the signal<br>before cases rise.</h1>
          <p>Compare local weather and vulnerability signals to plan earlier dengue prevention across the wards in your dataset.</p>
          <span class="hero-chip">{data['ward_id'].nunique()} wards · {len(cities)} cities · data through {latest_date:%d %b %Y}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if reference_ts.date() > latest_date.date():
        st.info(
            f"Scenario week is after the latest observation ({latest_date:%d %b %Y}). "
            "The app uses the latest available ward record as its baseline; weather controls are hypothetical."
        )
    elif prediction_week.date() > latest_date.date():
        st.info(
            f"The source data ends on {latest_date:%d %b %Y}. This forecast extends beyond observed cases and weather; "
            "it uses the latest ward record plus the editable scenario inputs."
        )
    elif observed_week.date() < reference_ts.date():
        st.info(f"Using the most recent source observation on or before your selection: {observed_week:%d %b %Y}.")

    st.markdown('<div class="section-kicker">Selected ward outlook</div>', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{html_escape(str(city))} <span style="color:#779238">/</span> {html_escape(str(ward_id))}</h2>',
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.05, 2.0], gap="large")
    with top_left:
        st.markdown(
            f"""
            <div class="risk-panel">
              <div class="label">Week +{horizon} · {prediction_week:%d %b %Y}</div>
              <div class="tier" style="color:{tier_color}"><span class="dot" style="background:{tier_color}"></span>{tier} risk</div>
              <div class="summary">Estimated {prediction['expected_cases']:.1f} cases per week</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Water-log risk", f"{selected['water_logging_risk_index']:.2f}", help="Ward-level source index")
        c2.metric("Sanitation", f"{selected['sanitation_score']:.2f}", help="Ward-level source index")
        c3.metric("Density / km²", f"{selected['population_density_sqkm']:,.0f}", help="Residents per km² in source data")
        st.markdown(
            f"<div class='soft-card'><h3>What the estimate uses</h3><p>Observed {observed_week:%d %b %Y} ward conditions, a {horizon}-week lead, rolling rainfall and climate values, recent dengue cases, and ward vulnerability. Scenario controls change the weather inputs without retraining.</p></div>",
            unsafe_allow_html=True,
        )

    metric = bundle["metrics"][horizon]
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Forecast cases / week", f"{prediction['expected_cases']:.1f}")
    k2.metric("MAE / week", f"{metric['mae']:.2f}", help=f"Holdout test dates start {metric['holdout_start']}")
    k3.metric("RMSE / week", f"{metric['rmse']:.2f}", help=f"Error over {metric['holdout_rows']:,} chronological holdout rows")
    k4.metric("Holdout rows", f"{metric['holdout_rows']:,}", help=f"Chronological evaluation from {metric['holdout_start']}")

    st.markdown("<br>", unsafe_allow_html=True)
    ward_tab, drivers_tab, method_tab = st.tabs(["Ward overview", "Weather & drivers", "Model notes"])
    with ward_tab:
        try:
            city_features = features.loc[features["city"] == city]
            grid = build_city_grid(city_features, city, reference_ts, bundle, ward_id, scenario)
            st.plotly_chart(make_risk_heatmap(grid, city), width="stretch", config={"displayModeBar": False})
            with st.expander("Ward estimates by horizon"):
                table = grid.copy()
                table["horizon"] = table["horizon"].map(lambda weeks: f"Week +{weeks}")
                table["expected_cases"] = table["expected_cases"].map(lambda value: f"{value:.1f}")
                st.dataframe(
                    table.rename(columns={"ward_id": "Ward", "horizon": "Horizon", "tier": "Tier", "expected_cases": "Cases / week"}),
                    width="stretch",
                    hide_index=True,
                )
        except Exception as exc:
            st.warning(f"Ward comparison is unavailable for this reference week: {exc}")

        history = features.loc[(features["ward_id"] == ward_id) & (features["date"] <= observed_week)].tail(52)
        st.plotly_chart(make_history_chart(history), width="stretch", config={"displayModeBar": False})

    with drivers_tab:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.plotly_chart(make_importance_chart(bundle, horizon), width="stretch", config={"displayModeBar": False})
            st.caption("Relative feature importance across this horizon's trained model. It describes model behavior, not causal effects.")
        with right:
            st.markdown("### Weather inputs for this scenario")
            w1, w2 = st.columns(2)
            w1.metric("Rainfall · 2 weeks", f"{scenario['rainfall_2w_mm']:.0f} mm")
            w2.metric("Rainfall · 4 weeks", f"{scenario['rainfall_4w_mm']:.0f} mm")
            w1.metric("Average temp · 2 weeks", f"{scenario['avg_temp_2w_c']:.1f} °C")
            w2.metric("Average temp · 4 weeks", f"{scenario['avg_temp_4w_c']:.1f} °C")
            w1.metric("Humidity · 2 weeks", f"{scenario['relative_humidity_2w_pct']:.0f}%")
            w2.metric("Humidity · 4 weeks", f"{scenario['relative_humidity_4w_pct']:.0f}%")
            cutoffs = prediction["cutoffs"]
            st.markdown(
                f"<div class='soft-card'><h3>Tier boundaries for week +{horizon}</h3><p>Medium ≥ {cutoffs['medium']:.1f} · High ≥ {cutoffs['high']:.1f} · Critical ≥ {cutoffs['critical']:.1f} cases per ward per week.</p></div>",
                unsafe_allow_html=True,
            )

    with method_tab:
        summary = bundle["data_summary"]
        st.markdown("### Chronological validation")
        st.write(
            f"The first 80% of weekly dates selected and trained the model. The final 20% began on "
            f"**{metric['holdout_start']}** and supplied **{metric['holdout_rows']:,}** ward-week rows for evaluation. "
            f"Training rows whose target date reached the holdout were removed, and this final period was not used to choose the model."
        )
        selection = bundle.get("selection", {}).get(horizon)
        if selection:
            selected_model = selection["selected"]
            st.markdown("### Model selection")
            st.write(
                f"Selected **{selected_model['estimator']}** configuration "
                f"**{selected_model['name']}** using the lowest mean RMSE "
                f"(**{selected_model['mean_rmse']:.2f} cases / ward-week**) across "
                f"**{len(selected_model['folds'])}** earlier chronological validation windows."
            )
            with st.expander("Candidate validation results"):
                candidates = pd.DataFrame(
                    [
                        {
                            "Candidate": candidate["name"],
                            "Estimator": candidate["estimator"],
                            "Mean RMSE": candidate["mean_rmse"],
                            "Mean MAE": candidate["mean_mae"],
                        }
                        for candidate in selection["candidates"]
                    ]
                )
                st.dataframe(
                    candidates,
                    column_config={
                        "Mean RMSE": st.column_config.NumberColumn(format="%.2f"),
                        "Mean MAE": st.column_config.NumberColumn(format="%.2f"),
                    },
                    width="stretch",
                    hide_index=True,
                    alt="Candidate forecasting models ranked by chronological validation error.",
                )
        else:
            st.caption("This model bundle predates chronological candidate selection.")
        st.markdown("### Data coverage")
        st.write(
            f"**{summary['rows']:,}** rows · **{summary['wards']}** wards · "
            f"{', '.join(summary['cities'])} · {summary['first_date']} to {summary['latest_date']}."
        )
        st.markdown("### Interpretation")
        st.write(
            "The model estimates weekly dengue case counts for one selected lead week. "
            "Risk tiers are based on training-period case-count quantiles. No weather API is called, "
            "and the source ends in 2025, so later dates are scenario projections from the latest row. "
            "Use this prototype to compare signals and plan review; it is not an official alert."
        )
        st.caption(
            f"Model artifact trained {bundle['trained_at_utc'][:10]} · "
            f"holdout starts {summary['holdout_start']} · local scikit-learn {bundle['versions']['scikit_learn']}"
        )

    st.markdown(
        "<p class='footnote'>WardWatch · Local prototype · No live weather feed or external prediction API · Data source: supplied hackathon dataset export.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

