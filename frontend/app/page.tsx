"use client";

import { useEffect, useState, useMemo } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Calendar,
  CheckCircle2,
  ChevronDown,
  CloudRain,
  Cpu,
  Database,
  Droplets,
  ExternalLink,
  Flame,
  Info,
  Layers,
  MapPin,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Sliders,
  Thermometer,
  TrendingUp,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface WeatherScenario {
  rainfall_1w_mm: number;
  rainfall_2w_mm: number;
  rainfall_4w_mm: number;
  avg_temp_2w_c: number;
  avg_temp_4w_c: number;
  relative_humidity_2w_pct: number;
  relative_humidity_4w_pct: number;
}

interface PredictionData {
  city: string;
  ward_id: string;
  horizon_weeks: number;
  reference_date: string;
  forecast_date: string;
  expected_cases: number;
  predicted_cases: number;
  tier: "Low" | "Medium" | "High" | "Critical";
  confidence_interval: {
    lower: number;
    upper: number;
  };
  cutoffs: {
    medium: number;
    high: number;
    critical: number;
  };
  model_name: string;
  validation_rmse: number;
  holdout_rmse: number;
  holdout_mae: number;
  drivers: Array<{
    feature: string;
    label: string;
    importance: number;
    value: number;
  }>;
  defaults: WeatherScenario;
}

interface WardHistoryItem {
  date: string;
  dengue_cases: number;
  rainfall_1w_mm: number;
  rainfall_2w_mm: number;
  rainfall_4w_mm: number;
  avg_temp_2w_c: number;
  relative_humidity_2w_pct: number;
}

interface GridItem {
  ward_id: string;
  horizon: number;
  tier: "Low" | "Medium" | "High" | "Critical";
  expected_cases: number;
}

interface WardInfo {
  ward_id: string;
  city: string;
  population_density_sqkm: number;
  water_logging_risk_index: number;
  sanitation_score: number;
}

export default function WardWatchDashboard() {
  // Navigation & filter state
  const [cities, setCities] = useState<string[]>(["Chattogram", "Gazipur", "Narayanganj"]);
  const [wardsByCity, setWardsByCity] = useState<Record<string, string[]>>({
    Chattogram: [
      "CCC_Ward_01", "CCC_Ward_02", "CCC_Ward_03", "CCC_Ward_04", "CCC_Ward_05",
      "CCC_Ward_06", "CCC_Ward_07", "CCC_Ward_08", "CCC_Ward_09", "CCC_Ward_10"
    ],
    Gazipur: ["GCC_Ward_01", "GCC_Ward_02", "GCC_Ward_03", "GCC_Ward_04", "GCC_Ward_05"],
    Narayanganj: ["NCC_Ward_01", "NCC_Ward_02", "NCC_Ward_03", "NCC_Ward_04", "NCC_Ward_05"],
  });

  const [selectedCity, setSelectedCity] = useState("Chattogram");
  const [selectedWard, setSelectedWard] = useState("CCC_Ward_01");
  const [referenceDate, setReferenceDate] = useState("2025-12-29");
  const [horizon, setHorizon] = useState<2 | 3 | 4>(2);
  const [activeTab, setActiveTab] = useState<"overview" | "drivers" | "architecture">("overview");

  // Backend state
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [scenarioOpen, setScenarioOpen] = useState(true);

  // Data state
  const [scenario, setScenario] = useState<WeatherScenario | null>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [history, setHistory] = useState<WardHistoryItem[]>([]);
  const [cityGrid, setCityGrid] = useState<GridItem[]>([]);
  const [wardMetadata, setWardMetadata] = useState<WardInfo[]>([]);

  // Selected ward static metadata
  const currentWardInfo = useMemo(() => {
    return (
      wardMetadata.find((w) => w.ward_id === selectedWard) || {
        ward_id: selectedWard,
        city: selectedCity,
        population_density_sqkm: 24500,
        water_logging_risk_index: 0.68,
        sanitation_score: 0.52,
      }
    );
  }, [wardMetadata, selectedWard, selectedCity]);

  // Initial metadata fetch
  useEffect(() => {
    async function fetchMeta() {
      try {
        const res = await fetch(`${API_BASE}/api/metadata`);
        if (res.ok) {
          const data = await res.json();
          if (data.cities) setCities(data.cities);
          if (data.wards_by_city) setWardsByCity(data.wards_by_city);
          if (data.available_dates?.max) setReferenceDate(data.available_dates.max);
          setBackendOnline(true);
        } else {
          setBackendOnline(false);
        }
      } catch {
        setBackendOnline(false);
      }
    }

    async function fetchWards() {
      try {
        const res = await fetch(`${API_BASE}/api/wards`);
        if (res.ok) {
          const data = await res.json();
          setWardMetadata(data);
        }
      } catch {}
    }

    fetchMeta();
    fetchWards();
  }, []);

  // Update selected ward if city changes
  const availableWards = useMemo(() => {
    return wardsByCity[selectedCity] || [];
  }, [wardsByCity, selectedCity]);

  useEffect(() => {
    if (availableWards.length > 0 && !availableWards.includes(selectedWard)) {
      setSelectedWard(availableWards[0]);
    }
  }, [selectedCity, availableWards, selectedWard]);

  // Fetch prediction whenever inputs change
  useEffect(() => {
    let active = true;

    async function loadPrediction() {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            city: selectedCity,
            ward_id: selectedWard,
            reference_date: referenceDate,
            horizon: horizon,
            weather_scenario: scenario,
          }),
        });

        if (res.ok) {
          const data = await res.json();
          if (active) {
            setPrediction(data);
            setBackendOnline(true);
            // If scenario was null, initialize with defaults
            if (!scenario) {
              setScenario(data.defaults);
            }
          }
        } else {
          if (active) setBackendOnline(false);
        }
      } catch {
        if (active) setBackendOnline(false);
      } finally {
        if (active) setLoading(false);
      }
    }

    loadPrediction();
    return () => {
      active = false;
    };
  }, [selectedCity, selectedWard, referenceDate, horizon, scenario]);

  // Fetch city grid
  useEffect(() => {
    async function loadGrid() {
      try {
        const res = await fetch(`${API_BASE}/api/city-overview`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            city: selectedCity,
            reference_date: referenceDate,
            horizon: horizon,
            selected_ward: selectedWard,
            weather_scenario: scenario,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          setCityGrid(data.grid || []);
        }
      } catch {}
    }
    loadGrid();
  }, [selectedCity, referenceDate, horizon, selectedWard, scenario]);

  // Fetch ward history
  useEffect(() => {
    async function loadHistory() {
      try {
        const res = await fetch(`${API_BASE}/api/ward-history?ward_id=${selectedWard}&weeks=52`);
        if (res.ok) {
          const data = await res.json();
          setHistory(data);
        }
      } catch {}
    }
    loadHistory();
  }, [selectedWard]);

  const handleResetScenario = () => {
    if (prediction?.defaults) {
      setScenario({ ...prediction.defaults });
    }
  };

  const updateScenario = (key: keyof WeatherScenario, value: number) => {
    if (!scenario) return;
    const updated = { ...scenario, [key]: value };

    // Maintain physical consistency: rainfall_1w <= rainfall_2w <= rainfall_4w
    if (key === "rainfall_1w_mm") {
      updated.rainfall_2w_mm = Math.max(updated.rainfall_2w_mm, value);
      updated.rainfall_4w_mm = Math.max(updated.rainfall_4w_mm, updated.rainfall_2w_mm);
    } else if (key === "rainfall_2w_mm") {
      updated.rainfall_1w_mm = Math.min(updated.rainfall_1w_mm, value);
      updated.rainfall_4w_mm = Math.max(updated.rainfall_4w_mm, value);
    } else if (key === "rainfall_4w_mm") {
      updated.rainfall_2w_mm = Math.min(updated.rainfall_2w_mm, value);
      updated.rainfall_1w_mm = Math.min(updated.rainfall_1w_mm, updated.rainfall_2w_mm);
    }

    setScenario(updated);
  };

  const getTierStyles = (tier: string) => {
    switch (tier) {
      case "Critical":
        return {
          bg: "bg-rose-500/15 border-rose-500/40 text-rose-400",
          glow: "shadow-[0_0_24px_rgba(244,63,94,0.25)]",
          badge: "bg-rose-500 text-slate-950 font-bold",
          dot: "bg-rose-500 animate-pulse",
        };
      case "High":
        return {
          bg: "bg-amber-500/15 border-amber-500/40 text-amber-400",
          glow: "shadow-[0_0_24px_rgba(245,158,11,0.2)]",
          badge: "bg-amber-500 text-slate-950 font-bold",
          dot: "bg-amber-500",
        };
      case "Medium":
        return {
          bg: "bg-lime-500/15 border-lime-500/40 text-lime-400",
          glow: "shadow-[0_0_24px_rgba(132,204,22,0.18)]",
          badge: "bg-lime-500 text-slate-950 font-bold",
          dot: "bg-lime-500",
        };
      default:
        return {
          bg: "bg-emerald-500/15 border-emerald-500/40 text-emerald-400",
          glow: "shadow-[0_0_24px_rgba(16,185,129,0.15)]",
          badge: "bg-emerald-500 text-slate-950 font-bold",
          dot: "bg-emerald-500",
        };
    }
  };

  const currentTier = prediction?.tier || "Low";
  const tierStyle = getTierStyles(currentTier);

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100">
      {/* Top Banner & Status Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-slate-950 font-black text-xl">
              🦟
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-emerald-400 via-teal-300 to-lime-300 bg-clip-text text-transparent">
                  WardWatch
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  ML 2.0
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium hidden sm:block">
                Bangladesh Dengue Early-Warning & Risk Prediction Engine
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendOnline
                    ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"
                    : "bg-amber-400 animate-pulse"
                }`}
              />
              <span className="text-slate-300 font-medium">
                {backendOnline ? "FastAPI Inference Online" : "Connecting to Local API..."}
              </span>
            </div>

            <a
              href="https://github.com/TOMAL5085/EDU-AI-Engineering-Hackathon-2026-Team-Baddie"
              target="_blank"
              rel="noreferrer"
              className="hidden md:flex items-center space-x-1.5 text-xs text-slate-400 hover:text-emerald-400 transition-colors px-2.5 py-1.5 rounded-lg hover:bg-slate-800/60"
            >
              <span>GitHub Repo</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Hero Section */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-emerald-950/40 to-slate-900 border border-slate-800/80 p-6 sm:p-8 shadow-2xl">
          <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
          <div className="relative z-10 max-w-3xl space-y-3">
            <div className="inline-flex items-center space-x-2 text-xs uppercase tracking-widest font-bold text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20">
              <Activity className="w-3.5 h-3.5" />
              <span>Predictive Epidemiological Surveillance</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-black tracking-tight text-white leading-tight">
              See the dengue signal <br className="hidden sm:block" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-lime-300">
                2 to 4 weeks before outbreaks peak.
              </span>
            </h1>
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
              Programmatic multi-week weather lag pipelines coupled with chronologically validated tree ensembles. 
              Estimates expected weekly case counts and risk tiers for municipal wards in Bangladesh without future data leakage.
            </p>

            <div className="flex flex-wrap gap-2 pt-2">
              <span className="text-xs px-2.5 py-1 rounded-md bg-slate-800/90 text-slate-300 border border-slate-700/60">
                🏙️ 20 Wards Covered
              </span>
              <span className="text-xs px-2.5 py-1 rounded-md bg-slate-800/90 text-slate-300 border border-slate-700/60">
                📍 Chattogram · Gazipur · Narayanganj
              </span>
              <span className="text-xs px-2.5 py-1 rounded-md bg-slate-800/90 text-slate-300 border border-slate-700/60">
                📅 2016 – 2025 (522 Weekly Records)
              </span>
              <span className="text-xs px-2.5 py-1 rounded-md bg-emerald-900/30 text-emerald-300 border border-emerald-700/50">
                ⚡ 100% CPU Local Inference
              </span>
            </div>
          </div>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Interactive Controls */}
          <div className="lg:col-span-4 space-y-5">
            {/* Primary Selectors Card */}
            <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-4 shadow-lg">
              <div className="flex items-center space-x-2 pb-2 border-b border-slate-800/80">
                <Sliders className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-bold tracking-wide uppercase text-slate-200">
                  Target Selection
                </h2>
              </div>

              {/* City Selector */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-400 flex items-center justify-between">
                  <span>City</span>
                  <MapPin className="w-3.5 h-3.5 text-slate-500" />
                </label>
                <div className="grid grid-cols-3 gap-1.5">
                  {cities.map((c) => (
                    <button
                      key={c}
                      onClick={() => setSelectedCity(c)}
                      className={`text-xs py-2 px-2.5 rounded-lg font-medium transition-all text-center ${
                        selectedCity === c
                          ? "bg-emerald-500 text-slate-950 font-bold shadow-md shadow-emerald-500/20"
                          : "bg-slate-800/80 hover:bg-slate-800 text-slate-300 border border-slate-700/50"
                      }`}
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </div>

              {/* Ward Selector */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-400 flex items-center justify-between">
                  <span>Ward Neighborhood</span>
                  <span className="text-[10px] text-slate-500">{availableWards.length} Available</span>
                </label>
                <div className="relative">
                  <select
                    value={selectedWard}
                    onChange={(e) => setSelectedWard(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700/80 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-mono"
                  >
                    {availableWards.map((w) => (
                      <option key={w} value={w}>
                        {w}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Reference Date Picker */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-400 flex items-center justify-between">
                  <span>Reference Week</span>
                  <Calendar className="w-3.5 h-3.5 text-slate-500" />
                </label>
                <input
                  type="date"
                  value={referenceDate}
                  min="2016-01-25"
                  max="2025-12-29"
                  onChange={(e) => setReferenceDate(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700/80 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
                <p className="text-[11px] text-slate-500">
                  Data bounds: Jan 2016 – Dec 2025. Future dates use latest observed baseline.
                </p>
              </div>

              {/* Forecast Horizon Toggle */}
              <div className="space-y-1.5 pt-1">
                <label className="text-xs font-semibold text-slate-400 flex items-center justify-between">
                  <span>Forecast Lead Time</span>
                  <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {([2, 3, 4] as const).map((h) => (
                    <button
                      key={h}
                      onClick={() => setHorizon(h)}
                      className={`text-xs py-2 rounded-lg font-bold transition-all border ${
                        horizon === h
                          ? "bg-gradient-to-r from-emerald-500 to-teal-500 text-slate-950 border-emerald-400 shadow-md shadow-emerald-500/25"
                          : "bg-slate-800/80 hover:bg-slate-800 text-slate-300 border-slate-700/60"
                      }`}
                    >
                      +{h} Weeks
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Weather Scenario Sliders Card */}
            <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-4 shadow-lg">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
                <div className="flex items-center space-x-2">
                  <CloudRain className="w-4 h-4 text-teal-400" />
                  <h3 className="text-sm font-bold tracking-wide uppercase text-slate-200">
                    Weather Scenario Controls
                  </h3>
                </div>
                <button
                  onClick={handleResetScenario}
                  title="Reset to observed historical values"
                  className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center space-x-1 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 transition"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Reset</span>
                </button>
              </div>

              <p className="text-[11px] text-slate-400">
                Simulate drought or high-monsoon conditions. Inputs instantly recompute ML inference without retraining.
              </p>

              {scenario ? (
                <div className="space-y-4 pt-1">
                  {/* Rainfall Sliders */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-300 flex items-center space-x-1.5">
                        <Droplets className="w-3.5 h-3.5 text-teal-400" />
                        <span>Rainfall · 1 week</span>
                      </span>
                      <span className="font-mono text-emerald-400 font-semibold">
                        {scenario.rainfall_1w_mm.toFixed(0)} mm
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={400}
                      step={5}
                      value={scenario.rainfall_1w_mm}
                      onChange={(e) => updateScenario("rainfall_1w_mm", parseFloat(e.target.value))}
                      className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />

                    <div className="flex items-center justify-between text-xs pt-1">
                      <span className="text-slate-300 flex items-center space-x-1.5">
                        <Droplets className="w-3.5 h-3.5 text-teal-400" />
                        <span>Rainfall · 2 weeks (rolling)</span>
                      </span>
                      <span className="font-mono text-emerald-400 font-semibold">
                        {scenario.rainfall_2w_mm.toFixed(0)} mm
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={600}
                      step={5}
                      value={scenario.rainfall_2w_mm}
                      onChange={(e) => updateScenario("rainfall_2w_mm", parseFloat(e.target.value))}
                      className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />

                    <div className="flex items-center justify-between text-xs pt-1">
                      <span className="text-slate-300 flex items-center space-x-1.5">
                        <Droplets className="w-3.5 h-3.5 text-teal-400" />
                        <span>Rainfall · 4 weeks (rolling)</span>
                      </span>
                      <span className="font-mono text-emerald-400 font-semibold">
                        {scenario.rainfall_4w_mm.toFixed(0)} mm
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1200}
                      step={10}
                      value={scenario.rainfall_4w_mm}
                      onChange={(e) => updateScenario("rainfall_4w_mm", parseFloat(e.target.value))}
                      className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />
                  </div>

                  <hr className="border-slate-800" />

                  {/* Temperature & Humidity */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-300 flex items-center space-x-1.5">
                        <Thermometer className="w-3.5 h-3.5 text-amber-400" />
                        <span>Avg Temp · 2 weeks</span>
                      </span>
                      <span className="font-mono text-amber-400 font-semibold">
                        {scenario.avg_temp_2w_c.toFixed(1)} °C
                      </span>
                    </div>
                    <input
                      type="range"
                      min={15}
                      max={40}
                      step={0.5}
                      value={scenario.avg_temp_2w_c}
                      onChange={(e) => updateScenario("avg_temp_2w_c", parseFloat(e.target.value))}
                      className="w-full accent-amber-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />

                    <div className="flex items-center justify-between text-xs pt-1">
                      <span className="text-slate-300 flex items-center space-x-1.5">
                        <Droplets className="w-3.5 h-3.5 text-blue-400" />
                        <span>Relative Humidity · 2 weeks</span>
                      </span>
                      <span className="font-mono text-blue-400 font-semibold">
                        {scenario.relative_humidity_2w_pct.toFixed(0)} %
                      </span>
                    </div>
                    <input
                      type="range"
                      min={35}
                      max={100}
                      step={1}
                      value={scenario.relative_humidity_2w_pct}
                      onChange={(e) => updateScenario("relative_humidity_2w_pct", parseFloat(e.target.value))}
                      className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />
                  </div>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-slate-500">
                  Loading weather baseline...
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Prediction Cards & Tabs */}
          <div className="lg:col-span-8 space-y-6">
            {/* Top Prediction Display Banner */}
            <div
              className={`rounded-2xl border p-6 sm:p-7 transition-all ${tierStyle.bg} ${tierStyle.glow}`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-400">
                    <span>{selectedCity}</span>
                    <span>/</span>
                    <span className="text-white font-mono">{selectedWard}</span>
                    <span>•</span>
                    <span>Week +{horizon}</span>
                    <span>•</span>
                    <span className="text-emerald-400 font-mono">
                      Target: {prediction?.forecast_date || "—"}
                    </span>
                  </div>

                  <div className="flex items-baseline space-x-3 pt-1">
                    <span className="text-4xl sm:text-5xl font-black tracking-tight text-white font-mono">
                      {prediction ? prediction.expected_cases.toFixed(1) : "—"}
                    </span>
                    <div className="text-sm text-slate-300">
                      <span className="font-bold">cases</span> / ward-week
                      <div className="text-xs text-slate-400">
                        95% CI: [{prediction?.confidence_interval.lower.toFixed(1)} –{" "}
                        {prediction?.confidence_interval.upper.toFixed(1)}] cases
                      </div>
                    </div>
                  </div>
                </div>

                {/* Risk Tier Badge */}
                <div className="flex sm:flex-col items-center sm:items-end justify-between sm:justify-center">
                  <div className="text-xs uppercase font-bold text-slate-400 pb-1 hidden sm:block">
                    Outbreak Risk Tier
                  </div>
                  <div
                    className={`inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-base font-black uppercase tracking-wider border shadow-md ${tierStyle.badge}`}
                  >
                    <span className={`w-2.5 h-2.5 rounded-full ${tierStyle.dot}`} />
                    <span>{currentTier} Risk</span>
                  </div>
                  <div className="text-[11px] text-slate-400 pt-1 text-right">
                    Threshold: ≥ {prediction?.cutoffs[currentTier.toLowerCase() as keyof typeof prediction.cutoffs] || 0} cases
                  </div>
                </div>
              </div>
            </div>

            {/* Metric Summary Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="rounded-xl bg-slate-900 border border-slate-800/80 p-3.5 space-y-1">
                <span className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">
                  Water-Log Risk
                </span>
                <div className="text-xl font-black text-slate-100 font-mono">
                  {currentWardInfo.water_logging_risk_index.toFixed(2)}
                </div>
                <div className="text-[10px] text-slate-500">Source hazard index</div>
              </div>

              <div className="rounded-xl bg-slate-900 border border-slate-800/80 p-3.5 space-y-1">
                <span className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">
                  Sanitation Score
                </span>
                <div className="text-xl font-black text-slate-100 font-mono">
                  {currentWardInfo.sanitation_score.toFixed(2)}
                </div>
                <div className="text-[10px] text-slate-500">Ward infrastructure</div>
              </div>

              <div className="rounded-xl bg-slate-900 border border-slate-800/80 p-3.5 space-y-1">
                <span className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">
                  Holdout MAE
                </span>
                <div className="text-xl font-black text-emerald-400 font-mono">
                  {prediction ? prediction.holdout_mae.toFixed(2) : "1.09"}
                </div>
                <div className="text-[10px] text-slate-500">Cases/wk (from 2024)</div>
              </div>

              <div className="rounded-xl bg-slate-900 border border-slate-800/80 p-3.5 space-y-1">
                <span className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">
                  Holdout RMSE
                </span>
                <div className="text-xl font-black text-teal-400 font-mono">
                  {prediction ? prediction.holdout_rmse.toFixed(2) : "2.09"}
                </div>
                <div className="text-[10px] text-slate-500">Final 20% validation</div>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="border-b border-slate-800 flex space-x-6">
              {[
                { id: "overview", label: "Ward Overview & Heatmap", icon: Layers },
                { id: "drivers", label: "Model Drivers & Features", icon: BarChart3 },
                { id: "architecture", label: "Validation & Architecture", icon: Cpu },
              ].map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as typeof activeTab)}
                    className={`pb-3 text-xs sm:text-sm font-bold flex items-center space-x-2 border-b-2 transition-all ${
                      isActive
                        ? "border-emerald-400 text-emerald-400"
                        : "border-transparent text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Tab 1: Ward Overview & Heatmap */}
            {activeTab === "overview" && (
              <div className="space-y-6">
                {/* Heatmap Card */}
                <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-4 shadow-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                        <span>City-Wide Ward Risk Heatmap · {selectedCity}</span>
                      </h3>
                      <p className="text-xs text-slate-400">
                        Multi-horizon risk comparison across all wards in {selectedCity}. Click any ward to view details.
                      </p>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400">
                          <th className="py-2.5 px-3 font-semibold">Ward</th>
                          <th className="py-2.5 px-3 font-semibold text-center">+2 Weeks</th>
                          <th className="py-2.5 px-3 font-semibold text-center">+3 Weeks</th>
                          <th className="py-2.5 px-3 font-semibold text-center">+4 Weeks</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono">
                        {availableWards.map((ward) => {
                          const w2 = cityGrid.find((g) => g.ward_id === ward && g.horizon === 2);
                          const w3 = cityGrid.find((g) => g.ward_id === ward && g.horizon === 3);
                          const w4 = cityGrid.find((g) => g.ward_id === ward && g.horizon === 4);

                          const isSelected = selectedWard === ward;

                          return (
                            <tr
                              key={ward}
                              onClick={() => setSelectedWard(ward)}
                              className={`cursor-pointer transition-colors ${
                                isSelected ? "bg-slate-800/90 font-bold" : "hover:bg-slate-800/40"
                              }`}
                            >
                              <td className="py-2.5 px-3 text-slate-200 flex items-center space-x-2">
                                {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
                                <span>{ward}</span>
                              </td>
                              {[w2, w3, w4].map((item, idx) => {
                                const tier = item?.tier || "Low";
                                const style = getTierStyles(tier);
                                return (
                                  <td key={idx} className="py-2.5 px-3 text-center">
                                    <span
                                      className={`inline-block px-2.5 py-1 rounded-md text-[11px] font-bold border ${style.bg}`}
                                    >
                                      {tier} ({item ? item.expected_cases.toFixed(1) : "—"})
                                    </span>
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* History Chart Mock / Visualizer */}
                <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-4 shadow-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Reported Case Trajectory · {selectedWard} (Past 52 Weeks)
                      </h3>
                      <p className="text-xs text-slate-400">
                        Historical observed weekly dengue counts with 4-week rolling rainfall signals.
                      </p>
                    </div>
                  </div>

                  {history.length > 0 ? (
                    <div className="space-y-3">
                      <div className="h-44 flex items-end space-x-1.5 pt-4 px-2 bg-slate-950/60 rounded-lg border border-slate-800/60">
                        {history.slice(-32).map((item, i) => {
                          const maxCase = Math.max(...history.map((h) => h.dengue_cases), 10);
                          const heightPct = Math.min(100, Math.max(8, (item.dengue_cases / maxCase) * 100));
                          const isHigh = item.dengue_cases > 5;

                          return (
                            <div
                              key={i}
                              title={`${item.date}: ${item.dengue_cases} cases, ${item.rainfall_4w_mm.toFixed(0)}mm rain`}
                              className="flex-1 flex flex-col items-center h-full justify-end group relative"
                            >
                              <div
                                style={{ height: `${heightPct}%` }}
                                className={`w-full rounded-t transition-all ${
                                  isHigh
                                    ? "bg-gradient-to-t from-amber-500 to-rose-500"
                                    : "bg-emerald-500/70 group-hover:bg-emerald-400"
                                }`}
                              />
                            </div>
                          );
                        })}
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-500 px-1 font-mono">
                        <span>{history[0]?.date}</span>
                        <div className="flex items-center space-x-4">
                          <span className="flex items-center space-x-1">
                            <span className="w-2 h-2 rounded-sm bg-emerald-500" />
                            <span>Low/Medium</span>
                          </span>
                          <span className="flex items-center space-x-1">
                            <span className="w-2 h-2 rounded-sm bg-rose-500" />
                            <span>Outbreak Surge</span>
                          </span>
                        </div>
                        <span>{history[history.length - 1]?.date}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="py-12 text-center text-xs text-slate-500">
                      Loading historical timeline...
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Tab 2: Drivers & Feature Importance */}
            {activeTab === "drivers" && (
              <div className="space-y-6">
                <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-4 shadow-lg">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                      <BarChart3 className="w-4 h-4 text-emerald-400" />
                      <span>Model-Wide Feature Importance (Week +{horizon})</span>
                    </h3>
                    <p className="text-xs text-slate-400">
                      Relative importance weights learned by the selected ensemble estimator. Explains model mechanics without claiming causal proof.
                    </p>
                  </div>

                  {prediction && prediction.drivers ? (
                    <div className="space-y-3 pt-2">
                      {prediction.drivers.map((driver, idx) => {
                        const widthPct = Math.round((driver.importance / (prediction.drivers[0]?.importance || 1)) * 100);
                        return (
                          <div key={idx} className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-slate-300 font-medium">{driver.label}</span>
                              <div className="space-x-3 text-slate-400 font-mono">
                                <span className="text-slate-500">Val: {driver.value.toFixed(1)}</span>
                                <span className="text-emerald-400 font-bold">{(driver.importance * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                              <div
                                style={{ width: `${widthPct}%` }}
                                className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full rounded-full transition-all duration-500"
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="py-12 text-center text-xs text-slate-500">Loading driver rankings...</div>
                  )}
                </div>

                {/* Tier Thresholds Breakdown */}
                <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-3 shadow-lg">
                  <h3 className="text-sm font-bold text-white">Outbreak Tier Boundaries (Week +{horizon})</h3>
                  <p className="text-xs text-slate-400">
                    Thresholds derived dynamically from training set 75th, 90th, and 97th percentiles:
                  </p>
                  <div className="grid grid-cols-4 gap-3 pt-2 font-mono text-center text-xs">
                    <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
                      <div className="text-[10px] uppercase font-bold text-slate-400">Low</div>
                      <div className="text-base font-black">{"<"} {prediction?.cutoffs.medium.toFixed(1)}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-lime-500/10 border border-lime-500/20 text-lime-300">
                      <div className="text-[10px] uppercase font-bold text-slate-400">Medium</div>
                      <div className="text-base font-black">≥ {prediction?.cutoffs.medium.toFixed(1)}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300">
                      <div className="text-[10px] uppercase font-bold text-slate-400">High</div>
                      <div className="text-base font-black">≥ {prediction?.cutoffs.high.toFixed(1)}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300">
                      <div className="text-[10px] uppercase font-bold text-slate-400">Critical</div>
                      <div className="text-base font-black">≥ {prediction?.cutoffs.critical.toFixed(1)}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 3: Model Architecture & Evaluation */}
            {activeTab === "architecture" && (
              <div className="space-y-6">
                {/* Chronological Validation Card */}
                <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-4 shadow-lg">
                  <div className="flex items-center space-x-2">
                    <ShieldAlert className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white">
                      Chronological Nested Validation & Leakage Prevention
                    </h3>
                  </div>
                  <div className="text-xs text-slate-300 space-y-2 leading-relaxed">
                    <p>
                      <strong>1. Untouched Final 20% Holdout:</strong> All dates beginning{" "}
                      <span className="font-mono text-emerald-400 font-semibold">2024-01-01</span> were strictly quarantined
                      and never touched during feature selection, model tuning, or hyperparameter exploration.
                    </p>
                    <p>
                      <strong>2. Expanding Inner Folds:</strong> Inside the 80% development period (2016 to 2023), three expanding
                      chronological validation folds were evaluated.
                    </p>
                    <p>
                      <strong>3. Target Date Purging:</strong> Rows whose forward target date crossed into a validation window or the
                      final holdout were strictly removed to guarantee zero future data leakage.
                    </p>
                  </div>

                  <div className="overflow-x-auto pt-2">
                    <table className="w-full text-xs text-left border border-slate-800 rounded-lg overflow-hidden">
                      <thead className="bg-slate-800/60 text-slate-300">
                        <tr>
                          <th className="py-2.5 px-3">Horizon</th>
                          <th className="py-2.5 px-3">Selected Model</th>
                          <th className="py-2.5 px-3">Validation RMSE</th>
                          <th className="py-2.5 px-3">Final Holdout MAE</th>
                          <th className="py-2.5 px-3">Final Holdout RMSE</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800 font-mono">
                        <tr>
                          <td className="py-2.5 px-3 font-bold text-emerald-400">+2 Weeks</td>
                          <td className="py-2.5 px-3">Voting Ensemble (ET + GB)</td>
                          <td className="py-2.5 px-3">2.43</td>
                          <td className="py-2.5 px-3">1.13</td>
                          <td className="py-2.5 px-3">2.20</td>
                        </tr>
                        <tr>
                          <td className="py-2.5 px-3 font-bold text-teal-400">+3 Weeks</td>
                          <td className="py-2.5 px-3">Extra Trees (Tuned Deep)</td>
                          <td className="py-2.5 px-3">2.32</td>
                          <td className="py-2.5 px-3 text-emerald-400 font-bold">1.09</td>
                          <td className="py-2.5 px-3 text-emerald-400 font-bold">2.09</td>
                        </tr>
                        <tr>
                          <td className="py-2.5 px-3 font-bold text-lime-400">+4 Weeks</td>
                          <td className="py-2.5 px-3">Extra Trees (Regularized)</td>
                          <td className="py-2.5 px-3">2.74</td>
                          <td className="py-2.5 px-3 text-emerald-400 font-bold">1.24</td>
                          <td className="py-2.5 px-3 text-emerald-400 font-bold">2.35</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[11px] text-slate-500 italic">
                    * Values reported in dengue cases per ward-week. Lower error is better; these are regression metrics, not accuracy percentages.
                  </p>
                </div>

                {/* Compute & Cost Efficiency Card */}
                <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-3 shadow-lg">
                  <div className="flex items-center space-x-2">
                    <Cpu className="w-4 h-4 text-teal-400" />
                    <h3 className="text-sm font-bold text-white">Compute & API Cost Efficiency</h3>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs pt-1">
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                      <div className="font-bold text-emerald-400">Zero External APIs</div>
                      <p className="text-slate-400 text-[11px]">
                        Operates 100% locally with zero paid third-party weather, LLM, or telemetry calls.
                      </p>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                      <div className="font-bold text-teal-400">In-Memory Caching</div>
                      <p className="text-slate-400 text-[11px]">
                        Datasets and feature frames are cached in memory on backend startup for sub-10ms inference.
                      </p>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                      <div className="font-bold text-lime-400">CPU-Parallel Training</div>
                      <p className="text-slate-400 text-[11px]">
                        Serialized joblib artifacts eliminate runtime GPU overhead or cloud dependencies.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-3">
          <div className="flex items-center space-x-2">
            <span>WardWatch · EDU AI Engineering Hackathon 2026 · Team Baddie</span>
          </div>
          <div>
            Built with Next.js App Router, Tailwind CSS, FastAPI, and Scikit-Learn.
          </div>
        </div>
      </footer>
    </div>
  );
}
