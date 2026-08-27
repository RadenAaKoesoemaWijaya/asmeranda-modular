"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useWorkflow } from "@/lib/workflow-store";
import { useT } from "@/lib/i18n";

const MODELS = [
  "RandomForest",
  "GradientBoosting",
  "LogisticRegression",
  "LinearRegression",
  "DecisionTree",
  "KNeighbors",
  "SVM",
  "XGBoost",
  "LightGBM",
  "CatBoost",
  "Voting",
  "Stacking",
];

const CV_METHODS = [
  { id: "kfold", label: "K-Fold (Standard)" },
  { id: "stratified", label: "Stratified K-Fold (Seimbang)" },
  { id: "timeseries", label: "Time Series Split" },
  { id: "loo", label: "Leave-One-Out" },
  { id: "none", label: "Tanpa CV (Train-Test Saja)" },
];

const HYPERPARAM_TEMPLATES = {
  RandomForest: {
    n_estimators: { min: 10, max: 500, default: 100, type: "int" },
    max_depth: { min: 2, max: 30, default: 10, type: "int" },
    min_samples_split: { min: 2, max: 20, default: 2, type: "int" },
    min_samples_leaf: { min: 1, max: 10, default: 1, type: "int" },
  },
  GradientBoosting: {
    n_estimators: { min: 10, max: 500, default: 100, type: "int" },
    learning_rate: { min: 0.005, max: 0.5, default: 0.1, type: "float" },
    max_depth: { min: 2, max: 15, default: 3, type: "int" },
    subsample: { min: 0.5, max: 1.0, default: 0.8, type: "float" },
  },
  LogisticRegression: {
    C: { min: 0.01, max: 20.0, default: 1.0, type: "float" },
    max_iter: { min: 100, max: 2000, default: 1000, type: "int" },
    solver: { options: ["lbfgs", "liblinear", "saga"], default: "lbfgs", type: "select" },
  },
  LinearRegression: {
    fit_intercept: { default: true, type: "boolean" },
  },
  DecisionTree: {
    max_depth: { min: 2, max: 30, default: 10, type: "int" },
    min_samples_split: { min: 2, max: 20, default: 2, type: "int" },
    min_samples_leaf: { min: 1, max: 10, default: 1, type: "int" },
  },
  KNeighbors: {
    n_neighbors: { min: 1, max: 30, default: 5, type: "int" },
    weights: { options: ["uniform", "distance"], default: "uniform", type: "select" },
  },
  SVM: {
    C: { min: 0.01, max: 20.0, default: 1.0, type: "float" },
    kernel: { options: ["linear", "rbf", "poly"], default: "rbf", type: "select" },
  },
  XGBoost: {
    n_estimators: { min: 10, max: 500, default: 100, type: "int" },
    learning_rate: { min: 0.005, max: 0.5, default: 0.1, type: "float" },
    max_depth: { min: 2, max: 15, default: 6, type: "int" },
  },
  LightGBM: {
    n_estimators: { min: 10, max: 500, default: 100, type: "int" },
    learning_rate: { min: 0.005, max: 0.5, default: 0.1, type: "float" },
    max_depth: { min: 2, max: 15, default: -1, type: "int" },
  },
  CatBoost: {
    iterations: { min: 10, max: 500, default: 100, type: "int" },
    learning_rate: { min: 0.005, max: 0.5, default: 0.1, type: "float" },
    depth: { min: 2, max: 12, default: 6, type: "int" },
  },
  Voting: {
    voting: { options: ["hard", "soft"], default: "soft", type: "select" },
  },
  Stacking: {
    final_estimator: { options: ["LogisticRegression", "LinearRegression"], default: "LogisticRegression", type: "select" },
  },
};

export default function TrainingPage() {
  const lang = useWorkflow((s) => s.language) || "id";
  const tr = useT(lang);
  const set = useWorkflow((s) => s.set);
  const stateId = useWorkflow((s) => s.stateId);
  const datasetName = useWorkflow((s) => s.datasetName);
  const problemType = useWorkflow((s) => s.problemType);
  const optimizationResults = useWorkflow((s) => s.optimizationResults);

  const isUnsupervised = problemType === "Clustering" || problemType === "Unsupervised";

  const [modelType, setModelType] = useState("RandomForest");
  const [cvMethod, setCvMethod] = useState("kfold");
  const [cvFolds, setCvFolds] = useState(5);
  const [hyperparams, setHyperparams] = useState({});
  const [showHyperparams, setShowHyperparams] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Advanced Tools States
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [evaluating, setEvaluating] = useState(false);
  const [learningCurveResult, setLearningCurveResult] = useState(null);
  const [generatingLearningCurve, setGeneratingLearningCurve] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparingModels, setComparingModels] = useState(false);

  // Sync hyperparams when modelType changes or optimized params available
  useEffect(() => {
    const template = HYPERPARAM_TEMPLATES[modelType] || {};
    const defaults = {};
    Object.keys(template).forEach((key) => {
      defaults[key] = template[key].default;
    });

    if (
      optimizationResults?.model_type === modelType &&
      optimizationResults?.best_params
    ) {
      setHyperparams({ ...defaults, ...optimizationResults.best_params });
    } else {
      setHyperparams(defaults);
    }
  }, [modelType, optimizationResults]);

  if (!stateId || !problemType) {
    return (
      <div className="card" style={{ maxWidth: 840, margin: "0 auto" }}>
        <h1 className="page-title">🧠 {tr("training.title")}</h1>
        <div className="alert alert-warning mt-6">
          <span>⚠️</span>
          <div>
            <strong>Tahap Preprocessing Diperlukan:</strong>
            <p style={{ marginTop: 4 }}>
              Selesaikan tahap <a href="/preprocessing" style={{ color: "#b45309", fontWeight: 600, textDecoration: "underline" }}>Preprocessing</a> terlebih dahulu sebelum melatih model machine learning.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (isUnsupervised) {
    return (
      <div className="card" style={{ maxWidth: 840, margin: "0 auto" }}>
        <h1 className="page-title">🧠 {tr("training.title")}</h1>
        <div className="alert alert-info mt-6">
          <span>ℹ️</span>
          <div>
            <strong>Mode Unsupervised Terdeteksi:</strong>
            <p style={{ marginTop: 4 }}>
              Pelatihan Model Supervised (Klasifikasi/Regresi) dinonaktifkan untuk tipe masalah <strong>{problemType}</strong>. Silakan beralih ke halaman <a href="/clustering" style={{ color: "var(--color-primary-700)", fontWeight: 600, textDecoration: "underline" }}>Clustering Analysis →</a> atau <a href="/advanced-ml" style={{ color: "var(--color-primary-700)", fontWeight: 600, textDecoration: "underline" }}>Advanced ML →</a>.
            </p>
          </div>
        </div>
      </div>
    );
  }

  async function runTraining() {
    setBusy(true);
    setError(null);
    setResult(null);
    setEvaluationResult(null);
    setLearningCurveResult(null);

    try {
      const r = await api.startTraining({
        state_id: stateId,
        model_type: modelType,
        problem_type: problemType,
        cv_method: cvMethod,
        cv_folds: Number(cvFolds),
        hyperparams: hyperparams,
      });

      if (!r.success) throw new Error(r.error || "Gagal melatih model.");
      setResult(r);
      set({
        modelId: r.model_id,
        modelType: modelType,
        metrics: r.metrics,
        cvScores: r.cv_scores,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function runEvaluation() {
    if (!result || !result.model_id) {
      setError("Latih model terlebih dahulu sebelum menjalankan evaluasi mendalam.");
      return;
    }

    setEvaluating(true);
    setError(null);
    try {
      const r = await api.evaluateModel({
        state_id: stateId,
        model_id: result.model_id,
        generate_plots: true,
        plot_types: ["confusion_matrix", "roc_curve", "feature_importance", "precision_recall_curve"],
      });
      if (!r.success) throw new Error(r.error);
      setEvaluationResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setEvaluating(false);
    }
  }

  async function runLearningCurve() {
    if (!result || !result.model_id) {
      setError("Latih model terlebih dahulu sebelum membuat kurva pembelajaran.");
      return;
    }

    setGeneratingLearningCurve(true);
    setError(null);
    try {
      const r = await api.generateLearningCurve({
        state_id: stateId,
        model_id: result.model_id,
        cv: 5,
      });
      if (!r.success) throw new Error(r.error || "Gagal menghasilkan Learning Curve");
      setLearningCurveResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setGeneratingLearningCurve(false);
    }
  }

  async function runModelComparison() {
    setComparingModels(true);
    setError(null);
    try {
      const r = await api.compareModels({
        state_id: stateId,
        cv_method: cvMethod,
        cv_folds: Number(cvFolds),
      });
      if (!r.success) throw new Error(r.error || "Gagal membandingkan model");
      setComparisonResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setComparingModels(false);
    }
  }

  return (
    <div style={{ maxWidth: 1120, margin: "0 auto" }}>
      {/* ── Header ── */}
      <div className="page-header">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h1 className="page-title">🧠 {tr("training.title")}</h1>
            <p className="page-subtitle">
              Latih algoritma machine learning mutakhir, validasi silang (cross-validation), evaluasi metrik kinerja, dan analisis diagnostik.
            </p>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <span className="sidebar-link-badge" style={{ background: "var(--color-primary-600)", fontSize: 13, padding: "6px 12px" }}>
              {problemType}
            </span>
            {datasetName && (
              <span className="sidebar-link-badge" style={{ background: "var(--color-slate-600)", fontSize: 13, padding: "6px 12px" }}>
                {datasetName}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ── Optimization Integration Banner ── */}
      {optimizationResults?.model_type && optimizationResults?.best_params && (
        <div
          style={{
            marginBottom: 24,
            padding: 16,
            background: "#eff6ff",
            borderRadius: "var(--radius-xl)",
            border: "1px solid #bfdbfe",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 24 }}>🎯</span>
            <div>
              <strong style={{ color: "#1e40af" }}>Parameter Teroptimasi Siap Digunakan!</strong>
              <div style={{ fontSize: 13, color: "#1e3a8a", marginTop: 2 }}>
                Model: <strong>{optimizationResults.model_type}</strong> | Skor Terbaik: <strong>{optimizationResults.best_score?.toFixed(4)}</strong>
              </div>
            </div>
          </div>

          {modelType === optimizationResults.model_type ? (
            <span style={{ color: "#15803d", fontWeight: 600, fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
              ✓ Hyperparameter optimal sudah terpasang otomatis
            </span>
          ) : (
            <button
              onClick={() => setModelType(optimizationResults.model_type)}
              className="btn btn-primary btn-sm"
            >
              Ganti ke {optimizationResults.model_type} & Pasang Parameter
            </button>
          )}
        </div>
      )}

      {/* ── Error Banner ── */}
      {error && (
        <div className="alert alert-error mb-6">
          <span>❌</span>
          <div>
            <strong>Terjadi Kendala:</strong>
            <p style={{ margin: 0 }}>{error}</p>
          </div>
        </div>
      )}

      {/* ── Training Configuration Card ── */}
      <div className="card mb-6">
        <h2 className="card-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>⚙️</span> Konfigurasi Pelatihan Model
        </h2>
        <p className="card-subtitle">
          Pilih arsitektur model dan strategi validasi silang (cross-validation) untuk menguji generalisasi model.
        </p>

        <div className="grid-3" style={{ gap: 16 }}>
          <div className="form-group">
            <label>{tr("training.model_type")}</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
            >
              {MODELS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>{tr("training.cv_method")}</label>
            <select
              value={cvMethod}
              onChange={(e) => setCvMethod(e.target.value)}
            >
              {CV_METHODS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          {cvMethod !== "none" && cvMethod !== "loo" && (
            <div className="form-group">
              <label>{tr("training.cv_folds")} (Jumlah Lipatan)</label>
              <input
                type="number"
                min="2"
                max="20"
                value={cvFolds}
                onChange={(e) => setCvFolds(e.target.value)}
              />
            </div>
          )}
        </div>

        {/* Hyperparameter Toggle */}
        <div style={{ marginTop: 8, marginBottom: 12 }}>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={showHyperparams}
              onChange={(e) => setShowHyperparams(e.target.checked)}
              style={{ width: "auto" }}
            />
            <strong>Kustomisasi Hyperparameter Manual (Tingkat Lanjut)</strong>
            {optimizationResults?.model_type === modelType && (
              <span className="sidebar-link-badge" style={{ background: "#10b981" }}>
                Nilai Teroptimasi Aktif
              </span>
            )}
          </label>
        </div>

        {/* Hyperparameter Inputs */}
        {showHyperparams && (
          <div
            style={{
              padding: 16,
              background: "var(--color-slate-50)",
              borderRadius: "var(--radius-lg)",
              border: "1px solid var(--color-slate-200)",
              marginBottom: 16,
            }}
          >
            <h4 style={{ fontSize: 14, fontWeight: 600, color: "var(--color-slate-800)", marginBottom: 12 }}>
              Parameter Spesifik untuk {modelType}:
            </h4>
            <div className="grid-2" style={{ gap: 12 }}>
              {Object.entries(HYPERPARAM_TEMPLATES[modelType] || {}).map(([paramName, config]) => (
                <div key={paramName} className="form-group" style={{ marginBottom: 0 }}>
                  <label style={{ fontSize: 13 }}>{paramName}</label>
                  {config.type === "select" ? (
                    <select
                      value={hyperparams[paramName] || config.default}
                      onChange={(e) =>
                        setHyperparams({
                          ...hyperparams,
                          [paramName]: e.target.value,
                        })
                      }
                    >
                      {config.options.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : config.type === "boolean" ? (
                    <select
                      value={hyperparams[paramName] ?? config.default}
                      onChange={(e) =>
                        setHyperparams({
                          ...hyperparams,
                          [paramName]: e.target.value === "true",
                        })
                      }
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  ) : (
                    <input
                      type="number"
                      step={config.type === "float" ? "0.01" : "1"}
                      min={config.min}
                      max={config.max}
                      value={hyperparams[paramName] ?? config.default}
                      onChange={(e) =>
                        setHyperparams({
                          ...hyperparams,
                          [paramName]:
                            config.type === "float"
                              ? parseFloat(e.target.value)
                              : parseInt(e.target.value),
                        })
                      }
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 8 }}>
          <button
            onClick={runTraining}
            disabled={busy}
            className="btn btn-primary btn-lg"
          >
            {busy ? "⏳ Melatih Model..." : `🚀 ${tr("training.start")} (${modelType})`}
          </button>

          <button
            onClick={runModelComparison}
            disabled={comparingModels || busy}
            className="btn btn-secondary btn-lg"
            style={{ background: "#faf5ff", borderColor: "#d8b4fe", color: "#6b21a8" }}
          >
            {comparingModels ? "⏳ Membandingkan Seluruh Model..." : "🏆 Bandingkan Semua Algoritma"}
          </button>
        </div>
      </div>

      {/* ── Training Success Results Card ── */}
      {result && result.success && (
        <div className="card mb-6" style={{ border: "1px solid #86efac", background: "linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%)" }}>
          <div className="flex-between mb-4">
            <div>
              <h2 className="card-title" style={{ color: "#15803d", display: "flex", alignItems: "center", gap: 8 }}>
                <span>✅</span> Pelatihan Model Sukses ({result.model_type || modelType})
              </h2>
              <p className="card-subtitle" style={{ color: "#166534" }}>
                Model berhasil dilatih pada data training dan dievaluasi pada data hold-out test.
              </p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="code" style={{ fontSize: 13 }}>ID: {result.model_id}</span>
              <button
                onClick={() => {
                  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_PATH || "/api/v1";
                  window.open(`${API_BASE}/training/models/${result.model_id}/download`, "_blank");
                }}
                className="btn btn-secondary btn-sm"
                title="Download Serialized .pkl File"
              >
                💾 Unduh .pkl
              </button>
            </div>
          </div>

          {/* Metric Cards Grid */}
          <div className="grid-4 mb-6" style={{ gap: 12 }}>
            {problemType === "Classification" ? (
              <>
                <div style={{ padding: 14, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0", boxShadow: "0 2px 4px rgba(0,0,0,0.02)" }}>
                  <div style={{ fontSize: 11, color: "#166534", fontWeight: 600 }}>ACCURACY</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                    {result.metrics?.accuracy !== undefined ? (result.metrics.accuracy * 100).toFixed(2) + "%" : "N/A"}
                  </div>
                </div>

                <div style={{ padding: 14, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0", boxShadow: "0 2px 4px rgba(0,0,0,0.02)" }}>
                  <div style={{ fontSize: 11, color: "#166534", fontWeight: 600 }}>BALANCED ACCURACY</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                    {result.metrics?.balanced_accuracy !== undefined ? (result.metrics.balanced_accuracy * 100).toFixed(2) + "%" : "N/A"}
                  </div>
                </div>

                <div style={{ padding: 14, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0", boxShadow: "0 2px 4px rgba(0,0,0,0.02)" }}>
                  <div style={{ fontSize: 11, color: "#166534", fontWeight: 600 }}>F1-SCORE (MACRO)</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                    {result.metrics?.f1_macro !== undefined ? result.metrics.f1_macro.toFixed(4) : "N/A"}
                  </div>
                </div>

                <div style={{ padding: 14, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0", boxShadow: "0 2px 4px rgba(0,0,0,0.02)" }}>
                  <div style={{ fontSize: 11, color: "#166534", fontWeight: 600 }}>MCC (CORRELATION)</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                    {result.metrics?.mcc !== undefined ? result.metrics.mcc.toFixed(4) : "N/A"}
                  </div>
                </div>
              </>
            ) : (
              <>
                <div style={{ padding: 14, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0" }}>
                  <div style={{ fontSize: 11, color: "#166534", fontWeight: 600 }}>R² SCORE</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                    {result.metrics?.r2 !== undefined ? result.metrics.r2.toFixed(4) : "N/A"}
                  </div>
                </div>

                <div style={{ padding: 14, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0" }}>
                  <div style={{ fontSize: 11, color: "#166534", fontWeight: 600 }}>RMSE</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                    {result.metrics?.rmse !== undefined ? result.metrics.rmse.toFixed(4) : "N/A"}
                  </div>
                </div>

                <div style={{ padding: 14, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0" }}>
                  <div style={{ fontSize: 11, color: "#166534", fontWeight: 600 }}>MAE</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                    {result.metrics?.mae !== undefined ? result.metrics.mae.toFixed(4) : "N/A"}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Cross Validation Summary */}
          {result.cv_scores && (
            <div style={{ padding: 16, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #e2e8f0", marginBottom: 20 }}>
              <h4 style={{ fontSize: 14, fontWeight: 600, color: "#334155", marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
                <span>📊</span> Hasil Cross-Validation ({result.cv_scores.method}):
              </h4>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 16, fontSize: 13, color: "#475569" }}>
                <div>Metrik: <strong>{result.cv_scores.scoring}</strong></div>
                <div>Rata-rata: <strong style={{ color: "#15803d", fontSize: 15 }}>{result.cv_scores.mean?.toFixed(4)}</strong></div>
                <div>Standar Deviasi: <strong>±{result.cv_scores.std?.toFixed(4)}</strong></div>
                {result.cv_scores.scores && (
                  <div style={{ display: "flex", gap: 4 }}>
                    {result.cv_scores.scores.map((s, i) => (
                      <span key={i} style={{ padding: "2px 6px", background: "#f1f5f9", borderRadius: 4, fontSize: 11 }}>
                        Fold {i + 1}: {s.toFixed(3)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Feature Importance Bar Chart */}
          {result.feature_importances && result.feature_importances.length > 0 && (
            <div style={{ padding: 16, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #e2e8f0", marginBottom: 20 }}>
              <h4 style={{ fontSize: 14, fontWeight: 600, color: "#334155", marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
                <span>🏆</span> Tingkat Kepentingan Fitur (Feature Importance):
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {result.feature_importances.slice(0, 10).map((f, i) => {
                  const maxImp = result.feature_importances[0].importance || 1;
                  const pct = Math.max(2, (f.importance / maxImp) * 100);
                  return (
                    <div key={f.feature} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <span style={{ width: 140, fontSize: 13, fontWeight: 500, color: "#334155", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={f.feature}>
                        {f.feature}
                      </span>
                      <div style={{ flex: 1, background: "#f1f5f9", height: 12, borderRadius: 6, overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            background: "linear-gradient(90deg, #3b82f6, #60a5fa)",
                            borderRadius: 6,
                          }}
                        />
                      </div>
                      <span style={{ width: 60, textAlign: "right", fontSize: 12, fontWeight: 600, color: "#1e40af" }}>
                        {f.importance.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Action Hub for Deep Evaluation, Learning Curve, Interpretation */}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", paddingTop: 8, borderTop: "1px solid #bbf7d0" }}>
            <button
              onClick={runEvaluation}
              disabled={evaluating}
              className="btn btn-primary"
              style={{ background: "#059669" }}
            >
              {evaluating ? "⏳ Menganalisis Evaluasi..." : "🔬 Evaluasi Lanjutan (ROC & Confusion Matrix)"}
            </button>

            <button
              onClick={runLearningCurve}
              disabled={generatingLearningCurve}
              className="btn btn-primary"
              style={{ background: "#0284c7" }}
            >
              {generatingLearningCurve ? "⏳ Membuat Kurva..." : "📈 Analisis Kurva Pembelajaran (Bias/Variance)"}
            </button>

            <a
              href="/shap"
              className="btn btn-secondary"
              style={{ background: "#f8fafc", color: "#1e293b" }}
            >
              📊 Buka Interpretasi SHAP →
            </a>

            <a
              href="/lime"
              className="btn btn-secondary"
              style={{ background: "#f8fafc", color: "#1e293b" }}
            >
              🔬 Buka Interpretasi LIME →
            </a>

            <a
              href="/inference"
              className="btn btn-primary"
              style={{ background: "#7c3aed", borderColor: "#6d28d9" }}
            >
              🔮 Deteksi Data Baru (Inferensi) →
            </a>
          </div>
        </div>
      )}

      {/* ── Detailed Evaluation Plots Card ── */}
      {evaluationResult && evaluationResult.success && (
        <div className="card mb-6">
          <h3 className="card-title mb-4">📊 Visualisasi Evaluasi Model Mendalam</h3>

          {evaluationResult.plots && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
                gap: 16,
              }}
            >
              {Object.entries(evaluationResult.plots).map(([plotName, plotBase64]) => (
                plotBase64 && (
                  <div
                    key={plotName}
                    style={{
                      background: "#fff",
                      padding: 16,
                      borderRadius: "var(--radius-lg)",
                      border: "1px solid var(--color-slate-200)",
                      boxShadow: "var(--shadow-sm)",
                    }}
                  >
                    <h5 style={{ fontSize: 14, fontWeight: 600, color: "var(--color-slate-800)", marginBottom: 12, textTransform: "capitalize" }}>
                      {plotName.replace(/_/g, " ")}
                    </h5>
                    <img
                      src={`data:image/png;base64,${plotBase64}`}
                      alt={plotName}
                      style={{ width: "100%", height: "auto", borderRadius: 6 }}
                    />
                  </div>
                )
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Learning Curve Diagnostics Card ── */}
      {learningCurveResult && learningCurveResult.success && (
        <div className="card mb-6">
          <h3 className="card-title mb-2">📈 Diagnostik Kurva Pembelajaran (Learning Curve)</h3>
          <div className="alert alert-info mb-4">
            <span>💡</span>
            <div>
              <strong>Diagnosa Model: {learningCurveResult.diagnosis}</strong>
              <p style={{ margin: 0, fontSize: 13 }}>
                Skor Training Akhir: <strong>{learningCurveResult.final_train_score?.toFixed(4)}</strong> | Skor Validasi Akhir: <strong>{learningCurveResult.final_test_score?.toFixed(4)}</strong> (Gap: {learningCurveResult.score_gap?.toFixed(4)})
              </p>
            </div>
          </div>

          {learningCurveResult.plot_base64 && (
            <div style={{ maxWidth: 720, margin: "0 auto", background: "#fff", padding: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}>
              <img
                src={`data:image/png;base64,${learningCurveResult.plot_base64}`}
                alt="Learning Curve Diagnostic"
                style={{ width: "100%", height: "auto" }}
              />
            </div>
          )}
        </div>
      )}

      {/* ── Model Comparison Leaderboard Card ── */}
      {comparisonResult && comparisonResult.success && (
        <div className="card mb-6">
          <div className="flex-between mb-4">
            <div>
              <h3 className="card-title">🏆 Papan Peringkat Benchmark Seluruh Model</h3>
              <p className="card-subtitle">
                Model terbaik berdasarkan metrik <strong>{comparisonResult.ranking_metric}</strong>:{" "}
                <span className="sidebar-link-badge" style={{ background: "#059669", fontSize: 13 }}>
                  👑 {comparisonResult.best_model?.model_type}
                </span>
              </p>
            </div>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Peringkat</th>
                  <th>Algoritma Model</th>
                  <th>Skor ({comparisonResult.ranking_metric})</th>
                  <th>Waktu Pelatihan</th>
                  <th>Tindakan</th>
                </tr>
              </thead>
              <tbody>
                {(comparisonResult.ranking || []).map((m, idx) => (
                  <tr key={m.model_type}>
                    <td>
                      <strong style={{ color: idx === 0 ? "#d97706" : "#475569" }}>
                        {idx === 0 ? "🥇 #1" : idx === 1 ? "🥈 #2" : idx === 2 ? "🥉 #3" : `#${idx + 1}`}
                      </strong>
                    </td>
                    <td>
                      <strong>{m.model_type}</strong>
                    </td>
                    <td style={{ fontWeight: 700, color: idx === 0 ? "#15803d" : "#1e40af" }}>
                      {m.metrics?.[comparisonResult.ranking_metric] !== undefined
                        ? Number(m.metrics[comparisonResult.ranking_metric]).toFixed(4)
                        : "N/A"}
                    </td>
                    <td>{m.train_time ? `${m.train_time.toFixed(2)}s` : "-"}</td>
                    <td>
                      <button
                        onClick={() => {
                          setModelType(m.model_type);
                          window.scrollTo({ top: 0, behavior: "smooth" });
                        }}
                        className="btn btn-secondary btn-sm"
                      >
                        Gunakan Model Ini
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
