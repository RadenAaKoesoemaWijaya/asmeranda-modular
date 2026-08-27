"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWorkflow } from "@/lib/workflow-store";
import { useT } from "@/lib/i18n";

const OPTIMIZATION_METHODS = ["grid_search", "random_search", "bayesian"];

const MODELS = [
  "RandomForest",
  "GradientBoosting",
  "LogisticRegression",
  "LinearRegression",
  "DecisionTree",
  "KNeighbors",
  "SVM",
];

export default function OptimizationPage() {
  const router = useRouter();
  const lang = useWorkflow((s) => s.language) || "id";
  const tr = useT(lang);
  const stateId = useWorkflow((s) => s.stateId);
  const problemType = useWorkflow((s) => s.problemType);
  const setOptimizedHyperparams = useWorkflow((s) => s.setOptimizedHyperparams);
  const set = useWorkflow((s) => s.set);

  const isUnsupervised =
    problemType === "Clustering" || problemType === "Unsupervised";

  const [modelType, setModelType] = useState("RandomForest");
  const [method, setMethod] = useState("grid_search");
  const [cvFolds, setCvFolds] = useState(5);
  const [nIter, setNIter] = useState(50);
  const [useAsync, setUseAsync] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [applied, setApplied] = useState(false);

  // Guard: preprocessing must be done first
  if (!stateId) {
    return (
      <div>
        <h1>{tr("optimization.title")}</h1>
        <div
          style={{
            padding: 16,
            background: "#fef3c7",
            borderRadius: 6,
            border: "1px solid #f59e0b",
            color: "#92400e",
          }}
        >
          ⚠ Selesaikan tahap <strong>Preprocessing</strong> terlebih dahulu sebelum
          menggunakan Optimasi Hyperparameter.{" "}
          <a href="/preprocessing" style={{ color: "#92400e", fontWeight: 600 }}>
            Buka Preprocessing →
          </a>
        </div>
      </div>
    );
  }

  // Guard: unsupervised cannot use this page
  if (isUnsupervised) {
    return (
      <div>
        <h1>{tr("optimization.title")}</h1>
        <div
          style={{
            padding: 16,
            background: "#f3f4f6",
            borderRadius: 6,
            border: "1px solid #d1d5db",
            color: "#374151",
          }}
        >
          🔒 Optimasi Hyperparameter hanya tersedia untuk{" "}
          <strong>Supervised Learning</strong> (Klasifikasi / Regresi).
          <br />
          Mode aktif: <strong>{problemType}</strong> — Gunakan halaman{" "}
          <a href="/clustering" style={{ color: "#1d4ed8", fontWeight: 600 }}>
            Clustering Analysis →
          </a>
        </div>
      </div>
    );
  }

  async function runOptimization() {
    setBusy(true);
    setError(null);
    setResult(null);
    setApplied(false);

    try {
      const payload = {
        state_id: stateId,
        model_type: modelType,
        problem_type: problemType,
        method: method,
        cv_folds: Number(cvFolds),
        n_iter: Number(nIter),
      };

      const r = useAsync
        ? await api.optimizeHyperparameters(payload)
        : await api.optimizeHyperparametersSync(payload);

      if (!r.success) throw new Error(r.error);
      setResult(r);

      // Persist optimization results + best_params to store
      set({
        optimizationResults: {
          best_params: r.best_params,
          best_score: r.best_score,
          method: r.method,
          model_type: modelType,
        },
      });
      if (r.best_params && typeof setOptimizedHyperparams === "function") {
        setOptimizedHyperparams(modelType, r.best_params);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function applyToTraining() {
    setApplied(true);
    setTimeout(() => router.push("/training"), 700);
  }

  return (
    <div>
      <h1>{tr("optimization.title")}</h1>
      <p style={{ color: "#64748b" }}>
        Temukan hyperparameter terbaik untuk model <strong>{problemType}</strong>{" "}
        sebelum pelatihan final.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 12,
          marginTop: 16,
          maxWidth: 720,
        }}
      >
        <label>
          Tipe Model
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            style={{ width: "100%" }}
          >
            {MODELS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        <label>
          Metode Optimasi
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            style={{ width: "100%" }}
          >
            {OPTIMIZATION_METHODS.map((m) => (
              <option key={m} value={m}>
                {m.replace(/_/g, " ").toUpperCase()}
              </option>
            ))}
          </select>
        </label>

        <label>
          CV Folds
          <input
            type="number"
            min="2"
            max="10"
            value={cvFolds}
            onChange={(e) => setCvFolds(e.target.value)}
            style={{ width: "100%" }}
          />
        </label>
      </div>

      {(method === "random_search" || method === "bayesian") && (
        <div style={{ marginTop: 12, maxWidth: 240 }}>
          <label>
            Iterasi
            <input
              type="number"
              min="10"
              max="200"
              value={nIter}
              onChange={(e) => setNIter(e.target.value)}
              style={{ width: "100%" }}
            />
          </label>
        </div>
      )}

      <div style={{ marginTop: 16 }}>
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={useAsync}
            onChange={(e) => setUseAsync(e.target.checked)}
          />
          Jalankan di background (async)
        </label>
      </div>

      <button
        onClick={runOptimization}
        disabled={busy}
        style={{
          marginTop: 16,
          padding: "10px 20px",
          background: "#1e40af",
          color: "#fff",
          border: "none",
          borderRadius: 6,
          cursor: busy ? "not-allowed" : "pointer",
        }}
      >
        {busy ? "Mengoptimasi..." : "🔧 Jalankan Optimasi"}
      </button>

      {error && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            background: "#fee2e2",
            color: "#991b1b",
            borderRadius: 6,
          }}
        >
          {error}
        </div>
      )}

      {result && result.success && (
        <div
          style={{
            marginTop: 16,
            padding: 16,
            background: "#dcfce7",
            borderRadius: 6,
            color: "#166534",
          }}
        >
          <h3>✅ Hasil Optimasi</h3>
          <p>
            Metode:{" "}
            <strong>{result.method?.replace(/_/g, " ").toUpperCase()}</strong>
          </p>
          <p>
            Best Score: <strong>{result.best_score?.toFixed(4)}</strong>
          </p>

          {result.best_params && (
            <div style={{ marginTop: 12 }}>
              <h4>Hyperparameter Terbaik:</h4>
              <pre
                style={{
                  background: "#0f172a",
                  color: "#e2e8f0",
                  padding: 12,
                  borderRadius: 4,
                  overflow: "auto",
                }}
              >
                {JSON.stringify(result.best_params, null, 2)}
              </pre>

              <div
                style={{
                  marginTop: 12,
                  padding: 12,
                  background: "#eff6ff",
                  borderRadius: 6,
                  border: "1px solid #bfdbfe",
                }}
              >
                <p
                  style={{
                    margin: "0 0 8px",
                    color: "#1d4ed8",
                    fontWeight: 600,
                  }}
                >
                  🚀 Hyperparameter berhasil disimpan ke store!
                </p>
                <button
                  onClick={applyToTraining}
                  disabled={applied}
                  style={{
                    padding: "8px 16px",
                    background: applied ? "#16a34a" : "#059669",
                    color: "#fff",
                    border: "none",
                    borderRadius: 6,
                    cursor: applied ? "default" : "pointer",
                    fontSize: 14,
                    fontWeight: 600,
                  }}
                >
                  {applied
                    ? "✓ Mengalihkan ke Pelatihan Model..."
                    : "→ Terapkan ke Pelatihan Model"}
                </button>
              </div>
            </div>
          )}

          {useAsync && (
            <p
              style={{
                marginTop: 12,
                fontStyle: "italic",
                color: "#64748b",
              }}
            >
              Optimasi berjalan di background. Hasil akan tersedia setelah
              selesai.
            </p>
          )}
        </div>
      )}
    </div>
  );
}