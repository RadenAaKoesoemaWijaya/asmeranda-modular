"use client";

import { useState } from "react";
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
];

const CV_METHODS = ["kfold", "stratified", "loo", "timeseries", "none"];

export default function TrainingPage() {
  const lang = useWorkflow((s) => s.language) || "id";
  const tr = useT(lang);
  const set = useWorkflow((s) => s.set);
  const stateId = useWorkflow((s) => s.stateId);
  const problemType = useWorkflow((s) => s.problemType);

  const [modelType, setModelType] = useState("RandomForest");
  const [cvMethod, setCvMethod] = useState("kfold");
  const [cvFolds, setCvFolds] = useState(5);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  if (!stateId || !problemType) {
    return (
      <div>
        <h1>{tr("training.title")}</h1>
        <p style={{ color: "#dc2626" }}>⚠ Jalankan preprocessing dulu.</p>
      </div>
    );
  }

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.startTraining({
        state_id: stateId,
        model_type: modelType,
        problem_type: problemType,
        cv_method: cvMethod,
        cv_folds: Number(cvFolds),
      });
      if (!r.success) throw new Error(r.error);
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

  return (
    <div>
      <h1>{tr("training.title")}</h1>
      <p style={{ color: "#64748b" }}>
        Problem type: <strong>{problemType}</strong>
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
          {tr("training.model_type")}
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
          {tr("training.cv_method")}
          <select
            value={cvMethod}
            onChange={(e) => setCvMethod(e.target.value)}
            style={{ width: "100%" }}
          >
            {CV_METHODS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        <label>
          {tr("training.cv_folds")}
          <input
            type="number"
            min="2"
            max="20"
            value={cvFolds}
            onChange={(e) => setCvFolds(e.target.value)}
            style={{ width: "100%" }}
          />
        </label>
      </div>

      <button
        onClick={run}
        disabled={busy}
        style={{
          marginTop: 20,
          padding: "10px 20px",
          background: "#1e40af",
          color: "#fff",
          border: "none",
          borderRadius: 6,
        }}
      >
        {busy ? tr("common.loading") : tr("training.start")}
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

      {result && (
        <div
          style={{
            marginTop: 16,
            padding: 16,
            background: "#dcfce7",
            borderRadius: 6,
            color: "#166534",
          }}
        >
          <h3>✓ {tr("common.success")}</h3>
          <p>
            Model ID: <code>{result.model_id}</code>
          </p>
          <h4>{tr("training.metrics")}</h4>
          <pre
            style={{
              background: "#0f172a",
              color: "#e2e8f0",
              padding: 12,
              borderRadius: 4,
              overflow: "auto",
            }}
          >
            {JSON.stringify(result.metrics, null, 2)}
          </pre>
          {result.cv_scores && (
            <>
              <h4>{tr("training.cv_scores")}</h4>
              <p>
                Method: {result.cv_scores.method} | Folds: {result.cv_scores.folds} |{" "}
                Scoring: {result.cv_scores.scoring}
              </p>
              <p>
                Mean: {Number(result.cv_scores.mean).toFixed(4)} | Std:{" "}
                {Number(result.cv_scores.std).toFixed(4)}
              </p>
            </>
          )}
          <p>
            Lanjut ke <strong>SHAP</strong> atau <strong>LIME</strong> untuk interpretasi.
          </p>
        </div>
      )}
    </div>
  );
}
