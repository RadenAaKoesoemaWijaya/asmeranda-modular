"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useWorkflow } from "@/lib/workflow-store";
import { useT } from "@/lib/i18n";

const CLUSTERING_METHODS = ["kmeans", "dbscan", "hierarchical", "spectral"];

export default function ClusteringPage() {
  const lang = useWorkflow((s) => s.language) || "id";
  const tr = useT(lang);
  const stateId = useWorkflow((s) => s.stateId);
  const problemType = useWorkflow((s) => s.problemType);
  const isUnsupervised = problemType === "Clustering" || problemType === "Unsupervised";

  const [method, setMethod] = useState("kmeans");
  const [nClusters, setNClusters] = useState(3);
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);
  const [maxK, setMaxK] = useState(10);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [optimalKResult, setOptimalKResult] = useState(null);
  const [error, setError] = useState(null);

  if (!stateId) {
    return (
      <div>
        <h1>{tr("clustering.title")}</h1>
        <div style={{ padding: 16, background: "#fef3c7", borderRadius: 6, border: "1px solid #f59e0b", color: "#92400e" }}>
          ⚠ Selesaikan tahap <strong>Preprocessing</strong> terlebih dahulu sebelum melakukan Clustering.{" "}
          <a href="/preprocessing" style={{ color: "#92400e", fontWeight: 600 }}>Buka Preprocessing →</a>
        </div>
      </div>
    );
  }

  if (problemType && !isUnsupervised) {
    return (
      <div>
        <h1>{tr("clustering.title")}</h1>
        <div style={{ padding: 16, background: "#fef3c7", borderRadius: 6, border: "1px solid #f59e0b", color: "#92400e" }}>
          ⚠ Halaman ini untuk <strong>Unsupervised Learning (Clustering)</strong>.<br />
          Mode aktif saat ini: <strong>{problemType}</strong> — pilih tipe masalah <em>Clustering</em> di halaman{" "}
          <a href="/preprocessing" style={{ color: "#92400e", fontWeight: 600 }}>Preprocessing →</a>{" "}
          untuk mengaktifkan analisis ini.
        </div>
      </div>
    );
  }

  async function runClustering() {
    setBusy(true);
    setError(null);
    setResult(null);

    try {
      const params =
        method === "kmeans" || method === "hierarchical" || method === "spectral"
          ? { n_clusters: Number(nClusters) }
          : { eps: Number(eps), min_samples: Number(minSamples) };

      const r = await api.performClustering({
        state_id: stateId,
        method: method,
        parameters: params,
      });

      if (!r.success) throw new Error(r.error);
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function findOptimalK() {
    setBusy(true);
    setError(null);
    setOptimalKResult(null);

    try {
      const r = await api.findOptimalK({
        state_id: stateId,
        method: "kmeans",
        parameters: { max_k: Number(maxK) },
      });

      if (!r.success) throw new Error(r.error);
      setOptimalKResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1>{tr("clustering.title")}</h1>
      <p style={{ color: "#64748b" }}>
        Temukan pola tersembunyi dalam data melalui analisis klasterisasi.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
          marginTop: 16,
          maxWidth: 720,
        }}
      >
        <label>
          Metode Clustering
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            style={{ width: "100%" }}
          >
            {CLUSTERING_METHODS.map((m) => (
              <option key={m} value={m}>
                {m.charAt(0).toUpperCase() + m.slice(1)}
              </option>
            ))}
          </select>
        </label>

        {(method === "kmeans" || method === "hierarchical" || method === "spectral") && (
          <label>
          Jumlah Klaster (k)
            <input
              type="number"
              min="2"
              max="20"
              value={nClusters}
              onChange={(e) => setNClusters(e.target.value)}
              style={{ width: "100%" }}
            />
          </label>
        )}

        {method === "dbscan" && (
          <>
            <label>
              EPS
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="2.0"
                value={eps}
                onChange={(e) => setEps(e.target.value)}
                style={{ width: "100%" }}
              />
            </label>
            <label>
              Min Samples
              <input
                type="number"
                min="2"
                max="20"
                value={minSamples}
                onChange={(e) => setMinSamples(e.target.value)}
                style={{ width: "100%" }}
              />
            </label>
          </>
        )}
      </div>

      <div
        style={{
          display: "flex",
          gap: 8,
          marginTop: 16,
        }}
      >
        <button
          onClick={runClustering}
          disabled={busy}
          style={{
            padding: "10px 20px",
            background: "#1e40af",
            color: "#fff",
            border: "none",
            borderRadius: 6,
          }}
        >
          {busy ? "Menjalankan..." : "🔬 Jalankan Clustering"}
        </button>

        {(method === "kmeans" || method === "hierarchical") && (
          <button
            onClick={findOptimalK}
            disabled={busy}
            style={{
              padding: "10px 20px",
              background: "#059669",
              color: "#fff",
              border: "none",
              borderRadius: 6,
            }}
          >
            {busy ? "Menganalisis..." : "📊 Cari K Optimal"}
          </button>
        )}
      </div>

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

      {optimalKResult && optimalKResult.success && (
        <div
          style={{
            marginTop: 16,
            padding: 16,
            background: "#f0fdf4",
            borderRadius: 6,
            border: "1px solid #16a34a",
          }}
        >
          <h3>Optimal K Analysis</h3>
          <p>
            Optimal K (Elbow Method):{" "}
            <strong>{optimalKResult.optimal_k_elbow}</strong>
          </p>
          <p>
            Optimal K (Silhouette):{" "}
            <strong>{optimalKResult.optimal_k_silhouette}</strong>
          </p>
          <div style={{ marginTop: 12 }}>
            <h4>Silhouette Scores by K:</h4>
            <ul>
              {optimalKResult.k_values.map((k, i) => (
                <li key={k}>
                  K={k}: {optimalKResult.silhouette_scores[i]?.toFixed(3)}
                </li>
              ))}
            </ul>
          </div>
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
          <h3>✅ Hasil Clustering</h3>
          <p>Metode: <strong>{result.method}</strong></p>
          <p>Jumlah Klaster: <strong>{result.metrics.n_clusters}</strong></p>
          <p>
            Silhouette Score:{" "}
            <strong>{result.metrics.silhouette_score?.toFixed(3)}</strong>
          </p>
          <p>
            Calinski-Harabasz Score:{" "}
            {result.metrics.calinski_harabasz_score?.toFixed(2)}
          </p>
          <p>
            Davies-Bouldin Score:{" "}
            {result.metrics.davies_bouldin_score?.toFixed(3)}
          </p>
          {result.metrics.n_noise > 0 && (
            <p>Noise Points (DBSCAN): {result.metrics.n_noise}</p>
          )}
          {result.metrics.cluster_sizes && (
            <div style={{ marginTop: 12 }}>
              <h4>Ukuran Klaster:</h4>
              <ul>
                {Object.entries(result.metrics.cluster_sizes).map(
                  ([cluster, size]) => (
                    <li key={cluster}>
                      Klaster {cluster}: {size} sampel
                    </li>
                  )
                )}
              </ul>
            </div>
          )}

          <div style={{ marginTop: 16, padding: 12, background: "#eff6ff", borderRadius: 6, border: "1px solid #bfdbfe" }}>
            <p style={{ margin: 0, color: "#1d4ed8", fontWeight: 600 }}>
              🎉 Clustering selesai!
            </p>
            <p style={{ margin: "6px 0 0", color: "#1d4ed8" }}>
              Lanjut ke{" "}
              <a href="/interpretation" style={{ color: "#1d4ed8", fontWeight: 600 }}>Interpretasi Model →</a>{" "}
              atau unduh hasil cluster.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}