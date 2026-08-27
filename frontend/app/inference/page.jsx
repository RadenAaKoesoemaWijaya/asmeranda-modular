"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useWorkflow } from "@/lib/workflow-store";
import { useT } from "@/lib/i18n";

export default function InferencePage() {
  const lang = useWorkflow((s) => s.language) || "id";
  const tr = useT(lang);
  const activeModelId = useWorkflow((s) => s.modelId);
  const activeModelType = useWorkflow((s) => s.modelType);
  const activeProblemType = useWorkflow((s) => s.problemType);
  const activeFeatureNames = useWorkflow((s) => s.featureNames) || [];

  // Model selection
  const [modelSource, setModelSource] = useState("platform"); // "platform" | "upload"
  const [selectedModelId, setSelectedModelId] = useState(activeModelId || "");
  const [availableModels, setAvailableModels] = useState([]);
  const [modelMetadata, setModelMetadata] = useState(null);
  const [uploadedModelFile, setUploadedModelFile] = useState(null);
  const [modelLoading, setModelLoading] = useState(false);

  // Prediction Mode
  const [predictionMode, setPredictionMode] = useState("form"); // "form" | "batch"

  // Form input mode state
  const [formInputs, setFormInputs] = useState({});
  const [singleResult, setSingleResult] = useState(null);
  const [singleBusy, setSingleBusy] = useState(false);

  // Batch file mode state
  const [batchFile, setBatchFile] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [batchBusy, setBatchBusy] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);

  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Fetch available models on load
  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    try {
      const res = await api.listModels();
      if (res) {
        const modelsList = Object.values(res);
        setAvailableModels(modelsList);
        if (!selectedModelId && modelsList.length > 0) {
          setSelectedModelId(modelsList[modelsList.length - 1].model_id);
        }
      }
    } catch (e) {
      console.error("Gagal memuat daftar model:", e);
    }
  }

  // When selected model changes, fetch its metadata and initialize form
  useEffect(() => {
    if (!selectedModelId) return;
    fetchModelDetails(selectedModelId);
  }, [selectedModelId]);

  async function fetchModelDetails(id) {
    try {
      const meta = await api.getModel(id);
      if (meta) {
        setModelMetadata(meta);
        const feats = meta.feature_names || [];
        const initInputs = {};
        feats.forEach((f) => {
          initInputs[f] = 0.0;
        });
        setFormInputs(initInputs);
      }
    } catch (e) {
      console.error("Gagal memuat metadata model:", e);
    }
  }

  // Handle uploading external .pkl file
  async function handleModelUpload(file) {
    if (!file) return;
    setModelLoading(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const res = await api.uploadModel(file);
      if (res && res.success) {
        setSuccessMsg(res.message || "Model .pkl berhasil diimpor!");
        await loadModels();
        setSelectedModelId(res.model_id);
        setModelSource("platform");
      } else {
        throw new Error(res.error || "Gagal mengimpor model");
      }
    } catch (e) {
      setError(e.message || "Terjadi kesalahan saat mengunggah model");
    } finally {
      setModelLoading(false);
    }
  }

  // Handle single record prediction
  async function handleSinglePredict(e) {
    if (e) e.preventDefault();
    if (!selectedModelId) {
      setError("Pilih atau unggah model terlebih dahulu");
      return;
    }
    setSingleBusy(true);
    setError(null);
    setSingleResult(null);

    try {
      const payload = [formInputs];
      const res = await api.predictWithModel(selectedModelId, payload);
      if (!res.success) {
        throw new Error(res.error || "Gagal melakukan prediksi");
      }
      setSingleResult(res);
    } catch (e) {
      setError(e.message || "Gagal melakukan prediksi data");
    } finally {
      setSingleBusy(false);
    }
  }

  // Handle batch file prediction
  async function handleBatchPredict() {
    if (!selectedModelId) {
      setError("Pilih atau unggah model terlebih dahulu");
      return;
    }
    if (!batchFile) {
      setError("Pilih file CSV, Excel, atau JSON data baru terlebih dahulu");
      return;
    }

    setBatchBusy(true);
    setBatchProgress(0);
    setError(null);
    setBatchResult(null);

    try {
      const res = await api.predictWithFile(selectedModelId, batchFile, (pct) => {
        setBatchProgress(pct);
      });

      if (!res.success) {
        throw new Error(res.error || "Gagal melakukan prediksi batch file");
      }
      setBatchResult(res);
      setSuccessMsg(res.message || "Prediksi batch selesai!");
    } catch (e) {
      setError(e.message || "Gagal melakukan deteksi batch pada file");
    } finally {
      setBatchBusy(false);
    }
  }

  // Download predictions table as CSV
  function downloadBatchCSV() {
    if (!batchResult || !batchResult.preview) return;
    const rows = batchResult.preview;
    if (rows.length === 0) return;

    const headers = Object.keys(rows[0]);
    const csvContent = [
      headers.join(","),
      ...rows.map((r) =>
        headers.map((h) => `"${String(r[h] ?? "").replace(/"/g, '""')}"`).join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `predictions_${selectedModelId}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  const currentFeatures =
    modelMetadata?.feature_names || activeFeatureNames || [];
  const currentProblemType =
    modelMetadata?.problem_type || activeProblemType || "Classification";

  return (
    <div style={{ maxWidth: 1120, margin: "0 auto" }}>
      {/* ── Page Header ── */}
      <div className="page-header">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h1 className="page-title">🔮 Deteksi & Inferensi Data Baru</h1>
            <p className="page-subtitle">
              Gunakan model machine learning yang telah dilatih atau di-download (.pkl) untuk mendeteksi anomali, klasifikasi target, atau estimasi regresi pada data baru secara interaktif atau batch file.
            </p>
          </div>
          {selectedModelId && (
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span className="sidebar-link-badge" style={{ background: "var(--color-primary-600)", fontSize: 13, padding: "6px 12px" }}>
                Model: {modelMetadata?.model_type || activeModelType || "ML Model"}
              </span>
              <button
                onClick={() => {
                  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_PATH || "/api/v1";
                  window.open(`${API_BASE}/training/models/${selectedModelId}/download`, "_blank");
                }}
                className="btn btn-secondary btn-sm"
                title="Download Serialized .pkl File"
              >
                💾 Unduh .pkl
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Notification Banners ── */}
      {error && (
        <div className="alert alert-error mb-6">
          <span>❌</span>
          <div>
            <strong>Terjadi Kesalahan:</strong>
            <p style={{ margin: 0 }}>{error}</p>
          </div>
        </div>
      )}

      {successMsg && (
        <div className="alert alert-success mb-6">
          <span>✅</span>
          <div>
            <strong>Sukses:</strong>
            <p style={{ margin: 0 }}>{successMsg}</p>
          </div>
        </div>
      )}

      {/* ── Section 1: Model Source & Selection ── */}
      <div className="card mb-6">
        <div className="flex-between mb-4">
          <h2 className="card-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span>📦</span> Pilih Model untuk Inferensi
          </h2>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => setModelSource("platform")}
              className={`btn btn-sm ${modelSource === "platform" ? "btn-primary" : "btn-secondary"}`}
            >
              Model Tersimpan ({availableModels.length})
            </button>
            <button
              onClick={() => setModelSource("upload")}
              className={`btn btn-sm ${modelSource === "upload" ? "btn-primary" : "btn-secondary"}`}
            >
              📤 Unggah Model Eksternal (.pkl)
            </button>
          </div>
        </div>

        {modelSource === "platform" ? (
          availableModels.length > 0 ? (
            <div className="grid-2" style={{ gap: 16 }}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label>Model Terdaftar:</label>
                <select
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                >
                  {availableModels.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.model_type} (ID: {m.model_id.slice(0, 10)}...) — {m.problem_type} ({m.n_features} fitur)
                    </option>
                  ))}
                </select>
              </div>

              {modelMetadata && (
                <div style={{ padding: 12, background: "var(--color-slate-50)", borderRadius: "var(--radius-md)", border: "1px solid var(--color-slate-200)", fontSize: 13 }}>
                  <div style={{ color: "var(--color-slate-700)" }}>
                    Tipe Masalah: <strong>{modelMetadata.problem_type}</strong> | Fitur Diperlukan: <strong>{currentFeatures.length} kolom</strong>
                  </div>
                  <div style={{ marginTop: 4, color: "var(--color-slate-500)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    Daftar Fitur: {currentFeatures.slice(0, 8).join(", ")}{currentFeatures.length > 8 ? "..." : ""}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="alert alert-info">
              <span>ℹ️</span>
              <div>
                <strong>Belum ada model tersimpan:</strong>
                <p style={{ marginTop: 4, marginBottom: 0 }}>
                  Latih model di menu <a href="/training" style={{ color: "#1d4ed8", fontWeight: 600, textDecoration: "underline" }}>Pelatihan Model</a> atau unggah file <code>.pkl</code> yang sudah diunduh.
                </p>
              </div>
            </div>
          )
        ) : (
          <div style={{ padding: 20, border: "2px dashed #cbd5e1", borderRadius: "var(--radius-lg)", background: "#f8fafc", textAlign: "center" }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>📤</div>
            <h4 style={{ margin: "0 0 4px 0", color: "#1e293b" }}>Unggah File Model (.pkl / .pickle)</h4>
            <p style={{ fontSize: 13, color: "#64748b", margin: "0 0 12px 0" }}>
              Gunakan file model yang telah Anda ekspor dari Asmeranda AI atau scikit-learn / XGBoost pipeline.
            </p>
            <input
              type="file"
              accept=".pkl,.pickle"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleModelUpload(e.target.files[0]);
                }
              }}
              disabled={modelLoading}
              style={{ display: "inline-block", maxWidth: 320 }}
            />
            {modelLoading && <p style={{ color: "#1d4ed8", fontSize: 13, marginTop: 8 }}>⏳ Memproses dan memvalidasi file model...</p>}
          </div>
        )}
      </div>

      {/* ── Section 2: Choose Prediction Method ── */}
      {selectedModelId && (
        <>
          <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
            <button
              onClick={() => setPredictionMode("form")}
              className={`btn btn-lg ${predictionMode === "form" ? "btn-primary" : "btn-secondary"}`}
              style={{ flex: 1 }}
            >
              📝 Input Satuan (Form Interaktif)
            </button>
            <button
              onClick={() => setPredictionMode("batch")}
              className={`btn btn-lg ${predictionMode === "batch" ? "btn-primary" : "btn-secondary"}`}
              style={{ flex: 1 }}
            >
              📁 Deteksi Sekaligus (Batch File CSV / Excel)
            </button>
          </div>

          {/* ── Mode 1: Interactive Form Input ── */}
          {predictionMode === "form" && (
            <div className="card mb-6">
              <h3 className="card-title mb-2">📝 Input Fitur Data Baru</h3>
              <p className="card-subtitle mb-4">
                Masukkan nilai atribut data baru untuk mendapatkan prediksi langsung beserta probabilitas keyakinan model.
              </p>

              <form onSubmit={handleSinglePredict}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                    gap: 16,
                    maxHeight: 420,
                    overflowY: "auto",
                    padding: 8,
                    background: "#f8fafc",
                    borderRadius: "var(--radius-lg)",
                    border: "1px solid #e2e8f0",
                    marginBottom: 20,
                  }}
                >
                  {currentFeatures.length > 0 ? (
                    currentFeatures.map((feat) => (
                      <div key={feat} className="form-group" style={{ marginBottom: 0 }}>
                        <label style={{ fontSize: 13, fontWeight: 600, color: "#334155" }} title={feat}>
                          {feat}
                        </label>
                        <input
                          type="number"
                          step="any"
                          value={formInputs[feat] ?? 0}
                          onChange={(e) =>
                            setFormInputs({
                              ...formInputs,
                              [feat]: parseFloat(e.target.value) || 0,
                            })
                          }
                          required
                          style={{ background: "#ffffff" }}
                        />
                      </div>
                    ))
                  ) : (
                    <div style={{ color: "#64748b", padding: 12 }}>
                      Tidak ada daftar fitur eksplisit, sistem akan menerima input numerik otomatis.
                    </div>
                  )}
                </div>

                <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                  <button
                    type="submit"
                    disabled={singleBusy}
                    className="btn btn-primary btn-lg"
                  >
                    {singleBusy ? "⏳ Menghitung Prediksi..." : "🚀 Deteksi / Prediksi Sekarang"}
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      const sample = {};
                      currentFeatures.forEach((f) => {
                        sample[f] = parseFloat((Math.random() * 5).toFixed(2));
                      });
                      setFormInputs(sample);
                    }}
                    className="btn btn-secondary btn-lg"
                  >
                    🎲 Isi Nilai Contoh (Random Sample)
                  </button>
                </div>
              </form>

              {/* Single Prediction Result Display */}
              {singleResult && singleResult.success && (
                <div
                  style={{
                    marginTop: 24,
                    padding: 20,
                    background: "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
                    borderRadius: "var(--radius-xl)",
                    border: "1px solid #86efac",
                  }}
                >
                  <h4 style={{ margin: "0 0 12px 0", color: "#166534", fontSize: 16, display: "flex", alignItems: "center", gap: 8 }}>
                    <span>🎯</span> Hasil Deteksi & Prediksi Model
                  </h4>

                  <div className="grid-2" style={{ gap: 16 }}>
                    <div style={{ padding: 16, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0" }}>
                      <div style={{ fontSize: 12, color: "#166534", fontWeight: 700 }}>NILAI PREDIKSI KELUARAN</div>
                      <div style={{ fontSize: 32, fontWeight: 800, color: "#15803d", marginTop: 4 }}>
                        {singleResult.predictions && singleResult.predictions[0] !== undefined
                          ? String(singleResult.predictions[0])
                          : "N/A"}
                      </div>
                    </div>

                    {singleResult.probabilities && singleResult.probabilities[0] && (
                      <div style={{ padding: 16, background: "#ffffff", borderRadius: "var(--radius-lg)", border: "1px solid #bbf7d0" }}>
                        <div style={{ fontSize: 12, color: "#166534", fontWeight: 700 }}>TINGKAT KEYAKINAN (CONFIDENCE)</div>
                        <div style={{ fontSize: 32, fontWeight: 800, color: "#1e40af", marginTop: 4 }}>
                          {(Math.max(...singleResult.probabilities[0]) * 100).toFixed(2)}%
                        </div>
                        <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>
                          Distribusi Probabilitas Kelas: [{singleResult.probabilities[0].map((p) => (p * 100).toFixed(1) + "%").join(" | ")}]
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Mode 2: Batch File Upload (CSV/Excel/JSON) ── */}
          {predictionMode === "batch" && (
            <div className="card mb-6">
              <h3 className="card-title mb-2">📁 Deteksi Data Sekaligus (Batch File)</h3>
              <p className="card-subtitle mb-4">
                Unggah file dataset baru dalam format CSV, Excel (.xlsx), atau JSON. Model akan mendeteksi setiap baris data dan menambahkan kolom <code>Prediction</code> serta <code>Confidence</code>.
              </p>

              <div
                style={{
                  padding: 24,
                  border: "2px dashed #93c5fd",
                  borderRadius: "var(--radius-lg)",
                  background: "#eff6ff",
                  textAlign: "center",
                  marginBottom: 20,
                }}
              >
                <div style={{ fontSize: 36, marginBottom: 8 }}>📄</div>
                <h4 style={{ margin: "0 0 6px 0", color: "#1e3a8a" }}>Pilih File Data Baru</h4>
                <p style={{ fontSize: 13, color: "#3b82f6", margin: "0 0 16px 0" }}>
                  Mendukung .csv, .xlsx, .xls, .json
                </p>
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setBatchFile(e.target.files[0]);
                    }
                  }}
                  style={{ display: "inline-block", maxWidth: 360 }}
                />
                {batchFile && (
                  <div style={{ marginTop: 12, fontSize: 14, fontWeight: 600, color: "#1e40af" }}>
                    ✓ File Terpilih: {batchFile.name} ({(batchFile.size / 1024).toFixed(1)} KB)
                  </div>
                )}
              </div>

              <button
                onClick={handleBatchPredict}
                disabled={!batchFile || batchBusy}
                className="btn btn-primary btn-lg"
              >
                {batchBusy ? `⏳ Memproses Batch Prediksi (${batchProgress}%)...` : "🚀 Jalankan Batch Deteksi & Prediksi"}
              </button>

              {/* Batch Prediction Results Preview & Download */}
              {batchResult && batchResult.success && (
                <div style={{ marginTop: 24 }}>
                  <div className="flex-between mb-4">
                    <div>
                      <h4 style={{ margin: 0, color: "#15803d", fontSize: 16 }}>
                        ✅ Prediksi Selesai ({batchResult.total_rows} Baris Berhasil Diproses)
                      </h4>
                      <p style={{ margin: 0, fontSize: 13, color: "#475569" }}>
                        Menampilkan pratinjau hasil prediksi dan skor probabilitas per baris:
                      </p>
                    </div>

                    <button
                      onClick={downloadBatchCSV}
                      className="btn btn-secondary btn-sm"
                      style={{ background: "#15803d", color: "#ffffff", borderColor: "#15803d" }}
                    >
                      💾 Unduh Hasil Lengkap (.CSV)
                    </button>
                  </div>

                  {batchResult.preview && batchResult.preview.length > 0 && (
                    <div className="table-container" style={{ maxHeight: 360, overflowY: "auto" }}>
                      <table>
                        <thead>
                          <tr>
                            <th>#</th>
                            {Object.keys(batchResult.preview[0]).map((col) => (
                              <th
                                key={col}
                                style={{
                                  background: col === "Prediction" || col === "Confidence" ? "#dcfce7" : undefined,
                                  color: col === "Prediction" || col === "Confidence" ? "#166534" : undefined,
                                  fontWeight: col === "Prediction" || col === "Confidence" ? 700 : 600,
                                }}
                              >
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {batchResult.preview.map((row, idx) => (
                            <tr key={idx}>
                              <td style={{ color: "#64748b", fontSize: 12 }}>{idx + 1}</td>
                              {Object.keys(batchResult.preview[0]).map((col) => (
                                <td
                                  key={col}
                                  style={{
                                    fontWeight: col === "Prediction" ? 700 : undefined,
                                    color: col === "Prediction" ? "#15803d" : col === "Confidence" ? "#1e40af" : undefined,
                                  }}
                                >
                                  {String(row[col] ?? "")}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
