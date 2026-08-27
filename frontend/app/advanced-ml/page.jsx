'use client';

import { useState, useEffect, useMemo } from 'react';
import { api } from '@/lib/api';
import { useWorkflow } from '@/lib/workflow-store';
import { useT } from '@/lib/i18n';

const COLOR_PALETTE = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1'
];

export default function AdvancedMLPage() {
  const lang = useWorkflow((s) => s.language) || "id";
  const tr = useT(lang);
  const stateId = useWorkflow((s) => s.stateId);
  const datasetName = useWorkflow((s) => s.datasetName);
  const storeNumericalColumns = useWorkflow((s) => s.numericalColumns) || [];
  const setStore = useWorkflow((s) => s.set);

  const [activeTab, setActiveTab] = useState('umap');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Available Columns from backend
  const [availableColumns, setAvailableColumns] = useState([]);
  const [numericalCols, setNumericalCols] = useState(storeNumericalColumns);

  // Results State
  const [umapResult, setUmapResult] = useState(null);
  const [hdbscanResult, setHdbscanResult] = useState(null);
  const [anomalyResult, setAnomalyResult] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);
  const [missingResult, setMissingResult] = useState(null);
  const [outlierResult, setOutlierResult] = useState(null);

  // Form States - UMAP / PCA / t-SNE
  const [umapMethod, setUmapMethod] = useState('umap');
  const [nComponents, setNComponents] = useState(2);
  const [nNeighbors, setNNeighbors] = useState(15);
  const [minDist, setMinDist] = useState(0.1);
  const [umapMetric, setUmapMetric] = useState('euclidean');
  const [perplexity, setPerplexity] = useState(30);

  // Form States - HDBSCAN
  const [minClusterSize, setMinClusterSize] = useState(5);
  const [minSamples, setMinSamples] = useState('');
  const [hdbscanMetric, setHdbscanMetric] = useState('euclidean');

  // Form States - Anomaly Detection
  const [anomalyMethod, setAnomalyMethod] = useState('isolation_forest');
  const [contamination, setContamination] = useState(0.1);
  const [nEstimators, setNEstimators] = useState(100);

  // Form States - Forecasting
  const [forecastTarget, setForecastTarget] = useState('');
  const [forecastPeriods, setForecastPeriods] = useState(10);
  const [forecastMethod, setForecastMethod] = useState('arima');

  // Form States - Utilities
  const [missingStrategy, setMissingStrategy] = useState('auto');
  const [numericImpute, setNumericImpute] = useState('mean');
  const [outlierMethod, setOutlierMethod] = useState('iqr');
  const [outlierThreshold, setOutlierThreshold] = useState(1.5);

  // Fetch Columns when stateId is ready
  useEffect(() => {
    if (!stateId) return;
    api.advancedML.getColumns(stateId)
      .then((res) => {
        if (res && res.success) {
          setAvailableColumns(res.columns || []);
          if (res.numerical_columns && res.numerical_columns.length > 0) {
            setNumericalCols(res.numerical_columns);
            if (!forecastTarget) {
              setForecastTarget(res.numerical_columns[0]);
            }
          }
        }
      })
      .catch((err) => console.warn("Failed to get state columns:", err));
  }, [stateId]);

  // Set default forecast target if numerical columns exist
  useEffect(() => {
    if (!forecastTarget && numericalCols.length > 0) {
      setForecastTarget(numericalCols[0]);
    }
  }, [numericalCols, forecastTarget]);

  // Handlers
  const handleRunUMAP = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.advancedML.runUMAP({
        state_id: stateId,
        method: umapMethod,
        n_components: Number(nComponents),
        n_neighbors: Number(nNeighbors),
        min_dist: Number(minDist),
        metric: umapMetric,
      });
      if (!res.success) throw new Error(res.error || 'Dimensionality reduction failed');
      setUmapResult(res);
      setStore({ advancedMLResults: { ...(useWorkflow.getState().advancedMLResults || {}), umap: res } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunHDBSCAN = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.advancedML.runHDBSCAN({
        state_id: stateId,
        min_cluster_size: Number(minClusterSize),
        min_samples: minSamples ? Number(minSamples) : null,
        metric: hdbscanMetric,
      });
      if (!res.success) throw new Error(res.error || 'HDBSCAN clustering failed');
      setHdbscanResult(res);
      setStore({ advancedMLResults: { ...(useWorkflow.getState().advancedMLResults || {}), hdbscan: res } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAnomalyDetection = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.advancedML.runAnomalyDetection({
        state_id: stateId,
        method: anomalyMethod,
        contamination: Number(contamination),
        n_estimators: Number(nEstimators),
      });
      if (!res.success) throw new Error(res.error || 'Anomaly detection failed');
      setAnomalyResult(res);
      setStore({ advancedMLResults: { ...(useWorkflow.getState().advancedMLResults || {}), anomaly: res } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunForecast = async () => {
    if (!forecastTarget) {
      setError(lang === 'id' ? 'Silakan pilih kolom target terlebih dahulu' : 'Please select a target column first');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.advancedML.runForecast({
        state_id: stateId,
        target_column: forecastTarget,
        periods: Number(forecastPeriods),
        method: forecastMethod,
      });
      if (!res.success) throw new Error(res.error || 'Forecasting failed');
      setForecastResult(res);
      setStore({ advancedMLResults: { ...(useWorkflow.getState().advancedMLResults || {}), forecast: res } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunMissingValues = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.advancedML.handleMissingValues({
        state_id: stateId,
        strategy: missingStrategy,
        numeric_strategy: numericImpute,
        categorical_strategy: 'mode',
        threshold: 0.5,
      });
      if (!res.success) throw new Error(res.error || 'Missing value handling failed');
      setMissingResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunDetectOutliers = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.advancedML.detectOutliers({
        state_id: stateId,
        method: outlierMethod,
        threshold: Number(outlierThreshold),
      });
      if (!res.success) throw new Error(res.error || 'Outlier detection failed');
      setOutlierResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'umap', icon: '🌌', label: tr('advanced_ml.umap') },
    { id: 'hdbscan', icon: '🧬', label: tr('advanced_ml.hdbscan') },
    { id: 'anomaly', icon: '🚨', label: tr('advanced_ml.anomaly') },
    { id: 'forecast', icon: '📈', label: tr('advanced_ml.forecast') },
    { id: 'utilities', icon: '🛠️', label: tr('advanced_ml.utilities') },
  ];

  if (!stateId) {
    return (
      <div className="card" style={{ maxWidth: 840, margin: '0 auto' }}>
        <h1 className="page-title">🚀 {tr('advanced_ml.title')}</h1>
        <p className="page-subtitle">{tr('advanced_ml.subtitle')}</p>
        <div className="alert alert-warning mt-6">
          <span>⚠️</span>
          <div>
            <strong>Tahap Preprocessing Diperlukan:</strong>
            <p style={{ marginTop: 4 }}>
              Silakan selesaikan tahap <a href="/preprocessing" style={{ color: '#b45309', fontWeight: 600, textDecoration: 'underline' }}>Preprocessing</a> terlebih dahulu agar data terstruktur siap diproses oleh modul Advanced ML.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 1120, margin: '0 auto' }}>
      {/* ── Page Header ── */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h1 className="page-title">🚀 {tr('advanced_ml.title')}</h1>
            <p className="page-subtitle">{tr('advanced_ml.subtitle')}</p>
          </div>
          {datasetName && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 14px', background: 'var(--color-primary-50)', borderRadius: 'var(--radius-full)', border: '1px solid var(--color-primary-200)', fontSize: 13, color: 'var(--color-primary-700)' }}>
              <span>📦 Dataset:</span>
              <span style={{ fontWeight: 600 }}>{datasetName}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Navigation Tabs ── */}
      <div style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 8, borderBottom: '1px solid var(--color-slate-200)', marginBottom: 24 }}>
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                setError(null);
              }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 18px',
                borderRadius: 'var(--radius-lg)',
                border: 'none',
                background: isActive ? 'var(--color-primary-600)' : 'transparent',
                color: isActive ? '#fff' : 'var(--color-slate-600)',
                fontWeight: isActive ? 600 : 500,
                fontSize: 14,
                cursor: 'pointer',
                transition: 'all var(--transition-fast)',
                boxShadow: isActive ? '0 4px 12px rgba(37, 99, 235, 0.25)' : 'none',
                whiteSpace: 'nowrap'
              }}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

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

      {/* ── Tab 1: UMAP & Dimensionality Reduction ── */}
      {activeTab === 'umap' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Controls Card */}
          <div className="card">
            <div className="flex-between mb-4">
              <div>
                <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>🌌</span> UMAP & Reduksi Dimensi Interaktif
                </h2>
                <p className="card-subtitle">
                  Proyeksikan data berdimensi banyak ke bidang 2D menggunakan topologi manifold nonlinear (UMAP), PCA, atau t-SNE.
                </p>
              </div>
              <span className="sidebar-link-badge" style={{ background: 'var(--color-primary-500)', padding: '4px 10px', fontSize: 11 }}>
                Non-linear Manifold
              </span>
            </div>

            <div className="grid-3" style={{ gap: 16 }}>
              <div className="form-group">
                <label>Algoritma Reduksi</label>
                <select
                  value={umapMethod}
                  onChange={(e) => setUmapMethod(e.target.value)}
                >
                  <option value="umap">UMAP (Uniform Manifold Approx & Proj)</option>
                  <option value="pca">PCA (Principal Component Analysis)</option>
                  <option value="tsne">t-SNE (t-Distributed Stochastic Neighbor)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Dimensi Output (n_components)</label>
                <select
                  value={nComponents}
                  onChange={(e) => setNComponents(Number(e.target.value))}
                >
                  <option value={2}>2 Dimensi (2D Plot)</option>
                  <option value={3}>3 Dimensi (3D Data)</option>
                </select>
              </div>

              {umapMethod === 'umap' && (
                <>
                  <div className="form-group">
                    <label>Jumlah Tetangga (n_neighbors): <strong>{nNeighbors}</strong></label>
                    <input
                      type="number"
                      min={2}
                      max={100}
                      value={nNeighbors}
                      onChange={(e) => setNNeighbors(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Jarak Minimum (min_dist): <strong>{minDist}</strong></label>
                    <input
                      type="number"
                      step={0.05}
                      min={0.01}
                      max={0.99}
                      value={minDist}
                      onChange={(e) => setMinDist(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Metrik Jarak (Distance Metric)</label>
                    <select
                      value={umapMetric}
                      onChange={(e) => setUmapMetric(e.target.value)}
                    >
                      <option value="euclidean">Euclidean</option>
                      <option value="cosine">Cosine</option>
                      <option value="manhattan">Manhattan</option>
                      <option value="correlation">Correlation</option>
                    </select>
                  </div>
                </>
              )}

              {umapMethod === 'tsne' && (
                <div className="form-group">
                  <label>Perplexity: <strong>{perplexity}</strong></label>
                  <input
                    type="number"
                    min={5}
                    max={100}
                    value={perplexity}
                    onChange={(e) => setPerplexity(e.target.value)}
                  />
                </div>
              )}
            </div>

            <div style={{ marginTop: 12 }}>
              <button
                onClick={handleRunUMAP}
                disabled={loading}
                className="btn btn-primary btn-lg"
              >
                {loading ? '⏳ Menghitung Reduksi Dimensi...' : `🚀 Jalankan ${umapMethod.toUpperCase()}`}
              </button>
            </div>
          </div>

          {/* Visualization Results */}
          {umapResult && umapResult.success && (
            <div className="card">
              <div className="flex-between mb-4">
                <h3 className="card-title">📊 Hasil Proyeksi 2D ({umapResult.method?.toUpperCase()})</h3>
                <div style={{ display: 'flex', gap: 8 }}>
                  <span className="sidebar-link-badge" style={{ background: '#059669' }}>
                    {umapResult.n_samples || (umapResult.points ? umapResult.points.length : 0)} Sampel
                  </span>
                  <span className="sidebar-link-badge" style={{ background: '#7c3aed' }}>
                    {umapResult.n_features || 'N'} Fitur Input
                  </span>
                </div>
              </div>

              {umapResult.parameters?.explained_variance_ratio && (
                <div className="alert alert-info mb-4" style={{ padding: '8px 12px', fontSize: 13 }}>
                  <span>ℹ️</span>
                  <span>
                    Total Variansi Terjelaskan: <strong>{((umapResult.parameters.total_explained_variance || 0) * 100).toFixed(1)}%</strong> (PC1: {((umapResult.parameters.explained_variance_ratio[0] || 0) * 100).toFixed(1)}%, PC2: {((umapResult.parameters.explained_variance_ratio[1] || 0) * 100).toFixed(1)}%)
                  </span>
                </div>
              )}

              {/* Interactive Scatter Plot */}
              <ScatterPlot points={umapResult.points || []} xLabel={umapResult.columns?.[0] || 'Dimensi 1'} yLabel={umapResult.columns?.[1] || 'Dimensi 2'} />

              {/* Coordinates Preview */}
              {umapResult.data && (
                <div style={{ marginTop: 24 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-700)', marginBottom: 8 }}>
                    📋 Cuplikan Koordinat Tereduksi (10 Sampel Pertama):
                  </h4>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Index</th>
                          {(umapResult.columns || ['Dim 1', 'Dim 2']).map((c) => (
                            <th key={c}>{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(umapResult.data || []).slice(0, 10).map((row, i) => (
                          <tr key={i}>
                            <td><strong>#{i}</strong></td>
                            {(umapResult.columns || Object.keys(row)).map((col) => (
                              <td key={col}>{typeof row[col] === 'number' ? row[col].toFixed(4) : row[col]}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Tab 2: HDBSCAN Clustering ── */}
      {activeTab === 'hdbscan' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div className="card">
            <div className="flex-between mb-4">
              <div>
                <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>🧬</span> HDBSCAN Clustering
                </h2>
                <p className="card-subtitle">
                  Hierarchical Density-Based Spatial Clustering of Applications with Noise. Mengelompokkan data berdasarkan kerapatan tanpa perlu menentukan jumlah cluster K di awal.
                </p>
              </div>
              <span className="sidebar-link-badge" style={{ background: '#10b981', padding: '4px 10px', fontSize: 11 }}>
                Density-Based
              </span>
            </div>

            <div className="grid-3" style={{ gap: 16 }}>
              <div className="form-group">
                <label>Ukuran Minimum Cluster (min_cluster_size)</label>
                <input
                  type="number"
                  min={2}
                  max={100}
                  value={minClusterSize}
                  onChange={(e) => setMinClusterSize(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label>Min Samples (Opsional, Default = min_cluster_size)</label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  placeholder="Otomatis"
                  value={minSamples}
                  onChange={(e) => setMinSamples(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label>Metrik Jarak (Metric)</label>
                <select
                  value={hdbscanMetric}
                  onChange={(e) => setHdbscanMetric(e.target.value)}
                >
                  <option value="euclidean">Euclidean</option>
                  <option value="manhattan">Manhattan</option>
                </select>
              </div>
            </div>

            <div style={{ marginTop: 12 }}>
              <button
                onClick={handleRunHDBSCAN}
                disabled={loading}
                className="btn btn-primary btn-lg"
              >
                {loading ? '⏳ Menjalankan HDBSCAN...' : '🔬 Jalankan HDBSCAN'}
              </button>
            </div>
          </div>

          {hdbscanResult && hdbscanResult.success && (
            <div className="card">
              <h3 className="card-title mb-4">🎯 Hasil Klasterisasi HDBSCAN</h3>

              {/* Metric Cards */}
              <div className="grid-4 mb-6" style={{ gap: 12 }}>
                <div style={{ padding: 14, background: '#eff6ff', borderRadius: 'var(--radius-lg)', border: '1px solid #bfdbfe' }}>
                  <div style={{ fontSize: 11, color: '#1e40af', fontWeight: 600, textTransform: 'uppercase' }}>Cluster Terbentuk</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#1e3a8a', marginTop: 4 }}>{hdbscanResult.n_clusters}</div>
                </div>

                <div style={{ padding: 14, background: '#fef2f2', borderRadius: 'var(--radius-lg)', border: '1px solid #fecaca' }}>
                  <div style={{ fontSize: 11, color: '#991b1b', fontWeight: 600, textTransform: 'uppercase' }}>Titik Noise (-1)</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#7f1d1d', marginTop: 4 }}>
                    {hdbscanResult.n_noise} <span style={{ fontSize: 13, fontWeight: 400 }}>({((hdbscanResult.metrics?.noise_ratio || 0) * 100).toFixed(1)}%)</span>
                  </div>
                </div>

                <div style={{ padding: 14, background: '#f0fdf4', borderRadius: 'var(--radius-lg)', border: '1px solid #bbf7d0' }}>
                  <div style={{ fontSize: 11, color: '#166534', fontWeight: 600, textTransform: 'uppercase' }}>Silhouette Score</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#14532d', marginTop: 4 }}>
                    {hdbscanResult.metrics?.silhouette_score !== undefined ? hdbscanResult.metrics.silhouette_score.toFixed(3) : 'N/A'}
                  </div>
                </div>

                <div style={{ padding: 14, background: '#faf5ff', borderRadius: 'var(--radius-lg)', border: '1px solid #e9d5ff' }}>
                  <div style={{ fontSize: 11, color: '#6b21a8', fontWeight: 600, textTransform: 'uppercase' }}>Davies-Bouldin</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#581c87', marginTop: 4 }}>
                    {hdbscanResult.metrics?.davies_bouldin_score !== undefined ? hdbscanResult.metrics.davies_bouldin_score.toFixed(3) : 'N/A'}
                  </div>
                </div>
              </div>

              {/* 2D Cluster Map */}
              {hdbscanResult.plot_points && hdbscanResult.plot_points.length > 0 && (
                <div className="mb-6">
                  <h4 style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-700)', marginBottom: 8 }}>
                    🗺️ Peta Sebaran Klaster (Proyeksi 2D PCA/Density):
                  </h4>
                  <ScatterPlot
                    points={hdbscanResult.plot_points.map((p) => ({
                      ...p,
                      target: p.cluster === -1 ? 'Noise (-1)' : `Cluster ${p.cluster}`
                    }))}
                    xLabel="Komponen 1"
                    yLabel="Komponen 2"
                  />
                </div>
              )}

              {/* Cluster Breakdown */}
              {hdbscanResult.cluster_sizes && (
                <div>
                  <h4 style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-700)', marginBottom: 12 }}>
                    📊 Distribusi Ukuran Klaster:
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {Object.entries(hdbscanResult.cluster_sizes).map(([name, count], i) => {
                      const total = Object.values(hdbscanResult.cluster_sizes).reduce((a, b) => a + b, 0);
                      const pct = total > 0 ? (count / total) * 100 : 0;
                      const isNoise = name.includes('Noise');
                      return (
                        <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <span style={{ width: 120, fontSize: 13, fontWeight: 600, color: isNoise ? '#64748b' : 'var(--color-slate-800)' }}>
                            {name}
                          </span>
                          <div style={{ flex: 1, background: '#f1f5f9', height: 12, borderRadius: 6, overflow: 'hidden' }}>
                            <div
                              style={{
                                width: `${pct}%`,
                                height: '100%',
                                background: isNoise ? '#94a3b8' : COLOR_PALETTE[i % COLOR_PALETTE.length],
                                borderRadius: 6,
                                transition: 'width 0.4s ease'
                              }}
                            />
                          </div>
                          <span style={{ width: 80, textAlign: 'right', fontSize: 13, fontWeight: 600 }}>
                            {count} ({pct.toFixed(1)}%)
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Tab 3: Anomaly Detection ── */}
      {activeTab === 'anomaly' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div className="card">
            <div className="flex-between mb-4">
              <div>
                <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>🚨</span> Deteksi Anomali & Outlier
                </h2>
                <p className="card-subtitle">
                  Identifikasi observasi yang menyimpang secara signifikan menggunakan algoritma ensemble Isolation Forest atau batas kernel One-Class SVM.
                </p>
              </div>
              <span className="sidebar-link-badge" style={{ background: '#ef4444', padding: '4px 10px', fontSize: 11 }}>
                Unsupervised Outlier
              </span>
            </div>

            <div className="grid-3" style={{ gap: 16 }}>
              <div className="form-group">
                <label>Metode Deteksi</label>
                <select
                  value={anomalyMethod}
                  onChange={(e) => setAnomalyMethod(e.target.value)}
                >
                  <option value="isolation_forest">Isolation Forest (Pohon Isolasi)</option>
                  <option value="one_class_svm">One-Class SVM (Support Vector Machine)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Estimasi Kontaminasi (Contamination): <strong>{((contamination || 0) * 100).toFixed(0)}%</strong></label>
                <input
                  type="number"
                  step={0.02}
                  min={0.01}
                  max={0.4}
                  value={contamination}
                  onChange={(e) => setContamination(e.target.value)}
                />
              </div>

              {anomalyMethod === 'isolation_forest' && (
                <div className="form-group">
                  <label>Jumlah Estimator (Trees)</label>
                  <input
                    type="number"
                    min={20}
                    max={500}
                    value={nEstimators}
                    onChange={(e) => setNEstimators(e.target.value)}
                  />
                </div>
              )}
            </div>

            <div style={{ marginTop: 12 }}>
              <button
                onClick={handleRunAnomalyDetection}
                disabled={loading}
                className="btn btn-primary btn-lg"
                style={{ background: '#dc2626' }}
              >
                {loading ? '⏳ Mendeteksi Anomali...' : '🚨 Jalankan Deteksi Anomali'}
              </button>
            </div>
          </div>

          {anomalyResult && anomalyResult.success && (
            <div className="card">
              <h3 className="card-title mb-4">🔍 Ringkasan Temuan Anomali</h3>

              <div className="grid-3 mb-6" style={{ gap: 16 }}>
                <div style={{ padding: 16, background: '#fef2f2', borderRadius: 'var(--radius-lg)', border: '1px solid #fecaca' }}>
                  <div style={{ fontSize: 12, color: '#991b1b', fontWeight: 600 }}>JUMLAH ANOMALI</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#dc2626', marginTop: 4 }}>
                    {anomalyResult.n_anomalies}
                  </div>
                  <div style={{ fontSize: 12, color: '#991b1b', marginTop: 2 }}>
                    dari {anomalyResult.n_total} total observasi
                  </div>
                </div>

                <div style={{ padding: 16, background: '#fffbeb', borderRadius: 'var(--radius-lg)', border: '1px solid #fde68a' }}>
                  <div style={{ fontSize: 12, color: '#92400e', fontWeight: 600 }}>TINGKAT ANOMALI (RATE)</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#d97706', marginTop: 4 }}>
                    {((anomalyResult.anomaly_rate || 0) * 100).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: 12, color: '#92400e', marginTop: 2 }}>
                    proporsi penyimpangan
                  </div>
                </div>

                <div style={{ padding: 16, background: '#f0fdf4', borderRadius: 'var(--radius-lg)', border: '1px solid #bbf7d0' }}>
                  <div style={{ fontSize: 12, color: '#166534', fontWeight: 600 }}>STATUS DATA NORMAL</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: '#16a34a', marginTop: 4 }}>
                    {anomalyResult.n_total - anomalyResult.n_anomalies}
                  </div>
                  <div style={{ fontSize: 12, color: '#166534', marginTop: 2 }}>
                    observasi sesuai pola lazim
                  </div>
                </div>
              </div>

              {/* 2D Projection */}
              {anomalyResult.plot_points && anomalyResult.plot_points.length > 0 && (
                <div className="mb-6">
                  <h4 style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-700)', marginBottom: 8 }}>
                    🗺️ Visualisasi Sebaran Anomali (Normal vs Anomali):
                  </h4>
                  <ScatterPlot
                    points={anomalyResult.plot_points.map((p) => ({
                      ...p,
                      target: p.is_anomaly ? '🚨 Anomali (Outlier)' : '✅ Normal'
                    }))}
                    xLabel="Komponen 1"
                    yLabel="Komponen 2"
                  />
                </div>
              )}

              {/* Outliers Table */}
              {anomalyResult.anomalous_samples && anomalyResult.anomalous_samples.length > 0 && (
                <div>
                  <h4 style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-700)', marginBottom: 8 }}>
                    📋 Daftar Baris Teridentifikasi Anomali (Paling Ekstrem):
                  </h4>
                  <div className="table-container" style={{ maxHeight: 360, overflowY: 'auto' }}>
                    <table>
                      <thead>
                        <tr>
                          <th>Row Index</th>
                          <th>Anomaly Score</th>
                          {Object.keys(anomalyResult.anomalous_samples[0])
                            .filter((k) => k !== '_anomaly_score' && k !== '_row_index')
                            .slice(0, 6)
                            .map((k) => (
                              <th key={k}>{k}</th>
                            ))}
                        </tr>
                      </thead>
                      <tbody>
                        {anomalyResult.anomalous_samples.map((row, idx) => (
                          <tr key={idx}>
                            <td><strong>#{row._row_index}</strong></td>
                            <td>
                              <span style={{ padding: '2px 8px', background: '#fee2e2', color: '#dc2626', borderRadius: 4, fontWeight: 600, fontSize: 12 }}>
                                {row._anomaly_score?.toFixed(4)}
                              </span>
                            </td>
                            {Object.keys(row)
                              .filter((k) => k !== '_anomaly_score' && k !== '_row_index')
                              .slice(0, 6)
                              .map((k) => (
                                <td key={k}>{typeof row[k] === 'number' ? row[k].toFixed(3) : String(row[k])}</td>
                              ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Tab 4: Time Series Forecasting ── */}
      {activeTab === 'forecast' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div className="card">
            <div className="flex-between mb-4">
              <div>
                <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>📈</span> Time Series Forecasting
                </h2>
                <p className="card-subtitle">
                  Peramalan nilai masa depan berdasarkan urutan data historis menggunakan ARIMA, Exponential Smoothing, Moving Average, atau Linear Trend.
                </p>
              </div>
              <span className="sidebar-link-badge" style={{ background: '#8b5cf6', padding: '4px 10px', fontSize: 11 }}>
                Sequential Forecasting
              </span>
            </div>

            <div className="grid-3" style={{ gap: 16 }}>
              <div className="form-group">
                <label>Kolom Target Prediksi</label>
                <select
                  value={forecastTarget}
                  onChange={(e) => setForecastTarget(e.target.value)}
                >
                  {numericalCols.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Horizon / Periode ke Depan (Steps)</label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={forecastPeriods}
                  onChange={(e) => setForecastPeriods(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label>Metode Peramalan</label>
                <select
                  value={forecastMethod}
                  onChange={(e) => setForecastMethod(e.target.value)}
                >
                  <option value="arima">ARIMA (AutoRegressive Integrated Moving Average)</option>
                  <option value="exp_smoothing">Exponential Smoothing (Holt-Winters / SES)</option>
                  <option value="moving_avg">Moving Average (Rata-rata Bergerak)</option>
                  <option value="linear">Linear Trend Extrapolation</option>
                  <option value="simple">Naive (Nilai Terakhir)</option>
                </select>
              </div>
            </div>

            <div style={{ marginTop: 12 }}>
              <button
                onClick={handleRunForecast}
                disabled={loading || !forecastTarget}
                className="btn btn-primary btn-lg"
                style={{ background: '#7c3aed' }}
              >
                {loading ? '⏳ Melakukan Peramalan...' : '📊 Jalankan Forecasting'}
              </button>
            </div>
          </div>

          {forecastResult && forecastResult.success && (
            <div className="card">
              <div className="flex-between mb-4">
                <h3 className="card-title">📈 Proyeksi Masa Depan ({forecastResult.method_name || forecastResult.method})</h3>
                <span className="sidebar-link-badge" style={{ background: '#7c3aed' }}>
                  Target: {forecastResult.parameters?.target_column}
                </span>
              </div>

              {/* Summary Stats */}
              <div className="grid-3 mb-6" style={{ gap: 12 }}>
                <div style={{ padding: 12, background: '#faf5ff', borderRadius: 'var(--radius-lg)', border: '1px solid #e9d5ff' }}>
                  <div style={{ fontSize: 11, color: '#6b21a8', fontWeight: 600 }}>NILAI TERAKHIR DIOBSERVASI</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: '#581c87', marginTop: 4 }}>
                    {forecastResult.last_observed?.toFixed(3)}
                  </div>
                </div>

                <div style={{ padding: 12, background: '#eff6ff', borderRadius: 'var(--radius-lg)', border: '1px solid #bfdbfe' }}>
                  <div style={{ fontSize: 11, color: '#1e40af', fontWeight: 600 }}>RATA-RATA HISTORIS</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: '#1e3a8a', marginTop: 4 }}>
                    {forecastResult.metrics?.mean?.toFixed(3)}
                  </div>
                </div>

                <div style={{ padding: 12, background: '#f0fdf4', borderRadius: 'var(--radius-lg)', border: '1px solid #bbf7d0' }}>
                  <div style={{ fontSize: 11, color: '#166534', fontWeight: 600 }}>PERIODE PREDIKSI</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: '#14532d', marginTop: 4 }}>
                    +{forecastResult.forecast_periods} Langkah
                  </div>
                </div>
              </div>

              {/* Forecast Line Chart */}
              <ForecastLineChart
                history={forecastResult.historical_data || []}
                forecast={forecastResult.forecast || []}
                lower={forecastResult.confidence_lower || []}
                upper={forecastResult.confidence_upper || []}
              />

              {/* Forecast Table */}
              <div style={{ marginTop: 24 }}>
                <h4 style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-700)', marginBottom: 8 }}>
                  📋 Tabel Nilai Proyeksi ({forecastResult.forecast_periods} Langkah):
                </h4>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Langkah (Step)</th>
                        <th>Nilai Prediksi</th>
                        <th>Batas Bawah (95% CI)</th>
                        <th>Batas Atas (95% CI)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(forecastResult.forecast || []).map((val, i) => (
                        <tr key={i}>
                          <td><strong>+{i + 1}</strong></td>
                          <td style={{ fontWeight: 600, color: '#7c3aed' }}>{val?.toFixed(4)}</td>
                          <td style={{ color: '#64748b' }}>{forecastResult.confidence_lower?.[i]?.toFixed(4) || '-'}</td>
                          <td style={{ color: '#64748b' }}>{forecastResult.confidence_upper?.[i]?.toFixed(4) || '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Tab 5: Data Utilities ── */}
      {activeTab === 'utilities' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Card A: Missing Values */}
          <div className="card">
            <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>🧹</span> Pembersihan Nilai Hilang (Missing Values Imputation)
            </h2>
            <p className="card-subtitle">
              Tangani nilai kosong atau NaN secara sistematis dengan imputasi statistik atau pembuangan fitur yang terlalu renggang.
            </p>

            <div className="grid-2" style={{ gap: 16 }}>
              <div className="form-group">
                <label>Strategi Utama</label>
                <select
                  value={missingStrategy}
                  onChange={(e) => setMissingStrategy(e.target.value)}
                >
                  <option value="auto">Auto (Imputasi Cerdas & Selektif)</option>
                  <option value="fill">Fill (Isi Semua Nilai Kosong)</option>
                  <option value="drop">Drop (Buang Baris/Kolom Hilang)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Metode Imputasi Numerik</label>
                <select
                  value={numericImpute}
                  onChange={(e) => setNumericImpute(e.target.value)}
                >
                  <option value="mean">Mean (Rata-rata)</option>
                  <option value="median">Median (Nilai Tengah)</option>
                  <option value="forward_fill">Forward Fill (Bawa Nilai Sebelumnya)</option>
                  <option value="backward_fill">Backward Fill (Bawa Nilai Sesudahnya)</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleRunMissingValues}
              disabled={loading}
              className="btn btn-primary"
              style={{ background: '#d97706' }}
            >
              {loading ? '⏳ Memproses...' : '✨ Jalankan Pembersihan Missing Values'}
            </button>

            {missingResult && missingResult.success && (
              <div className="alert alert-success mt-4">
                <span>✅</span>
                <div>
                  <strong>Pembersihan Selesai:</strong>
                  <p style={{ margin: 0 }}>
                    Dimensi Sebelum: {missingResult.original_shape?.join(' × ')} ➔ Dimensi Sesudah: {missingResult.new_shape?.join(' × ')}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Card B: Outlier Detection */}
          <div className="card">
            <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>🎯</span> Deteksi Outlier Berbasis Statistik
            </h2>
            <p className="card-subtitle">
              Pindai kolom numerik untuk menemukan pencilan ekstrim menggunakan rentang Interkuartil (IQR) atau Z-Score.
            </p>

            <div className="grid-2" style={{ gap: 16 }}>
              <div className="form-group">
                <label>Metode Deteksi</label>
                <select
                  value={outlierMethod}
                  onChange={(e) => setOutlierMethod(e.target.value)}
                >
                  <option value="iqr">IQR (Interquartile Range Rule)</option>
                  <option value="zscore">Z-Score (Standard Deviations)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Threshold Pengali: <strong>{outlierThreshold}</strong></label>
                <input
                  type="number"
                  step={0.1}
                  min={1.0}
                  max={5.0}
                  value={outlierThreshold}
                  onChange={(e) => setOutlierThreshold(e.target.value)}
                />
              </div>
            </div>

            <button
              onClick={handleRunDetectOutliers}
              disabled={loading}
              className="btn btn-primary"
              style={{ background: '#ea580c' }}
            >
              {loading ? '⏳ Memindai Outlier...' : '🔍 Pindai Outlier Seluruh Kolom'}
            </button>

            {outlierResult && outlierResult.success && outlierResult.outlier_info && (
              <div style={{ marginTop: 16 }}>
                <h4 style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-700)', marginBottom: 8 }}>
                  📊 Hasil Analisis Pencilan per Kolom:
                </h4>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Nama Kolom</th>
                        <th>Jumlah Outlier</th>
                        <th>Persentase (%)</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(outlierResult.outlier_info).map(([col, info]) => {
                        const count = info.outlier_count || 0;
                        const pct = info.outlier_percentage || 0;
                        return (
                          <tr key={col}>
                            <td><strong>{col}</strong></td>
                            <td>{count} baris</td>
                            <td>{pct.toFixed(2)}%</td>
                            <td>
                              {count > 0 ? (
                                <span style={{ padding: '2px 8px', background: '#fee2e2', color: '#b91c1c', borderRadius: 4, fontSize: 12, fontWeight: 600 }}>
                                  ⚠️ {count} Terdeteksi
                                </span>
                              ) : (
                                <span style={{ padding: '2px 8px', background: '#dcfce7', color: '#15803d', borderRadius: 4, fontSize: 12, fontWeight: 600 }}>
                                  ✓ Bersih
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive SVG Scatter Plot Component
// ─────────────────────────────────────────────────────────────────────────────
function ScatterPlot({ points = [], xLabel = 'X', yLabel = 'Y' }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  if (!points || points.length === 0) {
    return <div style={{ padding: 24, textAlign: 'center', color: '#94a3b8' }}>Tidak ada data titik koordinat.</div>;
  }

  // Calculate bounds
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  // Group by target for legend
  const distinctTargets = Array.from(new Set(points.map((p) => p.target || 'Default')));
  const colorMap = {};
  distinctTargets.forEach((t, i) => {
    colorMap[t] = COLOR_PALETTE[i % COLOR_PALETTE.length];
  });

  const width = 640;
  const height = 360;
  const padding = 40;

  const toSvgX = (x) => padding + ((x - minX) / spanX) * (width - 2 * padding);
  const toSvgY = (y) => height - padding - ((y - minY) / spanY) * (height - 2 * padding);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Legend */}
      {distinctTargets.length > 1 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, padding: '8px 12px', background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }}>
          {distinctTargets.map((t) => (
            <div key={t} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: colorMap[t] }} />
              <span style={{ fontWeight: 600, color: '#334155' }}>{t}</span>
            </div>
          ))}
        </div>
      )}

      {/* SVG Canvas */}
      <div style={{ position: 'relative', width: '100%', maxWidth: width, margin: '0 auto', background: '#ffffff', borderRadius: 12, border: '1px solid #e2e8f0', overflow: 'hidden', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.02)' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Grid lines */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#cbd5e1" strokeWidth={1} />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#cbd5e1" strokeWidth={1} />

          {/* Points */}
          {points.map((p, idx) => {
            const cx = toSvgX(p.x);
            const cy = toSvgY(p.y);
            const color = colorMap[p.target || 'Default'] || '#3b82f6';
            const isHovered = hoveredPoint && hoveredPoint.index === p.index;

            return (
              <circle
                key={idx}
                cx={cx}
                cy={cy}
                r={isHovered ? 6 : 3.5}
                fill={color}
                fillOpacity={0.8}
                stroke={isHovered ? '#ffffff' : 'none'}
                strokeWidth={isHovered ? 2 : 0}
                style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                onMouseEnter={() => setHoveredPoint(p)}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            );
          })}
        </svg>

        {/* Floating Tooltip */}
        {hoveredPoint && (
          <div
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              background: 'rgba(15, 23, 42, 0.9)',
              color: '#fff',
              padding: '6px 12px',
              borderRadius: 6,
              fontSize: 12,
              pointerEvents: 'none',
              boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
            }}
          >
            <div>Index: <strong>#{hoveredPoint.index}</strong></div>
            <div>X: <strong>{hoveredPoint.x.toFixed(3)}</strong> | Y: <strong>{hoveredPoint.y.toFixed(3)}</strong></div>
            {hoveredPoint.target && <div>Label: <strong style={{ color: '#60a5fa' }}>{hoveredPoint.target}</strong></div>}
          </div>
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', padding: '0 8px' }}>
        <span>Axis: {xLabel}</span>
        <span>Axis: {yLabel}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive SVG Forecast Line Chart Component
// ─────────────────────────────────────────────────────────────────────────────
function ForecastLineChart({ history = [], forecast = [], lower = [], upper = [] }) {
  const width = 640;
  const height = 320;
  const padding = 40;

  const totalPoints = history.length + forecast.length;
  if (totalPoints === 0) return null;

  const allVals = [...history, ...forecast, ...lower, ...upper].filter((v) => typeof v === 'number' && !isNaN(v));
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const spanVal = maxVal - minVal || 1;

  const toX = (idx) => padding + (idx / (totalPoints - 1 || 1)) * (width - 2 * padding);
  const toY = (val) => height - padding - ((val - minVal) / spanVal) * (height - 2 * padding);

  // History path
  const histPath = history.map((val, i) => `${i === 0 ? 'M' : 'L'} ${toX(i)} ${toY(val)}`).join(' ');

  // Forecast path
  const fcastStartIdx = history.length > 0 ? history.length - 1 : 0;
  const fcastPoints = history.length > 0 ? [history[history.length - 1], ...forecast] : forecast;
  const fcastPath = fcastPoints.map((val, i) => `${i === 0 ? 'M' : 'L'} ${toX(fcastStartIdx + i)} ${toY(val)}`).join(' ');

  // Confidence area polygon
  let confAreaPath = '';
  if (lower.length > 0 && upper.length > 0 && lower.length === upper.length) {
    const topPoints = upper.map((v, i) => `${toX(history.length + i)} ${toY(v)}`);
    const bottomPoints = lower.map((v, i) => `${toX(history.length + i)} ${toY(v)}`).reverse();
    confAreaPath = `M ${topPoints.join(' L ')} L ${bottomPoints.join(' L ')} Z`;
  }

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: width, margin: '0 auto', background: '#ffffff', borderRadius: 12, border: '1px solid #e2e8f0', padding: 12, overflow: 'hidden' }}>
      <div style={{ display: 'flex', gap: 16, fontSize: 12, marginBottom: 8, justifyContent: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 3, background: '#3b82f6', borderRadius: 2 }} />
          <span style={{ fontWeight: 600, color: '#334155' }}>Data Historis</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 3, background: '#8b5cf6', borderRadius: 2 }} />
          <span style={{ fontWeight: 600, color: '#7c3aed' }}>Proyeksi Prediksi</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 8, background: 'rgba(139, 92, 246, 0.15)', borderRadius: 2 }} />
          <span style={{ color: '#64748b' }}>Batas Keyakinan (95% CI)</span>
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* Grid lines */}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#cbd5e1" strokeWidth={1} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#cbd5e1" strokeWidth={1} />

        {/* Confidence Area */}
        {confAreaPath && <path d={confAreaPath} fill="rgba(139, 92, 246, 0.15)" />}

        {/* History Line */}
        {histPath && <path d={histPath} fill="none" stroke="#3b82f6" strokeWidth={2.5} />}

        {/* Forecast Line */}
        {fcastPath && <path d={fcastPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} strokeDasharray="5,5" />}

        {/* Points */}
        {history.map((val, i) => (
          <circle key={`h-${i}`} cx={toX(i)} cy={toY(val)} r={2.5} fill="#3b82f6" />
        ))}
        {forecast.map((val, i) => (
          <circle key={`f-${i}`} cx={toX(history.length + i)} cy={toY(val)} r={3.5} fill="#8b5cf6" />
        ))}
      </svg>
    </div>
  );
}