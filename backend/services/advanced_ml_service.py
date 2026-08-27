"""
Advanced ML Service - Core features including UMAP, PCA, t-SNE, HDBSCAN, Anomaly Detection, and Time Series Forecasting.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

logger = logging.getLogger("asmeranda.services.advanced_ml")


class AdvancedMLService:
    """Service for advanced ML features including dimensionality reduction, clustering, anomaly detection, and forecasting."""

    def __init__(self):
        self.scaler = StandardScaler()

    def _extract_numeric_scaled(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract numeric columns and scale them."""
        num_df = data.select_dtypes(include=[np.number])
        if num_df.empty:
            raise ValueError("Data tidak memiliki kolom numerik yang valid.")
        cols = num_df.columns.tolist()
        scaler = StandardScaler()
        # Handle NaN with mean imputation if any remains
        cleaned = num_df.fillna(num_df.mean()).values
        scaled = scaler.fit_transform(cleaned)
        return scaled, cols

    def umap_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        targets: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform UMAP dimensionality reduction.
        """
        try:
            scaled_data, cols = self._extract_numeric_scaled(data)
            n_samples = len(scaled_data)

            # Check if umap-learn is available
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(n_neighbors, max(2, n_samples - 1)),
                    min_dist=min_dist,
                    metric=metric,
                    random_state=random_state,
                )
                embedding = reducer.fit_transform(scaled_data)
                method_used = "umap"
            except ImportError:
                logger.warning("umap-learn not available, falling back to PCA")
                return self.pca_dimensionality_reduction(data, n_components, targets=targets)
            except Exception as umap_err:
                logger.warning(f"UMAP execution warning ({umap_err}), falling back to PCA")
                return self.pca_dimensionality_reduction(data, n_components, targets=targets)

            columns = [f"UMAP_{i+1}" for i in range(n_components)]
            result_df = pd.DataFrame(embedding, columns=columns)

            points = []
            for idx, row in enumerate(embedding):
                pt = {
                    "index": int(idx),
                    "x": float(row[0]),
                    "y": float(row[1]) if n_components > 1 else 0.0,
                }
                if n_components > 2:
                    pt["z"] = float(row[2])
                if targets is not None and idx < len(targets):
                    pt["target"] = str(targets[idx])
                points.append(pt)

            return {
                "success": True,
                "data": result_df.to_dict(orient="records"),
                "points": points,
                "columns": columns,
                "method": method_used,
                "parameters": {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "metric": metric,
                },
                "original_shape": list(data.shape),
                "reduced_shape": list(embedding.shape),
                "n_samples": n_samples,
                "n_features": len(cols),
            }

        except Exception as e:
            logger.error(f"UMAP dimensionality reduction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "umap",
            }

    def pca_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        n_components: int = 2,
        targets: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform PCA dimensionality reduction.
        """
        try:
            scaled_data, cols = self._extract_numeric_scaled(data)
            n_components_valid = min(n_components, scaled_data.shape[1], scaled_data.shape[0])
            pca = PCA(n_components=n_components_valid, random_state=42)
            embedding = pca.fit_transform(scaled_data)

            columns = [f"PC_{i+1}" for i in range(n_components_valid)]
            result_df = pd.DataFrame(embedding, columns=columns)

            points = []
            for idx, row in enumerate(embedding):
                pt = {
                    "index": int(idx),
                    "x": float(row[0]),
                    "y": float(row[1]) if n_components_valid > 1 else 0.0,
                }
                if n_components_valid > 2:
                    pt["z"] = float(row[2])
                if targets is not None and idx < len(targets):
                    pt["target"] = str(targets[idx])
                points.append(pt)

            var_ratios = pca.explained_variance_ratio_.tolist()
            return {
                "success": True,
                "data": result_df.to_dict(orient="records"),
                "points": points,
                "columns": columns,
                "method": "pca",
                "parameters": {
                    "n_components": n_components_valid,
                    "explained_variance_ratio": var_ratios,
                    "total_explained_variance": float(sum(var_ratios)),
                },
                "original_shape": list(data.shape),
                "reduced_shape": list(embedding.shape),
                "n_samples": len(scaled_data),
                "n_features": len(cols),
            }

        except Exception as e:
            logger.error(f"PCA dimensionality reduction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "pca",
            }

    def tsne_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        n_components: int = 2,
        perplexity: float = 30.0,
        targets: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform t-SNE dimensionality reduction.
        """
        try:
            scaled_data, cols = self._extract_numeric_scaled(data)
            n_samples = len(scaled_data)
            perp = min(perplexity, max(1.0, float(n_samples - 1) / 3.0))

            tsne = TSNE(
                n_components=n_components,
                perplexity=perp,
                random_state=42,
                init="pca",
                learning_rate="auto",
            )
            embedding = tsne.fit_transform(scaled_data)

            columns = [f"tSNE_{i+1}" for i in range(n_components)]
            result_df = pd.DataFrame(embedding, columns=columns)

            points = []
            for idx, row in enumerate(embedding):
                pt = {
                    "index": int(idx),
                    "x": float(row[0]),
                    "y": float(row[1]) if n_components > 1 else 0.0,
                }
                if n_components > 2:
                    pt["z"] = float(row[2])
                if targets is not None and idx < len(targets):
                    pt["target"] = str(targets[idx])
                points.append(pt)

            return {
                "success": True,
                "data": result_df.to_dict(orient="records"),
                "points": points,
                "columns": columns,
                "method": "tsne",
                "parameters": {
                    "n_components": n_components,
                    "perplexity": perp,
                },
                "original_shape": list(data.shape),
                "reduced_shape": list(embedding.shape),
                "n_samples": n_samples,
                "n_features": len(cols),
            }
        except Exception as e:
            logger.error(f"t-SNE dimensionality reduction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "tsne",
            }

    def hdbscan_clustering(
        self,
        data: pd.DataFrame,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
    ) -> Dict[str, Any]:
        """
        Perform HDBSCAN clustering with evaluation metrics and 2D visual projection.
        """
        try:
            scaled_data, cols = self._extract_numeric_scaled(data)

            # Check if hdbscan is available
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(2, min_cluster_size),
                    min_samples=min_samples,
                    metric=metric,
                    prediction_data=True,
                )
                labels = clusterer.fit_predict(scaled_data)
                probabilities = clusterer.probabilities_.tolist() if hasattr(clusterer, "probabilities_") else None
                method_used = "hdbscan"
            except (ImportError, Exception) as hdb_err:
                logger.warning(f"HDBSCAN not available or failed ({hdb_err}), falling back to DBSCAN")
                return self.dbscan_clustering(data, eps=0.5, min_samples=min_cluster_size, metric=metric)

            # Calculate cluster statistics
            labels_arr = np.array(labels)
            unique_labels = set(labels_arr)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = int((labels_arr == -1).sum())

            # Cluster sizes
            cluster_sizes = {}
            for cl in sorted(unique_labels):
                cl_name = "Noise (-1)" if cl == -1 else f"Cluster {cl}"
                cluster_sizes[cl_name] = int((labels_arr == cl).sum())

            # Evaluation metrics (only if at least 2 non-noise clusters and enough points)
            metrics = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": float(n_noise / len(labels_arr)) if len(labels_arr) > 0 else 0.0,
            }
            if n_clusters >= 2:
                non_noise_mask = labels_arr != -1
                if non_noise_mask.sum() > n_clusters:
                    try:
                        metrics["silhouette_score"] = float(silhouette_score(scaled_data[non_noise_mask], labels_arr[non_noise_mask]))
                    except Exception:
                        pass
                    try:
                        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(scaled_data[non_noise_mask], labels_arr[non_noise_mask]))
                    except Exception:
                        pass
                    try:
                        metrics["davies_bouldin_score"] = float(davies_bouldin_score(scaled_data[non_noise_mask], labels_arr[non_noise_mask]))
                    except Exception:
                        pass

            # Precalculate 2D coordinates for visual plotting
            try:
                pca2 = PCA(n_components=2, random_state=42)
                coords2d = pca2.fit_transform(scaled_data)
                plot_points = [
                    {
                        "index": int(i),
                        "x": float(coords2d[i, 0]),
                        "y": float(coords2d[i, 1]),
                        "cluster": int(labels_arr[i]),
                        "probability": float(probabilities[i]) if probabilities else 1.0,
                    }
                    for i in range(len(scaled_data))
                ]
            except Exception:
                plot_points = []

            return {
                "success": True,
                "labels": labels_arr.tolist(),
                "method": method_used,
                "parameters": {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "metric": metric,
                },
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "cluster_sizes": cluster_sizes,
                "metrics": metrics,
                "plot_points": plot_points,
                "cluster_probabilities": probabilities,
            }

        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "hdbscan",
            }

    def dbscan_clustering(
        self,
        data: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> Dict[str, Any]:
        """
        Perform DBSCAN clustering (fallback for HDBSCAN).
        """
        try:
            scaled_data, cols = self._extract_numeric_scaled(data)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = clusterer.fit_predict(scaled_data)
            labels_arr = np.array(labels)

            unique_labels = set(labels_arr)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = int((labels_arr == -1).sum())

            cluster_sizes = {}
            for cl in sorted(unique_labels):
                cl_name = "Noise (-1)" if cl == -1 else f"Cluster {cl}"
                cluster_sizes[cl_name] = int((labels_arr == cl).sum())

            metrics = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": float(n_noise / len(labels_arr)) if len(labels_arr) > 0 else 0.0,
            }
            if n_clusters >= 2:
                non_noise_mask = labels_arr != -1
                if non_noise_mask.sum() > n_clusters:
                    try:
                        metrics["silhouette_score"] = float(silhouette_score(scaled_data[non_noise_mask], labels_arr[non_noise_mask]))
                    except Exception:
                        pass

            try:
                pca2 = PCA(n_components=2, random_state=42)
                coords2d = pca2.fit_transform(scaled_data)
                plot_points = [
                    {
                        "index": int(i),
                        "x": float(coords2d[i, 0]),
                        "y": float(coords2d[i, 1]),
                        "cluster": int(labels_arr[i]),
                    }
                    for i in range(len(scaled_data))
                ]
            except Exception:
                plot_points = []

            return {
                "success": True,
                "labels": labels_arr.tolist(),
                "method": "dbscan",
                "parameters": {
                    "eps": eps,
                    "min_samples": min_samples,
                    "metric": metric,
                },
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "cluster_sizes": cluster_sizes,
                "metrics": metrics,
                "plot_points": plot_points,
            }

        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "dbscan",
            }

    def isolation_forest_anomaly_detection(
        self,
        data: pd.DataFrame,
        contamination: float = 0.1,
        n_estimators: int = 100,
    ) -> Dict[str, Any]:
        """
        Perform anomaly detection using Isolation Forest.
        """
        try:
            scaled_data, cols = self._extract_numeric_scaled(data)
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42,
            )
            predictions = iso_forest.fit_predict(scaled_data)
            scores = iso_forest.score_samples(scaled_data)

            # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
            anomaly_labels = (predictions == -1).astype(int)
            n_anomalies = int(anomaly_labels.sum())
            anomaly_rate = float(anomaly_labels.mean())

            # Anomalous indices and samples preview
            anom_indices = np.where(anomaly_labels == 1)[0].tolist()
            anom_samples = []
            for idx in anom_indices[:50]:  # Top 50 anomalous
                row_dict = data.iloc[idx].to_dict()
                row_dict["_anomaly_score"] = float(scores[idx])
                row_dict["_row_index"] = int(idx)
                anom_samples.append(row_dict)

            # Sort by highest anomaly score severity (lowest score_samples value)
            anom_samples.sort(key=lambda x: x["_anomaly_score"])

            # 2D projection for visualization
            try:
                pca2 = PCA(n_components=2, random_state=42)
                coords2d = pca2.fit_transform(scaled_data)
                plot_points = [
                    {
                        "index": int(i),
                        "x": float(coords2d[i, 0]),
                        "y": float(coords2d[i, 1]),
                        "is_anomaly": bool(anomaly_labels[i] == 1),
                        "score": float(scores[i]),
                    }
                    for i in range(len(scaled_data))
                ]
            except Exception:
                plot_points = []

            return {
                "success": True,
                "anomaly_labels": anomaly_labels.tolist(),
                "anomaly_scores": scores.tolist(),
                "method": "isolation_forest",
                "parameters": {
                    "contamination": contamination,
                    "n_estimators": n_estimators,
                },
                "n_anomalies": n_anomalies,
                "n_total": len(scaled_data),
                "anomaly_rate": anomaly_rate,
                "anomalous_indices": anom_indices,
                "anomalous_samples": anom_samples,
                "plot_points": plot_points,
            }

        except Exception as e:
            logger.error(f"Isolation Forest anomaly detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "isolation_forest",
            }

    def one_class_svm_anomaly_detection(
        self,
        data: pd.DataFrame,
        nu: float = 0.1,
        kernel: str = "rbf",
    ) -> Dict[str, Any]:
        """
        Perform anomaly detection using One-Class SVM.
        """
        try:
            scaled_data, cols = self._extract_numeric_scaled(data)
            svm = OneClassSVM(nu=min(0.5, max(0.01, nu)), kernel=kernel)
            predictions = svm.fit_predict(scaled_data)
            scores = svm.decision_function(scaled_data)

            anomaly_labels = (predictions == -1).astype(int)
            n_anomalies = int(anomaly_labels.sum())
            anomaly_rate = float(anomaly_labels.mean())

            anom_indices = np.where(anomaly_labels == 1)[0].tolist()
            anom_samples = []
            for idx in anom_indices[:50]:
                row_dict = data.iloc[idx].to_dict()
                row_dict["_anomaly_score"] = float(scores[idx])
                row_dict["_row_index"] = int(idx)
                anom_samples.append(row_dict)

            anom_samples.sort(key=lambda x: x["_anomaly_score"])

            try:
                pca2 = PCA(n_components=2, random_state=42)
                coords2d = pca2.fit_transform(scaled_data)
                plot_points = [
                    {
                        "index": int(i),
                        "x": float(coords2d[i, 0]),
                        "y": float(coords2d[i, 1]),
                        "is_anomaly": bool(anomaly_labels[i] == 1),
                        "score": float(scores[i]),
                    }
                    for i in range(len(scaled_data))
                ]
            except Exception:
                plot_points = []

            return {
                "success": True,
                "anomaly_labels": anomaly_labels.tolist(),
                "anomaly_scores": scores.tolist(),
                "method": "one_class_svm",
                "parameters": {
                    "nu": nu,
                    "kernel": kernel,
                },
                "n_anomalies": n_anomalies,
                "n_total": len(scaled_data),
                "anomaly_rate": anomaly_rate,
                "anomalous_indices": anom_indices,
                "anomalous_samples": anom_samples,
                "plot_points": plot_points,
            }

        except Exception as e:
            logger.error(f"One-Class SVM anomaly detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "one_class_svm",
            }

    def basic_forecasting(
        self,
        data: pd.DataFrame,
        target_column: str,
        periods: int = 10,
        method: str = "arima",
    ) -> Dict[str, Any]:
        """
        Perform time series forecasting with multiple algorithms and confidence bands.
        """
        try:
            if target_column not in data.columns:
                return {
                    "success": False,
                    "error": f"Kolom '{target_column}' tidak ditemukan dalam dataset",
                    "method": method,
                }

            series_raw = data[target_column].dropna().values
            if len(series_raw) < 3:
                return {
                    "success": False,
                    "error": "Data time series terlalu pendek (minimal 3 observasi).",
                    "method": method,
                }

            series = np.asarray(series_raw, dtype=float)
            n = len(series)
            std_err = float(np.std(series)) if np.std(series) > 0 else 1.0

            method_lower = method.lower()
            if method_lower in ("arima", "sarima", "auto_arima"):
                # Check statsmodels
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    model = ARIMA(series, order=(1, 1, 1))
                    fitted = model.fit()
                    forecast_res = fitted.get_forecast(steps=periods)
                    forecast = forecast_res.predicted_mean.tolist()
                    conf = forecast_res.conf_int(alpha=0.05)
                    conf_lower = conf[:, 0].tolist()
                    conf_upper = conf[:, 1].tolist()
                    method_name = "ARIMA (1, 1, 1)"
                except Exception as arima_err:
                    logger.warning(f"ARIMA fit fallback ({arima_err}), using Holt-Winters / Linear")
                    from scipy import stats
                    x = np.arange(n)
                    slope, intercept, _, _, _ = stats.linregress(x, series)
                    forecast = (slope * np.arange(n, n + periods) + intercept).tolist()
                    conf_lower = [f - 1.96 * std_err * np.sqrt(i + 1) for i, f in enumerate(forecast)]
                    conf_upper = [f + 1.96 * std_err * np.sqrt(i + 1) for i, f in enumerate(forecast)]
                    method_name = "ARIMA (Trend Extrapolation Fallback)"

            elif method_lower in ("exp_smoothing", "exponential_smoothing", "ses"):
                try:
                    from statsmodels.tsa.api import SimpleExpSmoothing
                    fit = SimpleExpSmoothing(series).fit()
                    forecast = fit.forecast(periods).tolist()
                    conf_lower = [f - 1.96 * std_err for f in forecast]
                    conf_upper = [f + 1.96 * std_err for f in forecast]
                    method_name = "Simple Exponential Smoothing"
                except Exception:
                    alpha = 0.3
                    last_val = series[-1]
                    forecast = [last_val for _ in range(periods)]
                    conf_lower = [f - 1.96 * std_err for f in forecast]
                    conf_upper = [f + 1.96 * std_err for f in forecast]
                    method_name = "Exponential Smoothing (Approx)"

            elif method_lower in ("moving_avg", "ma"):
                window = min(7, n)
                ma_val = float(np.mean(series[-window:]))
                forecast = [ma_val for _ in range(periods)]
                conf_lower = [ma_val - 1.96 * std_err for _ in range(periods)]
                conf_upper = [ma_val + 1.96 * std_err for _ in range(periods)]
                method_name = f"Moving Average (Window {window})"

            elif method_lower in ("linear", "trend"):
                from scipy import stats
                x = np.arange(n)
                slope, intercept, _, _, _ = stats.linregress(x, series)
                forecast = (slope * np.arange(n, n + periods) + intercept).tolist()
                conf_lower = [f - 1.96 * std_err * np.sqrt((i + 1) * 0.5) for i, f in enumerate(forecast)]
                conf_upper = [f + 1.96 * std_err * np.sqrt((i + 1) * 0.5) for i, f in enumerate(forecast)]
                method_name = "Linear Trend Extrapolation"

            else:  # simple / last value
                last_val = float(series[-1])
                forecast = [last_val for _ in range(periods)]
                conf_lower = [last_val - 1.96 * std_err for _ in range(periods)]
                conf_upper = [last_val + 1.96 * std_err for _ in range(periods)]
                method_name = "Naive (Last Value)"

            # Last 50 historical points for clean chart display
            display_history = series[-100:].tolist() if n > 100 else series.tolist()

            return {
                "success": True,
                "forecast": [float(v) for v in forecast],
                "confidence_lower": [float(v) for v in conf_lower],
                "confidence_upper": [float(v) for v in conf_upper],
                "historical_data": [float(v) for v in display_history],
                "total_observations": n,
                "last_observed": float(series[-1]),
                "method": method,
                "method_name": method_name,
                "parameters": {
                    "periods": periods,
                    "target_column": target_column,
                },
                "metrics": {
                    "mean": float(np.mean(series)),
                    "std": float(np.std(series)),
                    "min": float(np.min(series)),
                    "max": float(np.max(series)),
                },
                "forecast_periods": periods,
            }

        except Exception as e:
            logger.error(f"Basic forecasting failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": method,
            }