import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt

# ``plot_gap_statistic`` menerima parameter ``language`` eksplisit
# (default: 'en') sehingga aman dipakai dari backend FastAPI tanpa
# Streamlit terpasang.

def calculate_comprehensive_clustering_metrics(X, labels, method_name=""):
    """Calculate comprehensive clustering evaluation metrics"""
    metrics = {}
    
    if len(set(labels)) <= 1:
        return {"error": "Only one cluster or noise found"}
    
    try:
        # Internal Validation Metrics
        metrics['silhouette_score'] = silhouette_score(X, labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        
        # Additional metrics
        metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        metrics['n_noise_points'] = (labels == -1).sum() if -1 in labels else 0
        
        # Cluster size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        metrics['cluster_size_std'] = np.std(counts)
        metrics['cluster_size_mean'] = np.mean(counts)
        
        # Compactness (average within-cluster sum of squares)
        compactness = 0
        for label in unique_labels:
            if label != -1:  # Exclude noise points
                cluster_points = X[labels == label]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    compactness += np.sum((cluster_points - centroid) ** 2)
        metrics['within_cluster_ss'] = compactness
        
        # Separation (minimum distance between cluster centroids)
        if len(unique_labels) > 2 or (len(unique_labels) == 2 and -1 not in unique_labels):
            centroids = []
            for label in unique_labels:
                if label != -1:
                    cluster_points = X[labels == label]
                    if len(cluster_points) > 0:
                        centroids.append(np.mean(cluster_points, axis=0))
            
            if len(centroids) > 1:
                centroid_distances = euclidean_distances(centroids)
                np.fill_diagonal(centroid_distances, np.inf)
                metrics['min_centroid_distance'] = np.min(centroid_distances)
        
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics

def gap_statistic(X, max_k=10, n_refs=5):
    """Calculate Gap Statistic for optimal k"""
    gaps = np.zeros(max_k-1)
    sks = np.zeros(max_k-1)
    
    for k in range(1, max_k+1):
        # Generate reference datasets
        ref_inertias = []
        for _ in range(n_refs):
            random_data = np.random.uniform(
                low=X.min(axis=0), 
                high=X.max(axis=0), 
                size=X.shape
            )
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(random_data)
            ref_inertias.append(km.inertia_)
        
        # Calculate for actual data
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        
        # Gap statistic
        gap = np.log(np.mean(ref_inertias)) - np.log(km.inertia_)
        gaps[k-1] = gap
        sks[k-1] = np.std(ref_inertias) * np.sqrt(1 + 1/n_refs)
    
    return gaps, sks

def find_optimal_k_with_gap(X, max_k=10):
    """Find optimal k using Gap Statistic"""
    gaps, sks = gap_statistic(X, max_k)
    
    # Find optimal k using gap statistic rule
    optimal_k = 2
    for i in range(len(gaps)-1):
        if gaps[i] >= gaps[i+1] - sks[i+1]:
            optimal_k = i + 2  # Convert from index to k value
            break
    
    # If no clear optimal found, use maximum gap
    if optimal_k == 2 and len(gaps) > 0:
        optimal_k = np.argmax(gaps) + 2
    
    return optimal_k, gaps, sks

def find_optimal_clusters_kmeans(X, max_k=10, use_gap_statistic=False):
    """Find optimal number of clusters for K-Means"""
    if len(X) < 3:
        return 2, {}
    
    max_k = min(max_k, len(X) - 1)
    metrics = {'k': [], 'inertia': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}
    
    # Use Gap Statistic if requested
    if use_gap_statistic:
        optimal_k, gaps, sks = find_optimal_k_with_gap(X, max_k)
        metrics['gaps'] = gaps
        metrics['sks'] = sks
        return optimal_k, metrics
    
    for k in range(2, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            metrics['k'].append(k)
            metrics['inertia'].append(kmeans.inertia_)
            
            if len(set(labels)) > 1:
                metrics['silhouette'].append(silhouette_score(X, labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
            else:
                metrics['silhouette'].append(0)
                metrics['calinski_harabasz'].append(0)
                metrics['davies_bouldin'].append(np.inf)
                
        except Exception as e:
            continue
    
    # Find optimal k using multiple criteria
    optimal_k = 2
    if metrics['silhouette']:
        # Weighted combination of metrics
        silhouette_scores = np.array(metrics['silhouette'])
        calinski_scores = np.array(metrics['calinski_harabasz'])
        davies_bouldin_scores = np.array(metrics['davies_bouldin'])
        
        # Normalize scores
        silhouette_norm = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores) + 1e-10)
        calinski_norm = (calinski_scores - np.min(calinski_scores)) / (np.max(calinski_scores) - np.min(calinski_scores) + 1e-10)
        davies_bouldin_norm = 1 - ((davies_bouldin_scores - np.min(davies_bouldin_scores)) / (np.max(davies_bouldin_scores) - np.min(davies_bouldin_scores) + 1e-10))
        
        # Combined score (higher is better)
        combined_score = silhouette_norm + calinski_norm + davies_bouldin_norm
        optimal_idx = np.argmax(combined_score)
        optimal_k = metrics['k'][optimal_idx]
    
    return optimal_k, metrics

def plot_gap_statistic(gaps, sks, k_range, language: str = "en"):
    """Visualize Gap Statistic for optimal k selection.

    Parameters
    ----------
    gaps, sks, k_range : list/np.ndarray
    language : str
        'id' untuk label Indonesia, 'en' untuk English. Default 'en'
        agar aman bila dipanggil dari backend tanpa Streamlit.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Gap values
    k_values = list(range(1, len(gaps) + 1))
    ax1.plot(k_values, gaps, 'bo-', label='Gap Statistic')
    ax1.errorbar(k_values, gaps, yerr=sks, fmt='bo', capsize=5, alpha=0.7)

    lang = language or "en"
    ax1.set_xlabel('Jumlah Cluster (k)' if lang == 'id' else 'Number of Clusters (k)')
    ax1.set_ylabel('Gap Statistic')
    ax1.set_title('Gap Statistic vs Jumlah Cluster' if lang == 'id' else 'Gap Statistic vs Number of Clusters')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Find and mark optimal k
    optimal_k_idx = np.argmax(gaps)
    optimal_k = optimal_k_idx + 1
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'k optimal = {optimal_k}' if lang == 'id' else f'optimal k = {optimal_k}')
    ax1.legend()
    
    # Plot Gap differences
    if len(gaps) > 1:
        gap_diffs = np.diff(gaps)
        ax2.plot(k_values[1:], gap_diffs, 'ro-', label='Gap Differences')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Jumlah Cluster (k)' if lang == 'id' else 'Number of Clusters (k)')
        ax2.set_ylabel('Gap Difference')
        ax2.set_title('Perbedaan Gap Statistic' if lang == 'id' else 'Gap Statistic Differences')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    return fig

def calculate_kprototypes_metrics(data, categorical_idx, clusters, kproto_model):
    """Calculate K-Prototypes specific evaluation metrics"""
    metrics = {}
    
    # Total cost (WCSS + categorical cost) from model
    if hasattr(kproto_model, 'cost_'):
        metrics['total_cost'] = kproto_model.cost_
    
    # Numerical WCSS (within-cluster sum of squares)
    numerical_mask = ~np.isin(np.arange(data.shape[1]), categorical_idx)
    numerical_data = data[:, numerical_mask]
    numerical_data = numerical_data.astype(float)
    
    wcss = 0
    unique_clusters = np.unique(clusters)
    cluster_centroids = {}
    
    for cluster in unique_clusters:
        if cluster != -1:  # Exclude noise points
            cluster_mask = clusters == cluster
            cluster_points = numerical_data[cluster_mask]
            
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                cluster_centroids[cluster] = centroid
                wcss += np.sum((cluster_points - centroid) ** 2)
    
    metrics['numerical_wcss'] = wcss
    metrics['n_clusters'] = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    # Categorical purity
    if len(categorical_idx) > 0:
        total_purity = 0
        valid_clusters = 0
        
        for cluster in unique_clusters:
            if cluster != -1:  # Exclude noise points
                cluster_mask = clusters == cluster
                cluster_data = data[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_purity = 0
                    
                    for cat_idx in categorical_idx:
                        if cat_idx < cluster_data.shape[1]:
                            values = cluster_data[:, cat_idx]
                            if len(values) > 0:
                                unique, counts = np.unique(values, return_counts=True)
                                mode_value = unique[np.argmax(counts)]
                                purity = np.sum(values == mode_value) / len(values)
                                cluster_purity += purity
                    
                    if len(categorical_idx) > 0:
                        total_purity += cluster_purity / len(categorical_idx)
                        valid_clusters += 1
        
        if valid_clusters > 0:
            metrics['categorical_purity'] = total_purity / valid_clusters
            metrics['categorical_purity_std'] = np.std([total_purity / valid_clusters]) if valid_clusters > 1 else 0
        else:
            metrics['categorical_purity'] = 0
            metrics['categorical_purity_std'] = 0
    else:
        metrics['categorical_purity'] = 0
        metrics['categorical_purity_std'] = 0
    
    if 'total_cost' in metrics and metrics['total_cost'] > 0:
        cost_component = 1 / (1 + metrics['total_cost'] / len(clusters))
        purity_component = metrics['categorical_purity']
        metrics['combined_score'] = 0.7 * cost_component + 0.3 * purity_component
    
    return metrics

def find_optimal_eps_dbscan(X, min_samples_range=range(3, 8)):
    """Find optimal eps parameter for DBSCAN"""
    if len(X) < 3:
        return 0.5, 3, {}
    
    k = 4
    if len(X) > k:
        distances = np.sort(np.mean(euclidean_distances(X)[:k], axis=1))
        diffs = np.diff(distances)
        elbow_idx = np.argmax(diffs) + 1 if len(diffs) > 0 else len(distances) // 2
        suggested_eps = distances[min(elbow_idx, len(distances) - 1)]
    else:
        suggested_eps = 0.5
    
    eps_range = np.linspace(max(0.1, suggested_eps * 0.5), min(5.0, suggested_eps * 2), 10)
    
    best_score = -1
    best_params = {'eps': 0.5, 'min_samples': 3}
    results = {'eps': [], 'min_samples': [], 'n_clusters': [], 'silhouette': [], 'n_noise': []}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                
                results['eps'].append(eps)
                results['min_samples'].append(min_samples)
                results['n_clusters'].append(n_clusters)
                results['n_noise'].append(n_noise)
                
                if n_clusters > 1 and n_noise < len(X) * 0.3:
                    silhouette = silhouette_score(X, labels)
                    results['silhouette'].append(silhouette)
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_params = {'eps': eps, 'min_samples': min_samples}
                else:
                    results['silhouette'].append(0)
                    
            except Exception as e:
                continue
    
    return best_params['eps'], best_params['min_samples'], results

def analyze_cluster_stability(X, labels, n_bootstrap=10, noise_level=0.05):
    """Analyze clustering stability with bootstrap"""
    if len(set(labels)) <= 1:
        return {"stability_score": 0, "message": "Not enough clusters for stability analysis"}
    
    original_labels = labels.copy()
    stability_scores = []
    
    for i in range(n_bootstrap):
        X_noisy = X + np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
        try:
            if len(set(original_labels)) > 1:
                kmeans = KMeans(n_clusters=len(set(original_labels)), random_state=42 + i, n_init=10)
                new_labels = kmeans.fit_predict(X_noisy)
                if len(set(new_labels)) > 1:
                    stability_score = adjusted_rand_score(original_labels, new_labels)
                    stability_scores.append(stability_score)
        except Exception as e:
            continue
    
    if stability_scores:
        return {
            "stability_score": np.mean(stability_scores),
            "stability_std": np.std(stability_scores),
            "n_bootstrap": len(stability_scores)
        }
    else:
        return {"stability_score": 0, "message": "Cannot calculate stability"}

def analyze_cluster_characteristics(X, labels, feature_names=None):
    """Analyze cluster characteristics based on key features"""
    if feature_names is None:
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
    
    cluster_profiles = {}
    unique_labels = np.unique(labels)
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
            
        cluster_mask = labels == cluster_id
        cluster_data = X[cluster_mask] if hasattr(X, 'iloc') else X[cluster_mask]
        
        if len(cluster_data) == 0:
            continue
            
        profile = {
            'size': np.sum(cluster_mask),
            'percentage': np.sum(cluster_mask) / len(labels) * 100,
            'mean_values': np.mean(cluster_data, axis=0) if hasattr(cluster_data, 'mean') else np.mean(cluster_data),
            'std_values': np.std(cluster_data, axis=0) if hasattr(cluster_data, 'std') else np.std(cluster_data),
            'min_values': np.min(cluster_data, axis=0) if hasattr(cluster_data, 'min') else np.min(cluster_data),
            'max_values': np.max(cluster_data, axis=0) if hasattr(cluster_data, 'max') else np.max(cluster_data)
        }
        
        if hasattr(cluster_data, 'shape') and len(cluster_data.shape) > 1:
            feature_stats = {}
            for i, feature_name in enumerate(feature_names[:cluster_data.shape[1]]):
                if hasattr(cluster_data, 'iloc'):
                    if isinstance(feature_name, str) and feature_name in cluster_data.columns:
                        col_data = cluster_data[feature_name]
                    else:
                        col_data = cluster_data.iloc[:, i]
                else:
                    col_data = cluster_data[:, i]
                    
                feature_stats[feature_name] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
            profile['feature_stats'] = feature_stats
        
        cluster_profiles[f'Cluster_{cluster_id}'] = profile
    
    return cluster_profiles

def generate_cluster_report(X, labels, algorithm_name, evaluation_metrics, stability_results, cluster_profiles):
    """Generate comprehensive clustering report"""
    report = []
    report.append("="*60)
    report.append(f"CLUSTERING ANALYSIS REPORT - {algorithm_name.upper()}")
    report.append("="*60)
    
    report.append(f"\nDATASET OVERVIEW:")
    report.append(f"Total samples: {len(labels)}")
    report.append(f"Number of features: {X.shape[1] if hasattr(X, 'shape') else len(X[0])}")
    report.append(f"Number of clusters: {len(np.unique(labels))}")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    report.append(f"\nCLUSTER DISTRIBUTION:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            report.append(f"  Noise points: {count} ({count/len(labels)*100:.1f}%)")
        else:
            report.append(f"  Cluster {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    report.append(f"\nEVALUATION METRICS:")
    for metric_name, value in evaluation_metrics.items():
        if value is not None:
            if isinstance(value, (int, float)):
                report.append(f"  {metric_name}: {value:.4f}")
            elif isinstance(value, dict):
                report.append(f"  {metric_name}: {str(value)}")
            else:
                report.append(f"  {metric_name}: {value}")
        else:
            report.append(f"  {metric_name}: Not available")
    
    report.append(f"\nSTABILITY ANALYSIS:")
    if stability_results:
        report.append(f"  Average Adjusted Rand Index: {stability_results.get('avg_ari', 'N/A')}")
        report.append(f"  Average Normalized Mutual Info: {stability_results.get('avg_nmi', 'N/A')}")
        report.append(f"  Stability Score: {stability_results.get('stability_score', 'N/A')}")
    else:
        report.append("  Stability analysis not performed")
    
    if cluster_profiles:
        report.append(f"\nCLUSTER CHARACTERISTICS:")
        for cluster_name, profile in cluster_profiles.items():
            report.append(f"\n  {cluster_name}:")
            report.append(f"    Size: {profile['size']} samples ({profile['percentage']:.1f}%)")
            
            if 'feature_stats' in profile:
                report.append(f"    Key features:")
                feature_means = [(name, stats['mean']) for name, stats in profile['feature_stats'].items()]
                feature_means.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for feature_name, mean_val in feature_means[:3]:
                    report.append(f"      {feature_name}: {mean_val:.2f}")
    
    report.append(f"\nRECOMMENDATIONS:")
    
    if len(unique_labels) == 1:
        report.append("  ⚠️  Only one cluster found - consider adjusting parameters")
    elif any(count < len(labels) * 0.05 for count in counts):
        report.append("  ⚠️  Some clusters are very small - check for outliers")
    
    if 'Silhouette Score' in evaluation_metrics and evaluation_metrics['Silhouette Score'] is not None:
        silhouette = evaluation_metrics['Silhouette Score']
        if silhouette > 0.5:
            report.append("  ✅ Strong cluster separation (Silhouette > 0.5)")
        elif silhouette > 0.25:
            report.append("  ℹ️  Moderate cluster separation (0.25 < Silhouette < 0.5)")
        else:
            report.append("  ⚠️  Weak cluster separation (Silhouette < 0.25)")
    
    if stability_results and 'stability_score' in stability_results:
        stability_score = stability_results['stability_score']
        if isinstance(stability_score, (int, float)):
            if stability_score > 0.8:
                report.append("  ✅ High cluster stability")
            elif stability_score > 0.6:
                report.append("  ℹ️  Moderate cluster stability")
            else:
                report.append("  ⚠️  Low cluster stability - results may vary")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)
