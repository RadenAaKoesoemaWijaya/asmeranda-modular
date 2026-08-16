"""Test clustering endpoint directly without using the server."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from backend.services.clustering_service import ClusteringService

# Create test data
data = pd.DataFrame(np.random.randn(100, 5))

# Test clustering service
service = ClusteringService()
result = service.perform_clustering(data, method='kmeans', n_clusters=3)

print("Clustering service test:")
print(f"Success: {result.get('success')}")
print(f"Method: {result.get('method')}")
print(f"Number of clusters: {result.get('metrics', {}).get('n_clusters')}")
print(f"Silhouette score: {result.get('metrics', {}).get('silhouette_score')}")

# Test optimal k
optimal_result = service.find_optimal_k(data, max_k=10)
print("\nOptimal K test:")
print(f"Success: {optimal_result.get('success')}")
print(f"Optimal K (elbow): {optimal_result.get('optimal_k_elbow')}")
print(f"Optimal K (silhouette): {optimal_result.get('optimal_k_silhouette')}")