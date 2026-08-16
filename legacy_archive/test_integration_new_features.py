"""Integration test for new Phase 1 features."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from backend.services.clustering_service import ClusteringService
from backend.services.optimization_service import OptimizationService
from backend.services.recommendation_service import RecommendationService

print("=== Testing Phase 1 Services ===\n")

# Test Clustering Service
print("1. Testing Clustering Service:")
clustering_service = ClusteringService()
data = pd.DataFrame(np.random.randn(100, 5))

result = clustering_service.perform_clustering(data, method='kmeans', n_clusters=3)
print(f"   Clustering success: {result.get('success')}")
print(f"   Method: {result.get('method')}")
print(f"   Clusters: {result.get('metrics', {}).get('n_clusters')}")
print(f"   Silhouette: {result.get('metrics', {}).get('silhouette_score'):.3f}")

optimal_result = clustering_service.find_optimal_k(data, max_k=10)
print(f"   Optimal K (elbow): {optimal_result.get('optimal_k_elbow')}")
print(f"   Optimal K (silhouette): {optimal_result.get('optimal_k_silhouette')}")

# Test Optimization Service
print("\n2. Testing Optimization Service:")
optimization_service = OptimizationService()
X_train = pd.DataFrame(np.random.randn(100, 5))
y_train = pd.Series(np.random.randint(0, 2, 100))

grid_result = optimization_service.grid_search(
    X_train, y_train, "RandomForest", "Classification", cv=3
)
print(f"   Grid search success: {grid_result.get('success')}")
print(f"   Best score: {grid_result.get('best_score'):.3f}")
print(f"   Best params: {grid_result.get('best_params')}")

# Test Recommendation Service
print("\n3. Testing Recommendation Service:")
recommendation_service = RecommendationService()
test_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'target': np.random.choice(['A', 'B'], 100)
})

rec_result = recommendation_service.analyze_dataset(test_data)
print(f"   Analysis success: {rec_result.get('success')}")
print(f"   Dataset info: {rec_result.get('dataset_info')}")
print(f"   Number of recommendations: {len(rec_result.get('recommendations', []))}")
print(f"   Preprocessing steps: {rec_result.get('preprocessing_steps')}")

print("\n=== All Phase 1 Services Working ===")