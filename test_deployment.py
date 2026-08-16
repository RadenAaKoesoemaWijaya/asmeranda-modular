"""Test deployment connectivity and functionality."""
import sys
sys.path.insert(0, '.')

import requests
import time

print("=== Deployment Testing ===\n")

# Test Backend Health
print("1. Testing Backend Health...")
try:
    r = requests.get("http://localhost:8001/health")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    backend_healthy = r.status_code == 200
except Exception as e:
    print(f"   Error: {e}")
    backend_healthy = False

# Test Backend OpenAPI
print("\n2. Testing Backend OpenAPI...")
try:
    r = requests.get("http://localhost:8001/openapi.json")
    data = r.json()
    print(f"   Total endpoints: {len(data['paths'])}")
    
    # Check for our new endpoints
    preprocessing_endpoints = [p for p in data['paths'].keys() if 'preprocessing' in p]
    training_endpoints = [p for p in data['paths'].keys() if 'training' in p]
    eda_endpoints = [p for p in data['paths'].keys() if 'eda' in p]
    
    print(f"   Preprocessing endpoints: {preprocessing_endpoints}")
    print(f"   Training endpoints: {training_endpoints}")
    print(f"   EDA endpoints: {eda_endpoints}")
    
    # Check for Phase 1 endpoints (integrated)
    has_clustering = any('cluster' in p for p in preprocessing_endpoints)
    has_optimization = any('optimize' in p for p in training_endpoints)
    has_recommendations = any('analyze' in p for p in eda_endpoints)
    
    print(f"   Has clustering endpoints: {has_clustering}")
    print(f"   Has optimization endpoints: {has_optimization}")
    print(f"   Has recommendations endpoints: {has_recommendations}")
    
except Exception as e:
    print(f"   Error: {e}")

# Test Clustering Endpoint Directly
print("\n3. Testing Clustering Endpoint...")
try:
    r = requests.post("http://localhost:8001/api/v1/preprocessing/cluster", 
                      json={"state_id": "test", "method": "kmeans", "parameters": {"n_clusters": 3}})
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:200]}")
    clustering_works = r.status_code in [200, 400]  # 400 is expected error for no data
except Exception as e:
    print(f"   Error: {e}")
    clustering_works = False

# Test Optimization Endpoint Directly
print("\n4. Testing Optimization Endpoint...")
try:
    r = requests.post("http://localhost:8001/api/v1/training/optimize-sync",
                      json={"state_id": "test", "model_type": "RandomForest", 
                            "problem_type": "Classification", "method": "grid_search", "cv_folds": 3})
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:200]}")
    optimization_works = r.status_code in [200, 400]
except Exception as e:
    print(f"   Error: {e}")
    optimization_works = False

# Test Recommendations Endpoint Directly
print("\n5. Testing Recommendations Endpoint...")
try:
    r = requests.post("http://localhost:8001/api/v1/eda/analyze",
                      json={"dataset_id": "test"})
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.text[:200]}")
    recommendations_works = r.status_code in [200, 404]
except Exception as e:
    print(f"   Error: {e}")
    recommendations_works = False

# Test Frontend
print("\n6. Testing Frontend...")
try:
    r = requests.get("http://localhost:3001")
    print(f"   Status: {r.status_code}")
    frontend_works = r.status_code == 200
except Exception as e:
    print(f"   Error: {e}")
    frontend_works = False

# Summary
print("\n=== Deployment Summary ===")
print(f"Backend Healthy: {'OK' if backend_healthy else 'FAIL'}")
print(f"Clustering API: {'OK' if clustering_works else 'FAIL'}")
print(f"Optimization API: {'OK' if optimization_works else 'FAIL'}")
print(f"Recommendations API: {'OK' if recommendations_works else 'FAIL'}")
print(f"Frontend Running: {'OK' if frontend_works else 'FAIL'}")

all_good = backend_healthy and clustering_works and optimization_works and recommendations_works and frontend_works
print(f"\nOverall Status: {'All systems operational' if all_good else 'Some issues detected'}")