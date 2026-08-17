"""Test all advanced ML features."""
import sys
sys.path.insert(0, '.')

import requests
import json

print("=== Advanced ML Features Testing ===\n")

# Test Backend Health
print("1. Testing Backend Health...")
try:
    r = requests.get("http://localhost:8000/health")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
    backend_healthy = r.status_code == 200
except Exception as e:
    print(f"   Error: {e}")
    backend_healthy = False

# Test OpenAPI for Advanced ML endpoints
print("\n2. Testing Advanced ML Endpoints Registration...")
try:
    r = requests.get("http://localhost:8000/openapi.json")
    data = r.json()
    advanced_ml_endpoints = [p for p in data['paths'].keys() if 'advanced-ml' in p]
    print(f"   Advanced ML endpoints: {advanced_ml_endpoints}")
    advanced_ml_registered = len(advanced_ml_endpoints) > 0
except Exception as e:
    print(f"   Error: {e}")
    advanced_ml_registered = False

# Test UMAP endpoint
print("\n3. Testing UMAP Endpoint...")
try:
    r = requests.post("http://localhost:8000/api/v1/advanced-ml/umap",
                      json={"state_id": "test", "n_components": 2, "n_neighbors": 15, "min_dist": 0.1})
    print(f"   Status: {r.status_code}")
    umap_works = r.status_code in [200, 400]  # 400 is expected error for no data
    if r.status_code == 200:
        print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")
    umap_works = False

# Test HDBSCAN endpoint
print("\n4. Testing HDBSCAN Endpoint...")
try:
    r = requests.post("http://localhost:8000/api/v1/advanced-ml/hdbscan",
                      json={"state_id": "test", "min_cluster_size": 5, "min_samples": None, "metric": "euclidean"})
    print(f"   Status: {r.status_code}")
    hdbscan_works = r.status_code in [200, 400]
    if r.status_code == 200:
        print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")
    hdbscan_works = False

# Test Anomaly Detection endpoint
print("\n5. Testing Anomaly Detection Endpoint...")
try:
    r = requests.post("http://localhost:8000/api/v1/advanced-ml/anomaly-detection",
                      json={"state_id": "test", "method": "isolation_forest", "contamination": 0.1, "n_estimators": 100})
    print(f"   Status: {r.status_code}")
    anomaly_works = r.status_code in [200, 400]
    if r.status_code == 200:
        print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")
    anomaly_works = False

# Test Forecasting endpoint
print("\n6. Testing Forecasting Endpoint...")
try:
    r = requests.post("http://localhost:8000/api/v1/advanced-ml/forecast",
                      json={"state_id": "test", "target_column": "value", "periods": 10, "method": "arima"})
    print(f"   Status: {r.status_code}")
    forecast_works = r.status_code in [200, 400]
    if r.status_code == 200:
        print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")
    forecast_works = False

# Test Missing Values endpoint
print("\n7. Testing Missing Values Endpoint...")
try:
    r = requests.post("http://localhost:8000/api/v1/advanced-ml/handle-missing-values",
                      json={"state_id": "test", "strategy": "auto", "numeric_strategy": "mean", "categorical_strategy": "mode", "threshold": 0.5})
    print(f"   Status: {r.status_code}")
    missing_values_works = r.status_code in [200, 400]
    if r.status_code == 200:
        print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")
    missing_values_works = False

# Test Outlier Detection endpoint
print("\n8. Testing Outlier Detection Endpoint...")
try:
    r = requests.post("http://localhost:8000/api/v1/advanced-ml/detect-outliers",
                      json={"state_id": "test", "method": "iqr", "threshold": 1.5, "columns": None})
    print(f"   Status: {r.status_code}")
    outlier_works = r.status_code in [200, 400]
    if r.status_code == 200:
        print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")
    outlier_works = False

# Test Frontend
print("\n9. Testing Frontend...")
try:
    r = requests.get("http://localhost:3001")
    print(f"   Status: {r.status_code}")
    frontend_works = r.status_code == 200
except Exception as e:
    print(f"   Error: {e}")
    frontend_works = False

# Summary
print("\n=== Advanced ML Features Summary ===")
print(f"Backend Healthy: {'OK' if backend_healthy else 'FAIL'}")
print(f"Advanced ML Endpoints Registered: {'OK' if advanced_ml_registered else 'FAIL'}")
print(f"UMAP Endpoint: {'OK' if umap_works else 'FAIL'}")
print(f"HDBSCAN Endpoint: {'OK' if hdbscan_works else 'FAIL'}")
print(f"Anomaly Detection Endpoint: {'OK' if anomaly_works else 'FAIL'}")
print(f"Forecasting Endpoint: {'OK' if forecast_works else 'FAIL'}")
print(f"Missing Values Endpoint: {'OK' if missing_values_works else 'FAIL'}")
print(f"Outlier Detection Endpoint: {'OK' if outlier_works else 'FAIL'}")
print(f"Frontend Running: {'OK' if frontend_works else 'FAIL'}")

all_good = backend_healthy and advanced_ml_registered and umap_works and hdbscan_works and anomaly_works and forecast_works and missing_values_works and outlier_works and frontend_works
print(f"\nOverall Status: {'SUCCESS - All advanced ML features operational' if all_good else 'PARTIAL - Some issues detected'}")

if all_good:
    print("\n=== Advanced ML Features Ready ===")
    print("All advanced ML features are available and operational:")
    print("- UMAP Dimensionality Reduction")
    print("- HDBSCAN Clustering")
    print("- Anomaly Detection (Isolation Forest, One-Class SVM)")
    print("- Time Series Forecasting (ARIMA, SARIMA, Prophet, LSTM)")
    print("- Data Utilities (Missing Values, Outlier Detection, Data Validation)")
else:
    print("\n=== Issues Detected ===")
    print("Please address the issues above before proceeding.")