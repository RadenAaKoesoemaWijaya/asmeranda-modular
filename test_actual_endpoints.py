"""Test the actual endpoints directly."""
import sys
sys.path.insert(0, '.')

import requests
import subprocess
import time

# Start the backend server
backend_process = subprocess.Popen(
    ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="C:\\asmeranda-modular"
)

# Wait for server to start
time.sleep(5)

try:
    # Test clustering endpoint (through preprocessing)
    print("Testing clustering endpoint through preprocessing...")
    r = requests.post("http://localhost:8000/api/v1/preprocessing/cluster", json={
        "state_id": "test",
        "method": "kmeans",
        "parameters": {"n_clusters": 3}
    })
    print(f"Clustering endpoint: {r.status_code} - {r.text[:200]}")

    # Test optimization endpoint (through training)
    print("\nTesting optimization endpoint through training...")
    r = requests.post("http://localhost:8000/api/v1/training/optimize-sync", json={
        "state_id": "test",
        "model_type": "RandomForest",
        "problem_type": "Classification",
        "method": "grid_search",
        "cv_folds": 3
    })
    print(f"Optimization endpoint: {r.status_code} - {r.text[:200]}")

    # Test recommendations endpoint (through eda)
    print("\nTesting recommendations endpoint through eda...")
    r = requests.post("http://localhost:8000/api/v1/eda/analyze", json={
        "dataset_id": "test"
    })
    print(f"Recommendations endpoint: {r.status_code} - {r.text[:200]}")

finally:
    # Kill the backend process
    backend_process.terminate()
    backend_process.wait()
    print("\nBackend server stopped")