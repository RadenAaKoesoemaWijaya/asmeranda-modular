"""Test script to verify new endpoints are working."""
import sys
sys.path.insert(0, '.')

import requests
import time

# Start the backend server
import subprocess
backend_process = subprocess.Popen(
    ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="C:\\asmeranda-modular"
)

# Wait for server to start
time.sleep(5)

try:
    # Test health endpoint
    r = requests.get("http://localhost:8000/health")
    print(f"Health endpoint: {r.status_code} - {r.json()}")

    # Test OpenAPI spec
    r = requests.get("http://localhost:8000/openapi.json")
    data = r.json()
    
    print(f"\nTotal endpoints: {len(data['paths'])}")
    print("All endpoints:")
    for path in data['paths'].keys():
        print(f"  {path}")
    
    # Check for new endpoints
    clustering_endpoints = [path for path in data['paths'].keys() if 'clustering' in path]
    optimization_endpoints = [path for path in data['paths'].keys() if 'optimization' in path]
    recommendations_endpoints = [path for path in data['paths'].keys() if 'recommendations' in path]
    
    print(f"\nClustering endpoints: {clustering_endpoints}")
    print(f"Optimization endpoints: {optimization_endpoints}")
    print(f"Recommendations endpoints: {recommendations_endpoints}")

finally:
    # Kill the backend process
    backend_process.terminate()
    backend_process.wait()
    print("\nBackend server stopped")