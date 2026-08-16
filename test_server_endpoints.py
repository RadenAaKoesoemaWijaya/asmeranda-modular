"""Test to check if the endpoints are actually registered in the running server."""
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
    # Get the OpenAPI spec to see all registered endpoints
    r = requests.get("http://localhost:8000/openapi.json")
    data = r.json()
    
    print("All registered endpoints:")
    for path in data['paths'].keys():
        print(f"  {path}")
    
    # Test all preprocessing endpoints
    print("\nTesting preprocessing endpoints:")
    for path in data['paths'].keys():
        if 'preprocessing' in path:
            print(f"  Found: {path}")
    
    # Test all training endpoints
    print("\nTesting training endpoints:")
    for path in data['paths'].keys():
        if 'training' in path:
            print(f"  Found: {path}")
    
    # Test all eda endpoints
    print("\nTesting eda endpoints:")
    for path in data['paths'].keys():
        if 'eda' in path:
            print(f"  Found: {path}")

finally:
    # Kill the backend process
    backend_process.terminate()
    backend_process.wait()
    print("\nBackend server stopped")