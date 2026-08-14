"""End-to-end test untuk FastAPI backend."""
import io
import sys

import pandas as pd


def main():
    import sys

    sys.path.insert(0, ".")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1) Health
    r = client.get("/health")
    print("GET /health ->", r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # 2) Upload dataset (CSV in-memory)
    df = pd.DataFrame(
        {
            "age": [25, 32, 47, 51, 22, 38, 44, 29, 60, 35, 41, 28],
            "salary": [50000, 60000, 80000, 90000, 45000, 70000, 85000, 55000, 95000, 65000, 75000, 52000],
            "department": ["IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "HR", "Finance", "IT", "HR", "IT"],
            "churn": [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        }
    )
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    files = {"file": ("test.csv", csv_buf.getvalue().encode("utf-8"), "text/csv")}
    r = client.post("/api/v1/datasets", files=files)
    print("POST /api/v1/datasets ->", r.status_code)
    assert r.status_code == 200, r.text
    upload = r.json()
    print("  metadata:", {k: upload["metadata"][k] for k in ("dataset_id", "rows", "columns")})
    assert upload["success"] is True
    dataset_id = upload["metadata"]["dataset_id"]

    # 3) List datasets
    r = client.get("/api/v1/datasets")
    print("GET /api/v1/datasets ->", r.status_code, r.json()["total"], "datasets")
    assert r.json()["total"] == 1

    # 4) EDA summary
    r = client.get(f"/api/v1/eda/{dataset_id}/summary")
    print("GET /api/v1/eda/{id}/summary ->", r.status_code)
    assert r.status_code == 200
    s = r.json()
    print("  shape:", s["shape"])
    print("  num cols:", list(s["describe_numeric"].keys()))
    assert s["success"] is True
    assert s["shape"]["rows"] == 12
    assert s["shape"]["columns"] == 4

    # 5) EDA correlation
    r = client.get(f"/api/v1/eda/{dataset_id}/correlation")
    print("GET /api/v1/eda/{id}/correlation ->", r.status_code)
    assert r.status_code == 200
    c = r.json()
    assert c["success"] is True
    print("  corr cols:", c["columns"])
    print("  matrix size:", len(c["matrix"]), "x", len(c["matrix"][0]) if c["matrix"] else 0)

    # 6) Preprocessing
    cfg = {
        "dataset_id": dataset_id,
        "target_column": "churn",
        "problem_type": "Classification",
        "scaling_method": "standard",
        "imputation_strategy": "mean",
        "apply_encoding": True,
        "test_size": 0.25,
        "random_state": 42,
    }
    r = client.post("/api/v1/preprocessing/run", json=cfg)
    print("POST /api/v1/preprocessing/run ->", r.status_code)
    assert r.status_code == 200
    pp = r.json()
    print("  success:", pp["success"])
    print("  state_id:", pp["state_id"])
    print("  n_train:", pp["n_samples_train"], "n_test:", pp["n_samples_test"])
    print("  n_features:", pp["n_features"])
    print("  steps:", pp["preprocessing_steps"])
    assert pp["success"] is True
    assert pp["n_samples_train"] == 9
    assert pp["n_samples_test"] == 3
    state_id = pp["state_id"]

    # 7) Training - RandomForest
    tcfg = {
        "state_id": state_id,
        "model_type": "RandomForest",
        "problem_type": "Classification",
        "cv_method": "kfold",
        "cv_folds": 3,
    }
    r = client.post("/api/v1/training/start", json=tcfg)
    print("POST /api/v1/training/start ->", r.status_code)
    assert r.status_code == 200
    tr = r.json()
    print("  success:", tr["success"])
    print("  model_id:", tr.get("model_id"))
    print("  metrics keys:", list((tr.get("metrics") or {}).keys()))
    print("  cv report:", (tr.get("cv_scores") or {}).get("mean"))
    assert tr["success"] is True
    model_id = tr["model_id"]

    # 8) List models
    r = client.get("/api/v1/training/models")
    print("GET /api/v1/training/models ->", r.status_code, "models:", list(r.json().keys()))
    assert model_id in r.json()

    # 8b) SHAP analysis
    r = client.post(
        "/api/v1/interpretation/shap",
        json={"model_id": model_id, "state_id": state_id, "max_samples": 50},
    )
    print("POST /api/v1/interpretation/shap ->", r.status_code)
    assert r.status_code == 200
    sh = r.json()
    assert sh["success"] is True
    print("  method:", sh.get("method"), "n_samples:", sh.get("n_samples"))
    print("  feature_importance items:", len(sh.get("feature_importance") or []))
    if sh.get("shap_values_summary"):
        print("  top SHAP feature:", sh["shap_values_summary"][0])

    # 8c) LIME analysis
    r = client.post(
        "/api/v1/interpretation/lime",
        json={"model_id": model_id, "state_id": state_id, "sample_index": 0, "num_features": 5},
    )
    print("POST /api/v1/interpretation/lime ->", r.status_code)
    assert r.status_code == 200
    li = r.json()
    assert li["success"] is True
    print("  explanation items:", len(li.get("explanation") or []))

    # 9) Get model metadata
    r = client.get(f"/api/v1/training/models/{model_id}")
    print("GET /api/v1/training/models/{id} ->", r.status_code)
    assert r.status_code == 200

    # 10) Delete model
    r = client.delete(f"/api/v1/training/models/{model_id}")
    print("DELETE /api/v1/training/models/{id} ->", r.status_code, r.json())
    assert r.json()["deleted"] is True

    # 12) Re-upload timeseries dataset
    np = __import__("numpy")
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    series = np.sin(np.linspace(0, 12 * np.pi, 60)) * 10 + 50 + np.random.normal(0, 0.5, 60)
    df2 = pd.DataFrame({"date": dates.astype(str), "value": series})
    csv_buf2 = io.StringIO()
    df2.to_csv(csv_buf2, index=False)
    files2 = {"file": ("ts.csv", csv_buf2.getvalue().encode("utf-8"), "text/csv")}
    r = client.post("/api/v1/datasets", files=files2)
    assert r.status_code == 200
    ts_id = r.json()["metadata"]["dataset_id"]
    print("POST /api/v1/datasets (timeseries) ->", r.status_code, "id:", ts_id)

    # 13) Timeseries detect
    r = client.get(f"/api/v1/timeseries/{ts_id}/detect")
    print("GET /api/v1/timeseries/{id}/detect ->", r.status_code)
    assert r.status_code == 200
    det = r.json()
    assert det["success"] is True
    print("  target:", det["target_column"], "n:", det["n_observations"], "anomalies:", det["n_anomalies"])

    # 14) Timeseries forecast
    r = client.get(f"/api/v1/timeseries/{ts_id}/forecast?target_column=value&horizon=5&method=drift")
    print("GET /api/v1/timeseries/{id}/forecast ->", r.status_code)
    assert r.status_code == 200
    fc = r.json()
    assert fc["success"] is True
    print("  method:", fc["method"], "horizon:", fc["horizon"], "len(forecast):", len(fc["forecast"]))
    assert len(fc["forecast"]) == 5

    # 15) Timeseries anomalies
    r = client.get(f"/api/v1/timeseries/{ts_id}/anomalies?target_column=value&contamination=0.05")
    print("GET /api/v1/timeseries/{id}/anomalies ->", r.status_code)
    assert r.status_code == 200
    an = r.json()
    assert an["success"] is True
    print("  n_anomalies:", an["n_anomalies"])

    # 16) Cleanup timeseries dataset
    r = client.delete(f"/api/v1/datasets/{ts_id}")
    print("DELETE /api/v1/datasets/{id} (ts) ->", r.status_code, r.json())
    assert r.json()["deleted"] is True

    # 17) Delete dataset
    r = client.delete(f"/api/v1/datasets/{dataset_id}")
    print("DELETE /api/v1/datasets/{id} ->", r.status_code, r.json())
    assert r.json()["deleted"] is True

    print()
    print("ALL E2E TESTS PASSED")


if __name__ == "__main__":
    main()
