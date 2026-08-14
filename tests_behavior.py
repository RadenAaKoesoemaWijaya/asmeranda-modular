"""Behavioral test untuk QW1-QW3.

Menguji bahwa API publik modul-modul yang di-refactor masih
memiliki perilaku yang benar - baik dengan maupun tanpa Streamlit.
"""
import importlib
import sys
import traceback
from typing import Any


def run(label: str, fn) -> bool:
    try:
        fn()
        print(f"[OK]   {label}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# core.state
# ---------------------------------------------------------------------------
def test_core_state():
    from core import state as cs

    # Default state
    s = cs.get_state()
    assert s["target_column"] is None
    assert s["numerical_columns"] == []

    # Custom state_id
    sid = cs.new_state_id()
    s2 = cs.get_state(sid)
    s2["target_column"] = "y"
    assert cs.get_state(sid)["target_column"] == "y"
    cs.delete_state(sid)
    assert cs.get_state(sid).get("target_column") is None

    # WorkflowState wrapper
    sid2 = cs.new_state_id()
    ws = cs.WorkflowState(sid2)
    ws["problem_type"] = "Classification"
    assert ws["problem_type"] == "Classification"
    ws.set(target_column="y", X_train=None)
    assert ws.get("target_column") == "y"
    cs.delete_state(sid2)


# ---------------------------------------------------------------------------
# core.notifications
# ---------------------------------------------------------------------------
def test_core_notifications():
    from core import notifications as cn

    bus = cn.get_bus()
    received = []

    def listener(n):
        received.append(n)

    unsub = bus.subscribe(listener)
    cn.info("hello info", source="test")
    cn.success("hello success", source="test")
    cn.warning("hello warning", source="test")
    cn.error("hello error", source="test")
    cn.write("hello write", source="test")
    unsub()

    assert len(received) == 5, f"expected 5, got {len(received)}"
    levels = [n.level for n in received]
    assert levels == ["info", "success", "warning", "error", "write"]


# ---------------------------------------------------------------------------
# session_manager (legacy + new)
# ---------------------------------------------------------------------------
def test_session_manager():
    from session_manager import SessionStateManager, SessionManager

    sm = SessionStateManager()
    sm.initialize_data_session(__import__("pandas").DataFrame({"a": [1, 2, 3]}))
    assert sm.state["data"] is not None

    sm2 = SessionManager(state={"target_column": "x"})
    assert sm2.state["target_column"] == "x"
    res = sm2.validate_data_flow("upload", "eda")
    # Setelah initialize, data di-set ke None
    assert "valid" in res

    # validate_data_flow with all required keys
    sm2.state["data"] = "non-empty"
    res = sm2.validate_data_flow("upload", "eda")
    assert res["valid"] is True


# ---------------------------------------------------------------------------
# workflow_validator
# ---------------------------------------------------------------------------
def test_workflow_validator():
    from workflow_validator import WorkflowValidator

    state = {"data": None}
    v = WorkflowValidator(state=state)
    res = v.validate_workflow_transition("upload", "eda")
    assert res["valid"] is False
    assert "data" in res["missing"]

    # Isi data
    import pandas as pd

    state["data"] = pd.DataFrame({"a": range(20), "b": range(20)})
    state["numerical_columns"] = ["a", "b"]
    state["categorical_columns"] = []
    state["X_train"] = pd.DataFrame({"a": range(80), "b": range(80)})
    state["X_test"] = pd.DataFrame({"a": range(20), "b": range(20)})
    state["y_train"] = pd.Series(range(80))
    state["y_test"] = pd.Series(range(20))
    state["problem_type"] = "Classification"
    state["model_results"] = []

    res = v.validate_workflow_transition("preprocessing", "training")
    assert res["valid"] is True, res

    res = v.validate_workflow_transition("training", "interpretation")
    assert res["valid"] is False  # model_results masih kosong


# ---------------------------------------------------------------------------
# task_manager
# ---------------------------------------------------------------------------
def test_task_manager():
    import time

    from task_manager import BackgroundTaskManager, get_executor

    tm = BackgroundTaskManager()
    assert get_executor() is get_executor()  # singleton

    def add(a, b):
        time.sleep(0.01)
        return a + b

    tid = tm.submit_task("test", add, 2, 3)
    fut = tm.get_task(tid)["future"]
    fut.result(timeout=2)
    assert tm.get_task(tid)["status"] == "completed"
    assert tm.get_task(tid)["result"] == 5


# ---------------------------------------------------------------------------
# error_handler
# ---------------------------------------------------------------------------
def test_error_handler():
    from error_handler import ErrorHandler

    captured = []

    def cb(info):
        captured.append(info)

    h = ErrorHandler(language="id", display_callback=cb)
    info = h.handle_error(ValueError("bad value"), context="unit_test")
    assert info["type"] == "invalid_parameter"
    assert "Pesan" in info["message"] or "valid" in info["message"]
    assert len(captured) == 1

    # English
    h.set_language("en")
    info = h.handle_error(MemoryError("OOM"), context="oom_test")
    assert info["type"] == "memory_error"
    assert len(captured) == 2


# ---------------------------------------------------------------------------
# advanced_ml / priority3
# ---------------------------------------------------------------------------
def test_advanced_ml_and_priority3():
    import advanced_ml
    import priority3_functions

    # advanced_ml: defensive flags
    assert hasattr(advanced_ml, "UMAP_AVAILABLE")
    # priority3_functions: stats
    stats = priority3_functions.get_interpretation_performance_stats()
    assert "shap" in stats
    assert "lime" in stats


# ---------------------------------------------------------------------------
# ml_engine.evaluation
# ---------------------------------------------------------------------------
def test_evaluation():
    from ml_engine.evaluation import (
        adjusted_r2_score,
        breusch_pagan_test,
        calculate_vif,
        get_model_type,
    )

    assert adjusted_r2_score(0.8, 100, 3) > 0
    df = __import__("pandas").DataFrame({"a": range(10), "b": [x * 2 for x in range(10)]})
    vif = calculate_vif(df)
    assert len(vif) == 2
    # get_model_type
    class FakeClf:
        def predict_proba(self, X):
            return [[0.5, 0.5]]

        classes_ = [0, 1]

    class FakeReg:
        def predict(self, X):
            return [0]

    assert get_model_type(FakeClf()) == "Classification"
    assert get_model_type(FakeReg()) == "Regression"


# ---------------------------------------------------------------------------
# ml_engine.clustering_utils / timeseries_utils
# ---------------------------------------------------------------------------
def test_clustering():
    from ml_engine.clustering_utils import (
        calculate_comprehensive_clustering_metrics,
        plot_gap_statistic,
    )
    import numpy as np

    X = np.random.rand(50, 2)
    labels = np.array([0] * 25 + [1] * 25)
    m = calculate_comprehensive_clustering_metrics(X, labels)
    assert "silhouette_score" in m

    fig = plot_gap_statistic([0.1, 0.3, 0.5], [0.05, 0.05, 0.05], [1, 2, 3], language="id")
    assert fig is not None


def test_timeseries_utils():
    from ml_engine.timeseries_utils import (
        build_forecast_summary,
        calculate_forecast_metrics,
    )
    import numpy as np
    import pandas as pd

    actual = np.linspace(0, 10, 20)
    pred = actual + np.random.normal(0, 0.1, 20)
    metrics = calculate_forecast_metrics(actual, pred)
    assert metrics["count"] == 20
    assert metrics["rmse"] < 1.0

    # build_forecast_summary
    df = pd.DataFrame({"y": actual})
    fr = {"Model A": {"forecast": pred}}
    s = build_forecast_summary(fr, "y", df)
    assert s is not None
    assert "best_rmse" in s


# ---------------------------------------------------------------------------
# utils.advanced_data_scaling & advanced_feature_transformation
# ---------------------------------------------------------------------------
def test_utils_scaling():
    from utils import advanced_data_scaling, advanced_feature_transformation
    import numpy as np
    import pandas as pd

    np.random.seed(0)
    df = pd.DataFrame(
        {
            "a": np.random.normal(0, 1, 50),
            "b": np.random.normal(5, 2, 50),
            "c": np.random.choice(["x", "y", "z"], 50),
        }
    )
    res = advanced_data_scaling(df, method="standard", use_streamlit=False)
    assert res["success"] is True
    assert "scaled_data" in res

    res2 = advanced_feature_transformation(
        df, method="auto", target_column="a", use_streamlit=False
    )
    assert res2["success"] is True


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
tests = [
    ("core.state", test_core_state),
    ("core.notifications", test_core_notifications),
    ("session_manager", test_session_manager),
    ("workflow_validator", test_workflow_validator),
    ("task_manager", test_task_manager),
    ("error_handler", test_error_handler),
    ("advanced_ml/priority3", test_advanced_ml_and_priority3),
    ("ml_engine.evaluation", test_evaluation),
    ("ml_engine.clustering_utils", test_clustering),
    ("ml_engine.timeseries_utils", test_timeseries_utils),
    ("utils.advanced_data_scaling/feature_transformation", test_utils_scaling),
]

ok = 0
fail = 0
for label, fn in tests:
    if run(label, fn):
        ok += 1
    else:
        fail += 1

print()
print(f"Summary: {ok} OK, {fail} FAIL")
sys.exit(0 if fail == 0 else 1)
