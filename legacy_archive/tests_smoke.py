"""Smoke test untuk QW1-QW3. Import semua modul yang sudah di-refactor."""
import importlib
import sys
import traceback


def safe_import(name):
    try:
        importlib.import_module(name)
        print(f"[OK]   {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
        return False


modules = [
    "core.state",
    "core.log",
    "core.notifications",
    "session_manager",
    "workflow_validator",
    "task_manager",
    "error_handler",
    "priority3_functions",
    "advanced_ml",
    "utils",
    "ml_engine.evaluation",
    "ml_engine.clustering_utils",
    "ml_engine.timeseries_utils",
]

ok = 0
fail = 0
for m in modules:
    if safe_import(m):
        ok += 1
    else:
        fail += 1

print()
print(f"Summary: {ok} OK, {fail} FAIL")
sys.exit(0 if fail == 0 else 1)
