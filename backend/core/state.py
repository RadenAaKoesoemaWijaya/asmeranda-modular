"""
Bridge module for Workflow state container in backend.core.
Delegates directly to root core.state to maintain single source of truth.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure root directory is on sys.path so root core is loaded
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import all core state symbols from root core.state
import importlib.util
_root_state_path = _PROJECT_ROOT / "core" / "state.py"
if _root_state_path.exists():
    _spec = importlib.util.spec_from_file_location("root_core_state", str(_root_state_path))
    if _spec is not None and _spec.loader is not None:
        _root_core_state = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_root_core_state)
        
        DEFAULT_KEYS = _root_core_state.DEFAULT_KEYS
        _LOCK = _root_core_state._LOCK
        _STATE_REGISTRY = _root_core_state._STATE_REGISTRY
        _states = _STATE_REGISTRY
        _new_state = _root_core_state._new_state
        _cleanup_expired_states = _root_core_state._cleanup_expired_states
        new_state_id = _root_core_state.new_state_id
        get_state = _root_core_state.get_state
        set_state = _root_core_state.set_state
        reset_state = _root_core_state.reset_state
        delete_state = _root_core_state.delete_state
        list_states = _root_core_state.list_states
        WorkflowState = _root_core_state.WorkflowState
        _save_state_metadata_to_disk = getattr(_root_core_state, "_save_state_metadata_to_disk", None)
        _load_state_from_disk = getattr(_root_core_state, "_load_state_from_disk", None)
        _PARTITIONS_DIR = getattr(_root_core_state, "_PARTITIONS_DIR", None)


def get_all_state_ids():
    with _LOCK:
        return list(_STATE_REGISTRY.keys())


def cleanup_old_states(max_age_seconds: int = 3600) -> int:
    res = _cleanup_expired_states()
    return int(res) if res is not None else 0


__all__ = [
    "DEFAULT_KEYS",
    "WorkflowState",
    "get_state",
    "set_state",
    "reset_state",
    "delete_state",
    "list_states",
    "new_state_id",
    "get_all_state_ids",
    "cleanup_old_states",
    "_cleanup_expired_states",
    "_states",
    "_LOCK",
    "_STATE_REGISTRY",
    "_save_state_metadata_to_disk",
    "_load_state_from_disk",
    "_PARTITIONS_DIR",
]

