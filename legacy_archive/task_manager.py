"""
Background task manager - framework-agnostic.

Executor di-cache manual (singleton modul-level). State task disimpan
di dict biasa (in-memory, thread-safe via lock). Streamlit context
injection hanya diupayakan sebagai best-effort agar legacy UI tidak
break - bila gagal di-skip silently.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

try:
    import streamlit as _st  # type: ignore

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    _st = None
    _ST_AVAILABLE = False

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx  # type: ignore

    _ADD_SCRIPT_RUN_CTX = True
except Exception:  # pragma: no cover
    add_script_run_ctx = None  # type: ignore
    _ADD_SCRIPT_RUN_CTX = False


# ---------------------------------------------------------------------------
# Module-level singleton executor (menggantikan @st.cache_resource)
# ---------------------------------------------------------------------------
_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None


def get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Kembalikan singleton ThreadPoolExecutor (max_workers=4)."""
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="asmeranda-bg"
                )
    return _EXECUTOR


# ---------------------------------------------------------------------------
# BackgroundTaskManager
# ---------------------------------------------------------------------------
class BackgroundTaskManager:
    """
    Task manager sederhana (in-memory).

    - state disimpan di ``self.tasks`` (dict, thread-safe via lock)
    - Setiap task punya id UUID dan field: name, status, result, error,
      start_time, future.
    - Untuk legacy Streamlit UI, juga mirror ke ``st.session_state.bg_tasks``
      agar kode lama yang membaca dari sana tetap jalan.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        # Mirror ke Streamlit session_state untuk backward compat
        if _ST_AVAILABLE:
            try:
                _st.session_state.setdefault("bg_tasks", {})  # type: ignore[union-attr]
            except Exception:
                pass

    def _mirror_write(self, task_id: str, payload: Dict[str, Any]) -> None:
        if not _ST_AVAILABLE:
            return
        try:
            _st.session_state["bg_tasks"][task_id] = payload  # type: ignore[index]
        except Exception:
            pass

    def _mirror_delete(self, task_id: str) -> None:
        if not _ST_AVAILABLE:
            return
        try:
            _st.session_state["bg_tasks"].pop(task_id, None)  # type: ignore[index]
        except Exception:
            pass

    def submit_task(self, task_name: str, func: Callable, *args: Any, **kwargs: Any) -> str:
        """Submit ``func`` ke background thread; kembalikan task_id."""
        task_id = uuid.uuid4().hex

        with self._lock:
            self.tasks[task_id] = {
                "name": task_name,
                "status": "queued",
                "result": None,
                "error": None,
                "start_time": time.time(),
                "future": None,
            }
            self._mirror_write(task_id, self.tasks[task_id])

        def task_wrapper(tid: str, fn: Callable, *a: Any, **kw: Any) -> None:
            try:
                with self._lock:
                    self.tasks[tid]["status"] = "running"
                    self._mirror_write(tid, self.tasks[tid])
                result = fn(*a, **kw)
                with self._lock:
                    self.tasks[tid]["status"] = "completed"
                    self.tasks[tid]["result"] = result
                    self._mirror_write(tid, self.tasks[tid])
            except Exception as exc:
                with self._lock:
                    self.tasks[tid]["status"] = "error"
                    self.tasks[tid]["error"] = str(exc)
                    self._mirror_write(tid, self.tasks[tid])

        future = get_executor().submit(task_wrapper, task_id, func, *args, **kwargs)

        # Best-effort: inject Streamlit context agar ``st.session_state``
        # masih bisa diakses di dalam thread. Gagal di-skip silently.
        if _ADD_SCRIPT_RUN_CTX and add_script_run_ctx is not None:
            try:
                add_script_run_ctx(future)
            except Exception:
                pass

        with self._lock:
            self.tasks[task_id]["future"] = future
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            # Salin agar caller tidak memodifikasi state internal
            return {k: v for k, v in task.items()}

    def clear_task(self, task_id: str) -> None:
        with self._lock:
            self.tasks.pop(task_id, None)
            self._mirror_delete(task_id)

    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {tid: {k: v for k, v in t.items()} for tid, t in self.tasks.items()}
