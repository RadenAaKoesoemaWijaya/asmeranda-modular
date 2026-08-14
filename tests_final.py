"""Final integration test - verifies the whole stack is buildable.

Run this after a fresh clone + install to confirm nothing is broken.

Checks:
1. Backend imports (no Streamlit needed for core)
2. Backend FastAPI app can start (via TestClient)
3. Frontend Next.js can build (next build)
4. All test files pass
"""
import os
import shutil
import subprocess
import sys


def header(label: str) -> None:
    print()
    print("=" * 60)
    print(label)
    print("=" * 60)


def run(cmd: list, cwd: str = None, timeout: int = 120) -> tuple:
    """Run command. Return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        return -1, "", f"TIMEOUT after {timeout}s"
    except Exception as exc:
        return -2, "", str(exc)


def main() -> int:
    project_root = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    # Temukan npx.cmd (di Windows, .CMD supaya subprocess bisa invoke)
    npx = shutil.which("npx.cmd") or shutil.which("npx") or "npx"
    if not npx or npx in ("", "npx"):
        # Fallback ke path absolut kalau ada
        candidates = [
            r"C:\Program Files\nodejs\npx.cmd",
            r"C:\Program Files\nodejs\npx",
        ]
        for c in candidates:
            if os.path.exists(c):
                npx = c
                break

    failures = []

    # ------------------------------------------------------------------
    header("1) Backend import smoke test (no Streamlit needed)")
    rc, out, err = run([py, "tests_smoke.py"], cwd=project_root)
    print(out)
    if rc != 0:
        failures.append(("smoke", rc))

    # ------------------------------------------------------------------
    header("2) Backend behavioral test")
    rc, out, err = run([py, "tests_behavior.py"], cwd=project_root)
    # Cari baris summary di akhir output
    summary_lines = [
        l
        for l in out.splitlines()
        if l.startswith("[") or "Summary" in l
    ]
    print("\n".join(summary_lines))
    if rc != 0:
        failures.append(("behavior", rc))

    # ------------------------------------------------------------------
    header("3) Backend E2E test (17 FastAPI endpoints)")
    rc, out, err = run([py, "tests_e2e.py"], cwd=project_root)
    important = [
        l
        for l in out.splitlines()
        if l.startswith("GET ")
        or l.startswith("POST ")
        or l.startswith("DELETE ")
        or "ALL E2E" in l
        or "FAIL" in l
    ]
    print("\n".join(important))
    if rc != 0:
        failures.append(("e2e", rc))

    # ------------------------------------------------------------------
    header("4) Frontend Next.js build")
    rc, out, err = run(
        [npx, "next", "build"],
        cwd=os.path.join(project_root, "frontend"),
        timeout=300,
    )
    important = [
        l
        for l in (out + err).splitlines()
        if "Compiled" in l
        or "First Load" in l
        or "Error" in l
        or "Failed" in l
        or "Route" in l
        or "/" == l.strip()
    ]
    print("\n".join(important[-30:]))
    if rc != 0:
        failures.append(("next-build", rc))

    # ------------------------------------------------------------------
    header("Summary")
    if failures:
        print("FAILED:")
        for name, code in failures:
            print(f"  - {name}: exit code {code}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
