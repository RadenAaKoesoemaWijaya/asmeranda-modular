import os
import socket
import sys


def ensure_project_venv():
    """Use the repo-local virtual environment when present."""
    if getattr(sys, "frozen", False):
        return

    app_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(app_root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(app_root, ".venv", "bin", "python")

    if os.path.exists(venv_python):
        current_python = os.path.realpath(sys.executable)
        target_python = os.path.realpath(venv_python)
        if current_python != target_python:
            os.chdir(app_root)
            os.execv(target_python, [target_python, os.path.abspath(__file__)] + sys.argv[1:])


def get_available_port(start_port=8501, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("localhost", port))
                return port
            except OSError:
                continue
    return start_port


def main():
    ensure_project_venv()

    import streamlit.web.cli as stcli

    # Deteksi path aplikasi (berguna saat sudah di-compile oleh PyInstaller)
    if getattr(sys, 'frozen', False):
        application_path = sys._MEIPASS
        # Jika menggunakan --onedir, sys._MEIPASS sama dengan folder executable berada
        executable_dir = os.path.dirname(sys.executable)
        application_path = executable_dir
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))

    # Ubah working directory ke direktori aplikasi
    # Agar Streamlit bisa menemukan app.py, folder pages/, dll
    os.chdir(application_path)

    port = get_available_port()

    # Argumen untuk Streamlit
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        f"--server.port={port}",
        "--server.address=localhost",
        "--global.developmentMode=false"
    ]

    # Jalankan Streamlit CLI
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
