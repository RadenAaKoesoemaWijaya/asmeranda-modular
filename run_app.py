import os
import sys
import streamlit.web.cli as stcli

def main():
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
    
    # Argumen untuk Streamlit
    sys.argv = [
        "streamlit", 
        "run", 
        "app.py", 
        "--server.port=8501", 
        "--server.address=localhost",
        "--global.developmentMode=false"
    ]
    
    # Jalankan Streamlit CLI
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
