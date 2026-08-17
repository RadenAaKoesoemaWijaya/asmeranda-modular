# Asmeranda AI

Platform machine learning modular berbasis enterprise untuk workflow data science end-to-end. Mulai dari upload dataset, eksplorasi data, preprocessing, pelatihan model, optimasi hyperparameter, interpretasi model (XAI), hingga deteksi anomali dan forecasting deret waktu — semua dalam satu platform terintegrasi dengan keamanan berbasis peran (RBAC).

---

## 🚀 Fitur Utama

### 🔐 Keamanan & Kontrol Akses
- **RBAC (Role-Based Access Control)**: Tiga tingkat akses — `Admin`, `Analyst`, dan `Viewer`.
- **JWT & API Key**: Token sesi terenkripsi dan kunci API layanan.
- **Validasi Password**: Aturan kompleksitas ketat (huruf besar, kecil, angka, simbol).
- **Security Middleware**: Header keamanan HTTP, pembatasan ukuran payload (anti-DoS), dan rate limiting via SlowAPI.
- **Audit Log**: Jejak aktivitas terstruktur di `security_audit.log`.

### 🧠 Machine Learning Supervised
- **9+ Algoritma**: RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting, SVM, DecisionTree, KNN, Regresi Logistik/Linear.
- **Validasi Silang**: K-Fold, Stratified K-Fold, Leave-One-Out, Time Series Split.
- **Optimasi Hiperparameter**: Grid Search, Random Search, dan Bayesian Optimization via Optuna.
- **Rekomendasi Otomatis**: Saran algoritma dan pipeline preprocessing berdasarkan karakteristik dataset.
- **Evaluasi Lengkap**: ROC-AUC, PR Curve, Confusion Matrix, MCC, MAPE, Balanced Accuracy, Learning Curve.

### 🔍 Machine Learning Unsupervised & Reduksi Dimensi
- **Clustering**: KMeans, DBSCAN, Hierarchical, Spectral, dan HDBSCAN.
- **Optimal-K**: Analisis otomatis via Elbow Method dan Silhouette Score.
- **Reduksi Dimensi**: UMAP dan PCA untuk visualisasi data berdimensi tinggi (2D/3D).

### 💡 Explainable AI (XAI)
- **SHAP**: Feature importance global menggunakan TreeExplainer, LinearExplainer, dan KernelExplainer.
- **LIME**: Penjelasan prediksi lokal per instance untuk data tabular.

### 📈 Time Series & Deteksi Anomali
- **Forecasting**: ARIMA, SARIMA, Prophet, LSTM, dan rata-rata bergerak dengan inferensi frekuensi otomatis.
- **Deteksi Anomali**: Isolation Forest, One-Class SVM, dan batas statistik rolling.

### 🧹 Pemrosesan Data & EDA
- **Inferensi Tipe Otomatis**: Deteksi kolom numerik, kategorik, datetime, dan teks.
- **Pipeline Preprocessing**: Imputasi nilai hilang, deteksi outlier, encoding kategorik, dan scaling fitur.
- **EDA Suite**: Statistik deskriptif, histogram distribusi, dan heatmap korelasi.

---

## 🛠️ Teknologi yang Digunakan

| Lapisan | Teknologi |
|---|---|
| **Backend API** | FastAPI, Pydantic v2, Uvicorn, Starlette, SlowAPI |
| **Engine Data** | Polars, Pandas, PyArrow, NumPy |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, statsmodels |
| **Explainable AI** | SHAP, LIME |
| **Visualisasi** | Matplotlib, Seaborn |
| **Keamanan & Auth** | PyJWT, Bcrypt, Cryptography, Passlib |
| **Frontend** | Next.js 14 (App Router), React 18, Zustand, Custom CSS |
| **Infra & DevOps** | Docker, Docker Compose, Nginx, Azure Container Apps, AWS, GCP |

---

## 📦 Instalasi & Cara Menjalankan

### Prasyarat
- **Python 3.11+**
- **Node.js 18+** dan `npm`
- **Docker Desktop** *(opsional, untuk deployment containerized)*

---

### Menjalankan di Lokal (Development)

#### 1. Setup Backend

```bash
# Masuk ke direktori project
cd asmeranda-modular

# Buat dan aktifkan virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependensi backend
pip install -r backend/requirements-backend.txt

# Salin konfigurasi environment
cp .env.example .env

# Jalankan server FastAPI
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Setup Frontend

```bash
# Buka terminal baru, masuk ke folder frontend
cd frontend

# Install paket Node.js
npm install

# Jalankan server Next.js
npm run dev
```

---

### Akses & Kredensial Default

| Layanan | URL |
|---|---|
| **Aplikasi Web (Frontend)** | http://localhost:3000 |
| **API Backend** | http://localhost:8000 |
| **Dokumentasi Swagger** | http://localhost:8000/docs |
| **ReDoc API** | http://localhost:8000/redoc |

| Akun | Username | Password | Role |
|---|---|---|---|
| **Admin** | `admin` | `Admin@Asmeranda2026!` | `admin` |

---

## 🐳 Deployment & Pemeliharaan dengan Docker

### 1. Menjalankan / Build Pertama Kali
```bash
# Build dan jalankan semua service di latar belakang (Backend + Frontend + Nginx)
docker compose up --build -d

# Cek status kesehatan kontainer (pastikan status backend "healthy")
docker compose ps
```

### 2. Menerapkan Update pada Aplikasi yang Sedang Berjalan
Jika ada perubahan kode atau konfigurasi (seperti `next.config.js`, dependensi, atau backend):
```bash
# Opsi A: Rebuild dan restart service secara mulus (Recommended)
docker compose up --build -d

# Opsi B: Rebuild service tertentu saja (misal hanya frontend atau backend)
docker compose up --build -d frontend
docker compose up --build -d backend

# Opsi C: Bersihkan kontainer lama lalu nyalakan ulang (Clean Restart)
docker compose down
docker compose up --build -d
```

### 3. Monitoring & Troubleshooting
```bash
# Lihat log real-time semua service
docker compose logs -f

# Lihat log service tertentu (contoh: backend atau frontend)
docker compose logs -f backend
docker compose logs -f frontend

# Hentikan semua kontainer
docker compose down
```

---

## 📊 Referensi Endpoint API

| Endpoint | Metode | Deskripsi |
|---|---|---|
| `/api/v1/auth/login` | `POST` | Login dan dapatkan JWT token |
| `/api/v1/auth/register` | `POST` | Registrasi pengguna baru |
| `/api/v1/auth/me` | `GET` | Lihat profil pengguna aktif |
| `/api/v1/datasets/upload` | `POST` | Upload dataset (CSV, XLSX, Parquet, JSON) |
| `/api/v1/datasets/list` | `GET` | Daftar semua dataset |
| `/api/v1/datasets/{id}/preview` | `GET` | Preview dataset dengan paginasi |
| `/api/v1/eda/summary` | `POST` | Statistik deskriptif dan audit missing value |
| `/api/v1/preprocessing/run` | `POST` | Jalankan imputasi, scaling, dan train-test split |
| `/api/v1/preprocessing/cluster` | `POST` | Clustering unsupervised |
| `/api/v1/training/start` | `POST` | Mulai pelatihan model (asinkron) |
| `/api/v1/training/models` | `GET` | Daftar dan metrik model yang sudah dilatih |
| `/api/v1/optimization/hyperparameters` | `POST` | Optimasi Bayesian dengan Optuna |
| `/api/v1/interpretation/shap` | `POST` | Hitung SHAP feature attribution |
| `/api/v1/interpretation/lime` | `POST` | Penjelasan prediksi lokal (LIME) |
| `/api/v1/timeseries/forecast` | `POST` | Pelatihan model & prediksi deret waktu |
| `/api/v1/advanced-ml/umap` | `POST` | Reduksi dimensi dengan UMAP |
| `/health` | `GET` | Cek status kesehatan layanan |

---

## 🧪 Testing & Verifikasi

```bash
# Jalankan seluruh test suite
pytest

# Jalankan test keamanan saja
pytest backend/tests/security/ -v

# Jalankan dengan laporan code coverage
pytest --cov=backend --cov-report=term-missing

# Verifikasi sistem end-to-end
python final_verification.py
```

---

## ☁️ Deployment ke Cloud

| Platform | Cara Deploy |
|---|---|
| **Azure Container Apps** | `deploy-to-azure.bat` (Windows) atau `./deploy-to-azure.sh` (Linux) |
| **AWS** | `./deploy-cloud-aws.sh` |
| **GCP** | `./deploy-cloud-gcp.sh` |
| **Docker Desktop** | `./deploy-docker-desktop.ps1` |

---

## 📝 Lisensi & Kontak

Perangkat lunak proprietary milik **PT. Asmer Sahabat Sukses**.

- **Email Support**: support@asmeranda.ai
- **Dokumentasi Interaktif**: Akses `/docs` setelah server berjalan

---

**Asmeranda AI — Platform Machine Learning Modular End-to-End**
© 2024–2026 PT. Asmer Sahabat Sukses. Seluruh hak dilindungi undang-undang.