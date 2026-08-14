# 🤖 Asmeranda AI

> **Machine Learning Platform Modular — PT. Asmer Sahabat Sukses**

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen)
![Stack](https://img.shields.io/badge/stack-FastAPI%20%2B%20Next.js%2014-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)

> ⚠️ **Hak Cipta Dilindungi.** Dilarang memperbanyak atau mendistribusikan
> tanpa izin tertulis dari **PT. Asmer Sahabat Sukses**.

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Unggulan](#-fitur-unggulan)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Alur Kerja ML](#-alur-kerja-ml)
- [Status Build & Verifikasi](#-status-build--verifikasi)
- [Cara Instalasi](#-cara-instalasi)
  - [Opsi A — Lokal](#opsi-a--instalasi-lokal)
  - [Opsi B — Docker Desktop](#opsi-b--docker-desktop)
  - [Opsi C — Azure Container Apps](#opsi-c--azure-container-apps)
  - [Opsi D — Vercel + Railway](#opsi-d--cloud-terpisah-vercel--railway)
- [Testing](#-testing)
- [Endpoint API](#-endpoint-api-fastapi-v1)
- [File Deployment](#-file-deployment)
- [Catatan & Batasan](#-catatan--batasan)
- [Pembaruan Terbaru](#-pembaruan-terbaru)
- [Lisensi](#-lisensi--hak-cipta)

---

## 🧠 Tentang Proyek

**Asmeranda AI** adalah platform Machine Learning berbasis web yang dirancang secara **modular**, memungkinkan pengguna untuk menjalankan seluruh pipeline analitik data — mulai dari unggah dataset hingga interpretasi model — melalui antarmuka yang intuitif tanpa perlu menulis kode.

Platform ini dibangun di atas:
- **Backend:** FastAPI (Python) — REST API async dengan dokumentasi OpenAPI otomatis
- **Frontend:** Next.js 14 App Router — antarmuka modern dengan state management Zustand
- **ML Engine:** Scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, LIME, Polars

---

## ✨ Fitur Unggulan

| Fitur | Deskripsi |
|---|---|
| **Backend FastAPI** | Auto-generated OpenAPI/Swagger di `/docs`, type-safe via Pydantic, async-ready, dependency injection |
| **Frontend Modern** | Next.js 14 App Router, Zustand state (persist ke localStorage), sidebar dengan aktivasi per workflow step |
| **Stack ML Lengkap** | Random Forest, XGBoost, LightGBM, CatBoost, SHAP, LIME, IsolationForest, ADF stationarity test |
| **Big Data Engine** | Polars (Apache Arrow multi-thread) menggantikan Pandas — hingga 10–100× lebih cepat, hemat memori |
| **Server-Side Pagination** | `pl.scan_parquet()` (LazyFrame) — hanya memuat data per halaman tanpa membebani RAM |
| **WebSocket Real-time** | Progress preprocessing di-broadcast secara real-time ke semua klien yang terhubung |
| **Docker Ready** | Backend + Frontend berjalan dengan satu perintah `docker compose up --build` |
| **Workflow Guard** | `WorkflowValidator` mencegah pengguna melompat ke tahap yang belum siap |
| **Defensive Fallbacks** | Bila pustaka C++ tidak tersedia, sistem auto-fallback ke Python native tanpa crash |
| **i18n Built-in** | Antarmuka tersedia dalam Bahasa Indonesia & English |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────┐
│                  FRONTEND (Next.js 14)           │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Upload  │  │   EDA    │  │  Model Train  │  │
│  │ Dataset  │  │ & Stats  │  │  & Interpret  │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       └─────────────┴────────────── ┘           │
│              Zustand State (localStorage)        │
└──────────────────────┬──────────────────────────┘
                       │ HTTP REST / WebSocket
┌──────────────────────▼──────────────────────────┐
│                 BACKEND (FastAPI)                │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Dataset  │  │  ML      │  │  Timeseries   │  │
│  │ Manager  │  │ Engine   │  │  & Anomaly    │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       └─────────────┴───────────────┘            │
│         Polars (Arrow Engine) + Scikit-learn      │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              DATA LAYER (disk/volume)            │
│   data/datasets/  *.parquet + *.meta.json        │
│   data/models/    *.pkl                          │
└─────────────────────────────────────────────────┘
```

---

## 🔄 Alur Kerja ML

Setiap langkah divalidasi oleh `WorkflowValidator` — pengguna tidak dapat melanjutkan ke tahap berikutnya sebelum tahap sebelumnya selesai.

```
┌───────────────────┐
│  1. Data Upload   │  ← CSV / XLSX / Parquet / JSON / TSV
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. EDA           │  ← Summary statistik, korelasi Pearson, missing values
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. Preprocessing │  ← Imputasi, scaling, encoding, train-test split
└────────┬──────────┘   (progress real-time via WebSocket)
         │
         ▼
┌───────────────────┐
│  4. Model Training│  ← Pilih algoritma + cross-validation
└──┬─────────────┬──┘
   │             │
   ▼             ▼
┌──────┐     ┌──────┐
│ SHAP │     │ LIME │   ← Interpretasi model (global & per-instance)
└──────┘     └──────┘
         │
         ▼
┌───────────────────┐
│  5. Time Series   │  ← Forecasting (naive/drift/mean) & deteksi anomali
└───────────────────┘
```

---

## ✅ Status Build & Verifikasi

Hasil verifikasi terakhir **(Juni 2026)**:

| Komponen | Status |
|---|---|
| Frontend `npm run build` | ✅ Berhasil |
| Backend smoke test (13 modul) | ✅ Lulus |
| Behavioral test (11 test) | ✅ Lulus |
| E2E API (17 endpoint) | ✅ Lulus |
| `tests_final.py` (aggregator) | ✅ Lulus |
| Docker build frontend | ✅ Berhasil |
| Docker Compose (`up --build`) | ✅ Siap |

Jalankan semua verifikasi sekaligus:

```bash
# Windows
py tests_final.py

# Linux / macOS
python tests_final.py
```

---

## 🚀 Cara Instalasi

Aplikasi mendukung **dual-mode deployment** — pilih sesuai kebutuhan:

| Opsi | Lingkungan | Cocok Untuk |
|---|---|---|
| **A** | Instalasi Lokal | Development & debugging (hot reload) |
| **B** | Docker Desktop | Production parity & isolation |
| **C** | Azure Container Apps | Enterprise cloud deployment |
| **D** | Vercel + Railway | Distributed cloud deployment |

> **Catatan:** Aplikasi sudah fully compatible dengan Docker Desktop dan instalasi lokal tanpa modifikasi tambahan.

---

### Opsi A — Instalasi Lokal

**Kebutuhan:** Python 3.11+, Node.js 20+, npm

```bash
# Terminal 1 — Backend (jalankan dari project root)
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements-backend.txt
py -m uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend (jalankan dari project root)
cd frontend
npm install
npm run dev
```

**Akses:** Frontend http://localhost:3000 | Backend http://localhost:8000 | API Docs http://localhost:8000/docs

> **Penting:** Jalankan kedua terminal dari direktori project root (`c:\asmeranda-modular`), bukan dari dalam folder `backend` atau `frontend`.

---

### Opsi B — Docker Desktop

**Kebutuhan:** Docker Desktop terinstal

```bash
docker compose up --build
```

**Akses:** Frontend http://localhost:3000 | Backend http://localhost:8000 | API Docs http://localhost:8000/docs

> Dataset tersimpan di volume Docker `asmeranda-data` dan survive restart container.

---

### Opsi C — Azure Container Apps

```bash
bash deploy-to-azure.sh    # Linux/WSL
deploy-to-azure.bat        # Windows
```

> 📄 Baca `AZURE_DEPLOYMENT_GUIDE.md` untuk panduan lengkap.

---

### Opsi D — Cloud Terpisah (Vercel + Railway)

**Backend (Railway):** Connect repo → Railway auto-detects `railway.toml` → Set env vars → Copy URL

**Frontend (Vercel):** Import repo → Set `NEXT_PUBLIC_API_BASE` → Deploy

---

## 🧪 Testing

```bash
# Windows (gunakan py)
py tests_smoke.py        # Smoke test: 13 modul import
py tests_behavior.py     # Behavioral test: 11 skenario
py tests_e2e.py          # E2E API test: 17 endpoint FastAPI
py tests_final.py        # Aggregator: jalankan semua test + next build

# Linux / macOS (gunakan python)
python tests_final.py

# Build frontend saja
cd frontend && npm run build
```

---

## 📡 Endpoint API (FastAPI v1)

Dokumentasi interaktif tersedia di **http://localhost:8000/docs** (Swagger UI).

| Method | Path | Deskripsi |
|---|---|---|
| `GET` | `/health` | Health check — cek service aktif |
| `GET` | `/api/v1/datasets` | List semua dataset |
| `POST` | `/api/v1/datasets` | Upload dataset (CSV/XLSX/Parquet/JSON/TSV) |
| `GET` | `/api/v1/datasets/{id}` | Metadata dataset tertentu |
| `DELETE` | `/api/v1/datasets/{id}` | Hapus dataset |
| `GET` | `/api/v1/eda/{id}/summary` | Ringkasan statistik EDA |
| `GET` | `/api/v1/eda/{id}/data` | Raw data dengan server-side pagination |
| `GET` | `/api/v1/eda/{id}/correlation` | Matriks korelasi Pearson |
| `POST` | `/api/v1/preprocessing/run` | Jalankan preprocessing |
| `POST` | `/api/v1/training/start` | Latih model ML |
| `GET` | `/api/v1/training/models` | List semua model |
| `GET` | `/api/v1/training/models/{id}` | Metadata model tertentu |
| `DELETE` | `/api/v1/training/models/{id}` | Hapus model |
| `POST` | `/api/v1/interpretation/shap` | Hitung SHAP (feature importance global) |
| `POST` | `/api/v1/interpretation/lime` | Hitung LIME (penjelasan per-instance) |
| `GET` | `/api/v1/timeseries/{id}/detect` | Analisis stasioneritas (ADF test) |
| `GET` | `/api/v1/timeseries/{id}/forecast` | Forecasting time series |
| `GET` | `/api/v1/timeseries/{id}/anomalies` | Deteksi anomali |
| `WS` | `/api/v1/ws/{dataset_id}` | WebSocket — progress preprocessing real-time |

---

## 📦 File Deployment

| File | Tujuan |
|---|---|
| `Dockerfile.backend` | Docker image untuk backend FastAPI |
| `frontend/Dockerfile` | Docker image untuk frontend Next.js (standalone mode) |
| `docker-compose.yml` | Orkestrasi lokal via Docker Desktop |
| `docker-compose.azure.yml` | Orkestrasi untuk Azure Container Apps |
| `railway.toml` | Konfigurasi deployment Railway (backend) |
| `vercel.json` | Konfigurasi deployment Vercel (frontend) |
| `.env.example` | Template environment variables |
| `requirements-backend.txt` | Dependensi Python untuk backend FastAPI |
| `deploy-to-azure.sh` | Script deploy ke Azure (Linux/WSL) |
| `deploy-to-azure.bat` | Script deploy ke Azure (Windows) |
| `AZURE_DEPLOYMENT_GUIDE.md` | Panduan lengkap setup Azure |

---

## ⚠️ Catatan & Batasan

| Topik | Keterangan |
|---|---|
| **State preprocessing** | State preprocessing sekarang persist ke disk (`data/states/*.json`) dan survive restart backend. |
| **Autentikasi** | Halaman `/login` masih placeholder. Backend MVP berjalan tanpa autentikasi aktif. |
| **Next.js vulnerabilities** | npm audit menunjukkan 16 vulnerabilities (Next.js 14.x). Jalankan `npm audit fix --force` untuk upgrade ke Next.js 16.x (breaking change). |
| **Persistensi data** | Dataset (`.parquet` + `.meta.json`) dan model (`.pkl`) tersimpan di `data/` dan bertahan setelah restart backend. |
| **Python di Windows** | Gunakan `py` (Python Launcher) jika perintah `python` tidak dikenali di terminal. |

---

## 📝 Pembaruan Terbaru

Berikut perbaikan dan peningkatan yang telah diterapkan (Juli 2026):

### Performance & Reliability
| # | Perubahan | Dampak |
|---|---|---|
| 1 | **Parallel Cross-Validation** — `n_jobs=-1` untuk semua CPU cores | Training 10-100× lebih cepat (tergantung CPU) |
| 2 | **State Persistence** — file-based JSON storage (`data/states/*.json`) | State survive restart backend |
| 3 | **Async Background Tasks** — training model jalan di background | Tidak blocking request handler |
| 4 | **Server-Side Pagination** — endpoint `GET /api/v1/eda/{id}/data` | Dataset besar tanpa membebani RAM |
| 5 | **Migrasi ke Polars** — data processing dengan Apache Arrow | Performa 10-100× lebih cepat |

### Security & Validation
| # | Perubahan | Dampak |
|---|---|---|
| 6 | **Production Safety Validation** — auto-check JWT secret, CORS, debug mode | Mencegah security risk di production |
| 7 | **Rate Limiting** — slowapi untuk API endpoints (10 uploads/5 training per minute) | Mencegah API abuse/DoS |
| 8 | **File Upload Size Validation** — cek size sebelum processing | Mencegah memory exhaustion |
| 9 | **Structured Error Logging** — context-aware logging (filename, size, error type) | Debugging lebih mudah |

### Previous Improvements (Juni 2026)
| # | Perubahan | Dampak |
|---|---|---|
| 10 | **Persistensi dataset/model** — metadata JSON + auto-scan startup | Data survive restart |
| 11 | **WebSocket Real-time Progress** — endpoint `/api/v1/ws/{dataset_id}` | Progress real-time tanpa polling |
| 12 | **Docker & Routing Fix** — `NEXT_PUBLIC_API_BASE` build arg | Docker Desktop siap pakai |

---

## 📄 Lisensi & Hak Cipta

© **PT. Asmer Sahabat Sukses** — Hak Cipta Dilindungi.

Software ini adalah produk **proprietary**. Dilarang memperbanyak, mendistribusikan, memodifikasi, atau menggunakan kode ini untuk kepentingan komersial tanpa izin tertulis dari PT. Asmer Sahabat Sukses.

Untuk informasi lisensi, hubungi: **PT. Asmer Sahabat Sukses**.
#   a s m e r a n d a - m o d u l a r  
 