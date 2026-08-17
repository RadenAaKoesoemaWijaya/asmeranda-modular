# Asmeranda AI

A modern, enterprise-ready modular machine learning platform for end-to-end data science workflows. Upload datasets, explore data, preprocess, train models, optimize hyperparameters, interpret results, and run advanced ML operations including time series forecasting, anomaly detection, clustering, and explainable AI (XAI)—backed by comprehensive role-based security hardening.

---

## 🚀 Key Features

### 🔐 Security & Access Control (RBAC)
- **Role-Based Access Control (RBAC)**: Support for `Admin`, `Data Scientist`, and `Viewer` roles.
- **JWT & API Key Authentication**: Secure token-based user sessions and API key verification.
- **Password Strength Validation**: Enforced password complexity rules (length, uppercase, lowercase, numbers, symbols).
- **Security Middlewares**:
  - `SecurityHeadersMiddleware`: HTTP security headers (`X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Content-Security-Policy`, `Strict-Transport-Security`).
  - `RequestSizeLimitMiddleware`: Payload size enforcement (HTTP 413 protection).
- **Rate Limiting**: Endpoint-specific rate limiting powered by SlowAPI.
- **Security Audit Trail**: Structured event logging (`security_audit.log`) for authentication, registration, permission checks, and payload violations.
- **Input Sanitization & Output Encoding**: XSS and injection prevention.

### 🧠 Core Machine Learning
- **Supervised Learning**: 9+ algorithms (RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting, SVM, DecisionTree, KNN, Logistic/Linear Regression).
- **Model Training & Cross-Validation**: K-Fold, Stratified K-Fold, Leave-One-Out, and Time Series splits.
- **Hyperparameter Optimization**: Grid Search, Random Search, and Bayesian Optimization with Optuna.
- **Intelligent Recommendations**: Automatic algorithm recommendations and parameter presets based on dataset characteristics.
- **Advanced Metrics & Evaluation**: ROC-AUC, PR Curves, Confusion Matrix, MCC, MAPE, Balanced Accuracy, Cohen's Kappa, and Learning Curves.
- **Model Comparison & Leaderboard**: Automatic comparison and ranking of trained models.

### 🔍 Unsupervised Learning & Dimensionality Reduction
- **Clustering**: KMeans, DBSCAN, Hierarchical, Spectral, and HDBSCAN.
- **Cluster Diagnostics**: Optimal-K analysis via Elbow method and Silhouette score evaluation.
- **Dimensionality Reduction**: UMAP and PCA for 2D/3D high-dimensional data visualization.

### 💡 Explainable AI (XAI)
- **SHAP (SHapley Additive exPlanations)**: Global feature importance with TreeExplainer, LinearExplainer, and KernelExplainer.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local prediction explanations for tabular data.
- **Interactive Visualizations**: Interactive feature attribution and prediction contribution plots.

### 📈 Time Series & Anomaly Detection
- **Time Series Forecasting**: ARIMA, SARIMA, Prophet, LSTM, and moving averages with automated frequency inference.
- **Anomaly Detection**: Isolation Forest, One-Class SVM, and rolling statistical bounds.
- **Stationarity & Preprocessing**: ADF tests, seasonal decomposition, and missing value interpolation.

### 🧹 Data Processing & Exploratory Data Analysis (EDA)
- **Automated Type Inference**: Automatic column type detection (numerical, categorical, datetime, text).
- **Preprocessing Pipeline**: Missing value imputation, outlier detection, categorical encoding, and feature scaling (StandardScaler, MinMaxScaler, RobustScaler).
- **EDA Suite**: Data summary statistics, distribution histograms, and correlation heatmaps.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Backend Framework** | FastAPI, Pydantic v2, Uvicorn, Starlette |
| **Security & Auth** | PyJWT, Passlib (Bcrypt / PBKDF2), SlowAPI, SQLite |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, statsmodels |
| **Explainable AI** | SHAP, LIME |
| **Data Processing** | Polars, Pandas, NumPy, PyArrow |
| **Frontend Framework** | Next.js 14 (App Router), React 18, Zustand |
| **Styling & UI** | Custom Responsive CSS Design System, Dynamic Components |
| **Container & Cloud** | Docker, Docker Compose, Nginx, Azure Container Apps, AWS, GCP |

---

## 📦 Installation & Setup

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** & `npm`
- **Docker Desktop** (optional, for containerized deployment)

---

### Local Development Setup

#### 1. Backend Setup

```bash
# Clone and enter directory
cd asmeranda-modular

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements-backend.txt

# Configure environment variables (optional, defaults available)
cp .env.example .env

# Run FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Or run using helper script:
# python backend/run_backend.py
```

#### 2. Frontend Setup

```bash
# Open new terminal and navigate to frontend
cd frontend

# Install npm packages
npm install

# Start Next.js development server
npm run dev
```

---

### Access Points & Default Credentials

- **Frontend Application**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (enabled in development/debug mode)
- **ReDoc API Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Default Admin Credentials (Auto-bootstrapped in local DB):
- **Username**: `admin`
- **Password**: `AdminPass123!`
- **Role**: `admin`

---

## 🐳 Docker Deployment

To build and run the complete multi-service stack with Docker Compose:

```bash
# Start all containers (Backend, Frontend, Nginx)
docker compose up --build -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop containers
docker compose down
```

---

## 📊 API Reference

| Prefix | Endpoint | Method | Description |
|---|---|---|---|
| **/api/v1/auth** | `/login` | `POST` | User login (JSON) & JWT access token generation |
| | `/token` | `POST` | OAuth2-compatible token login |
| | `/register` | `POST` | Register new user with password validation |
| | `/me` | `GET` | Get current authenticated user profile |
| | `/users` | `GET` | List all users (Admin role required) |
| | `/verify-key` | `POST` | Validate custom API key |
| **/api/v1/datasets** | `/upload` | `POST` | Upload dataset (CSV, Excel, Parquet, JSON) |
| | `/list` | `GET` | List all stored datasets |
| | `/{dataset_id}` | `GET` | Get dataset details and column information |
| | `/{dataset_id}/preview` | `GET` | Paginated dataset preview |
| **/api/v1/eda** | `/summary` | `POST` | Compute statistical summary & missingness |
| | `/correlations` | `POST` | Compute correlation matrix |
| | `/distributions` | `POST` | Generate column histograms and distributions |
| **/api/v1/preprocessing** | `/run` | `POST` | Run feature engineering & transformation |
| | `/cluster` | `POST` | Execute clustering (KMeans, DBSCAN, etc.) |
| | `/optimal-k` | `GET` | Calculate optimal cluster count |
| **/api/v1/training** | `/start` | `POST` | Start supervised model training |
| | `/models` | `GET` | List all trained models |
| | `/models/{model_id}` | `GET` | Get trained model metrics & metadata |
| | `/evaluate` | `POST` | Run comprehensive model evaluation |
| | `/learning-curve` | `POST` | Generate learning curves |
| | `/compare` | `POST` | Compare multiple trained models |
| | `/optimize` | `POST` | Execute Optuna hyperparameter optimization |
| **/api/v1/recommendations** | `/models` | `POST` | Get smart algorithm recommendations |
| | `/pipeline` | `POST` | Recommend complete preprocessing pipeline |
| **/api/v1/optimization** | `/hyperparameters` | `POST` | Run parameter search & tuning |
| **/api/v1/interpretation** | `/shap` | `POST` | Generate SHAP feature attributions |
| | `/lime` | `POST` | Generate LIME instance explanations |
| **/api/v1/timeseries** | `/forecast` | `POST` | Train & predict time series forecasts |
| | `/anomaly-detection` | `POST` | Detect time-series anomalies |
| **/api/v1/advanced-ml** | `/umap` | `POST` | Dimensionality reduction via UMAP |
| | `/hdbscan` | `POST` | HDBSCAN clustering |
| | `/anomaly-detection` | `POST` | Tabular anomaly detection (Isolation Forest, SVM) |
| | `/handle-missing-values`| `POST` | Missing data imputation utilities |
| | `/detect-outliers` | `POST` | Outlier detection |
| **/api/v1/ws** | `/ws` | `WS` | Real-time WebSocket event streaming |
| **/health** | `/` | `GET` | Service health status check |

---

## 🏗️ Architecture & Project Structure

```
asmeranda-modular/
├── backend/
│   ├── api/v1/                  # FastAPI REST and WebSocket routes
│   │   ├── auth.py              # Authentication, registration & user management
│   │   ├── datasets.py          # Dataset ingestion & preview
│   │   ├── eda.py               # Exploratory data analysis
│   │   ├── preprocessing.py     # Preprocessing and clustering
│   │   ├── training.py          # Supervised training & evaluation
│   │   ├── optimization.py      # Optuna hyperparameter tuning
│   │   ├── recommendations.py   # AI-assisted model & pipeline recommendations
│   │   ├── interpretation.py    # SHAP and LIME explainability
│   │   ├── timeseries.py        # Time series forecasting & anomaly detection
│   │   ├── advanced_ml.py       # UMAP, HDBSCAN, Outliers
│   │   ├── health.py            # Health probe endpoint
│   │   └── ws.py                # WebSocket handler
│   ├── core/                    # Core configuration and security
│   │   ├── auth.py              # User authentication, RBAC, JWT, SQLite store
│   │   ├── config.py            # Pydantic Settings & environment loader
│   │   ├── security_audit.py    # Structured audit logging
│   │   ├── security_utils.py    # Input sanitization, password validator
│   │   └── session_manager.py   # Session handling
│   ├── schemas/                 # Pydantic request/response schemas
│   ├── services/                # Business logic & ML computational engines
│   ├── tests/                   # Pytest test suite (unit, integration, security)
│   ├── Dockerfile               # Backend container image definition
│   ├── requirements-backend.txt # Python dependency specification
│   └── main.py                  # FastAPI application entrypoint & middleware stack
├── frontend/
│   ├── app/                     # Next.js 14 App Router pages
│   │   ├── login/               # User login & authentication page
│   │   ├── data-upload/         # Dataset upload & management
│   │   ├── eda/                 # Exploratory data analysis dashboard
│   │   ├── preprocessing/       # Feature engineering & scaling
│   │   ├── training/            # Model training & comparison
│   │   ├── optimization/        # Hyperparameter tuning
│   │   ├── recommendations/     # AI model recommendations
│   │   ├── clustering/          # Unsupervised clustering & optimal-k
│   │   ├── shap/                # SHAP global & local explanations
│   │   ├── lime/                # LIME local prediction explanations
│   │   ├── timeseries/          # Time series forecasting
│   │   ├── advanced-ml/         # Advanced ML operations (UMAP, etc.)
│   │   └── page.jsx             # Home / Dashboard overview
│   ├── components/              # Reusable React components (Sidebar, Navbar, etc.)
│   ├── lib/                     # API client, state store (Zustand), i18n
│   └── package.json             # Frontend dependency specification
├── nginx/                       # Nginx reverse proxy configuration
├── azure/                       # Azure deployment templates and scripts
├── docker-compose.yml           # Multi-container orchestration
├── pytest.ini                   # Pytest test runner configuration
└── README.md                    # Project documentation
```

---

## 🧪 Testing & Quality Assurance

Run the test suite using `pytest`:

```bash
# Run all tests
pytest

# Run security test suite
pytest backend/tests/security/ -v

# Run unit tests only
pytest -m unit

# Run integration tests
pytest backend/tests/integration/ -v

# Run with test coverage report
pytest --cov=backend --cov-report=term-missing

# Run end-to-end verification script
python final_verification.py
```

---

## ☁️ Cloud & Production Deployment

For comprehensive deployment instructions across different environments, refer to [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md):

- **Microsoft Azure**: Deploy to Azure Container Apps via `deploy-to-azure.sh` or `deploy-to-azure.bat`.
- **Amazon Web Services (AWS)**: Deploy using `deploy-cloud-aws.sh`.
- **Google Cloud Platform (GCP)**: Deploy using `deploy-cloud-gcp.sh`.
- **Docker Desktop**: Deploy locally with `deploy-docker-desktop.ps1`.

---

## 🤝 Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Run tests and lint checks (`pytest`).
5. Push to the branch (`git push origin feature/AmazingFeature`).
6. Open a Pull Request.

---

## 📝 License & Support

This project is proprietary software developed by **PT. Asmer Sahabat Sukses**.

- **Email**: support@asmeranda.ai
- **Documentation**: Inline docstrings & Swagger UI at `/docs`

---

**Asmeranda AI — End-to-End Modular Machine Learning Platform**  
© 2024–2026 PT. Asmer Sahabat Sukses. All rights reserved.