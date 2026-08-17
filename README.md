# Asmeranda AI

A comprehensive modular machine learning platform for end-to-end data science workflows. Upload datasets, explore data, preprocess, train models, interpret results, and run advanced ML operations including time series forecasting, anomaly detection, and explainable AI.

## 🚀 Key Features

### Core Machine Learning
- **Supervised Learning**: 9+ algorithms (RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting, SVM, DecisionTree, KNN, Logistic/Linear Regression)
- **Unsupervised Learning**: Clustering (KMeans, DBSCAN, Hierarchical, Spectral) with optimal-k analysis
- **Advanced ML**: UMAP dimensionality reduction, HDBSCAN clustering, ensemble methods (Voting, Stacking)
- **Hyperparameter Optimization**: Grid search, randomized search, Bayesian optimization with Optuna
- **Advanced Metrics**: MCC, MAPE, Balanced Accuracy, Cohen's Kappa, Learning Curves
- **Model Comparison**: Automatic model selection and performance ranking

### Explainable AI (XAI)
- **SHAP**: Global feature importance with multiple explainers
- **LIME**: Local instance explanations
- **Comprehensive Visualization**: Confusion matrix, ROC curves, precision-recall curves, feature importance plots, learning curves

### Time Series & Anomaly Detection
- **Forecasting**: ARIMA, SARIMA, Prophet, LSTM, and simple methods
- **Anomaly Detection**: Isolation Forest, One-Class SVM, rolling statistics
- **Advanced Features**: Time series preprocessing, frequency adjustment, stationarity testing

### Data Processing
- **Advanced Preprocessing**: Missing value handling, outlier detection, data validation
- **Data Type Detection**: Automatic column type inference
- **Feature Engineering**: Scaling, encoding, train/test split with multiple CV methods

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI with Pydantic for validation
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels, Optuna
- **XAI Libraries**: SHAP, LIME
- **Data Processing**: Pandas, Polars, NumPy, PyArrow
- **API Documentation**: OpenAPI/Swagger

### Frontend
- **Framework**: Next.js 14 with App Router
- **State Management**: Zustand with persistence
- **Styling**: Custom CSS with responsive design
- **API Client**: Fetch-based with error handling

### Deployment
- **Containerization**: Docker Compose
- **Reverse Proxy**: Nginx
- **Cloud Ready**: AWS and GCP configurations available

## 📦 Installation

### Prerequisites
- Python 3.11+ (backend)
- Node.js 18+ (frontend)
- Docker Desktop (optional, for containerized deployment)

### Local Development Setup

#### Backend
```bash
# Navigate to project root
cd C:\asmeranda-modular

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r backend/requirements-backend.txt

# Start backend server
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Access points:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Docker Deployment

```bash
# Build and start all services
docker compose up --build

# Stop services
docker compose down
```

## 🎯 Usage Guide

### 1. Upload Dataset
- Navigate to "Data Upload" in the sidebar
- Upload CSV, Excel, Parquet, or JSON files
- Review dataset metadata and preview

### 2. Exploratory Data Analysis (EDA)
- View summary statistics, data types, and missing values
- Analyze correlation matrix
- Preview raw data with pagination
- Examine data distributions

### 3. Preprocessing
- Select target column and problem type (Classification/Regression/Forecasting)
- Configure scaling method (StandardScaler, MinMaxScaler, RobustScaler)
- Set imputation strategy for missing values
- Configure train/test split ratio
- Run preprocessing to prepare data for modeling

### 4. Model Training
- Choose from 9+ machine learning algorithms
- Configure hyperparameters with templates
- Select cross-validation method (K-Fold, Stratified, LOO, Time Series)
- Train model with background processing
- View training metrics and cross-validation scores

### 5. Model Evaluation
- Run comprehensive evaluation with advanced metrics
- View visualization plots (confusion matrix, ROC curves, etc.)
- Generate learning curves to detect overfitting/underfitting
- Compare multiple models automatically

### 6. Explainable AI
- Use SHAP for global feature importance
- Use LIME for local instance explanations
- View interactive plots and explanations

### 7. Advanced Features
- **Clustering**: KMeans, DBSCAN, Hierarchical, Spectral with optimal-k analysis
- **Optimization**: Hyperparameter tuning with Grid Search, Random Search, Bayesian Optimization
- **Advanced ML**: UMAP, HDBSCAN, Ensemble methods (Voting, Stacking)
- **Time Series**: Forecasting and anomaly detection
- **Data Utilities**: Missing value handling, outlier detection, data validation

## 📊 API Endpoints

### Training & Evaluation
- `POST /api/v1/training/start` - Train a model
- `GET /api/v1/training/models` - List all trained models
- `GET /api/v1/training/models/{model_id}` - Get model metadata
- `POST /api/v1/training/evaluate` - Evaluate model performance
- `POST /api/v1/training/learning-curve` - Generate learning curve
- `POST /api/v1/training/compare` - Compare multiple models
- `POST /api/v1/training/optimize` - Hyperparameter optimization

### Interpretation
- `POST /api/v1/interpretation/shap` - Generate SHAP explanations
- `POST /api/v1/interpretation/lime` - Generate LIME explanations

### Advanced ML
- `POST /api/v1/advanced-ml/umap` - UMAP dimensionality reduction
- `POST /api/v1/advanced-ml/hdbscan` - HDBSCAN clustering
- `POST /api/v1/advanced-ml/anomaly-detection` - Anomaly detection
- `POST /api/v1/advanced-ml/forecast` - Time series forecasting
- `POST /api/v1/advanced-ml/handle-missing-values` - Missing value handling
- `POST /api/v1/advanced-ml/detect-outliers` - Outlier detection

### Preprocessing & Clustering
- `POST /api/v1/preprocessing/run` - Run preprocessing pipeline
- `POST /api/v1/preprocessing/cluster` - Perform clustering
- `GET /api/v1/preprocessing/optimal-k` - Find optimal cluster count

## 🏗️ Architecture

```
asmeranda-modular/
├── backend/                 # FastAPI backend
│   ├── api/v1/             # API routes
│   ├── core/               # Configuration and utilities
│   ├── schemas/            # Pydantic models
│   └── services/           # Business logic
├── frontend/               # Next.js frontend
│   ├── app/                # Pages (App Router)
│   ├── components/         # React components
│   └── lib/                # Utilities and API client
├── core/                   # Shared state and utilities
├── ml_engine/              # ML helper functions
├── data/                   # Dataset storage
└── docker-compose.yml      # Docker configuration
```

## 🔧 Configuration

### Backend Configuration (`backend/core/config.py`)
- Database paths
- API keys and secrets
- CORS settings
- Rate limiting
- Logging configuration

### Frontend Configuration (`frontend/.env.local`)
- API base URL
- Feature flags
- Environment settings

## 🧪 Testing

### Run Automated Tests
```bash
# Test basic functionality
python test_basic_functionality.py

# Test advanced ML features
python test_advanced_ml.py

# Test supervised ML enhancements
python test_enhanced_supervised_ml.py

# Final verification
python final_verification.py
```

### Manual Testing
1. Access the application at http://localhost:3000
2. Upload a sample dataset
3. Complete the full workflow: Upload → EDA → Preprocessing → Training → Evaluation → XAI
4. Test advanced features in the "Advanced ML" section

## 🚀 Deployment

### Local Deployment
```bash
# Using Docker Compose
docker compose up --build

# Or manual setup
# Start backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Start frontend
cd frontend
npm run dev
```

### Cloud Deployment
- **AWS**: Use `deploy-cloud-aws.sh` script
- **GCP**: Use `deploy-cloud-gcp.sh` script
- **Docker Desktop**: Use `deploy-docker-desktop.ps1` script

See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

## 📈 Performance Optimization

### Backend
- Parallel cross-validation with `n_jobs=-1`
- Sample limiting for memory-intensive operations
- Background task processing for long-running operations
- Efficient data processing with Polars

### Frontend
- Code splitting with Next.js App Router
- Optimized bundle size
- Lazy loading of components
- Client-side state persistence

## 🔒 Security

- Rate limiting on API endpoints
- Input validation with Pydantic
- CORS configuration
- Secure file upload handling
- Environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure functionality
5. Submit a pull request

## 📝 License

This project is proprietary software developed by PT. Asmer Sahabat Sukses.

## 🆘 Support

For support and documentation:
- Email: support@asmeranda.ai
- Documentation: See inline code documentation and API docs at `/docs`
- Issues: Report through the internal issue tracking system

## 🎯 Roadmap

### Completed ✅
- Core supervised ML functionality
- Explainable AI (SHAP, LIME)
- Time series forecasting and anomaly detection
- Advanced ML features (UMAP, HDBSCAN, Ensemble methods)
- Advanced metrics and learning curves
- Model comparison and automatic selection

### Future Enhancements 🔮
- Real-time model monitoring
- Automated feature engineering
- Neural network architectures
- Multi-modal data support
- Advanced XAI techniques
- Production deployment pipeline

---

**Asmeranda AI - End-to-End Machine Learning Platform**  
© 2024 PT. Asmer Sahabat Sukses