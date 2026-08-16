import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, LeaveOneOut, LeavePOut, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, BaggingRegressor, VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from kmodes.kprototypes import KPrototypes
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import shap
import pickle
import os
from PIL import Image
import io
import time
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from auth_db import auth_db
from captcha_utils import captcha_gen, verify_captcha
import smtplib
import ssl
from email.message import EmailMessage

# Import modular components
from modules import (
    log_feature, safe_rerun
)
 
from utils import (prepare_timeseries_data, check_stationarity, plot_timeseries_analysis, 
                   analyze_trend_seasonality_cycle, plot_pattern_analysis,
                   implement_shap_classification, handle_multiclass_shap,
                   implement_lime_classification, detect_model_type,
                   prepare_forecasting_data_for_interpretation,
                   interpret_forecasting_model, create_forecasting_interpretation_dashboard,
                   check_model_compatibility, create_shap_visualization,
                   validate_clustering_parameters, get_model_interpretation_recommendations)
from priority2_functions import improved_data_preprocessing_for_interpretation, create_interpretation_report
from priority3_functions import (InterpretationCache, optimized_shap_for_large_dataset, 
                               optimized_lime_for_large_dataset, batch_interpretation,
                               create_interactive_shap_plot, create_interactive_lime_plot,
                               get_interpretation_performance_stats, interpretation_cache)
from param_presets import get_available_presets, get_preset_params, get_all_presets, save_custom_preset, load_custom_presets, export_preset_to_json, import_preset_from_json, create_preset_summary

# Import modular support files for enhanced workflow management
try:
    from session_manager import SessionManager
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False

try:
    from data_type_detector import DataTypeDetector
    DATA_TYPE_DETECTOR_AVAILABLE = True
except ImportError:
    DATA_TYPE_DETECTOR_AVAILABLE = False

try:
    from workflow_validator import WorkflowValidator
    WORKFLOW_VALIDATOR_AVAILABLE = True
except ImportError:
    WORKFLOW_VALIDATOR_AVAILABLE = False

try:
    from error_handler import ErrorHandler
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMB_AVAILABLE = True
except ImportError:
    IMB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# XGBoost imports
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM imports
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# CatBoost imports
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Initialize translation
TRANSLATIONS = {
    'en': {
        'app_title': 'Comprehensive Machine Learning App',
        'app_description': 'This application helps you analyze your data, preprocess it for machine learning, train models, and interpret the results using eXplainable AI.',
        # Add more translations here
    },
    'id': {
        'app_title': 'Aplikasi Machine Learning Komprehensif',
        'app_description': 'Aplikasi ini membantu Anda menganalisis data, memprosesnya untuk machine learning, melatih model, dan menginterpretasikan hasil menggunakan eXplainable AI.',
        # Add more translations here
    }
}

# Set page configuration
st.set_page_config(
    page_title="Asmeranda AI - Intelligent ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for luxurious premium UI - Dark/Gold/Emerald Theme
premium_css = """
<style>
/* Modern Luxurious Premium UI - Deep Glassmorphism & Gold/Emerald Accents */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

:root {
    --primary: #f8fafc;       /* Bright text for dark theme */
    --text-secondary: #cbd5e1;
    --bg-main: #0B0F19;       /* Very deep luxurious navy/slate */
    --bg-secondary: #111827;
    --accent-gold: #D4AF37;   /* Classic Gold */
    --accent-gold-glow: rgba(212, 175, 55, 0.4);
    --primary-green: #10b981; /* Emerald Green */
    --card-bg: rgba(17, 24, 39, 0.7);
    --card-border: rgba(212, 175, 55, 0.2);
}

/* Global Dynamic Background */
body, .stApp {
    background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 100%);
    color: var(--primary);
    font-family: 'Outfit', sans-serif;
}

/* Headers & Titles with Shimmer Effect */
h1, h2, h3 {
    font-weight: 800;
    letter-spacing: 0.02em;
    background: linear-gradient(90deg, #F3E5AB, var(--accent-gold), #F3E5AB);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmerText 6s linear infinite;
}

@keyframes shimmerText {
    to { background-position: 200% center; }
}

/* Sidebar with Glassmorphism */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(25px);
    border-right: 1px solid var(--card-border);
    box-shadow: 5px 0 30px rgba(0,0,0,0.5);
}

/* Glassmorphism Cards */
[data-testid="stVerticalBlock"] > div {
    background: var(--card-bg);
    backdrop-filter: blur(16px) saturate(180%);
    border-radius: 24px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--card-border);
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

[data-testid="stVerticalBlock"] > div:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 50px 0 rgba(212, 175, 55, 0.15);
    border-color: rgba(212, 175, 55, 0.4);
}

/* Premium Dynamic Buttons */
button, .stButton>button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: 1px solid rgba(16, 185, 129, 0.5) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
    font-size: 0.85rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    position: relative;
    overflow: hidden;
    z-index: 1;
}

/* Button Micro-animation (Pulse & Shine) */
button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.4), transparent);
    transition: all 0.6s ease;
    z-index: -1;
}

button:hover::before {
    left: 100%;
}

button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
}

button:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* Primary Button Override (for Gold Actions) */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #F3E5AB 0%, #D4AF37 100%) !important;
    color: #0f172a !important;
    border: none !important;
    box-shadow: 0 4px 15px var(--accent-gold-glow) !important;
}

.stButton>button[kind="primary"]:hover {
    box-shadow: 0 8px 25px rgba(212, 175, 55, 0.6) !important;
}

/* Modern Input Fields */
input, select, textarea, .stSelectbox > div > div > div, .stNumberInput > div > div > div {
    background: rgba(15, 23, 42, 0.6) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

input:focus, select:focus, textarea:focus, .stSelectbox > div > div > div:focus-within {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 2px var(--accent-gold-glow) !important;
    background: rgba(15, 23, 42, 0.9) !important;
}

/* Custom Tabs */
[data-baseweb="tab-list"] {
    background: rgba(0,0,0,0.3);
    border-radius: 15px;
    padding: 6px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.05);
}

[data-baseweb="tab"] {
    border-radius: 10px !important;
    border: none !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background: var(--accent-gold) !important;
    color: #0f172a !important;
    box-shadow: 0 4px 12px var(--accent-gold-glow) !important;
}

/* Smooth Progress Bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #D4AF37, #F3E5AB);
    border-radius: 20px;
    transition: width 0.5s ease;
}

/* Stat Metrics Pop */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border-left: 4px solid var(--accent-gold);
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    transition: transform 0.3s ease, border-color 0.3s ease;
}

[data-testid="stMetric"]:hover {
    transform: scale(1.03);
    border-left-color: var(--primary-green);
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--card-border);
}
</style>
"""
st.markdown(premium_css, unsafe_allow_html=True)

# Initialize modular support components
if SESSION_MANAGER_AVAILABLE:
    session_manager = SessionManager()
    
if DATA_TYPE_DETECTOR_AVAILABLE:
    data_type_detector = DataTypeDetector()
    
if WORKFLOW_VALIDATOR_AVAILABLE:
    workflow_validator = WorkflowValidator()
    
if ERROR_HANDLER_AVAILABLE:
    error_handler = ErrorHandler(language='id', auth_db=auth_db)  # Pass auth_db for error logging

# Language toggle button
if 'language' not in st.session_state:
    st.session_state.language = 'id'  # Default language is Indonesian

col1, col2 = st.columns([0.9, 0.1])

# Authentication functions are now imported from modules.auth
# render_loginizer, logout_user, send_otp_email imported from modules

# Ensure super admin exists before authentication
auth_db.ensure_super_admin_exists()

# Initialize session directly without loginizer for direct access
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = True
    st.session_state.current_username = 'local_user'

# ---------------------------
# Super Admin Dashboard Tab (Optional)
# ---------------------------
# render_admin_dashboard() # Disabled by default since login is removed

with col2:
    # Language toggle button
    if st.button('ID' if st.session_state.language == 'en' else 'EN'):
        st.session_state.language = 'id' if st.session_state.language == 'en' else 'en'


# App title and description with premium styling
_hero_html = (
    '<div style="text-align: center; padding: 1rem 0 0.5rem 0;">'
    '<h1 style="font-size: 2.2rem; margin-bottom: 0;">&#x1F916; Asmeranda AI</h1>'
    '<h2 style="font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; color: var(--text-secondary);">'
    "Intelligent Machine Learning Platform"
    "</h2>"
    '<div style="width: 60px; height: 3px; background: linear-gradient(90deg, var(--primary-green), var(--accent-gold)); margin: 1rem auto; border-radius: 2px;"></div>'
    "</div>"
)
st.markdown(_hero_html, unsafe_allow_html=True)

# Initialize session state variables centrally
from session_manager import SessionStateManager
SessionStateManager()

