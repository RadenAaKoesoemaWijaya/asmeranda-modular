import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_predict_model(model_file, data):
    """Memuat model dari file .pkl dan melakukan prediksi pada data baru"""
    try:
        # Load the model
        model = pickle.load(model_file)
        
        # Get model type
        model_type = type(model).__name__
        
        # Prepare data for prediction
        if hasattr(model, 'feature_names_in_'):
            # If model has feature names, use them
            required_features = model.feature_names_in_
            if set(required_features).issubset(set(data.columns)):
                X = data[required_features]
            else:
                missing_features = set(required_features) - set(data.columns)
                raise ValueError(f"Missing required features: {missing_features}")
        else:
            # Use all numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in data")
            X = data[numeric_cols]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            
            # Get prediction probabilities if available (for classification)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X)
                except:
                    pass
            
            return {
                'success': True,
                'predictions': predictions,
                'probabilities': probabilities,
                'model_type': model_type,
                'n_samples': len(predictions),
                'features_used': X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
            }
        else:
            raise ValueError("Model does not have predict method")
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def adjusted_r2_score(r2, n, k):
    """Hitung Adjusted R²."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def calculate_vif(X):
    """Hitung Variance Inflation Factor (VIF) untuk setiap fitur."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def breusch_pagan_test(y_true, y_pred, X):
    """Tampilkan Uji Heteroskedastisitas (Breusch-Pagan)"""
    residuals = y_true - y_pred
    X_const = sm.add_constant(X)
    bp_test = het_breuschpagan(residuals, X_const)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    return dict(zip(labels, bp_test))

def get_model_type(model, problem_type: str = None):
    """Menentukan jenis model (klasifikasi atau regresi) berdasarkan model.

    Parameters
    ----------
    model : estimator sklearn-like
    problem_type : str | None
        ``'Classification'`` / ``'Regression'`` / ``'Forecasting'``.
        Jika None, fungsi akan coba membaca dari ``core.state``
        (yang bridge ke ``st.session_state`` di legacy UI).
    """
    try:
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            return "Classification"
        if hasattr(model, "predict") and not hasattr(model, "classes_"):
            return "Regression"
    except Exception:
        pass

    # Fallback: ambil dari state (bisa Streamlit atau registry)
    if problem_type is None:
        try:
            from core.state import get_state

            problem_type = get_state().get("problem_type")
        except Exception:
            problem_type = None
    return problem_type
