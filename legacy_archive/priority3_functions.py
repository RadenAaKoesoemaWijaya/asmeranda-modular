"""
Priority 3 Functions for SHAP/LIME Performance Optimization
This module contains advanced functions for better performance and interactivity.
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime, timedelta
import hashlib
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Catatan: decorator ``@st.cache_data`` sudah dihapus dari modul ini
# karena modul ini sekarang bisa dipanggil dari backend FastAPI.
# Untuk performa, cache manual dilakukan via class ``InterpretationCache``
# di bawah ini.


class InterpretationCache:
    """
    Cache system for SHAP/LIME interpretation results
    """
    
    def __init__(self, cache_dir="interpretation_cache", max_age_hours=24):
        """
        Initialize cache system
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cache files
        max_age_hours : int
            Maximum age of cache files in hours
        """
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if not exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, model, X_data, method, **kwargs):
        """
        Generate unique cache key based on model, data, and parameters
        
        Returns:
        --------
        str
            Unique cache key
        """
        # Create hash from model type, data shape, and parameters
        model_info = f"{type(model).__name__}_{str(model)}"
        data_info = f"{X_data.shape}_{X_data.columns.tolist() if hasattr(X_data, 'columns') else 'no_columns'}"
        params_info = f"{method}_{sorted(kwargs.items())}"
        
        combined_info = f"{model_info}_{data_info}_{params_info}"
        return hashlib.md5(combined_info.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get file path for cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path):
        """Check if cache file exists and is not too old"""
        if not os.path.exists(cache_path):
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        return file_age < timedelta(hours=self.max_age_hours)
    
    def get(self, model, X_data, method, **kwargs):
        """
        Get cached interpretation result
        
        Parameters:
        -----------
        model : object
            Model used for interpretation
        X_data : pandas.DataFrame
            Data used for interpretation
        method : str
            Interpretation method ('shap' or 'lime')
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        dict or None
            Cached result or None if not found/invalid
        """
        cache_key = self._get_cache_key(model, X_data, method, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                cached_data['from_cache'] = True
                return cached_data
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        return None
    
    def set(self, model, X_data, method, result, **kwargs):
        """
        Cache interpretation result
        
        Parameters:
        -----------
        model : object
            Model used for interpretation
        X_data : pandas.DataFrame
            Data used for interpretation
        method : str
            Interpretation method ('shap' or 'lime')
        result : dict
            Result to cache
        **kwargs : dict
            Additional parameters
        """
        cache_key = self._get_cache_key(model, X_data, method, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Remove cache metadata before saving
            result_to_save = result.copy()
            result_to_save.pop('from_cache', None)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(result_to_save, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def clear_cache(self):
        """Clear all cache files"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            print(f"Error clearing cache: {e}")


def optimized_shap_for_large_dataset(model, X_data, max_samples=1000, background_samples=100, 
                                   cache=None, random_state=42):
    """
    Optimized SHAP implementation for large datasets
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_data : pandas.DataFrame
        Data for SHAP analysis
    max_samples : int, optional
        Maximum number of samples to analyze
    background_samples : int, optional
        Number of background samples for SHAP
    cache : InterpretationCache, optional
        Cache system for storing results
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing SHAP results and metadata
    """
    import shap
    
    result = {
        'success': False,
        'shap_values': None,
        'explainer': None,
        'feature_importance': None,
        'sample_info': {},
        'optimization_info': {},
        'from_cache': False
    }
    
    try:
        # Check cache first
        if cache:
            cached_result = cache.get(model, X_data, 'shap', 
                                    max_samples=max_samples, 
                                    background_samples=background_samples)
            if cached_result:
                return cached_result
        
        # Sample data for large datasets
        if len(X_data) > max_samples:
            X_sample = X_data.sample(n=max_samples, random_state=random_state)
            result['sample_info']['original_size'] = len(X_data)
            result['sample_info']['sampled_size'] = max_samples
            result['sample_info']['sampling_method'] = 'random'
        else:
            X_sample = X_data.copy()
            result['sample_info']['original_size'] = len(X_data)
            result['sample_info']['sampled_size'] = len(X_data)
            result['sample_info']['sampling_method'] = 'none'
        
        # Prepare background data
        if len(X_sample) > background_samples:
            X_background = X_sample.sample(n=background_samples, random_state=random_state)
        else:
            X_background = X_sample
        
        result['optimization_info']['background_size'] = len(X_background)
        
        # Start timing
        start_time = time.time()
        
        # Choose explainer based on model type
        model_type = type(model).__name__.lower()
        
        if 'randomforest' in model_type or 'gradientboosting' in model_type or 'xgb' in model_type:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model, X_background)
            result['optimization_info']['explainer_type'] = 'TreeExplainer'
        else:
            # Use KernelExplainer for other models with subset
            explainer = shap.KernelExplainer(model.predict, X_background)
            result['optimization_info']['explainer_type'] = 'KernelExplainer'
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance
        if isinstance(shap_values, list):
            # Multi-class case
            mean_abs_shap = np.mean([np.abs(values) for values in shap_values], axis=0)
        else:
            # Binary/regression case
            mean_abs_shap = np.abs(shap_values)
        
        feature_importance = list(zip(X_sample.columns, np.mean(mean_abs_shap, axis=0)))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate timing
        end_time = time.time()
        result['optimization_info']['computation_time'] = end_time - start_time
        result['optimization_info']['samples_per_second'] = len(X_sample) / (end_time - start_time)
        
        # Fill result
        result.update({
            'success': True,
            'shap_values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'X_sample': X_sample,
            'X_background': X_background
        })
        
        # Cache result
        if cache:
            cache.set(model, X_data, 'shap', result, 
                     max_samples=max_samples, 
                     background_samples=background_samples)
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
    
    return result


def optimized_lime_for_large_dataset(model, X_data, max_samples=100, num_features=10,
                                     cache=None, random_state=42):
    """
    Optimized LIME implementation for large datasets
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_data : pandas.DataFrame
        Data for LIME analysis
    max_samples : int, optional
        Maximum number of samples to analyze
    num_features : int, optional
        Number of features to explain
    cache : InterpretationCache, optional
        Cache system for storing results
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing LIME results and metadata
    """
    import lime
    import lime.lime_tabular
    
    result = {
        'success': False,
        'explanations': [],
        'explainer': None,
        'sample_info': {},
        'optimization_info': {},
        'from_cache': False
    }
    
    try:
        # Check cache first
        if cache:
            cached_result = cache.get(model, X_data, 'lime', 
                                    max_samples=max_samples, 
                                    num_features=num_features)
            if cached_result:
                return cached_result
        
        # Sample data for large datasets
        if len(X_data) > max_samples:
            X_sample = X_data.sample(n=max_samples, random_state=random_state)
            result['sample_info']['original_size'] = len(X_data)
            result['sample_info']['sampled_size'] = max_samples
            result['sample_info']['sampling_method'] = 'random'
        else:
            X_sample = X_data.copy()
            result['sample_info']['original_size'] = len(X_data)
            result['sample_info']['sampled_size'] = len(X_data)
            result['sample_info']['sampling_method'] = 'none'
        
        # Start timing
        start_time = time.time()
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_sample.values,
            feature_names=X_sample.columns.tolist(),
            mode='regression' if not hasattr(model, 'classes_') else 'classification',
            random_state=random_state
        )
        
        # Generate explanations for multiple samples
        n_explanations = min(10, len(X_sample))  # Limit to 10 explanations
        explanations = []
        
        for i in range(n_explanations):
            try:
                # Determine prediction function
                if hasattr(model, 'predict_proba'):
                    predict_fn = model.predict_proba
                else:
                    predict_fn = model.predict
                
                explanation = explainer.explain_instance(
                    X_sample.iloc[i].values,
                    predict_fn,
                    num_features=num_features
                )
                
                explanations.append({
                    'sample_index': i,
                    'explanation': explanation,
                    'predicted': model.predict(X_sample.iloc[i:i+1])[0],
                    'feature_importance': explanation.as_list()
                })
                
            except Exception as e:
                explanations.append({
                    'sample_index': i,
                    'error': str(e),
                    'explanation': None
                })
        
        # Calculate timing
        end_time = time.time()
        result['optimization_info']['computation_time'] = end_time - start_time
        result['optimization_info']['explanations_per_second'] = n_explanations / (end_time - start_time)
        
        # Fill result
        result.update({
            'success': True,
            'explanations': explanations,
            'explainer': explainer,
            'n_successful': len([exp for exp in explanations if 'error' not in exp]),
            'n_failed': len([exp for exp in explanations if 'error' in exp])
        })
        
        # Cache result
        if cache:
            cache.set(model, X_data, 'lime', result, 
                     max_samples=max_samples, 
                     num_features=num_features)
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
    
    return result


def batch_interpretation(model, X_data_list, method='shap', **kwargs):
    """
    Batch processing for multiple datasets
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_data_list : list of pandas.DataFrame
        List of datasets to interpret
    method : str, optional
        Interpretation method ('shap' or 'lime')
    **kwargs : dict
        Additional parameters for interpretation
        
    Returns:
    --------
    list
        List of interpretation results
    """
    results = []
    
    for i, X_data in enumerate(X_data_list):
        try:
            if method == 'shap':
                result = optimized_shap_for_large_dataset(model, X_data, **kwargs)
            elif method == 'lime':
                result = optimized_lime_for_large_dataset(model, X_data, **kwargs)
            else:
                result = {'success': False, 'error': f'Unknown method: {method}'}
            
            result['batch_index'] = i
            results.append(result)
            
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'batch_index': i
            })
    
    return results


def create_interactive_shap_plot(shap_values, X_data, feature_names=None, max_display=20):
    """
    Create interactive SHAP plots for Streamlit
    
    Parameters:
    -----------
    shap_values : array or list
        SHAP values
    X_data : pandas.DataFrame
        Data used for SHAP
    feature_names : list, optional
        List of feature names
    max_display : int, optional
        Maximum number of features to display
        
    Returns:
    --------
    dict
        Dictionary containing plot configurations
    """
    import shap
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    if feature_names is None:
        feature_names = X_data.columns.tolist()
    
    plots = {}
    
    try:
        # Feature importance plot
        if isinstance(shap_values, list):
            # Multi-class case
            mean_abs_shap = np.mean([np.abs(values) for values in shap_values], axis=0)
        else:
            # Binary/regression case
            mean_abs_shap = np.abs(shap_values)
        
        feature_importance = np.mean(mean_abs_shap, axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(max_display)
        
        # Interactive bar plot
        fig_importance = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title='SHAP Feature Importance',
            color='importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=max(400, max_display * 25))
        plots['importance'] = fig_importance
        
        # Summary plot data for interactive visualization
        if isinstance(shap_values, list):
            shap_values_flat = shap_values[0]  # Use first class for demo
        else:
            shap_values_flat = shap_values
        
        # Create scatter plot for feature values vs SHAP values
        scatter_data = []
        for i, feature in enumerate(feature_names[:max_display]):
            feature_values = X_data.iloc[:, i].values
            feature_shap = shap_values_flat[:, i]
            
            scatter_data.append(pd.DataFrame({
                'feature_value': feature_values,
                'shap_value': feature_shap,
                'feature': feature
            }))
        
        if scatter_data:
            scatter_df = pd.concat(scatter_data, ignore_index=True)
            
            fig_scatter = px.scatter(
                scatter_df,
                x='feature_value',
                y='shap_value',
                color='feature',
                title='Feature Values vs SHAP Values',
                hover_data=['feature']
            )
            plots['scatter'] = fig_scatter
        
        plots['success'] = True
        
    except Exception as e:
        plots = {'success': False, 'error': str(e)}
    
    return plots


def create_interactive_lime_plot(lime_explanations, max_display=10):
    """
    Create interactive LIME plots for Streamlit
    
    Parameters:
    -----------
    lime_explanations : list
        List of LIME explanations
    max_display : int, optional
        Maximum number of features to display
        
    Returns:
    --------
    dict
        Dictionary containing plot configurations
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    plots = {}
    
    try:
        # Aggregate feature importance across explanations
        feature_scores = {}
        
        for exp in lime_explanations:
            if exp.get('explanation') and exp.get('feature_importance'):
                for feature, score in exp['feature_importance']:
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(abs(score))
        
        # Calculate average importance
        avg_importance = {}
        for feature, scores in feature_scores.items():
            avg_importance[feature] = np.mean(scores)
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'importance': list(avg_importance.values())
        }).sort_values('importance', ascending=False).head(max_display)
        
        # Interactive bar plot
        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='LIME Feature Importance (Average Across Samples)',
            color='importance',
            color_continuous_scale='plasma'
        )
        fig_importance.update_layout(height=max(400, max_display * 25))
        plots['importance'] = fig_importance
        
        # Create explanation details for individual samples
        successful_exps = [exp for exp in lime_explanations if exp.get('explanation')]
        
        if successful_exps:
            # Sample selection dropdown
            sample_options = [f"Sample {exp['sample_index']}" for exp in successful_exps]
            plots['sample_options'] = sample_options
            plots['explanations'] = successful_exps
        
        plots['success'] = True
        
    except Exception as e:
        plots = {'success': False, 'error': str(e)}
    
    return plots

def get_interpretation_performance_stats():
    """
    Get performance statistics for interpretation methods
    
    Returns:
    --------
    dict
        Performance statistics
    """
    return {
        'shap': {
            'tree_explainer': 'Fast for tree-based models',
            'kernel_explainer': 'Slower but model-agnostic',
            'recommended_max_samples': 1000,
            'memory_usage': 'Medium to High'
        },
        'lime': {
            'tabular_explainer': 'Medium speed',
            'recommended_max_samples': 100,
            'memory_usage': 'Low to Medium'
        },
        'optimization_tips': [
            'Use sampling for large datasets (>10,000 samples)',
            'Cache results for repeated analyses',
            'Use TreeExplainer for tree-based models',
            'Limit number of features for visualization',
            'Consider batch processing for multiple analyses'
        ]
    }


def monitor_interpretation_performance(func):
    """
    Decorator to monitor interpretation performance
    
    Parameters:
    -----------
    func : callable
        Function to monitor
        
    Returns:
    --------
    callable
        Wrapped function with performance monitoring
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # Add performance metrics to result
        if isinstance(result, dict):
            result['performance'] = {
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': datetime.now().isoformat()
            }
        
        return result
    
    return wrapper


def get_memory_usage():
    """
    Get current memory usage in MB
    
    Returns:
    --------
    float
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0


# Initialize global cache
interpretation_cache = InterpretationCache()
