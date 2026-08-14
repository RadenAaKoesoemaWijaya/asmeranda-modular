import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
import warnings
warnings.filterwarnings('ignore')

# Notifikasi terpusat - lihat ``core.notifications``. Di legacy UI,
# Streamlit tetap terpasang dan ``core.notifications`` otomatis
# meneruskan ke ``st.info``/``success``/``warning``/``error``. Di
# backend FastAPI, notifikasi hanya dicatat ke logger.
from core.notifications import info as _notify_info
from core.notifications import success as _notify_success
from core.notifications import warning as _notify_warning
from core.notifications import error as _notify_error
from core.notifications import write as _notify_write

def is_timeseries(data, column_name):
    """
    Mendeteksi apakah kolom tertentu dalam dataframe merupakan data timeseries
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame yang berisi data
    column_name : str
        Nama kolom yang akan diperiksa
        
    Returns:
    --------
    bool
        True jika kolom tersebut kemungkinan merupakan data timeseries, False jika tidak
    """
    # Cek apakah kolom tersebut dapat dikonversi ke datetime
    try:
        pd.to_datetime(data[column_name])
        return True
    except:
        pass
    
    # Cek apakah nama kolom mengandung kata kunci terkait waktu
    time_keywords = ['date', 'time', 'year', 'month', 'day', 'tanggal', 'waktu', 'tahun', 'bulan', 'hari']
    if any(keyword in column_name.lower() for keyword in time_keywords):
        return True
    
    return False

def detect_timeseries_columns(data):
    """
    Mendeteksi kolom-kolom yang kemungkinan merupakan data timeseries dalam dataframe
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame yang berisi data
        
    Returns:
    --------
    list
        Daftar nama kolom yang kemungkinan merupakan data timeseries
    """
    timeseries_columns = []
    
    for col in data.columns:
        if is_timeseries(data, col):
            timeseries_columns.append(col)
    
    return timeseries_columns

@st.cache_data(show_spinner=False)
def prepare_timeseries_data(data, date_column, target_column, freq=None):
    """
    Menyiapkan data timeseries untuk analisis dan forecasting
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame yang berisi data
    date_column : str
        Nama kolom tanggal/waktu
    target_column : str
        Nama kolom target yang akan diprediksi
    freq : str, optional
        Frekuensi data (D untuk harian, W untuk mingguan, M untuk bulanan, dll.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame yang telah disiapkan untuk analisis timeseries
    """
    # Buat salinan data
    ts_data = data.copy()
    
    # Konversi kolom tanggal ke datetime
    ts_data[date_column] = pd.to_datetime(ts_data[date_column])
    
    # Urutkan data berdasarkan tanggal
    ts_data = ts_data.sort_values(by=date_column)
    
    # Set tanggal sebagai index
    ts_data = ts_data.set_index(date_column)
    
    # Resampling data jika frekuensi ditentukan
    if freq is not None:
        ts_data = ts_data.resample(freq)[target_column].mean().to_frame()
    
    return ts_data

def check_stationarity(timeseries):
    """
    Memeriksa stasioneritas data timeseries menggunakan uji Augmented Dickey-Fuller
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan diperiksa
        
    Returns:
    --------
    dict
        Hasil uji stasioneritas
    """
    # Hapus nilai missing
    clean_timeseries = timeseries.dropna()
    
    # Cek apakah data konstan (semua nilai sama)
    if len(clean_timeseries.unique()) <= 1:
        return {
            'Test Statistic': None,
            'p-value': None,
            'Lags Used': None,
            'Number of Observations Used': len(clean_timeseries),
            'Critical Values': None,
            'Stationary': True,  # Data konstan dianggap stasioner
            'Message': 'Data konstan (semua nilai sama) - tidak dapat dilakukan uji stasioneritas'
        }
    
    # Cek apakah data terlalu pendek
    if len(clean_timeseries) < 3:
        return {
            'Test Statistic': None,
            'p-value': None,
            'Lags Used': None,
            'Number of Observations Used': len(clean_timeseries),
            'Critical Values': None,
            'Stationary': True,
            'Message': 'Data terlalu pendek untuk uji stasioneritas'
        }
    
    try:
        # Uji Augmented Dickey-Fuller
        result = adfuller(clean_timeseries)
        
        # Siapkan output
        output = {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Number of Observations Used': result[3],
            'Critical Values': result[4],
            'Stationary': result[1] <= 0.05,
            'Message': None
        }
        
        return output
        
    except Exception as e:
        return {
            'Test Statistic': None,
            'p-value': None,
            'Lags Used': None,
            'Number of Observations Used': len(clean_timeseries),
            'Critical Values': None,
            'Stationary': True,
            'Message': f'Error dalam uji stasioneritas: {str(e)}'
        }

def plot_timeseries_analysis(timeseries, figsize=(15, 12)):
    """
    Membuat plot analisis timeseries (data asli, ACF, PACF)
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan dianalisis
    figsize : tuple, optional
        Ukuran gambar
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi plot analisis
    """
    # Hapus nilai missing dan cek jumlah data
    clean_timeseries = timeseries.dropna()
    
    # Cek apakah data terlalu sedikit untuk ACF/PACF
    if len(clean_timeseries) < 2:
        # Hanya plot data asli jika terlalu sedikit data
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.plot(clean_timeseries)
        ax.set_title('Data Timeseries (Data terlalu sedikit untuk ACF/PACF)')
        plt.tight_layout()
        return fig

def interpret_forecasting_model(model, X_train, y_train, X_test, feature_names=None, 
                               method='shap', n_samples=100, random_state=42):
    """
    Fungsi untuk interpretasi model forecasting menggunakan SHAP atau LIME
    
    Parameters:
    -----------
    model : sklearn model
        Model forecasting yang telah dilatih
    X_train : pandas.DataFrame atau numpy.array
        Data training
    y_train : pandas.Series atau numpy.array
        Target training
    X_test : pandas.DataFrame atau numpy.array
        Data testing untuk interpretasi
    feature_names : list, optional
        Nama fitur
    method : str, default='shap'
        Metode interpretasi ('shap' atau 'lime')
    n_samples : int, default=100
        Jumlah sampel untuk interpretasi
    random_state : int, default=42
        Random state untuk reproduktivitas
        
    Returns:
    --------
    dict
        Hasil interpretasi yang berisi:
        - shap_values atau lime_explanations
        - feature_importance
        - summary_plot
    """
    
    # Validasi input
    if method not in ['shap', 'lime']:
        raise ValueError("Method harus 'shap' atau 'lime'")
    
    # Konversi ke numpy array jika perlu
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Siapkan nama fitur
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Pilih sampel untuk interpretasi
    if len(X_test) > n_samples:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test.copy()
    
    results = {}
    
    if method == 'shap':
        try:
            import shap
            
            # Buat explainer berdasarkan tipe model
            if hasattr(model, 'tree_'):  # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:  # Other models
                # Untuk forecasting, kita gunakan KernelExplainer dengan background data
                background = shap.sample(X_train, min(100, len(X_train)))
                explainer = shap.KernelExplainer(model.predict, background)
            
            # Hitung SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Simpan hasil
            results['shap_values'] = shap_values
            results['explainer'] = explainer
            results['X_sample'] = X_sample
            results['feature_names'] = feature_names
            
            # Hitung feature importance (rata-rata absolute SHAP values)
            if len(shap_values.shape) == 3:  # Multi-output
                feature_importance = np.mean(np.abs(shap_values), axis=(0, 2))
            elif len(shap_values.shape) == 2:  # Single output
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            else:  # 1D array
                feature_importance = np.abs(shap_values)
            
            results['feature_importance'] = dict(zip(feature_names, feature_importance))
            
            # Buat summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            results['summary_plot'] = plt.gcf()
            plt.close()
            
        except ImportError:
            raise ImportError("SHAP tidak terinstal. Silakan install dengan: pip install shap")
        except Exception as e:
            raise Exception(f"Error dalam menghitung SHAP values: {str(e)}")
    
    elif method == 'lime':
        try:
            import lime
            import lime.lime_tabular
            
            # Buat LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                mode='regression',
                random_state=random_state
            )
            
            # Generate explanations untuk beberapa sampel
            lime_explanations = []
            n_lime_samples = min(10, len(X_sample))  # Batasi untuk efisiensi
            
            for i in range(n_lime_samples):
                explanation = explainer.explain_instance(
                    X_sample[i], 
                    model.predict,
                    num_features=len(feature_names)
                )
                lime_explanations.append(explanation)
            
            # Simpan hasil
            results['lime_explanations'] = lime_explanations
            results['explainer'] = explainer
            results['X_sample'] = X_sample[:n_lime_samples]
            results['feature_names'] = feature_names
            
            # Ekstrak feature importance dari LIME
            feature_importance = {}
            for exp in lime_explanations:
                for feature, weight in exp.as_list():
                    # Ekstrak nama fitur dari string LIME
                    feature_name = feature.split('=')[0].strip()
                    if feature_name in feature_importance:
                        feature_importance[feature_name] += abs(weight)
                    else:
                        feature_importance[feature_name] = abs(weight)
            
            # Rata-ratakan importance
            for feature in feature_importance:
                feature_importance[feature] /= len(lime_explanations)
            
            results['feature_importance'] = feature_importance
            
        except ImportError:
            raise ImportError("LIME tidak terinstal. Silakan install dengan: pip install lime")
        except Exception as e:
            raise Exception(f"Error dalam menghitung LIME explanations: {str(e)}")
    
    return results

def validate_clustering_parameters(data, n_clusters=None, eps=None, min_samples=None, method='kmeans'):
    """
    Validasi parameter clustering untuk memastikan nilai yang masuk akal
    
    Parameters:
    -----------
    data : array-like
        Data yang akan diclustering
    n_clusters : int, optional
        Jumlah cluster yang diinginkan
    eps : float, optional
        Parameter eps untuk DBSCAN
    min_samples : int, optional
        Parameter min_samples untuk DBSCAN
    method : str
        Metode clustering yang digunakan ('kmeans', 'dbscan', 'hierarchical', 'spectral')
        
    Returns:
    --------
    dict
        Dictionary berisi parameter yang telah divalidasi dan pesan peringatan jika ada
    """
    import numpy as np
    
    result = {'valid': True, 'warnings': [], 'errors': [], 'parameters': {}}
    
    if data is None or len(data) == 0:
        result['valid'] = False
        result['errors'].append("Data tidak boleh kosong")
        return result
    
    n_samples = len(data)
    
    # Validasi berdasarkan metode
    if method in ['kmeans', 'hierarchical', 'spectral'] and n_clusters is not None:
        # Validasi n_clusters
        if n_clusters < 2:
            result['warnings'].append("Jumlah cluster minimal adalah 2, menggunakan nilai default 2")
            result['parameters']['n_clusters'] = 2
        elif n_clusters > n_samples // 2:
            result['warnings'].append(f"Jumlah cluster terlalu besar ({n_clusters} > {n_samples // 2}), menggunakan nilai default 3")
            result['parameters']['n_clusters'] = 3
        else:
            result['parameters']['n_clusters'] = n_clusters
            
    elif method == 'dbscan':
        # Validasi eps
        if eps is not None:
            if eps <= 0:
                result['errors'].append("Parameter eps harus lebih besar dari 0")
                result['valid'] = False
            else:
                result['parameters']['eps'] = eps
        
        # Validasi min_samples
        if min_samples is not None:
            if min_samples < 1:
                result['warnings'].append("min_samples minimal adalah 1, menggunakan nilai default 5")
                result['parameters']['min_samples'] = 5
            elif min_samples > n_samples // 2:
                result['warnings'].append(f"min_samples terlalu besar ({min_samples} > {n_samples // 2}), menggunakan nilai default 5")
                result['parameters']['min_samples'] = 5
            else:
                result['parameters']['min_samples'] = min_samples
    
    # Validasi umum untuk semua metode
    if n_samples < 10:
        result['warnings'].append("Jumlah sampel sangat sedikit, hasil clustering mungkin tidak optimal")
    
    if n_samples < 50 and method == 'spectral':
        result['warnings'].append("Spectral clustering membutuhkan minimal 50 sampel untuk hasil yang optimal")
    
    return result

def get_clustering_recommendations(data, method='kmeans', language='id'):
    """
    Memberikan rekomendasi parameter clustering berdasarkan karakteristik data
    
    Parameters:
    -----------
    data : array-like
        Data yang akan diclustering
    method : str
        Metode clustering yang digunakan
    language : str
        Bahasa untuk pesan ('id' atau 'en')
        
    Returns:
    --------
    dict
        Dictionary berisi rekomendasi parameter
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    
    messages = {
        'id': {
            'data_size': "Ukuran data: {n_samples} sampel, {n_features} fitur",
            'recommended_k': "Rekomendasi jumlah cluster: {k}",
            'recommended_eps': "Rekomendasi eps: {eps:.2f}",
            'recommended_min_samples': "Rekomendasi min_samples: {min_samples}",
            'data_sparsity': "Data tampak {density} (rata-rata jarak: {avg_distance:.2f})",
            'dense': "padat",
            'sparse': "jarang",
            'moderate': "sedang"
        },
        'en': {
            'data_size': "Data size: {n_samples} samples, {n_features} features",
            'recommended_k': "Recommended number of clusters: {k}",
            'recommended_eps': "Recommended eps: {eps:.2f}",
            'recommended_min_samples': "Recommended min_samples: {min_samples}",
            'data_sparsity': "Data appears {density} (average distance: {avg_distance:.2f})",
            'dense': "dense",
            'sparse': "sparse",
            'moderate': "moderate"
        }
    }
    
    msg = messages.get(language, messages['id'])
    result = {}
    
    if data is None or len(data) == 0:
        return result
    
    n_samples, n_features = data.shape if hasattr(data, 'shape') else (len(data), len(data[0]) if data else 0)
    result['data_info'] = msg['data_size'].format(n_samples=n_samples, n_features=n_features)
    
    # Rekomendasi untuk K-Means, Hierarchical, Spectral
    if method in ['kmeans', 'hierarchical', 'spectral']:
        # Gunakan elbow method sebagai dasar rekomendasi
        recommended_k = min(int(np.sqrt(n_samples / 2)), 10, n_samples // 5)
        result['recommended_k'] = recommended_k
        result['k_recommendation'] = msg['recommended_k'].format(k=recommended_k)
    
    # Rekomendasi untuk DBSCAN
    if method == 'dbscan':
        # Gunakan k-nearest neighbors untuk estimasi eps
        try:
            if n_samples > 10:
                nbrs = NearestNeighbors(n_neighbors=min(5, n_samples-1))
                nbrs.fit(data)
                distances, _ = nbrs.kneighbors(data)
                
                # Sort distances and find elbow
                k_distances = np.sort(distances[:, -1])
                
                # Estimasi eps (ambil persentil ke-90)
                recommended_eps = np.percentile(k_distances, 90)
                result['recommended_eps'] = recommended_eps
                result['eps_recommendation'] = msg['recommended_eps'].format(eps=recommended_eps)
                
                # Rekomendasi min_samples
                recommended_min_samples = min(5, n_samples // 10)
                result['recommended_min_samples'] = recommended_min_samples
                result['min_samples_recommendation'] = msg['recommended_min_samples'].format(min_samples=recommended_min_samples)
                
                # Analisis kepadatan data
                avg_distance = np.mean(k_distances)
                if avg_distance < 0.5:
                    density = msg['dense']
                elif avg_distance > 2.0:
                    density = msg['sparse']
                else:
                    density = msg['moderate']
                
                result['data_density'] = msg['data_sparsity'].format(density=density, avg_distance=avg_distance)
        except Exception as e:
            result['error'] = f"Error dalam estimasi parameter: {str(e)}"
    
    return result

def create_forecasting_interpretation_dashboard(interpretation_results, method='shap'):
    """
    Membuat dashboard visualisasi untuk hasil interpretasi forecasting
    
    Parameters:
    -----------
    interpretation_results : dict
        Hasil dari interpret_forecasting_model()
    method : str, default='shap'
        Metode interpretasi yang digunakan
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi dashboard interpretasi
    """
    
    if method == 'shap':
        return create_shap_forecasting_dashboard(interpretation_results)
    elif method == 'lime':
        return create_lime_forecasting_dashboard(interpretation_results)

def advanced_missing_value_analysis(data, target_column=None, language='id'):
    """
    Analisis mendalam untuk missing values dengan rekomendasi strategi penanganan
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset yang akan dianalisis
    target_column : str, optional
        Nama kolom target untuk analisis khusus
    language : str
        Bahasa untuk output ('id' atau 'en')
        
    Returns:
    --------
    dict
        Dictionary berisi analisis dan rekomendasi
    """
    messages = {
        'id': {
            'analysis_title': 'Analisis Missing Values',
            'missing_by_column': 'Missing Values per Kolom',
            'missing_pattern': 'Pola Missing Values',
            'missing_correlation': 'Korelasi Missing Values',
            'recommendations': 'Rekomendasi Strategi',
            'complete_case': 'Complete Case Analysis',
            'simple_imputation': 'Simple Imputation',
            'advanced_imputation': 'Advanced Imputation',
            'model_based': 'Model-based Imputation',
            'delete_rows': 'Hapus Baris',
            'delete_columns': 'Hapus Kolom',
            'missing_indicator': 'Fitur Indikator Missing',
            'low_missing': 'Rendah (<5%)',
            'moderate_missing': 'Sedang (5-20%)',
            'high_missing': 'Tinggi (20-40%)',
            'very_high_missing': 'Sangat Tinggi (>40%)',
            'random_pattern': 'Pola Acak (MCAR)',
            'systematic_pattern': 'Pola Sistematis (MAR)',
            'nonignorable_pattern': 'Pola Tidak Dapat Diabaikan (MNAR)'
        },
        'en': {
            'analysis_title': 'Missing Values Analysis',
            'missing_by_column': 'Missing Values per Column',
            'missing_pattern': 'Missing Values Pattern',
            'missing_correlation': 'Missing Values Correlation',
            'recommendations': 'Strategy Recommendations',
            'complete_case': 'Complete Case Analysis',
            'simple_imputation': 'Simple Imputation',
            'advanced_imputation': 'Advanced Imputation',
            'model_based': 'Model-based Imputation',
            'delete_rows': 'Delete Rows',
            'delete_columns': 'Delete Columns',
            'missing_indicator': 'Missing Indicator Features',
            'low_missing': 'Low (<5%)',
            'moderate_missing': 'Moderate (5-20%)',
            'high_missing': 'High (20-40%)',
            'very_high_missing': 'Very High (>40%)',
            'random_pattern': 'Random Pattern (MCAR)',
            'systematic_pattern': 'Systematic Pattern (MAR)',
            'nonignorable_pattern': 'Non-ignorable Pattern (MNAR)'
        }
    }
    
    msg = messages.get(language, messages['id'])
    result = {}
    
    if data is None or data.empty:
        result['error'] = 'Dataset kosong atau tidak valid' if language == 'id' else 'Dataset is empty or invalid'
        return result
    
    # Basic missing value statistics
    missing_stats = data.isnull().sum()
    missing_percentages = (missing_stats / len(data)) * 100
    
    result['missing_summary'] = {
        'total_missing': missing_stats.sum(),
        'columns_with_missing': (missing_stats > 0).sum(),
        'total_percentage': (missing_stats.sum() / (len(data) * len(data.columns))) * 100,
        'by_column': missing_percentages[missing_percentages > 0].to_dict()
    }
    
    # Categorize missing value levels
    missing_categories = {}
    for col, pct in missing_percentages.items():
        if pct > 0:
            if pct < 5:
                missing_categories[col] = msg['low_missing']
            elif pct < 20:
                missing_categories[col] = msg['moderate_missing']
            elif pct < 40:
                missing_categories[col] = msg['high_missing']
            else:
                missing_categories[col] = msg['very_high_missing']
    
    result['missing_categories'] = missing_categories
    
    # Analyze missing patterns
    missing_matrix = data.isnull()
    
    # Check for patterns in missing values
    if len(data) > 10:
        # MCAR test (Little's test simplified)
        missing_corr = missing_matrix.corr()
        high_corr_pairs = []
        
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation threshold
                    high_corr_pairs.append({
                        'columns': (missing_corr.columns[i], missing_corr.columns[j]),
                        'correlation': corr_val
                    })
        
        result['missing_correlations'] = high_corr_pairs
        
        # Pattern classification
        if len(high_corr_pairs) == 0:
            result['missing_pattern'] = msg['random_pattern']
        elif any(abs(pair['correlation']) > 0.7 for pair in high_corr_pairs):
            result['missing_pattern'] = msg['nonignorable_pattern']
        else:
            result['missing_pattern'] = msg['systematic_pattern']
    
    # Generate recommendations
    recommendations = []
    
    for col, pct in missing_percentages.items():
        if pct > 0:
            col_recommendations = []
            
            if pct < 5:
                col_recommendations.append(msg['simple_imputation'])
                if target_column != col:
                    col_recommendations.append(msg['missing_indicator'])
            elif pct < 20:
                col_recommendations.append(msg['advanced_imputation'])
                col_recommendations.append(msg['missing_indicator'])
            elif pct < 40:
                col_recommendations.append(msg['model_based'])
                col_recommendations.append(msg['advanced_imputation'])
            else:
                col_recommendations.append(msg['delete_columns'])
                col_recommendations.append(msg['model_based'])
            
            recommendations.append({
                'column': col,
                'missing_percentage': pct,
                'category': missing_categories[col],
                'strategies': col_recommendations
            })
    
    result['recommendations'] = recommendations
    
    return result

def advanced_missing_value_imputation(data, strategy='auto', target_column=None, language='id'):
    """
    Imputasi missing values dengan strategi yang canggih dan otomatis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset yang akan diimputasi
    strategy : str
        Strategi imputasi ('auto', 'simple', 'knn', 'iterative', 'mice', 'model')
    target_column : str, optional
        Nama kolom target untuk imputasi yang mempertimbangkan target
    language : str
        Bahasa untuk output
        
    Returns:
    --------
    dict
        Dictionary berisi data yang telah diimputasi dan informasi proses
    """
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import KNNImputer, IterativeImputer
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        # Fallback to simple imputation if advanced methods not available
        strategy = 'simple'
    
    result = {'success': False, 'data': None, 'strategy_used': strategy, 'imputation_info': {}}
    
    if data is None or data.empty:
        result['error'] = 'Dataset kosong' if language == 'id' else 'Dataset is empty'
        return result
    
    # Auto strategy selection
    if strategy == 'auto':
        missing_analysis = advanced_missing_value_analysis(data, target_column, language)
        
        # Determine best strategy based on missing patterns
        total_missing_pct = missing_analysis['missing_summary']['total_percentage']
        
        if total_missing_pct < 5:
            strategy = 'simple'
        elif total_missing_pct < 20:
            strategy = 'knn'
        elif total_missing_pct < 40:
            strategy = 'iterative'
        else:
            strategy = 'model'
    
    data_imputed = data.copy()
    imputation_info = {}
    
    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Imputation strategies
    if strategy == 'simple':
        # Simple imputation with mean/median/mode
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                # Use median for skewed data, mean for normal data
                if abs(data[col].skew()) > 1:
                    imputed_value = data[col].median()
                else:
                    imputed_value = data[col].mean()
                
                data_imputed[col] = data_imputed[col].fillna(imputed_value)
                imputation_info[col] = {'method': 'mean/median', 'value': imputed_value}
        
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                imputed_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data_imputed[col] = data_imputed[col].fillna(imputed_value)
                imputation_info[col] = {'method': 'mode', 'value': imputed_value}
    
    elif strategy == 'knn':
        # KNN Imputation for numerical columns
        if numerical_cols:
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            data_imputed[numerical_cols] = knn_imputer.fit_transform(data[numerical_cols])
            
            for col in numerical_cols:
                if data[col].isnull().sum() > 0:
                    imputation_info[col] = {'method': 'knn', 'n_neighbors': 5}
        
        # Simple imputation for categorical columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                imputed_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data_imputed[col] = data_imputed[col].fillna(imputed_value)
                imputation_info[col] = {'method': 'mode', 'value': imputed_value}
    
    elif strategy == 'iterative' or strategy == 'mice':
        # Iterative imputation (MICE-like)
        if numerical_cols:
            iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
            data_imputed[numerical_cols] = iterative_imputer.fit_transform(data[numerical_cols])
            
            for col in numerical_cols:
                if data[col].isnull().sum() > 0:
                    imputation_info[col] = {'method': 'iterative', 'max_iter': 10}
        
        # Simple imputation for categorical columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                imputed_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data_imputed[col] = data_imputed[col].fillna(imputed_value)
                imputation_info[col] = {'method': 'mode', 'value': imputed_value}
    
    elif strategy == 'model':
        # Model-based imputation using Random Forest
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                # Prepare data for model training
                train_data = data.dropna(subset=[col])
                missing_mask = data[col].isnull()
                
                if len(train_data) > 10:  # Minimum samples for training
                    # Prepare features (exclude target column if it's the current column)
                    feature_cols = [c for c in numerical_cols + categorical_cols if c != col and data[c].isnull().sum() < len(data) * 0.5]
                    
                    if len(feature_cols) > 0:
                        # Handle categorical features
                        train_features = train_data[feature_cols].copy()
                        test_features = data.loc[missing_mask, feature_cols].copy()
                        
                        # Simple imputation for features
                        for feat_col in feature_cols:
                            if feat_col in numerical_cols and train_features[feat_col].isnull().sum() > 0:
                                train_features[feat_col] = train_features[feat_col].fillna(train_data[feat_col].median())
                                test_features[feat_col] = test_features[feat_col].fillna(data[feat_col].median())
                            elif feat_col in categorical_cols and train_features[feat_col].isnull().sum() > 0:
                                train_features[feat_col] = train_features[feat_col].fillna(train_data[feat_col].mode()[0])
                                test_features[feat_col] = test_features[feat_col].fillna(data[feat_col].mode()[0])
                        
                        # Train Random Forest model
                        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                        rf_model.fit(train_features, train_data[col])
                        
                        # Predict missing values
                        predicted_values = rf_model.predict(test_features)
                        data_imputed.loc[missing_mask, col] = predicted_values
                        
                        imputation_info[col] = {'method': 'random_forest', 'n_estimators': 50}
                    else:
                        # Fallback to simple imputation
                        imputed_value = data[col].median()
                        data_imputed[col] = data_imputed[col].fillna(imputed_value)
                        imputation_info[col] = {'method': 'median', 'value': imputed_value}
        
        # Simple imputation for categorical columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                imputed_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data_imputed[col] = data_imputed[col].fillna(imputed_value)
                imputation_info[col] = {'method': 'mode', 'value': imputed_value}
    
    result['success'] = True
    result['data'] = data_imputed
    result['imputation_info'] = imputation_info
    result['strategy_used'] = strategy
    
    return result

def advanced_outlier_detection(data, method='auto', contamination=0.1, language='id'):
    """
    Deteksi outlier yang canggih dengan multiple methods dan analisis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset yang akan dianalisis
    method : str
        Metode deteksi ('auto', 'iqr', 'zscore', 'isolation_forest', 'lof', 'dbscan')
    contamination : float
        Proporsi outlier yang diharapkan (0.0 - 0.5)
    language : str
        Bahasa untuk output
        
    Returns:
    --------
    dict
        Dictionary berisi hasil deteksi outlier dan analisis
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        # Fallback to basic methods
        method = 'iqr'
    
    messages = {
        'id': {
            'outlier_analysis': 'Analisis Outlier',
            'method_used': 'Metode yang digunakan',
            'outlier_count': 'Jumlah outlier yang terdeteksi',
            'outlier_percentage': 'Persentase outlier',
            'affected_columns': 'Kolom yang terdampak',
            'recommendations': 'Rekomendasi',
            'remove_outliers': 'Hapus outlier',
            'cap_outliers': 'Batasi outlier',
            'keep_outliers': 'Pertahankan outlier',
            'investigate_outliers': 'Investigasi lebih lanjut',
            'extreme_outliers': 'Outlier ekstrem terdeteksi',
            'moderate_outliers': 'Outlier moderat terdeteksi',
            'few_outliers': 'Sedikit outlier terdeteksi'
        },
        'en': {
            'outlier_analysis': 'Outlier Analysis',
            'method_used': 'Method used',
            'outlier_count': 'Number of outliers detected',
            'outlier_percentage': 'Outlier percentage',
            'affected_columns': 'Affected columns',
            'recommendations': 'Recommendations',
            'remove_outliers': 'Remove outliers',
            'cap_outliers': 'Cap outliers',
            'keep_outliers': 'Keep outliers',
            'investigate_outliers': 'Further investigation',
            'extreme_outliers': 'Extreme outliers detected',
            'moderate_outliers': 'Moderate outliers detected',
            'few_outliers': 'Few outliers detected'
        }
    }
    
    msg = messages.get(language, messages['id'])
    result = {'success': False, 'outliers': {}, 'analysis': {}, 'recommendations': []}
    
    if data is None or data.empty:
        result['error'] = 'Dataset kosong' if language == 'id' else 'Dataset is empty'
        return result
    
    # Select numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numerical_cols:
        result['error'] = 'Tidak ada kolom numerik untuk analisis outlier' if language == 'id' else 'No numerical columns for outlier analysis'
        return result
    
    outliers_dict = {}
    method_used = method
    
    # Auto method selection based on data characteristics
    if method == 'auto':
        n_samples = len(data)
        n_features = len(numerical_cols)
        
        if n_samples < 50:
            method = 'iqr'
        elif n_samples < 1000 and n_features < 10:
            method = 'isolation_forest'
        elif n_features >= 10:
            method = 'lof'
        else:
            method = 'isolation_forest'
    
    # Apply outlier detection methods
    for col in numerical_cols:
        col_data = data[col].dropna()
        
        if len(col_data) < 10:  # Skip columns with too few values
            continue
        
        outliers_mask = None
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            
        elif method == 'zscore':
            # Z-Score method
            mean = col_data.mean()
            std = col_data.std()
            
            if std > 0:
                z_scores = np.abs((data[col] - mean) / std)
                outliers_mask = z_scores > 3.0  # 3 standard deviations
            
        elif method == 'isolation_forest':
            # Isolation Forest
            try:
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outliers_mask = iso_forest.fit_predict(data[[col]].fillna(data[col].median())) == -1
            except:
                # Fallback to IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        
        elif method == 'lof':
            # Local Outlier Factor
            try:
                # Need at least n_neighbors + 1 samples
                n_neighbors = min(20, len(col_data) - 1)
                if n_neighbors >= 5:
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                    outliers_mask = lof.fit_predict(data[[col]].fillna(data[col].median())) == -1
                else:
                    # Fallback to IQR
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            except:
                # Fallback to IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        
        elif method == 'dbscan':
            # DBSCAN for outlier detection
            try:
                # Standardize the data
                scaler = StandardScaler()
                col_scaled = scaler.fit_transform(data[[col]].fillna(data[col].median()))
                
                # Apply DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels = dbscan.fit_predict(col_scaled)
                outliers_mask = labels == -1  # -1 indicates outliers
            except:
                # Fallback to IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        
        if outliers_mask is not None:
            outliers_dict[col] = {
                'indices': data[outliers_mask].index.tolist(),
                'count': outliers_mask.sum(),
                'percentage': (outliers_mask.sum() / len(data)) * 100,
                'method': method
            }
    
    # Generate analysis and recommendations
    total_outliers = sum(info['count'] for info in outliers_dict.values())
    total_percentage = (total_outliers / (len(data) * len(outliers_dict))) * 100 if outliers_dict else 0
    
    result['success'] = True
    result['outliers'] = outliers_dict
    result['analysis'] = {
        'method_used': method_used,
        'total_outliers': total_outliers,
        'total_percentage': total_percentage,
        'affected_columns': list(outliers_dict.keys())
    }
    
    # Generate recommendations
    if total_percentage > 10:
        recommendations = [msg['extreme_outliers'], msg['investigate_outliers']]
    elif total_percentage > 5:
        recommendations = [msg['moderate_outliers'], msg['cap_outliers']]
    else:
        recommendations = [msg['few_outliers'], msg['keep_outliers']]
    
    result['recommendations'] = recommendations
    
    return result

def advanced_imbalanced_data_handling(X, y, method='auto', sampling_strategy='auto', random_state=42, language='id'):
    """
    Penanganan dataset yang tidak seimbang dengan metode yang canggih
    
    Parameters:
    -----------
    X : array-like
        Fitur dataset
    y : array-like
        Target dataset
    method : str
        Metode penyeimbangan ('auto', 'ros', 'rus', 'smote', 'adasyn', 'borderline_smote', 'smoteenn', 'smotetomek')
    sampling_strategy : str or float
        Strategi sampling ('auto', 'minority', 'not_minority', 'all', or float)
    random_state : int
        Random state untuk reproducibility
    language : str
        Bahasa untuk output
        
    Returns:
    --------
    dict
        Dictionary berisi data yang telah diseimbangkan dan informasi proses
    """
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.metrics import classification_report_imbalanced
        import pandas as pd
    except ImportError:
        result = {'success': False, 'error': 'imblearn tidak tersedia' if language == 'id' else 'imblearn not available'}
        return result
    
    messages = {
        'id': {
            'imbalance_analysis': 'Analisis Ketidakseimbangan',
            'original_distribution': 'Distribusi Original',
            'balanced_distribution': 'Distribusi Setelah Penyeimbangan',
            'method_used': 'Metode yang digunakan',
            'imbalance_ratio': 'Rasio ketidakseimbangan',
            'improvement': 'Peningkatan',
            'minority_class': 'Kelas minoritas',
            'majority_class': 'Kelas mayoritas',
            'synthetic_samples': 'Sampel sintetis yang dibuat',
            'removed_samples': 'Sampel yang dihapus',
            'recommendations': 'Rekomendasi',
            'severe_imbalance': 'Ketidakseimbangan parah terdeteksi',
            'moderate_imbalance': 'Ketidakseimbangan sedang terdeteksi',
            'slight_imbalance': 'Ketidakseimbangan ringan terdeteksi',
            'ensemble_recommendation': 'Gunakan ensemble methods',
            'threshold_recommendation': 'Pertimbangkan threshold tuning',
            'cost_sensitive_recommendation': 'Gunakan cost-sensitive learning'
        },
        'en': {
            'imbalance_analysis': 'Imbalance Analysis',
            'original_distribution': 'Original Distribution',
            'balanced_distribution': 'Distribution After Balancing',
            'method_used': 'Method used',
            'imbalance_ratio': 'Imbalance ratio',
            'improvement': 'Improvement',
            'minority_class': 'Minority class',
            'majority_class': 'Majority class',
            'synthetic_samples': 'Synthetic samples created',
            'removed_samples': 'Samples removed',
            'recommendations': 'Recommendations',
            'severe_imbalance': 'Severe imbalance detected',
            'moderate_imbalance': 'Moderate imbalance detected',
            'slight_imbalance': 'Slight imbalance detected',
            'ensemble_recommendation': 'Use ensemble methods',
            'threshold_recommendation': 'Consider threshold tuning',
            'cost_sensitive_recommendation': 'Use cost-sensitive learning'
        }
    }
    
    msg = messages.get(language, messages['id'])
    result = {'success': False, 'X_balanced': None, 'y_balanced': None, 'analysis': {}, 'recommendations': []}
    
    if X is None or y is None:
        result['error'] = 'Data tidak valid' if language == 'id' else 'Invalid data'
        return result
    
    # Convert to pandas if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Analyze original distribution
    original_counts = pd.Series(y).value_counts().sort_index()
    imbalance_ratio = original_counts.max() / original_counts.min() if original_counts.min() > 0 else float('inf')
    minority_class = original_counts.idxmin()
    majority_class = original_counts.idxmax()
    
    # Analyze mode (no balancing, just analysis)
    if method == 'analyze':
        # Calculate Gini coefficient for class distribution
        proportions = original_counts.values / len(y)
        gini = 1 - np.sum(proportions**2)
        
        # Determine imbalance level
        if imbalance_ratio > 10:
            level = msg['severe_imbalance']
        elif imbalance_ratio > 3:
            level = msg['moderate_imbalance']
        else:
            level = msg['slight_imbalance']
            
        # Count minority/majority classes (simple heuristic)
        avg_count = len(y) / len(original_counts)
        minority_classes_count = len(original_counts[original_counts < avg_count])
        majority_classes_count = len(original_counts[original_counts >= avg_count])
        
        result['success'] = True
        result['analysis'] = {
            'imbalance_level': level,
            'minority_classes': minority_classes_count,
            'majority_classes': majority_classes_count,
            'gini_coefficient': gini,
            'original_distribution': original_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }
        
        # Recommendations
        if imbalance_ratio > 3:
            result['recommendations'] = [msg['ensemble_recommendation'], msg['threshold_recommendation']]
        if len(original_counts) > 2:
            result['recommendations'].append("Consider multiclass-specific strategies" if language == 'en' else "Pertimbangkan strategi khusus multiclass")
            
        return result

    # Auto method selection and mapping generic terms
    if method == 'over':
        method = 'smote'
    elif method == 'under':
        method = 'rus'
    elif method == 'combine':
        method = 'smoteenn'
    elif method == 'hybrid':
        method = 'smotetomek'
        
    if method == 'auto':
        if imbalance_ratio > 10:
            method = 'smote'
        elif imbalance_ratio > 5:
            method = 'borderline_smote'
        elif imbalance_ratio > 3:
            method = 'ros'
        else:
            # No balancing needed
            result['success'] = True
            result['X_balanced'] = X
            result['y_balanced'] = y
            result['analysis'] = {
                'method_used': 'none',
                'original_distribution': original_counts.to_dict(),
                'imbalance_ratio_original': imbalance_ratio,
                'imbalance_ratio_balanced': imbalance_ratio,
                'reason': 'Rasio ketidakseimbangan terlalu kecil' if language == 'id' else 'Imbalance ratio too small'
            }
            return result
    
    # Select and configure the resampler
    resampler = None
    sampler_config = {}
    
    if sampling_strategy == 'auto':
        # Auto-determine sampling strategy
        if method in ['ros', 'smote', 'adasyn', 'borderline_smote']:
            sampling_strategy = 'minority'  # Oversample minority
        elif method in ['rus']:
            sampling_strategy = 'majority'  # Undersample majority
        else:
            sampling_strategy = 'auto'
    
    try:
        if method == 'ros':
            resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'rus':
            resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'smote':
            resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'adasyn':
            resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'borderline_smote':
            resampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'smoteenn':
            resampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'smotetomek':
            resampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
        else:
            result['error'] = f'Metode {method} tidak didukung' if language == 'id' else f'Method {method} not supported'
            return result
        
        # Apply resampling
        X_balanced, y_balanced = resampler.fit_resample(X, y)
        
        # Analyze results
        balanced_counts = pd.Series(y_balanced).value_counts().sort_index()
        new_imbalance_ratio = balanced_counts.max() / balanced_counts.min()
        
        # Calculate improvements
        samples_created = len(y_balanced) - len(y)
        samples_removed = len(y) - len(y_balanced) if len(y_balanced) < len(y) else 0
        
        result['success'] = True
        result['X_balanced'] = X_balanced
        result['y_balanced'] = y_balanced
        result['analysis'] = {
            'method_used': method,
            'original_distribution': original_counts.to_dict(),
            'balanced_distribution': balanced_counts.to_dict(),
            'imbalance_ratio_original': imbalance_ratio,
            'imbalance_ratio_balanced': new_imbalance_ratio,
            'improvement': imbalance_ratio - new_imbalance_ratio,
            'samples_change': samples_created if samples_created > 0 else -samples_removed
        }
        
        # Generate recommendations based on results
        if new_imbalance_ratio > 5:
            recommendations = [msg['severe_imbalance'], msg['ensemble_recommendation'], msg['cost_sensitive_recommendation']]
        elif new_imbalance_ratio > 3:
            recommendations = [msg['moderate_imbalance'], msg['threshold_recommendation']]
        else:
            recommendations = [msg['slight_imbalance']]
        
        result['recommendations'] = recommendations
        
    except Exception as e:
        result['error'] = f'Error dalam penyeimbangan: {str(e)}' if language == 'id' else f'Error in balancing: {str(e)}'
        return result
    
    return result

def create_shap_forecasting_dashboard(results):
    """
    Membuat dashboard SHAP untuk forecasting
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature Importance
    ax1 = plt.subplot(2, 2, 1)
    feature_importance = results['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    ax1.barh(features, importance)
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title('Feature Importance (Forecasting)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SHAP Summary Plot (jika tersedia)
    ax2 = plt.subplot(2, 2, 2)
    if 'summary_plot' in results:
        # Tampilkan summary plot yang sudah dibuat
        ax2.text(0.5, 0.5, 'Summary plot tersedia di objek results', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('SHAP Summary Plot')
    else:
        # Buat summary plot sederhana
        shap_values = results['shap_values']
        X_sample = results['X_sample']
        feature_names = results['feature_names']
        
        # Plot rata-rata absolute SHAP values
        if len(shap_values.shape) == 2:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values))
        
        ax2.bar(range(len(feature_names)), mean_shap)
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.set_ylabel('Mean |SHAP Value|')
        ax2.set_title('SHAP Feature Impact')
    
    # Plot 3: SHAP Values Distribution
    ax3 = plt.subplot(2, 2, 3)
    shap_values = results['shap_values']
    
    if len(shap_values.shape) == 2:
        # Flatten SHAP values untuk distribusi
        flat_shap = shap_values.flatten()
        ax3.hist(flat_shap, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('SHAP Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of SHAP Values')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Distribusi SHAP tidak tersedia\nuntuk bentuk data ini', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Sample Interpretation
    ax4 = plt.subplot(2, 2, 4)
    X_sample = results['X_sample']
    feature_names = results['feature_names']
    
    # Tampilkan interpretasi untuk sampel pertama
    if len(X_sample) > 0:
        sample_idx = 0
        sample_features = X_sample[sample_idx]
        
        # Buat bar plot untuk nilai fitur sampel
        ax4.bar(range(len(feature_names)), sample_features)
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=45)
        ax4.set_ylabel('Feature Value')
        ax4.set_title(f'Sample Features (Index {sample_idx})')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_lime_forecasting_dashboard(results):
    """
    Membuat dashboard LIME untuk forecasting
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature Importance
    ax1 = plt.subplot(2, 2, 1)
    feature_importance = results['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    ax1.barh(features, importance)
    ax1.set_xlabel('Mean |LIME Weight|')
    ax1.set_title('Feature Importance (LIME)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LIME Explanation untuk sampel pertama
    ax2 = plt.subplot(2, 2, 2)
    lime_explanations = results['lime_explanations']
    
    if lime_explanations:
        first_exp = lime_explanations[0]
        features, weights = zip(*first_exp.as_list())
        
        # Buat horizontal bar plot
        y_pos = np.arange(len(features))
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        ax2.barh(y_pos, weights, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features)
        ax2.set_xlabel('LIME Weight')
        ax2.set_title('LIME Explanation (Sample 0)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Distribusi bobot LIME
    ax3 = plt.subplot(2, 2, 3)
    all_weights = []
    for exp in lime_explanations:
        for _, weight in exp.as_list():
            all_weights.append(weight)
    
    ax3.hist(all_weights, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('LIME Weight')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of LIME Weights')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Rata-rata absolute importance per fitur
    ax4 = plt.subplot(2, 2, 4)
    feature_importance = results['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    ax4.bar(range(len(features)), importance)
    ax4.set_xticks(range(len(features)))
    ax4.set_xticklabels(features, rotation=45)
    ax4.set_ylabel('Mean |LIME Weight|')
    ax4.set_title('Average Feature Importance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_trend_seasonality_cycle(timeseries, period=None):
    """
    Menganalisis tren, seasonality, dan siklus dalam data timeseries
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan dianalisis
    period : int, optional
        Periode seasonal (jika diketahui)
        
    Returns:
    --------
    dict
        Hasil analisis tren, seasonality, dan siklus
    """
    # Hapus nilai missing
    clean_timeseries = timeseries.dropna()
    
    if len(clean_timeseries) < 4:
        return {
            'trend_detected': False,
            'seasonality_detected': False,
            'cycle_detected': False,
            'trend_strength': 0,
            'seasonality_strength': 0,
            'cycle_strength': 0,
            'message': 'Data terlalu sedikit untuk analisis tren, seasonality, dan siklus'
        }
    
    try:
        # Gunakan periode default jika tidak ditentukan
        if period is None:
            # Estimasi periode berdasarkan panjang data
            period = min(12, len(clean_timeseries) // 4)
            if period < 2:
                period = 2
        
        # Decompose time series
        decomposition = seasonal_decompose(clean_timeseries, model='additive', period=period)
        
        # Hitung kekuatan tren, seasonality, dan siklus
        trend_strength = np.var(decomposition.trend.dropna()) / np.var(clean_timeseries)
        seasonal_strength = np.var(decomposition.seasonal.dropna()) / np.var(clean_timeseries)
        residual_strength = np.var(decomposition.resid.dropna()) / np.var(clean_timeseries)
        
        # Deteksi tren menggunakan regresi linear
        x = np.arange(len(clean_timeseries))
        slope, _ = np.polyfit(x, clean_timeseries.values, 1)
        trend_detected = abs(slope) > 0.01 * np.std(clean_timeseries)
        
        # Deteksi seasonality berdasarkan kekuatan seasonal
        seasonality_detected = seasonal_strength > 0.1
        
        # Deteksi siklus menggunakan periodogram
        frequencies, power_spectrum = periodogram(clean_timeseries)
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip frequency 0
        dominant_period = 1 / frequencies[dominant_freq_idx] if frequencies[dominant_freq_idx] > 0 else None
        
        # Deteksi siklus berdasarkan periode dominan yang signifikan
        cycle_detected = False
        if dominant_period is not None and dominant_period > period:
            max_power = power_spectrum[dominant_freq_idx]
            avg_power = np.mean(power_spectrum)
            cycle_detected = max_power > 3 * avg_power  # Threshold untuk deteksi siklus
        
        return {
            'trend_detected': trend_detected,
            'seasonality_detected': seasonality_detected,
            'cycle_detected': cycle_detected,
            'trend_strength': float(trend_strength),
            'seasonality_strength': float(seasonality_strength),
            'cycle_strength': float(residual_strength),
            'trend_slope': float(slope),
            'dominant_cycle_period': float(dominant_period) if dominant_period else None,
            'decomposition': decomposition,
            'message': None
        }
        
    except Exception as e:
        return {
            'trend_detected': False,
            'seasonality_detected': False,
            'cycle_detected': False,
            'trend_strength': 0,
            'seasonality_strength': 0,
            'cycle_strength': 0,
            'message': f'Error dalam analisis: {str(e)}'
        }

def plot_pattern_analysis(timeseries, period=None, figsize=(20, 16)):
    """
    Membuat visualisasi komprehensif untuk analisis pola timeseries
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan dianalisis
    period : int, optional
        Periode seasonal
    figsize : tuple, optional
        Ukuran gambar
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi analisis pola lengkap
    """
    # Hapus nilai missing
    clean_timeseries = timeseries.dropna()
    
    if len(clean_timeseries) < 4:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.plot(clean_timeseries)
        ax.set_title('Data Timeseries (Data terlalu sedikit untuk analisis pola)')
        plt.tight_layout()
        return fig
    
    # Gunakan periode default jika tidak ditentukan
    if period is None:
        period = min(12, len(clean_timeseries) // 4)
        if period < 2:
            period = 2
    
    # Buat figure dengan subplots
    fig = plt.figure(figsize=figsize)
    
    # Subplot 1: Original data with trend line
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(clean_timeseries.index, clean_timeseries.values, label='Original Data', alpha=0.7)
    
    # Add trend line
    x_numeric = np.arange(len(clean_timeseries))
    z = np.polyfit(x_numeric, clean_timeseries.values, 1)
    p = np.poly1d(z)
    ax1.plot(clean_timeseries.index, p(x_numeric), "r--", label=f'Trend (slope: {z[0]:.4f})', linewidth=2)
    ax1.set_title('Original Data with Trend Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Seasonal decomposition
    try:
        decomposition = seasonal_decompose(clean_timeseries, model='additive', period=period)
        
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(decomposition.trend.dropna().index, decomposition.trend.dropna().values, color='red')
        ax2.set_title('Trend Component')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(decomposition.seasonal.dropna().index, decomposition.seasonal.dropna().values, color='green')
        ax3.set_title('Seasonal Component')
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(4, 2, 4)
        ax4.plot(decomposition.resid.dropna().index, decomposition.resid.dropna().values, color='orange')
        ax4.set_title('Residual Component')
        ax4.grid(True, alpha=0.3)
        
    except Exception as e:
        ax2 = plt.subplot(4, 2, 2)
        ax2.text(0.5, 0.5, f'Error in decomposition:\n{str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes)
        ax2.set_title('Decomposition Error')
    
    # Subplot 5: Seasonal subseries plot
    ax5 = plt.subplot(4, 2, 5)
    if period <= len(clean_timeseries):
        seasonal_data = []
        for i in range(period):
            seasonal_values = clean_timeseries.iloc[i::period]
            if len(seasonal_values) > 0:
                seasonal_data.append(seasonal_values.values)
        
        if seasonal_data:
            ax5.boxplot(seasonal_data, labels=[f'P{i+1}' for i in range(len(seasonal_data))])
            ax5.set_title(f'Seasonal Pattern (Period={period})')
            ax5.set_xlabel('Seasonal Period')
            ax5.set_ylabel('Value')
    
    # Subplot 6: Year-over-year growth (if applicable)
    ax6 = plt.subplot(4, 2, 6)
    if len(clean_timeseries) >= 365:  # At least one year of data
        try:
            # Calculate year-over-year growth
            yoy_growth = clean_timeseries.pct_change(periods=365).dropna() * 100
            ax6.plot(yoy_growth.index, yoy_growth.values, color='purple')
            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax6.set_title('Year-over-Year Growth Rate (%)')
            ax6.grid(True, alpha=0.3)
        except:
            ax6.text(0.5, 0.5, 'YoY growth not applicable', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax6.transAxes)
    
    # Subplot 7: Moving averages
    ax7 = plt.subplot(4, 2, 7)
    window_short = min(7, len(clean_timeseries) // 4)
    window_long = min(30, len(clean_timeseries) // 2)
    
    if window_short >= 2:
        short_ma = clean_timeseries.rolling(window=window_short).mean()
        ax7.plot(clean_timeseries.index, clean_timeseries.values, alpha=0.5, label='Original')
        ax7.plot(short_ma.index, short_ma.values, label=f'MA-{window_short}', linewidth=2)
        
        if window_long > window_short:
            long_ma = clean_timeseries.rolling(window=window_long).mean()
            ax7.plot(long_ma.index, long_ma.values, label=f'MA-{window_long}', linewidth=2)
        
        ax7.set_title('Moving Averages')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # Subplot 8: Distribution by time period
    ax8 = plt.subplot(4, 2, 8)
    if hasattr(clean_timeseries.index, 'month'):
        # Group by month
        monthly_data = clean_timeseries.groupby(clean_timeseries.index.month)
        monthly_means = monthly_data.mean()
        monthly_stds = monthly_data.std()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax8.bar(range(1, 13), [monthly_means.get(i, 0) for i in range(1, 13)], 
                yerr=[monthly_stds.get(i, 0) for i in range(1, 13)],
                capsize=5)
        ax8.set_xticks(range(1, 13))
        ax8.set_xticklabels(months)
        ax8.set_title('Seasonal Distribution by Month')
        ax8.set_ylabel('Mean Value')
    
    plt.tight_layout()
    return fig

def create_simplified_timeseries_plot(clean_timeseries, figsize=(12, 8)):
    """
    Create simplified timeseries plot for small datasets
    """
    if len(clean_timeseries) < 5:
        # Plot data asli dan ACF saja jika data cukup untuk ACF tapi tidak untuk PACF
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot data asli
        axes[0].plot(clean_timeseries)
        axes[0].set_title('Data Timeseries')
        
        # Plot ACF
        try:
            plot_acf(clean_timeseries, ax=axes[1])
            axes[1].set_title('Autocorrelation Function (ACF)')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error plotting ACF: {str(e)}', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('ACF - Error')
        
        plt.tight_layout()
        return fig

def create_full_timeseries_plot(clean_timeseries, figsize=(12, 8)):
    """
    Create full timeseries plot with ACF and PACF
    """
    if len(clean_timeseries) >= 5:
        # Plot lengkap jika data cukup
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot data asli
        axes[0].plot(clean_timeseries)
        axes[0].set_title('Data Timeseries')
        
        # Plot ACF
        try:
            plot_acf(clean_timeseries, ax=axes[1])
            axes[1].set_title('Autocorrelation Function (ACF)')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error plotting ACF: {str(e)}', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('ACF - Error')
        
        # Plot PACF
        try:
            plot_pacf(clean_timeseries, ax=axes[2])
            axes[2].set_title('Partial Autocorrelation Function (PACF)')
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Error plotting PACF: {str(e)}', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('PACF - Error')
        
        plt.tight_layout()
        return fig

def create_features_from_date(df, date_column):
    """
    Membuat fitur-fitur dari kolom tanggal (tahun, bulan, hari, dll.)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data
    date_column : str
        Nama kolom tanggal
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan fitur-fitur tambahan dari tanggal
    """
    # Buat salinan data
    df_new = df.copy()
    
    # Konversi kolom tanggal ke datetime jika belum
    if not pd.api.types.is_datetime64_any_dtype(df_new[date_column]):
        df_new[date_column] = pd.to_datetime(df_new[date_column])
    
    # Ekstrak fitur dari tanggal
    df_new['year'] = df_new[date_column].dt.year
    df_new['month'] = df_new[date_column].dt.month
    df_new['day'] = df_new[date_column].dt.day
    df_new['dayofweek'] = df_new[date_column].dt.dayofweek
    df_new['quarter'] = df_new[date_column].dt.quarter
    df_new['is_month_start'] = df_new[date_column].dt.is_month_start.astype(int)
    df_new['is_month_end'] = df_new[date_column].dt.is_month_end.astype(int)

    return df_new

def advanced_data_scaling(data, method='auto', target_column=None, language='id', use_streamlit=True):
    """
    Melakukan scaling dan transformasi data dengan metode canggih
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data yang akan discaling dan ditransformasi
    method : str
        Metode scaling yang digunakan ('auto', 'standard', 'minmax', 'robust', 
        'quantile', 'power', 'log', 'boxcox', 'yeojohnson')
    target_column : str, optional
        Nama kolom target untuk transformasi yang mempertimbangkan target
    language : str
        Bahasa untuk pesan output ('id' atau 'en')
    use_streamlit : bool
        Apakah menggunakan Streamlit untuk output (default: True)
        
    Returns:
    --------
    dict
        Dictionary berisi data hasil scaling, scaler object, dan informasi transformasi
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
    from scipy import stats
    import warnings
    
    messages = {
        'id': {
            'auto_select': "Memilih metode scaling otomatis berdasarkan karakteristik data...",
            'standard_selected': "StandardScaler dipilih untuk data dengan distribusi normal",
            'robust_selected': "RobustScaler dipilih untuk data dengan outlier",
            'quantile_selected': "QuantileTransformer dipilih untuk data tidak normal",
            'power_selected': "PowerTransformer dipilih untuk data dengan skewness tinggi",
            'log_warning': "Transformasi log tidak dapat diterapkan pada data dengan nilai negatif",
            'boxcox_warning': "Transformasi Box-Cox memerlikan data positif",
            'yeojohnson_selected': "Yeo-Johnson dipilih untuk transformasi umum",
            'scaling_completed': "Scaling dan transformasi selesai",
            'skewness_detected': "Skewness terdeteksi: {:.2f}",
            'kurtosis_detected': "Kurtosis terdeteksi: {:.2f}",
            'outlier_detected': "Outlier terdeteksi: {} dari {} sampel",
            'normal_distribution': "Distribusi mendekati normal",
            'non_normal_distribution': "Distribusi tidak normal, transformasi diperlukan"
        },
        'en': {
            'auto_select': "Selecting automatic scaling method based on data characteristics...",
            'standard_selected': "StandardScaler selected for normally distributed data",
            'robust_selected': "RobustScaler selected for data with outliers",
            'quantile_selected': "QuantileTransformer selected for non-normal data",
            'power_selected': "PowerTransformer selected for highly skewed data",
            'log_warning': "Log transformation cannot be applied to data with negative values",
            'boxcox_warning': "Box-Cox transformation requires positive data",
            'yeojohnson_selected': "Yeo-Johnson selected for general transformation",
            'scaling_completed': "Scaling and transformation completed",
            'skewness_detected': "Skewness detected: {:.2f}",
            'kurtosis_detected': "Kurtosis detected: {:.2f}",
            'outlier_detected': "Outliers detected: {} of {} samples",
            'normal_distribution': "Distribution approximately normal",
            'non_normal_distribution': "Distribution not normal, transformation required"
        }
    }
    
    msg = messages.get(language, messages['id'])
    
    try:
        # Analisis karakteristik data numerik
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {
                'success': False,
                'message': 'Tidak ada data numerik untuk discaling' if language == 'id' else 'No numerical data to scale',
                'original_data': data
            }
        
        # Analisis distribusi dan outlier
        skewness = numeric_data.skew().abs().mean()
        kurtosis = numeric_data.kurtosis().abs().mean()
        
        # Deteksi outlier menggunakan IQR
        outlier_count = 0
        total_samples = len(numeric_data)
        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_data[col] < (Q1 - 1.5 * IQR)) | (numeric_data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        
        # Pilih metode scaling otomatis
        if method == 'auto':
            if use_streamlit:
                _notify_info(msg['auto_select'])

            # Logika pemilihan metode
            if outlier_count > 0.05 * total_samples:  # >5% outlier
                method = 'robust'
                if use_streamlit:
                    _notify_info(msg['robust_selected'])
            elif skewness > 1.0:  # High skewness
                method = 'power'
                if use_streamlit:
                    _notify_info(msg['power_selected'])
            elif skewness > 0.5:  # Moderate skewness
                method = 'quantile'
                if use_streamlit:
                    _notify_info(msg['quantile_selected'])
            else:  # Approximately normal
                method = 'standard'
                if use_streamlit:
                    _notify_info(msg['standard_selected'])

        # Informasi analisis data
        if use_streamlit:
            _notify_write(f"**Analisis Data:**")
            _notify_write(f"- {msg['skewness_detected'].format(skewness)}")
            _notify_write(f"- {msg['kurtosis_detected'].format(kurtosis)}")
            _notify_write(f"- {msg['outlier_detected'].format(outlier_count, total_samples)}")

            if skewness < 0.5 and abs(kurtosis) < 3:
                _notify_success(msg['normal_distribution'])
            else:
                _notify_warning(msg['non_normal_distribution'])
        
        # Terapkan scaling sesuai metode yang dipilih
        scaled_data = numeric_data.copy()
        scaler_info = {}
        
        if method == 'standard':
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(numeric_data)
            scaled_data[numeric_data.columns] = scaled_values
            scaler_info = {'method': 'StandardScaler', 'scaler': scaler}
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(numeric_data)
            scaled_data[numeric_data.columns] = scaled_values
            scaler_info = {'method': 'MinMaxScaler', 'scaler': scaler}
            
        elif method == 'robust':
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(numeric_data)
            scaled_data[numeric_data.columns] = scaled_values
            scaler_info = {'method': 'RobustScaler', 'scaler': scaler}
            
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
            scaled_values = scaler.fit_transform(numeric_data)
            scaled_data[numeric_data.columns] = scaled_values
            scaler_info = {'method': 'QuantileTransformer', 'scaler': scaler}
            
        elif method == 'power':
            # Gunakan PowerTransformer untuk transformasi umum
            scaler = PowerTransformer(method='yeo-johnson')
            
            # Cek apakah Box-Cox bisa digunakan (semua data positif)
            if (numeric_data > 0).all().all():
                try:
                    scaler = PowerTransformer(method='box-cox')
                    scaler_info = {'method': 'PowerTransformer (Box-Cox)', 'scaler': scaler}
                except:
                    scaler = PowerTransformer(method='yeo-johnson')
                    scaler_info = {'method': 'PowerTransformer (Yeo-Johnson)', 'scaler': scaler}
            else:
                scaler_info = {'method': 'PowerTransformer (Yeo-Johnson)', 'scaler': scaler}
            
            scaled_values = scaler.fit_transform(numeric_data)
            scaled_data[numeric_data.columns] = scaled_values
            
        elif method == 'log':
            # Transformasi logaritmik
            if (numeric_data <= 0).any().any():
                if use_streamlit:
                    _notify_warning(msg['log_warning'])
                # Gunakan shift untuk membuat semua data positif
                shift_value = abs(numeric_data.min().min()) + 1
                log_data = np.log(numeric_data + shift_value)
                scaled_data[numeric_data.columns] = log_data
                scaler_info = {'method': 'Log Transformation (shifted)', 'shift_value': shift_value}
            else:
                log_data = np.log(numeric_data)
                scaled_data[numeric_data.columns] = log_data
                scaler_info = {'method': 'Log Transformation'}

        elif method == 'boxcox':
            # Transformasi Box-Cox
            if (numeric_data <= 0).any().any():
                if use_streamlit:
                    _notify_warning(msg['boxcox_warning'])
                # Fallback ke Yeo-Johnson
                scaler = PowerTransformer(method='yeo-johnson')
                scaled_values = scaler.fit_transform(numeric_data)
                scaled_data[numeric_data.columns] = scaled_values
                scaler_info = {'method': 'PowerTransformer (Yeo-Johnson) - Fallback dari Box-Cox', 'scaler': scaler}
            else:
                scaler = PowerTransformer(method='box-cox')
                scaled_values = scaler.fit_transform(numeric_data)
                scaled_data[numeric_data.columns] = scaled_values
                scaler_info = {'method': 'PowerTransformer (Box-Cox)', 'scaler': scaler}

        elif method == 'yeojohnson':
            scaler = PowerTransformer(method='yeo-johnson')
            scaled_values = scaler.fit_transform(numeric_data)
            scaled_data[numeric_data.columns] = scaled_values
            scaler_info = {'method': 'PowerTransformer (Yeo-Johnson)', 'scaler': scaler}

        # Gabungkan data non-numerik kembali
        result_data = data.copy()
        result_data[numeric_data.columns] = scaled_data

        # Hitung statistik setelah scaling
        after_skewness = scaled_data.skew().abs().mean()
        after_kurtosis = scaled_data.kurtosis().abs().mean()

        if use_streamlit:
            _notify_success(msg['scaling_completed'])
        
        return {
            'success': True,
            'scaled_data': result_data,
            'scaler_info': scaler_info,
            'original_data': data,
            'method_used': method,
            'statistics': {
                'before': {'skewness': float(skewness), 'kurtosis': float(kurtosis)},
                'after': {'skewness': float(after_skewness), 'kurtosis': float(after_kurtosis)},
                'improvement': {
                    'skewness_reduction': float(skewness - after_skewness),
                    'kurtosis_reduction': float(abs(kurtosis) - abs(after_kurtosis))
                }
            },
            'outlier_info': {
                'count': int(outlier_count),
                'percentage': float(outlier_count / total_samples * 100)
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'original_data': data,
            'message': f'Error dalam scaling data: {str(e)}'
        }

def advanced_feature_transformation(data, categorical_features=None, numerical_features=None, 
                                  target_column=None, method='auto', language='id', use_streamlit=True):
    """
    Melakukan transformasi fitur canggih untuk preprocessing
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data yang akan ditransformasi
    categorical_features : list
        Daftar fitur kategorikal
    numerical_features : list  
        Daftar fitur numerik
    target_column : str, optional
        Nama kolom target
    method : str
        Metode transformasi ('auto', 'polynomial', 'interaction', 'binning', 'encoding')
    language : str
        Bahasa untuk pesan output ('id' atau 'en')
    use_streamlit : bool
        Apakah menggunakan Streamlit untuk output (default: True)
        
    Returns:
    --------
    dict
        Dictionary berisi data hasil transformasi dan informasi transformasi
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    messages = {
        'id': {
            'auto_select': "Memilih transformasi otomatis berdasarkan tipe fitur...",
            'polynomial_selected': "Transformasi polynomial dipilih untuk fitur numerik",
            'interaction_selected': "Transformasi interaksi dipilih untuk fitur kategorikal",
            'binning_selected': "Binning dipilih untuk fitur numerik dengan banyak nilai unik",
            'encoding_selected': "Encoding otomatis dipilih untuk fitur kategorikal",
            'no_categorical': "Tidak ada fitur kategorikal untuk ditransformasi",
            'no_numerical': "Tidak ada fitur numerik untuk ditransformasi",
            'transformation_completed': "Transformasi fitur selesai",
            'features_created': "{} fitur baru dibuat dari {} fitur asli",
            'polynomial_degree': "Derajat polynomial: {}",
            'bins_created': "{} bins dibuat untuk fitur {}",
            'interaction_features': "{} fitur interaksi dibuat"
        },
        'en': {
            'auto_select': "Selecting automatic transformation based on feature types...",
            'polynomial_selected': "Polynomial transformation selected for numerical features",
            'interaction_selected': "Interaction transformation selected for categorical features",
            'binning_selected': "Binning selected for numerical features with many unique values",
            'encoding_selected': "Automatic encoding selected for categorical features",
            'no_categorical': "No categorical features to transform",
            'no_numerical': "No numerical features to transform",
            'transformation_completed': "Feature transformation completed",
            'features_created': "{} new features created from {} original features",
            'polynomial_degree': "Polynomial degree: {}",
            'bins_created': "{} bins created for feature {}",
            'interaction_features': "{} interaction features created"
        }
    }
    
    msg = messages.get(language, messages['id'])
    
    try:
        result_data = data.copy()
        transformation_info = []
        
        # Deteksi fitur otomatis jika tidak ditentukan
        if categorical_features is None:
            categorical_features = result_data.select_dtypes(include=['object', 'category']).columns.tolist()
        if numerical_features is None:
            numerical_features = result_data.select_dtypes(include=[np.number]).columns.tolist()
            # Hapus target column dari numerical features jika ada
            if target_column and target_column in numerical_features:
                numerical_features.remove(target_column)
        
        if use_streamlit:
            _notify_info(msg['auto_select'])

        # Transformasi fitur numerik
        if numerical_features:
            if method == 'auto':
                # Analisis karakteristik fitur numerik
                for feature in numerical_features:
                    unique_count = result_data[feature].nunique()
                    skewness = abs(result_data[feature].skew())

                    if unique_count > 20 and skewness > 1.0:
                        # Binning untuk fitur dengan banyak nilai unik dan skewness tinggi
                        from sklearn.preprocessing import KBinsDiscretizer

                        # Tentukan jumlah bins otomatis
                        n_bins = min(10, unique_count // 5)
                        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

                        binned_feature = discretizer.fit_transform(result_data[[feature]])
                        result_data[f'{feature}_binned'] = binned_feature.flatten()

                        transformation_info.append({
                            'feature': feature,
                            'method': 'binning',
                            'n_bins': n_bins,
                            'strategy': 'quantile'
                        })

                        if use_streamlit:
                            _notify_success(msg['bins_created'].format(n_bins, feature))

                    elif skewness > 0.5:
                        # Polynomial features untuk fitur dengan skewness moderat
                        poly = PolynomialFeatures(degree=2, include_bias=False)

                        # Hanya gunakan fitur ini untuk polynomial
                        poly_features = poly.fit_transform(result_data[[feature]])

                        # Ambil fitur polynomial (selain yang pertama - bias)
                        if poly_features.shape[1] > 1:
                            for i in range(1, poly_features.shape[1]):
                                result_data[f'{feature}_poly_{i}'] = poly_features[:, i]

                            transformation_info.append({
                                'feature': feature,
                                'method': 'polynomial',
                                'degree': 2
                            })

                            if use_streamlit:
                                _notify_success(msg['polynomial_degree'].format(2))

            elif method == 'polynomial':
                # Transformasi polynomial untuk semua fitur numerik
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_data = poly.fit_transform(result_data[numerical_features])

                # Buat nama fitur polynomial
                feature_names = poly.get_feature_names_out(numerical_features)

                # Tambahkan fitur polynomial ke dataframe
                for i, name in enumerate(feature_names[len(numerical_features):]):  # Skip original features
                    result_data[f'poly_{name}'] = poly_data[:, len(numerical_features) + i]

                transformation_info.append({
                    'method': 'polynomial',
                    'original_features': numerical_features,
                    'new_features': feature_names[len(numerical_features):].tolist(),
                    'degree': 2
                })

                if use_streamlit:
                    _notify_success(msg['polynomial_degree'].format(2))
        
        # Transformasi fitur kategorikal
        if categorical_features:
            if method == 'auto' or method == 'encoding':
                from sklearn.preprocessing import OneHotEncoder
                
                for feature in categorical_features:
                    unique_count = result_data[feature].nunique()
                    
                    if unique_count <= 10:  # One-hot encoding untuk kategori dengan sedikit nilai unik
                        encoder = OneHotEncoder(sparse_output=False, drop='first')
                        encoded_features = encoder.fit_transform(result_data[[feature]])
                        
                        # Buat nama fitur baru
                        categories = encoder.categories_[0][1:]  # Skip first category (reference)
                        for i, category in enumerate(categories):
                            result_data[f'{feature}_{category}'] = encoded_features[:, i]
                        
                        transformation_info.append({
                            'feature': feature,
                            'method': 'onehot_encoding',
                            'n_categories': unique_count,
                            'n_new_features': len(categories)
                        })
                        
                    else:  # Frequency encoding untuk kategori dengan banyak nilai unik
                        freq_encoding = result_data[feature].value_counts()
                        result_data[f'{feature}_freq'] = result_data[feature].map(freq_encoding)
                        
                        transformation_info.append({
                            'feature': feature,
                            'method': 'frequency_encoding',
                            'n_categories': unique_count
                        })
                
                if use_streamlit:
                    _notify_success(msg['encoding_selected'])

        # Hitung statistik transformasi
        original_feature_count = len(data.columns)
        new_feature_count = len(result_data.columns)
        features_created = new_feature_count - original_feature_count
        
        if use_streamlit:
            _notify_success(msg['transformation_completed'])
            _notify_info(msg['features_created'].format(features_created, original_feature_count))
        
        return {
            'success': True,
            'transformed_data': result_data,
            'transformation_info': transformation_info,
            'original_data': data,
            'method_used': method,
            'statistics': {
                'original_features': original_feature_count,
                'new_features': new_feature_count,
                'features_created': features_created
            },
            'feature_types': {
                'categorical': categorical_features,
                'numerical': numerical_features
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'original_data': data,
            'message': f'Error dalam transformasi fitur: {str(e)}'
        }

def interpret_statistical_model(model, model_type='arima', language='id'):
    """
    Interpretasi khusus untuk model statistik seperti ARIMA, SARIMA, Exponential Smoothing
    
    Parameters:
    -----------
    model : object
        Model statistik yang telah dilatih
    model_type : str
        Tipe model ('arima', 'sarima', 'exponential_smoothing', dll.)
    language : str
        Bahasa untuk output ('id' atau 'en')
        
    Returns:
    --------
    dict
        Dictionary berisi interpretasi model statistik
    """
    messages = {
        'id': {
            'title': 'Interpretasi Model Statistik',
            'coefficients': 'Koefisien Model',
            'ar_terms': 'Koefisien AutoRegresif (AR)',
            'ma_terms': 'Koefisien Moving Average (MA)', 
            'seasonal_terms': 'Koefisien Musiman',
            'trend': 'Komponen Tren',
            'seasonality': 'Komponen Musiman',
            'residuals': 'Analisis Residual',
            'goodness_of_fit': 'Goodness of Fit',
            'aic': 'AIC (Akaike Information Criterion)',
            'bic': 'BIC (Bayesian Information Criterion)',
            'mse': 'Mean Squared Error',
            'no_coefficients': 'Tidak dapat mengekstrak koefisien dari model',
            'recommendations': 'Rekomendasi',
            'check_residuals': 'Periksa plot residual untuk asumsi model',
            'check_normality': 'Periksa normalitas residual',
            'check_autocorrelation': 'Periksa autokorelasi residual',
            'use_decomposition': 'Gunakan dekomposisi deret waktu untuk analisis komponen'
        },
        'en': {
            'title': 'Statistical Model Interpretation',
            'coefficients': 'Model Coefficients',
            'ar_terms': 'AutoRegressive (AR) Coefficients',
            'ma_terms': 'Moving Average (MA) Coefficients',
            'seasonal_terms': 'Seasonal Coefficients',
            'trend': 'Trend Component',
            'seasonality': 'Seasonality Component',
            'residuals': 'Residual Analysis',
            'goodness_of_fit': 'Goodness of Fit',
            'aic': 'AIC (Akaike Information Criterion)',
            'bic': 'BIC (Bayesian Information Criterion)',
            'mse': 'Mean Squared Error',
            'no_coefficients': 'Cannot extract coefficients from model',
            'recommendations': 'Recommendations',
            'check_residuals': 'Check residual plots for model assumptions',
            'check_normality': 'Check residual normality',
            'check_autocorrelation': 'Check residual autocorrelation',
            'use_decomposition': 'Use time series decomposition for component analysis'
        }
    }
    
    lang = messages.get(language, messages['id'])
    result = {
        'title': lang['title'],
        'model_type': model_type,
        'coefficients': {},
        'goodness_of_fit': {},
        'interpretation': {},
        'recommendations': [],
        'success': False
    }
    
    try:
        # Ekstrak koefisien berdasarkan tipe model
        if hasattr(model, 'params'):
            result['coefficients']['params'] = model.params
            
        if hasattr(model, 'arparams'):
            result['coefficients']['ar_terms'] = model.arparams
            
        if hasattr(model, 'maparams'):
            result['coefficients']['ma_terms'] = model.maparams
            
        if hasattr(model, 'seasonalarparams'):
            result['coefficients']['seasonal_ar'] = model.seasonalarparams
            
        if hasattr(model, 'seasonalmaparams'):
            result['coefficients']['seasonal_ma'] = model.seasonalmaparams
            
        # Ekstrak goodness of fit metrics
        if hasattr(model, 'aic'):
            result['goodness_of_fit']['aic'] = model.aic
            
        if hasattr(model, 'bic'):
            result['goodness_of_fit']['bic'] = model.bic
            
        if hasattr(model, 'mse'):
            result['goodness_of_fit']['mse'] = model.mse
            
        if hasattr(model, 'mse_resid'):
            result['goodness_of_fit']['mse'] = model.mse_resid
            
        # Interpretasi berdasarkan tipe model
        if 'arima' in model_type.lower():
            result['interpretation']['model'] = 'ARIMA (AutoRegressive Integrated Moving Average)'
            result['interpretation']['ar_component'] = 'Menggambarkan hubungan antara observasi saat ini dengan observasi sebelumnya'
            result['interpretation']['ma_component'] = 'Menggambarkan hubungan antara kesalahan saat ini dengan kesalahan sebelumnya'
            
        elif 'sarima' in model_type.lower():
            result['interpretation']['model'] = 'SARIMA (Seasonal ARIMA)'
            result['interpretation']['seasonal_component'] = 'Menangkap pola musiman dalam data'
            
        elif 'exponential' in model_type.lower():
            result['interpretation']['model'] = 'Exponential Smoothing'
            result['interpretation']['smoothing'] = 'Memberikan bobot yang lebih besar pada observasi terbaru'
            
        # Rekomendasi analisis lanjutan
        result['recommendations'] = [
            lang['check_residuals'],
            lang['check_normality'],
            lang['check_autocorrelation'],
            lang['use_decomposition']
        ]
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['message'] = lang['no_coefficients']
        result['success'] = False
        
    return result

def _is_statistical_model(model):
    """
    Mengecek apakah model adalah model statistik (ARIMA, SARIMA, dll.)
    
    Parameters:
    -----------
    model : object
        Model yang akan dicek
        
    Returns:
    --------
    bool
        True jika model adalah model statistik
    """
    model_type = type(model).__name__.lower()
    module_name = type(model).__module__.lower()
    
    # Check for statistical model indicators
    statistical_indicators = [
        'arima', 'sarima', 'exponential', 'holt', 'winters',
        'statsmodels', 'pmdarima', 'tbats', 'bats'
    ]
    
    # Check model attributes that indicate statistical model
    statistical_attributes = [
        'params', 'arparams', 'maparams', 'seasonalarparams',
        'seasonalmaparams', 'aic', 'bic', 'hqic'
    ]
    
    # Check if any indicator is in model type or module
    for indicator in statistical_indicators:
        if indicator in model_type or indicator in module_name:
            return True
    
    # Check if model has statistical attributes
    for attr in statistical_attributes:
        if hasattr(model, attr):
            return True
    
    return False

def _is_neural_network_model(model):
    """
    Mengecek apakah model adalah neural network
    
    Parameters:
    -----------
    model : object
        Model yang akan dicek
        
    Returns:
    --------
    bool
        True jika model adalah neural network
    """
    model_type = type(model).__name__.lower()
    module_name = type(model).__module__.lower()
    
    # Check for common neural network indicators
    neural_network_indicators = [
        'mlp', 'neural', 'keras', 'tensorflow', 'torch', 'skorch',
        'sequential', 'model', 'network', 'dense', 'conv', 'lstm'
    ]
    
    # Check model attributes that indicate neural network
    neural_network_attributes = [
        'layers', 'weights', 'activation', 'optimizer', 'loss',
        'fit_generator', 'predict_generator', 'forward'
    ]
    
    # Check if any indicator is in model type or module
    for indicator in neural_network_indicators:
        if indicator in model_type or indicator in module_name:
            return True
    
    # Check if model has neural network attributes
    for attr in neural_network_attributes:
        if hasattr(model, attr):
            return True
    
    # Special check for sklearn MLP models
    if hasattr(model, 'hidden_layer_sizes') or hasattr(model, 'activation'):
        return True
    
    return False

def check_model_compatibility(model, method='shap', language='id'):
    """
    Mengecek kompatibilitas model dengan metode interpretasi SHAP atau LIME
    Versi simplified yang lebih permissive dan user-friendly

    Parameters:
    -----------
    model : object
        Model yang akan dicek kompatibilitasnya
    method : str, optional
        Metode interpretasi ('shap' atau 'lime')
    language : str, optional
        Bahasa untuk pesan error ('id' atau 'en')

    Returns:
    --------
    dict
        Dictionary berisi:
        - 'compatible': Boolean apakah model kompatibel
        - 'message': Pesan penjelasan
        - 'suggestion': Saran alternatif
        - 'error_type': Tipe error jika tidak kompatibel
        - 'confidence': Tingkat kepercayaan compatibility (high/medium/low)
    """
    # Pesan error dalam berbagai bahasa
    messages = {
        'id': {
            'no_predict': "Model tidak memiliki method 'predict'. Tidak dapat digunakan untuk interpretasi.",
            'clustering': "Model clustering tidak cocok untuk interpretasi prediktif.",
            'dimensionality_reduction': "Model reduksi dimensi tidak cocok untuk interpretasi prediktif.",
            'neural_network_shap': "Model neural network akan menggunakan SHAP DeepExplainer atau fallback ke KernelExplainer.",
            'complex_ensemble': "Model ensemble kompleks mungkin tidak didukung. Coba model individual.",
            'generally_supported': "Model kemungkinan besar didukung untuk interpretasi {method}.",
            'try_alternative': "Jika gagal, coba metode {alternative} atau model yang lebih sederhana.",
            'suggestion_prefix': "Saran: "
        },
        'en': {
            'no_predict': "Model does not have 'predict' method. Cannot be used for interpretation.",
            'clustering': "Clustering models are not suitable for predictive interpretation.",
            'dimensionality_reduction': "Dimensionality reduction models are not suitable for predictive interpretation.",
            'neural_network_shap': "Neural network models will use SHAP DeepExplainer or fallback to KernelExplainer.",
            'complex_ensemble': "Complex ensemble models may not be supported. Try individual models.",
            'generally_supported': "Model is likely supported for {method} interpretation.",
            'try_alternative': "If it fails, try {alternative} method or simpler models.",
            'suggestion_prefix': "Suggestion: "
        }
    }

    lang = messages.get(language, messages['id'])

    try:
        # Basic requirement check
        if not hasattr(model, 'predict'):
            return {
                'compatible': False,
                'message': lang['no_predict'],
                'suggestion': lang['suggestion_prefix'] + "Gunakan model dengan method 'predict'.",
                'error_type': 'missing_predict_method',
                'confidence': 'low'
            }

        # Get model information
        model_type = type(model).__name__.lower()
        module_name = type(model).__module__.lower()

        # Clear non-predictive models (strict check)
        clearly_unsupported = [
            'kmeans', 'dbscan', 'hierarchical', 'agglomerative',  # Clustering
            'pca', 'tsne', 'umap', 'lda', 'nmf'  # Dimensionality reduction
        ]

        for unsupported in clearly_unsupported:
            if unsupported in model_type or unsupported in module_name:
                if 'cluster' in unsupported:
                    error_msg = lang['clustering']
                else:
                    error_msg = lang['dimensionality_reduction']

                return {
                    'compatible': False,
                    'message': error_msg,
                    'suggestion': lang['suggestion_prefix'] + "Gunakan model prediktif seperti Random Forest atau Logistic Regression.",
                    'error_type': 'unsupported_model_type',
                    'confidence': 'low'
                }

        # Neural networks (now supported with DeepExplainer)
        neural_indicators = ['mlp', 'neural', 'dense', 'lstm', 'gru', 'cnn', 'rnn']
        is_neural = _is_neural_network_model(model)

        if is_neural:
            if method == 'shap':
                return {
                    'compatible': True,
                    'message': lang['neural_network_shap'],
                    'suggestion': lang['suggestion_prefix'] + "Model akan menggunakan SHAP DeepExplainer atau fallback ke KernelExplainer.",
                    'error_type': None,
                    'confidence': 'high'
                }
            else:  # LIME
                return {
                    'compatible': True,
                    'message': lang['generally_supported'].format(method=method.upper()),
                    'suggestion': lang['suggestion_prefix'] + "Pastikan data input dalam format numerik.",
                    'error_type': None,
                    'confidence': 'medium'
                }

        # Complex ensembles (warning but allow)
        complex_ensembles = ['voting', 'stacking', 'blending']
        is_complex_ensemble = any(ensemble in model_type for ensemble in complex_ensembles)

        if is_complex_ensemble:
            return {
                'compatible': True,
                'message': lang['generally_supported'].format(method=method.upper()),
                'suggestion': lang['suggestion_prefix'] + lang['complex_ensemble'] + " " + lang['try_alternative'].format(alternative='LIME' if method == 'shap' else 'SHAP'),
                'error_type': 'complex_ensemble',
                'confidence': 'medium'
            }

        # Statistical models (special handling)
        is_statistical = _is_statistical_model(model)
        
        if is_statistical:
            if method == 'shap':
                return {
                    'compatible': False,
                    'message': "Model statistik tidak mendukung interpretasi SHAP secara langsung. Gunakan interpretasi khusus statistik.",
                    'suggestion': lang['suggestion_prefix'] + "Gunakan fungsi interpret_statistical_model() atau coba dengan LIME.",
                    'error_type': 'statistical_model',
                    'confidence': 'high'
                }
            else:  # LIME
                return {
                    'compatible': True,
                    'message': "Model statistik dapat menggunakan LIME dengan pendekatan khusus.",
                    'suggestion': lang['suggestion_prefix'] + "LIME akan menggunakan mode regresi untuk interpretasi.",
                    'error_type': None,
                    'confidence': 'medium'
                }

        # Common supported models (high confidence)
        well_supported = [
            'randomforest', 'xgb', 'lgbm', 'catboost',  # Tree-based
            'linear', 'logistic', 'ridge', 'lasso', 'elastic',  # Linear
            'svm', 'svc', 'svr',  # SVM
            'knn', 'nearest',  # KNN
            'decisiontree', 'extratree',  # Tree models
            'gaussiannb', 'multinomialnb', 'bernoullinb'  # Naive Bayes
        ]

        is_well_supported = any(supported in model_type for supported in well_supported)

        if is_well_supported:
            suggestions = []
            if method == 'shap' and any(tree in model_type for tree in ['randomforest', 'xgb', 'lgbm', 'decisiontree']):
                suggestions.append("Gunakan TreeExplainer untuk hasil optimal")
            elif method == 'lime':
                suggestions.append("Pastikan data tidak ada missing values")

            return {
                'compatible': True,
                'message': lang['generally_supported'].format(method=method.upper()),
                'suggestion': lang['suggestion_prefix'] + ' '.join(suggestions) if suggestions else "",
                'error_type': None,
                'confidence': 'high'
            }

        # Unknown models (permissive approach)
        return {
            'compatible': True,
            'message': f"Model {model_type} akan dicoba dengan {method.upper()}.",
            'suggestion': lang['suggestion_prefix'] + "Jika gagal, coba model yang lebih umum seperti Random Forest.",
            'error_type': 'unknown_model_type',
            'confidence': 'low'
        }

    except Exception as e:
        return {
            'compatible': False,
            'message': f"Error saat mengecek kompatibilitas: {str(e)}",
            'suggestion': lang['suggestion_prefix'] + "Periksa model dan coba lagi.",
            'error_type': 'compatibility_check_error',
            'confidence': 'low'
        }

def get_model_interpretation_recommendations(model, language='id'):
    """
    Memberikan rekomendasi metode interpretasi yang sesuai untuk model

    
    Parameters:
    -----------
    model : object
        Model yang akan direkomendasikan metode interpretasinya
    language : str, optional
        Bahasa untuk pesan ('id' atau 'en')
        
    Returns:
    --------
    dict
        Dictionary berisi rekomendasi interpretasi
    """
    
    lang = language if language in ['id', 'en'] else 'id'
    
    # Dapatkan tipe model
    model_type = type(model).__name__.lower()
    module_name = type(model).__module__.lower()
    
    # Rekomendasi berdasarkan tipe model
    recommendations = {
        'tree_based': {
            'primary': 'SHAP (TreeExplainer)',
            'secondary': 'LIME',
            'reason': 'Model berbasis pohon sangat cocok dengan SHAP TreeExplainer karena cepat dan akurat',
            'alternatives': ['Feature Importance', 'Permutation Importance']
        },
        'linear': {
            'primary': 'SHAP (LinearExplainer)',
            'secondary': 'LIME',
            'reason': 'Model linear memiliki interpretasi yang sangat baik dengan SHAP LinearExplainer',
            'alternatives': ['Coefficients', 'Feature Importance']
        },
        'neural_network': {
            'primary': 'LIME',
            'secondary': 'SHAP (DeepSHAP)',
            'reason': 'Neural network lebih cocok dengan LIME untuk interpretasi lokal',
            'alternatives': ['Integrated Gradients', 'Grad-CAM']
        },
        'ensemble': {
            'primary': 'SHAP',
            'secondary': 'LIME',
            'reason': 'Ensemble model dapat diinterpretasi dengan SHAP untuk hasil global yang baik',
            'alternatives': ['Feature Importance', 'Permutation Importance']
        },
        'statistical': {
            'primary': 'LIME',
            'secondary': 'Interpretasi Statistik',
            'reason': 'Model statistik lebih cocok dengan LIME atau interpretasi khusus statistik',
            'alternatives': ['Analisis Koefisien', 'Dekomposisi Time Series', 'Analisis Residual']
        },
        'unsupported': {
            'primary': 'Tidak tersedia',
            'secondary': 'Tidak tersedia',
            'reason': 'Model ini tidak cocok untuk interpretasi otomatis',
            'alternatives': ['Analisis manual', 'Visualisasi data']
        }
    }
    
    # Kategorikan model
    if _is_statistical_model(model):
        category = 'statistical'
    elif any(tree in model_type for tree in ['randomforest', 'xgb', 'lgbm', 'catboost', 'decisiontree', 'extratree']):
        category = 'tree_based'
    elif any(linear in model_type for linear in ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
        category = 'linear'
    elif any(nn in model_type for nn in ['mlp', 'neural', 'dense', 'lstm', 'gru']) or 'keras' in module_name or 'torch' in module_name:
        category = 'neural_network'
    elif any(ensemble in model_type for ensemble in ['voting', 'bagging', 'boosting']):
        category = 'ensemble'
    elif any(unsupported in model_type for unsupported in ['kmeans', 'pca', 'tsne']):
        category = 'unsupported'
    else:
        category = 'ensemble'  # Default ke ensemble untuk model yang tidak dikenal
    
    rec = recommendations[category]
    
    if lang == 'id':
        return {
            'primary_method': rec['primary'],
            'secondary_method': rec['secondary'],
            'reason': rec['reason'],
            'alternatives': rec['alternatives'],
            'explanation': f"Model Anda tergolong {category.replace('_', ' ')}. "
                          f"Metode yang paling direkomendasikan adalah {rec['primary']}. "
                          f"Alasan: {rec['reason']}"
        }
    else:
        return {
            'primary_method': rec['primary'],
            'secondary_method': rec['secondary'],
            'reason': rec['reason'],
            'alternatives': rec['alternatives'],
            'explanation': f"Your model is categorized as {category.replace('_', ' ')}. "
                          f"The most recommended method is {rec['primary']}. "
                          f"Reason: {rec['reason']}"
        }

def implement_shap_classification(model, X_sample, X_train=None, language='id', problem_type=None, class_names=None, feature_names=None):
    """
    Implementasi SHAP untuk model klasifikasi dengan penanganan multi-class
    
    Parameters:
    -----------
    model : sklearn model
        Model yang telah dilatih
    X_sample : pandas.DataFrame
        Sample data untuk interpretasi
    X_train : pandas.DataFrame, optional
        Data training untuk background SHAP
    language : str, optional
        Bahasa untuk pesan ('id' atau 'en')
    problem_type : str, optional
        'binary' atau 'multiclass' (auto-detected jika None)
    class_names : list, optional
        Nama kelas untuk klasifikasi
    feature_names : list, optional
        Nama fitur
        
    Returns:
    --------
    dict
        Dictionary berisi SHAP values dan informasi interpretasi
    """
    import shap
    import numpy as np
    
    # Auto-detect problem type if not specified
    if problem_type is None:
        if hasattr(model, 'classes_'):
            if len(model.classes_) == 2:
                problem_type = 'binary'
            else:
                problem_type = 'multiclass'
        else:
            problem_type = 'binary'  # Default assumption
    
    # Use provided class_names or extract from model
    if class_names is None and hasattr(model, 'classes_'):
        class_names = list(model.classes_)
    
    result = {
        'shap_values': None,
        'expected_value': None,
        'explainer': None,
        'problem_type': problem_type,
        'class_names': class_names,
        'feature_names': feature_names or X_sample.columns.tolist(),
        'success': False
    }
    
    try:
        # Tentukan jenis explainer berdasarkan model
        if hasattr(model, 'tree_'):  # Tree-based models
            explainer = shap.TreeExplainer(model, data=X_train if X_train is not None else X_sample)
        elif hasattr(model, 'coef_'):  # Linear models
            explainer = shap.LinearExplainer(model, data=X_train if X_train is not None else X_sample)
        elif _is_neural_network_model(model):  # Neural networks
            try:
                # Try DeepExplainer first for neural networks
                background_data = X_train if X_train is not None else shap.sample(X_sample, min(100, len(X_sample)))
                explainer = shap.DeepExplainer(model, background_data)
            except Exception:
                # Fallback to KernelExplainer if DeepExplainer fails
                background_data = X_train if X_train is not None else shap.sample(X_sample, min(50, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict_proba, background_data)
        else:  # General models - use KernelExplainer with better background
            background_data = X_train if X_train is not None else shap.sample(X_sample, min(50, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
        
        # Hitung SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Penanganan untuk berbagai jenis output
        if problem_type == 'binary':
            # Binary classification: SHAP values untuk kelas positif
            if isinstance(shap_values, list) and len(shap_values) == 2:
                result['shap_values'] = shap_values[1]  # Kelas positif
                result['expected_value'] = explainer.expected_value[1] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                result['shap_values'] = shap_values
                result['expected_value'] = explainer.expected_value if hasattr(explainer, 'expected_value') else None
                
        elif problem_type == 'multiclass':
            # Multi-class classification
            if isinstance(shap_values, list):
                result['shap_values'] = shap_values  # List of SHAP values untuk setiap kelas
                result['expected_value'] = explainer.expected_value if hasattr(explainer, 'expected_value') else None
                result['n_classes'] = len(shap_values)
            else:
                result['shap_values'] = shap_values
                result['expected_value'] = explainer.expected_value if hasattr(explainer, 'expected_value') else None
                result['n_classes'] = len(class_names) if class_names else 2
        
        result['explainer'] = explainer
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        
    return result

def handle_multiclass_shap(shap_values, predicted_class=None, method='individual', class_names=None):
    """
    Penanganan SHAP values untuk multi-class classification
    
    Parameters:
    -----------
    shap_values : list or array
        SHAP values dari model multi-class
    predicted_class : int, optional
        Kelas yang diprediksi untuk focus
    method : str, optional
        'individual' (fokus pada kelas tertentu), 'average' (rata-rata semua kelas), atau 'max_importance' (kelas dengan importance tertinggi)
    class_names : list, optional
        Nama kelas untuk display
        
    Returns:
    --------
    dict
        Dictionary berisi processed SHAP values
    """
    import numpy as np
    
    result = {}
    
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        
        if method == 'individual' and predicted_class is not None:
            # Fokus pada kelas yang diprediksi
            if 0 <= predicted_class < n_classes:
                result['shap_values_focused'] = shap_values[predicted_class]
                result['class_focused'] = predicted_class
                result['method'] = 'individual'
                result['class_name'] = class_names[predicted_class] if class_names and predicted_class < len(class_names) else f'Class {predicted_class}'
            else:
                result['error'] = f'Predicted class {predicted_class} out of range [0, {n_classes-1}]'
                result['method'] = 'error'
            
        elif method == 'average':
            # Rata-rata absolute SHAP values untuk semua kelas
            avg_shap = np.mean([np.abs(values) for values in shap_values], axis=0)
            result['shap_values_average'] = avg_shap
            result['method'] = 'average'
            result['n_classes'] = n_classes
            
        elif method == 'max_importance':
            # Pilih kelas dengan total importance tertinggi
            class_importance = []
            for i, class_shap in enumerate(shap_values):
                total_importance = np.sum(np.abs(class_shap))
                class_importance.append((total_importance, i))
            
            max_importance_class = max(class_importance, key=lambda x: x[0])[1]
            result['shap_values_focused'] = shap_values[max_importance_class]
            result['class_focused'] = max_importance_class
            result['method'] = 'max_importance'
            result['class_name'] = class_names[max_importance_class] if class_names and max_importance_class < len(class_names) else f'Class {max_importance_class}'
            result['importance_score'] = class_importance[max_importance_class][0]
            
        else:
            # Simpan semua SHAP values
            result['shap_values_all'] = shap_values
            result['n_classes'] = n_classes
            result['method'] = 'all'
            
    else:
        result['shap_values'] = shap_values
        result['method'] = 'single'
        
    return result

def create_shap_visualization(shap_values, X_sample, feature_names=None, class_names=None, 
                           problem_type='binary', selected_class=None, max_display=10):
    """
    Membuat visualisasi SHAP yang robust untuk berbagai jenis output
    
    Parameters:
    -----------
    shap_values : array or list
        SHAP values
    X_sample : pandas.DataFrame
        Sample data
    feature_names : list, optional
        Nama fitur
    class_names : list, optional
        Nama kelas
    problem_type : str
        'binary' atau 'multiclass'
    selected_class : int, optional
        Kelas yang dipilih untuk multi-class
    max_display : int
        Maksimal fitur yang ditampilkan
        
    Returns:
    --------
    dict
        Dictionary berisi figure dan informasi visualisasi
    """
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    
    result = {
        'figures': {},
        'success': False,
        'error': None
    }
    
    try:
        if feature_names is None:
            feature_names = X_sample.columns.tolist()
        
        # Handle multi-class SHAP values
        if problem_type == 'multiclass' and isinstance(shap_values, list):
            if selected_class is not None and 0 <= selected_class < len(shap_values):
                # Use selected class
                shap_to_plot = shap_values[selected_class]
                class_name = class_names[selected_class] if class_names and selected_class < len(class_names) else f'Class {selected_class}'
            else:
                # Use first class as default
                shap_to_plot = shap_values[0]
                class_name = class_names[0] if class_names else 'Class 0'
        else:
            # Binary classification or single array
            shap_to_plot = shap_values
            class_name = class_names[0] if class_names else 'Prediction'
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_to_plot, X_sample, feature_names=feature_names, 
                         plot_type="bar", max_display=max_display, show=False)
        result['figures']['summary_bar'] = plt.gcf()
        plt.close()
        
        # Create detailed summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_to_plot, X_sample, feature_names=feature_names, 
                         max_display=max_display, show=False)
        result['figures']['summary_detailed'] = plt.gcf()
        plt.close()
        
        # Create feature importance ranking
        if len(shap_to_plot.shape) == 2:
            mean_abs_shap = np.mean(np.abs(shap_to_plot), axis=0)
        else:
            mean_abs_shap = np.abs(shap_to_plot)
        
        # Sort features by importance
        feature_importance = list(zip(feature_names, mean_abs_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        result['feature_importance'] = feature_importance[:max_display]
        result['class_name'] = class_name
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        
    return result

def implement_lime_classification(model, X_sample, y_sample=None, problem_type='binary', 
                                class_names=None, feature_names=None, num_features=10):
    """
    Implementasi LIME untuk model klasifikasi
    
    Parameters:
    -----------
    model : sklearn model
        Model yang telah dilatih
    X_sample : pandas.DataFrame
        Sample data untuk interpretasi
    y_sample : array, optional
        Target untuk sample data
    problem_type : str, optional
        'binary' atau 'multiclass'
    class_names : list, optional
        Nama kelas untuk klasifikasi
    feature_names : list, optional
        Nama fitur
    num_features : int, optional
        Jumlah fitur terpenting yang akan ditampilkan
        
    Returns:
    --------
    dict
        Dictionary berisi LIME explanations dan informasi interpretasi
    """
    try:
        import lime
        import lime.lime_tabular
        
        result = {
            'explanations': [],
            'problem_type': problem_type,
            'class_names': class_names,
            'feature_names': feature_names or X_sample.columns.tolist(),
            'num_features': num_features
        }
        
        # Buat LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_sample.values,
            feature_names=result['feature_names'],
            class_names=class_names,
            mode='classification'
        )
        
        # Fungsi prediksi untuk LIME
        if problem_type == 'binary':
            predict_fn = lambda x: model.predict_proba(x)[:, 1]  # Probabilitas kelas positif
        else:
            predict_fn = model.predict_proba  # Semua probabilitas untuk multi-class
        
        # Generate explanations untuk beberapa sample
        n_samples = min(5, len(X_sample))  # Maksimal 5 sample
        
        for i in range(n_samples):
            explanation = explainer.explain_instance(
                X_sample.iloc[i].values,
                predict_fn,
                num_features=num_features,
                top_labels=1 if problem_type == 'binary' else len(class_names)
            )
            
            result['explanations'].append({
                'sample_index': i,
                'predicted_class': model.predict(X_sample.iloc[i:i+1])[0],
                'predicted_proba': model.predict_proba(X_sample.iloc[i:i+1])[0],
                'explanation': explanation
            })
        
        result['explainer'] = explainer
        result['predict_fn'] = predict_fn
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        
    return result

def detect_model_type(model):
    """
    Mendeteksi jenis model (classification, regression, atau forecasting)
    
    Parameters:
    -----------
    model : sklearn/statsmodels model
        Model yang akan dideteksi
        
    Returns:
    --------
    str
        'classification', 'regression', atau 'forecasting'
    """
    # Cek apakah model memiliki classes_ (indikator classification)
    if hasattr(model, 'classes_'):
        return 'classification'
    
    # Cek apakah model memiliki predict_proba (indikator classification)
    if hasattr(model, 'predict_proba'):
        return 'classification'
    
    # Cek model forecasting (ARIMA, SARIMA, dsb)
    model_name = type(model).__name__.lower()
    forecasting_models = ['arima', 'sarima', 'sarimax', 'holtwinters', 'exponentialsmoothing', 'var', 'vecm']
    if any(fm in model_name for fm in forecasting_models):
        return 'forecasting'
    
    # Default ke regression
    return 'regression'

def prepare_forecasting_data_for_interpretation(X_train, X_test, selected_features, sample_idx):
    """
    Mempersiapkan data forecasting untuk interpretasi SHAP/LIME
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Data training
    X_test : pandas.DataFrame
        Data testing  
    selected_features : list
        Daftar fitur yang dipilih
    sample_idx : int
        Indeks sampel yang akan dijelaskan
        
    Returns:
    --------
    dict
        Dictionary berisi data yang siap untuk interpretasi
    """
    try:
        # Filter data berdasarkan fitur yang dipilih
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Pastikan sample_idx valid
        if sample_idx >= len(X_test_selected):
            sample_idx = len(X_test_selected) - 1
        
        # Ambil sampel yang akan dijelaskan
        sample = X_test_selected.iloc[sample_idx]
        
        result = {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'sample': sample,
            'sample_idx': sample_idx,
            'success': True
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def validate_data_for_ml(df, target_column=None, feature_columns=None, 
                        max_missing_ratio=0.5, min_samples=10, 
                        handle_missing='auto', verbose=True):
    """
    Validasi dan membersihkan data untuk machine learning dengan pendekatan yang konsisten
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang akan divalidasi
    target_column : str, optional
        Nama kolom target (jika ada)
    feature_columns : list, optional
        Daftar kolom fitur yang akan digunakan (jika None, gunakan semua kolom numerik)
    max_missing_ratio : float, optional
        Rasio missing value maksimum yang diizinkan per kolom (default: 0.5)
    min_samples : int, optional
        Jumlah minimum sampel yang dibutuhkan (default: 10)
    handle_missing : str, optional
        Metode penanganan missing value ('auto', 'drop', 'impute', 'none')
    verbose : bool, optional
        Apakah akan menampilkan informasi proses validasi
        
    Returns:
    --------
    dict
        Dictionary berisi:
        - 'data': DataFrame yang sudah divalidasi dan dibersihkan
        - 'target_column': Nama kolom target (jika ada)
        - 'feature_columns': Daftar kolom fitur yang digunakan
        - 'missing_handled': Informasi bagaimana missing value ditangani
        - 'warnings': Daftar peringatan jika ada
        - 'errors': Daftar error jika ada
    """
    
    result = {
        'data': None,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'missing_handled': {},
        'warnings': [],
        'errors': []
    }
    
    # Validasi input
    if df is None or df.empty:
        result['errors'].append("DataFrame kosong atau None")
        return result
    
    if len(df) < min_samples:
        result['errors'].append(f"Jumlah sampel ({len(df)}) kurang dari minimum ({min_samples})")
        return result
    
    # Buat salinan data
    df_clean = df.copy()
    
    # Identifikasi kolom fitur jika tidak ditentukan
    if feature_columns is None:
        # Gunakan semua kolom numerik kecuali target column
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            feature_columns = [col for col in numeric_columns if col != target_column]
        else:
            feature_columns = numeric_columns
    
    result['feature_columns'] = feature_columns
    
    # Validasi target column jika ada
    if target_column and target_column not in df_clean.columns:
        result['errors'].append(f"Kolom target '{target_column}' tidak ditemukan")
        return result
    
    # Validasi feature columns
    missing_features = [col for col in feature_columns if col not in df_clean.columns]
    if missing_features:
        result['errors'].append(f"Kolom fitur tidak ditemukan: {missing_features}")
        return result
    
    # Analisis missing values
    columns_to_check = feature_columns.copy()
    if target_column:
        columns_to_check.append(target_column)
    
    missing_analysis = df_clean[columns_to_check].isnull().sum()
    missing_ratio = missing_analysis / len(df_clean)
    
    if verbose:
        print("Analisis Missing Values:")
        for col in columns_to_check:
            ratio = missing_ratio[col]
            count = missing_analysis[col]
            print(f"  {col}: {count} missing ({ratio:.2%})")
    
    # Identifikasi kolom dengan terlalu banyak missing values
    high_missing_cols = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    
    if high_missing_cols:
        result['warnings'].append(f"Kolom dengan missing > {max_missing_ratio:.0%}: {high_missing_cols}")
        
        if handle_missing == 'auto':
            # Drop kolom dengan terlalu banyak missing
            df_clean = df_clean.drop(columns=high_missing_cols)
            result['missing_handled']['dropped_columns'] = high_missing_cols
            
            # Update feature columns
            feature_columns = [col for col in feature_columns if col not in high_missing_cols]
            result['feature_columns'] = feature_columns
            
            if verbose:
                print(f"Menghapus kolom dengan terlalu banyak missing: {high_missing_cols}")
    
    # Tangani missing values pada kolom yang tersisa
    if handle_missing == 'auto':
        # Gunakan strategi yang sesuai untuk setiap tipe data
        for col in feature_columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Untuk data numerik, gunakan median
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    result['missing_handled'][col] = f'filled_with_median({median_val:.2f})'
                    
            elif df_clean[col].dtype == 'object':
                # Untuk data kategorikal, gunakan mode
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    result['missing_handled'][col] = f'filled_with_mode({mode_val})'
        
        # Tangani missing values pada target column
        if target_column and df_clean[target_column].isnull().any():
            if df_clean[target_column].dtype in ['float64', 'int64']:
                median_val = df_clean[target_column].median()
                df_clean[target_column] = df_clean[target_column].fillna(median_val)
                result['missing_handled'][target_column] = f'filled_with_median({median_val:.2f})'
            else:
                mode_val = df_clean[target_column].mode().iloc[0] if not df_clean[target_column].mode().empty else 'unknown'
                df_clean[target_column] = df_clean[target_column].fillna(mode_val)
                result['missing_handled'][target_column] = f'filled_with_mode({mode_val})'
                
    elif handle_missing == 'drop':
        # Hapus baris dengan missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=columns_to_check)
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            result['missing_handled']['dropped_rows'] = dropped_rows
            if verbose:
                print(f"Menghapus {dropped_rows} baris dengan missing values")
                
    elif handle_missing == 'impute':
        # Gunakan imputasi yang lebih canggih (mirip dengan yang ada di app.py)
        for col in columns_to_check:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['float64', 'int64']:
                    # Gunakan interpolasi untuk data time series jika memungkinkan
                    try:
                        df_clean[col] = df_clean[col].interpolate(method='linear')
                        df_clean[col] = df_clean[col].ffill().bfill()
                        result['missing_handled'][col] = 'interpolated_and_filled'
                    except:
                        # Fallback ke median
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        result['missing_handled'][col] = f'filled_with_median({median_val:.2f})'
                else:
                    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    result['missing_handled'][col] = f'filled_with_mode({mode_val})'
    
    # Validasi final
    final_missing = df_clean[columns_to_check].isnull().sum().sum()
    if final_missing > 0:
        result['warnings'].append(f"Masih ada {final_missing} missing values setelah cleaning")
        if verbose:
            print(f"Peringatan: Masih ada {final_missing} missing values")
    
    # Validasi ukuran data akhir
    if len(df_clean) < min_samples:
        result['errors'].append(f"Jumlah sampel akhir ({len(df_clean)}) kurang dari minimum ({min_samples})")
        return result
    
    # Hasil akhir
    result['data'] = df_clean
    
    if verbose:
        print(f"\nValidasi selesai:")
        print(f"  Jumlah sampel awal: {len(df)}")
        print(f"  Jumlah sampel akhir: {len(df_clean)}")
        print(f"  Kolom fitur: {len(feature_columns)}")
        if target_column:
            print(f"  Kolom target: {target_column}")
        if result['missing_handled']:
            print(f"  Missing values ditangani: {result['missing_handled']}")
        if result['warnings']:
            print(f"  Peringatan: {result['warnings']}")
    
    return result

def create_lag_features(df, target_column, lag_periods=[1, 2, 3, 7, 14, 30], fill_method='ffill'):
    """
    Membuat fitur lag dari kolom target
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data
    target_column : str
        Nama kolom target
    lag_periods : list, optional
        Daftar periode lag yang akan dibuat
    fill_method : str, optional
        Metode untuk mengisi NaN yang dihasilkan dari lag (ffill, bfill, zero, mean, median)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan fitur-fitur lag tambahan
    """
    # Buat salinan data
    df_new = df.copy()
    
    # Buat fitur lag
    for lag in lag_periods:
        df_new[f'{target_column}_lag_{lag}'] = df_new[target_column].shift(lag)
    
    # Tangani NaN yang dihasilkan dari operasi lag
    lag_columns = [f'{target_column}_lag_{lag}' for lag in lag_periods]
    
    if fill_method == 'ffill':
        df_new[lag_columns] = df_new[lag_columns].ffill()
    elif fill_method == 'bfill':
        df_new[lag_columns] = df_new[lag_columns].bfill()
    elif fill_method == 'zero':
        df_new[lag_columns] = df_new[lag_columns].fillna(0)
    elif fill_method == 'mean':
        for col in lag_columns:
            df_new[col] = df_new[col].fillna(df_new[col].mean())
    elif fill_method == 'median':
        for col in lag_columns:
            df_new[col] = df_new[col].fillna(df_new[col].median())
    else:
        # Default: gunakan forward fill dan backward fill
        df_new[lag_columns] = df_new[lag_columns].ffill().bfill()
    
    return df_new

def create_rolling_features(df, target_column, windows=[7, 14, 30], fill_method='ffill', min_periods=None):
    """
    Membuat fitur rolling (rata-rata, standar deviasi, min, max) dari kolom target
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data
    target_column : str
        Nama kolom target
    windows : list, optional
        Daftar ukuran window untuk rolling
    fill_method : str, optional
        Metode untuk mengisi NaN yang dihasilkan dari rolling (ffill, bfill, zero, mean, median)
    min_periods : int, optional
        Jumlah minimum observasi yang dibutuhkan untuk menghitung rolling statistics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan fitur-fitur rolling tambahan
    """
    # Buat salinan data
    df_new = df.copy()
    
    # Set min_periods jika tidak ditentukan (setengah dari window size)
    if min_periods is None:
        min_periods = 1  # Minimal 1 observasi untuk menghitung
    
    # Buat fitur rolling
    for window in windows:
        df_new[f'{target_column}_rolling_mean_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).mean()
        df_new[f'{target_column}_rolling_std_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).std()
        df_new[f'{target_column}_rolling_min_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).min()
        df_new[f'{target_column}_rolling_max_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).max()
    
    # Tangani NaN yang dihasilkan dari operasi rolling
    rolling_columns = []
    for window in windows:
        rolling_columns.extend([
            f'{target_column}_rolling_mean_{window}',
            f'{target_column}_rolling_std_{window}',
            f'{target_column}_rolling_min_{window}',
            f'{target_column}_rolling_max_{window}'
        ])
    
    if fill_method == 'ffill':
        df_new[rolling_columns] = df_new[rolling_columns].ffill()
    elif fill_method == 'bfill':
        df_new[rolling_columns] = df_new[rolling_columns].bfill()
    elif fill_method == 'zero':
        df_new[rolling_columns] = df_new[rolling_columns].fillna(0)
    elif fill_method == 'mean':
        for col in rolling_columns:
            df_new[col] = df_new[col].fillna(df_new[col].mean())
    elif fill_method == 'median':
        for col in rolling_columns:
            df_new[col] = df_new[col].fillna(df_new[col].median())
    else:
        # Default: gunakan forward fill dan backward fill
        df_new[rolling_columns] = df_new[rolling_columns].ffill().bfill()
    
    return df_new

def extract_sensor_features(df, group_cols=None, target_cols=None, aggregations=['mean', 'median', 'std', 'min', 'max', 'skew', 'kurt']):
    """
    Ekstraksi ciri statistik (agregasi) dari data sensor mentah (log)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data sensor mentah
    group_cols : list, optional
        Daftar kolom untuk pengelompokan (misal: Sensor_ID, Device_ID)
    target_cols : list, optional
        Daftar kolom numerik yang akan diekstraksi cirinya
    aggregations : list, optional
        Daftar fungsi agregasi statistik yang akan diterapkan
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame yang berisi hasil ekstraksi ciri (agregasi)
    """
    if df is None or df.empty:
        return None
        
    df_new = df.copy()
    
    # Deteksi kolom numerik jika target_cols tidak ditentukan
    if target_cols is None:
        target_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
        # Hapus group_cols dari target_cols jika ada
        if group_cols:
            # Pastikan group_cols adalah list
            if isinstance(group_cols, str):
                group_cols = [group_cols]
            target_cols = [col for col in target_cols if col not in group_cols]
            
    if not target_cols:
        return df_new
        
    # Lakukan agregasi
    if group_cols:
        # Pastikan group_cols adalah list
        if isinstance(group_cols, str):
            group_cols = [group_cols]
            
        # Gunakan groupby untuk agregasi
        agg_df = df_new.groupby(group_cols)[target_cols].agg(aggregations)
        
        # Flatten multi-index columns: 'Column_mean', 'Column_std', etc.
        agg_df.columns = [f'{col}_{stat}' for col, stat in agg_df.columns]
        
        # Reset index agar group_cols menjadi kolom biasa
        agg_df = agg_df.reset_index()
        return agg_df
    else:
        # Agregasi global (seluruh dataset menjadi satu baris fitur)
        agg_results = []
        for col in target_cols:
            for stat in aggregations:
                try:
                    val = df_new[col].agg(stat)
                    agg_results.append({f'{col}_{stat}': val})
                except:
                    continue
        
        # Combine into one row
        combined_row = {}
        for res in agg_results:
            combined_row.update(res)
            
        return pd.DataFrame([combined_row])


def _slope_integral_minmax_for_series(times, values):
    """
    Hitung slope (regresi linear), integral (trapezoidal), min, max untuk satu seri nilai vs waktu.
    times: array-like (numeric atau datetime yang dijadikan numeric)
    values: array-like numerik
    Returns: dict dengan keys slope, integral, min, max
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & np.isfinite(times)
    if mask.sum() < 2:
        return {'slope': np.nan, 'integral': np.nan, 'min': np.nan, 'max': np.nan}
    t = times[mask]
    y = values[mask]
    # Slope: regresi linear y = a*t + b
    try:
        slope, _ = np.polyfit(t, y, 1)
    except Exception:
        slope = np.nan
    # Integral: trapezoidal
    integral = np.trapz(y, t)
    return {'slope': float(slope), 'integral': float(integral), 'min': float(np.nanmin(y)), 'max': float(np.nanmax(y))}


def extract_sensor_features_slope_integral_minmax(df, time_column, group_cols=None, target_cols=None, window_size=None):
    """
    Ekstraksi ciri data sensor berdasarkan slope, integral, dan min-max pada range waktu tertentu.
    Untuk setiap kelompok (atau seluruh data) dan setiap kolom target, dihitung:
    - slope: kemiringan regresi linear nilai vs waktu
    - integral: luas di bawah kurva (trapezoidal)
    - min, max: nilai minimum dan maksimum dalam range waktu

    Parameters
    ----------
    df : pandas.DataFrame
        Data sensor (harus ada kolom waktu)
    time_column : str
        Nama kolom yang berisi waktu (akan diurutkan)
    group_cols : list, optional
        Kolom pengelompokan (misal: ID sensor). Jika None, seluruh data dianggap satu kelompok.
    target_cols : list, optional
        Kolom numerik yang diekstrak. Default: semua kolom numerik selain time dan group.
    window_size : int, optional
        Banyak baris per jendela waktu. Jika None, satu jendela = seluruh range per kelompok.
        Jika diisi, setiap kelompok dibagi per jendela lalu slope/integral/min/max diagregasi
        (slope/integral: rata-rata, min: min dari min, max: max dari max).

    Returns
    -------
    pandas.DataFrame
        Satu baris per kelompok (atau satu baris jika tanpa group), kolom: group_cols + {target}_slope, {target}_integral, {target}_min, {target}_max
    """
    if df is None or df.empty or time_column not in df.columns:
        return None

    df_work = df.copy()
    # Konversi waktu ke numerik untuk perhitungan
    if pd.api.types.is_datetime64_any_dtype(df_work[time_column]):
        t_numeric = df_work[time_column].astype(np.int64) / 1e9  # detik
    else:
        t_numeric = pd.to_numeric(df_work[time_column], errors='coerce')
    df_work['_t_numeric_'] = t_numeric

    if target_cols is None:
        target_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [c for c in target_cols if c != time_column and c != '_t_numeric_']
        if group_cols:
            target_cols = [c for c in target_cols if c not in (group_cols if isinstance(group_cols, list) else [group_cols])]
    if not target_cols:
        return None

    if group_cols is None:
        group_cols = []
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    def compute_features(g):
        g = g.sort_values('_t_numeric_')
        t = g['_t_numeric_'].values
        rows = []
        if window_size is None or len(g) <= window_size:
            for col in target_cols:
                if col not in g.columns:
                    continue
                res = _slope_integral_minmax_for_series(t, g[col].values)
                rows.append({f'{col}_slope': res['slope'], f'{col}_integral': res['integral'], f'{col}_min': res['min'], f'{col}_max': res['max']})
        else:
            slopes, integrals, mins, maxs = {c: [] for c in target_cols}, {c: [] for c in target_cols}, {c: [] for c in target_cols}, {c: [] for c in target_cols}
            for start in range(0, len(g) - 1):
                end = min(start + window_size, len(g))
                w = g.iloc[start:end]
                tw = w['_t_numeric_'].values
                for col in target_cols:
                    if col not in w.columns:
                        continue
                    res = _slope_integral_minmax_for_series(tw, w[col].values)
                    slopes[col].append(res['slope'])
                    integrals[col].append(res['integral'])
                    mins[col].append(res['min'])
                    maxs[col].append(res['max'])
            for col in target_cols:
                rows.append({
                    f'{col}_slope': np.nanmean(slopes[col]) if slopes[col] else np.nan,
                    f'{col}_integral': np.nanmean(integrals[col]) if integrals[col] else np.nan,
                    f'{col}_min': np.nanmin(mins[col]) if mins[col] else np.nan,
                    f'{col}_max': np.nanmax(maxs[col]) if maxs[col] else np.nan,
                })
        if not rows:
            return pd.DataFrame()
        out = {}
        for r in rows:
            out.update(r)
        return pd.DataFrame([out])

    if group_cols:
        grouped = df_work.groupby(group_cols, dropna=False)
        results = []
        for name, g in grouped:
            if isinstance(name, tuple):
                row = {k: v for k, v in zip(group_cols, name)}
            else:
                row = {group_cols[0]: name}
            feats = compute_features(g)
            if feats is not None and not feats.empty:
                row.update(feats.iloc[0].to_dict())
            results.append(row)
        out_df = pd.DataFrame(results)
    else:
        feats = compute_features(df_work)
        if feats is None or feats.empty:
            return None
        out_df = feats

    return out_df


def normalize_sensor_features(df, feature_columns=None, method='minmax'):
    """
    Normalisasi kolom fitur numerik (hasil ekstraksi sensor) sebelum EDA/ML.
    method: 'minmax' (0-1) atau 'standard' (z-score).
    """
    if df is None or df.empty:
        return df
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not feature_columns:
        return df
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    df_out = df.copy()
    X = df_out[feature_columns].fillna(df_out[feature_columns].mean())
    df_out[feature_columns] = scaler.fit_transform(X)
    return df_out