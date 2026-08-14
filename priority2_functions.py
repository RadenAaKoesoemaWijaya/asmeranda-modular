"""
Priority 2 Functions for SHAP/LIME Improvement
This module contains improved functions for better SHAP and LIME implementation.
"""

import pandas as pd
import numpy as np


def improved_data_preprocessing_for_interpretation(X_data, model=None, method='shap'):
    """
    Preprocessing data yang lebih baik untuk metode interpretasi SHAP/LIME
    
    Parameters:
    -----------
    X_data : pandas.DataFrame
        Data yang akan diproses
    model : object, optional
        Model yang akan digunakan untuk interpretasi
    method : str, optional
        Metode interpretasi ('shap' atau 'lime')
        
    Returns:
    --------
    dict
        Dictionary berisi data yang sudah diproses dan informasi preprocessing
    """
    result = {
        'X_processed': None,
        'preprocessing_steps': [],
        'warnings': [],
        'errors': [],
        'success': False
    }
    
    try:
        if X_data is None or X_data.empty:
            result['errors'].append("No data provided for preprocessing")
            return result
        
        # Buat salinan data
        X_processed = X_data.copy()
        
        # Identifikasi tipe kolom
        categorical_cols = X_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        result['preprocessing_steps'].append(f"Found {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")
        
        # Preprocessing untuk SHAP
        if method == 'shap':
            # SHAP memerlukan semua data numerik
            if categorical_cols:
                result['preprocessing_steps'].append(f"Applying one-hot encoding to {len(categorical_cols)} categorical columns for SHAP")
                X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=False)
                result['warnings'].append(f"Categorical features encoded: {categorical_cols}")
            
            # Pastikan semua data numerik
            for col in X_processed.columns:
                if X_processed[col].dtype == 'object':
                    try:
                        X_processed[col] = pd.factorize(X_processed[col])[0].astype(float)
                        result['preprocessing_steps'].append(f"Factorized column: {col}")
                    except Exception as e:
                        result['errors'].append(f"Failed to convert column {col}: {str(e)}")
                        continue
            
            # Handle missing values untuk SHAP
            if X_processed.isnull().any().any():
                missing_cols = X_processed.columns[X_processed.isnull().any()].tolist()
                for col in missing_cols:
                    if X_processed[col].dtype in [np.float64, np.int64]:
                        # Gunakan median untuk numerik
                        median_val = X_processed[col].median()
                        X_processed[col] = X_processed[col].fillna(median_val)
                        result['preprocessing_steps'].append(f"Filled {col} with median: {median_val}")
                    else:
                        # Drop kolom dengan terlalu banyak missing
                        missing_ratio = X_processed[col].isnull().sum() / len(X_processed)
                        if missing_ratio > 0.5:
                            X_processed = X_processed.drop(columns=[col])
                            result['warnings'].append(f"Dropped column {col} due to high missing ratio: {missing_ratio:.2%}")
                        else:
                            # Fill dengan mode untuk kategorikal
                            mode_val = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'unknown'
                            X_processed[col] = X_processed[col].fillna(mode_val)
                            result['preprocessing_steps'].append(f"Filled {col} with mode: {mode_val}")
        
        # Preprocessing untuk LIME
        elif method == 'lime':
            # LIME lebih fleksibel dengan data kategorikal
            if categorical_cols:
                result['preprocessing_steps'].append(f"Keeping {len(categorical_cols)} categorical columns for LIME (LIME can handle categorical data)")
                result['warnings'].append(f"LIME will process categorical features directly: {categorical_cols}")
            
            # Handle missing values untuk LIME
            if X_processed.isnull().any().any():
                missing_cols = X_processed.columns[X_processed.isnull().any()].tolist()
                for col in missing_cols:
                    missing_ratio = X_processed[col].isnull().sum() / len(X_processed)
                    if missing_ratio > 0.3:
                        # Drop kolom dengan terlalu banyak missing
                        X_processed = X_processed.drop(columns=[col])
                        result['warnings'].append(f"Dropped column {col} due to high missing ratio: {missing_ratio:.2%}")
                    else:
                        if X_processed[col].dtype in [np.float64, np.int64]:
                            # Gunakan median untuk numerik
                            median_val = X_processed[col].median()
                            X_processed[col] = X_processed[col].fillna(median_val)
                            result['preprocessing_steps'].append(f"Filled {col} with median: {median_val}")
                        else:
                            # Fill dengan mode untuk kategorikal
                            mode_val = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'unknown'
                            X_processed[col] = X_processed[col].fillna(mode_val)
                            result['preprocessing_steps'].append(f"Filled {col} with mode: {mode_val}")
        
        # Validasi akhir
        if X_processed.empty:
            result['errors'].append("No valid columns remaining after preprocessing")
            return result
        
        # Check untuk nilai infinite atau sangat besar
        for col in X_processed.columns:
            if X_processed[col].dtype in [np.float64, np.int64]:
                if np.isinf(X_processed[col]).any():
                    X_processed[col] = X_processed[col].replace([np.inf, -np.inf], np.nan)
                    median_val = X_processed[col].median()
                    X_processed[col] = X_processed[col].fillna(median_val)
                    result['preprocessing_steps'].append(f"Handled infinite values in {col}")
        
        result['X_processed'] = X_processed
        result['success'] = True
        result['final_shape'] = X_processed.shape
        result['final_columns'] = X_processed.columns.tolist()
        
    except Exception as e:
        result['errors'].append(f"Preprocessing error: {str(e)}")
        result['success'] = False
    
    return result


def create_interpretation_report(interpretation_result, method='shap', language='id'):
    """
    Membuat laporan interpretasi yang komprehensif
    
    Parameters:
    -----------
    interpretation_result : dict
        Hasil interpretasi dari SHAP atau LIME
    method : str, optional
        Metode yang digunakan ('shap' atau 'lime')
    language : str, optional
        Bahasa untuk laporan ('id' atau 'en')
        
    Returns:
    --------
    dict
        Dictionary berisi laporan interpretasi
    """
    import pandas as pd
    import numpy as np
    
    # Pesan dalam berbagai bahasa
    messages = {
        'id': {
            'report_title': 'Laporan Interpretasi Model',
            'success': 'Berhasil',
            'failed': 'Gagal',
            'features_analyzed': 'Fitur Dianalisis',
            'samples_processed': 'Sampel Diproses',
            'method_used': 'Metode yang Digunakan',
            'model_type': 'Tipe Model',
            'preprocessing_steps': 'Langkah Preprocessing',
            'warnings': 'Peringatan',
            'errors': 'Error',
            'feature_importance': 'Importansi Fitur',
            'top_features': 'Fitur Teratas',
            'recommendations': 'Rekomendasi'
        },
        'en': {
            'report_title': 'Model Interpretation Report',
            'success': 'Success',
            'failed': 'Failed',
            'features_analyzed': 'Features Analyzed',
            'samples_processed': 'Samples Processed',
            'method_used': 'Method Used',
            'model_type': 'Model Type',
            'preprocessing_steps': 'Preprocessing Steps',
            'warnings': 'Warnings',
            'errors': 'Errors',
            'feature_importance': 'Feature Importance',
            'top_features': 'Top Features',
            'recommendations': 'Recommendations'
        }
    }
    
    lang = messages.get(language, messages['id'])
    
    report = {
        'success': interpretation_result.get('success', False),
        'method': method.upper(),
        'summary': {},
        'details': {},
        'recommendations': []
    }
    
    if interpretation_result.get('success', False):
        # Informasi dasar
        report['summary']['status'] = lang['success']
        report['summary']['method'] = f"{lang['method_used']}: {method.upper()}"
        
        if method == 'shap':
            if 'feature_importance' in interpretation_result:
                importance_data = interpretation_result['feature_importance']
                if importance_data:
                    df_importance = pd.DataFrame(importance_data, columns=['Feature', 'Importance'])
                    report['details'][lang['feature_importance']] = df_importance
                    
                    # Top 5 features
                    top_features = df_importance.head(5)
                    report['details'][lang['top_features']] = top_features
                    
                    # Rekomendasi berdasarkan hasil
                    if len(top_features) > 0:
                        top_feature = top_features.iloc[0]['Feature']
                        report['recommendations'].append(f"Fokus pada fitur '{top_feature}' yang memiliki importance tertinggi")
                        
                        if method == 'shap':
                            report['recommendations'].append("Pertimbangkan untuk menggunakan TreeExplainer jika model berbasis pohon")
                            report['recommendations'].append("Validasi SHAP values dengan domain knowledge")
        
        elif method == 'lime':
            if 'explanations' in interpretation_result:
                explanations = interpretation_result['explanations']
                successful_exps = [exp for exp in explanations if exp.get('explanation') is not None]
                failed_exps = [exp for exp in explanations if exp.get('error') is not None]
                
                report['summary'][lang['samples_processed']] = f"{len(successful_exps)} successful, {len(failed_exps)} failed"
                
                if failed_exps > 0:
                    report['recommendations'].append("Periksa kembali data input dan model compatibility")
                    report['recommendations'].append("Pertimbangkan untuk preprocessing data yang lebih baik")
                
                if successful_exps > 0:
                    # Analisis feature importance dari LIME explanations
                    feature_scores = {}
                    for exp in successful_exps:
                        if exp.get('explanation'):
                            for feature in exp['explanation'].as_list():
                                feature_name = feature[0]
                                score = abs(feature[1])
                                if feature_name not in feature_scores:
                                    feature_scores[feature_name] = []
                                feature_scores[feature_name].append(score)
                    
                    # Rata-rata score per feature
                    avg_scores = {feat: np.mean(scores) for feat, scores in feature_scores.items()}
                    sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    if sorted_features:
                        report['details'][lang['feature_importance']] = pd.DataFrame(
                            sorted_features[:10], 
                            columns=['Feature', 'Average Importance']
                        )
                        
                        report['recommendations'].append(f"Fitur '{sorted_features[0][0]}' paling berpengaruh berdasarkan analisis LIME")
    else:
        report['summary']['status'] = lang['failed']
        if 'error' in interpretation_result:
            report['details'][lang['errors']] = interpretation_result['error']
            report['recommendations'].append("Periksa kembali model dan data yang digunakan")
            report['recommendations'].append("Pastikan semua library yang diperlukan sudah terinstall")
    
    return report
