import streamlit as st
import pandas as pd
import numpy as np
import json

                if selected_preset != "None":
                    preset_params = get_preset_params(model_name, selected_preset)
                    summary = create_preset_summary(model_name, selected_preset, preset_params)
                    st.markdown(summary)
        
        st.markdown("---")
    
    # Tambahkan fitur impor/ekspor preset
    with st.expander("💾 Impor/Ekspor Preset" if st_session.language == 'id' else "💾 Import/Export Presets"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Impor Preset" if st_session.language == 'id' else "Import Preset")
            uploaded_file = st.file_uploader(
                "Pilih file JSON preset:" if st_session.language == 'id' else "Choose preset JSON file:",
                type=['json'],
                key=f"preset_import_{model_name}"
            )
            if uploaded_file is not None:
                try:
                    imported_data = json.load(uploaded_file)
                    if st.button("Impor" if st_session.language == 'id' else "Import"):
                        if all(key in imported_data for key in ["model_type", "preset_name", "parameters"]):
                            custom_ranges.update(imported_data["parameters"])
                            st.success(f"Preset '{imported_data['preset_name']}' berhasil diimpor!" if st_session.language == 'id' else f"Preset '{imported_data['preset_name']}' successfully imported!")
                except Exception as e:
                    st.error(f"Error mengimpor preset: {str(e)}" if st_session.language == 'id' else f"Error importing preset: {str(e)}")
        
        with col2:
            st.subheader("Ekspor Preset" if st_session.language == 'id' else "Export Preset")
            if custom_ranges:
                preset_name_export = st.text_input(
                    "Nama preset untuk diekspor:" if st_session.language == 'id' else "Preset name to export:",
                    value=f"custom_{model_name.lower()}",
                    key=f"preset_export_name_{model_name}"
                )
                if st.button("Ekspor ke JSON" if st_session.language == 'id' else "Export to JSON"):
                    filename = export_preset_to_json(model_name, preset_name_export, custom_ranges)
                    if filename:
                        with open(filename, 'rb') as f:
                            st.download_button(
                                label="Unduh File JSON" if st_session.language == 'id' else "Download JSON File",
                                data=f.read(),
                                file_name=filename,
                                mime="application/json"
                            )
                        st.success(f"Preset berhasil diekspor ke {filename}!" if st_session.language == 'id' else f"Preset successfully exported to {filename}!")
            else:
                st.info("Tidak ada parameter kustom untuk diekspor" if st_session.language == 'id' else "No custom parameters to export")
        
        # Model-specific parameter ranges
        if model_name == "Random Forest":
            st_session.subheader("Rentang Parameter Kustom - Random Forest" if st_session.language == 'id' else "Custom Parameter Ranges - Random Forest")
            
            col1, col2 = st_session.columns(2)
            with col1:
                n_estimators_range = st_session.text_input(
                    "Jumlah pohon (n_estimators):" if st_session.language == 'id' else "Number of trees (n_estimators):",
                    placeholder="50:300:50",
                    help="Format: min:max:step" if st_session.language == 'id' else "Format: min:max:step"
                )
                max_depth_range = st_session.text_input(
                    "Kedalaman maksimum (max_depth):" if st_session.language == 'id' else "Maximum depth (max_depth):",
                    placeholder="3:20:1",
                    help="Format: min:max:step" if st_session.language == 'id' else "Format: min:max:step"
                )
                min_samples_split_range = st_session.text_input(
                    "Min samples split:" if st_session.language == 'id' else "Min samples split:",
                    placeholder="2:10:1"
                )
            with col2:
                min_samples_leaf_range = st_session.text_input(
                    "Min samples leaf:" if st_session.language == 'id' else "Min samples leaf:",
                    placeholder="1:5:1"
                )
                max_features_range = st_session.text_input(
                    "Max features:" if st_session.language == 'id' else "Max features:",
                    placeholder="sqrt,log2,None",
                    help="Pisahkan dengan koma" if st_session.language == 'id' else "Separate with commas"
                )
                bootstrap_range = st_session.text_input(
                    "Bootstrap:" if st_session.language == 'id' else "Bootstrap:",
                    placeholder="True,False"
                )
            
            # Parse ranges
            if n_estimators_range:
                custom_ranges['n_estimators'] = parse_custom_range(n_estimators_range, 'int')
            if max_depth_range:
                custom_ranges['max_depth'] = parse_custom_range(max_depth_range, 'int')
            if min_samples_split_range:
                custom_ranges['min_samples_split'] = parse_custom_range(min_samples_split_range, 'int')
            if min_samples_leaf_range:
                custom_ranges['min_samples_leaf'] = parse_custom_range(min_samples_leaf_range, 'int')
            if max_features_range:
                custom_ranges['max_features'] = parse_custom_range(max_features_range, 'categorical')
            if bootstrap_range:
                custom_ranges['bootstrap'] = parse_custom_range(bootstrap_range, 'categorical')
                
        return custom_ranges

def merge_custom_param_ranges(default_param_grid, custom_param_ranges):
    """
    Menggabungkan parameter grid default dengan custom parameter ranges
    
    Parameters:
    -----------
    default_param_grid : dict
        Parameter grid default dari aplikasi
    custom_param_ranges : dict
        Custom parameter ranges dari user input
        
    Returns:
    --------
    dict: Parameter grid yang sudah digabung
    """
    if custom_param_ranges is None or not custom_param_ranges:
        return default_param_grid
    
    # Salin parameter grid default
    merged_grid = default_param_grid.copy()
    
    # Override dengan custom parameter ranges
    for param, custom_range in custom_param_ranges.items():
        if custom_range is not None and len(custom_range) > 0:
            merged_grid[param] = custom_range
    
    return merged_grid

def validate_param_ranges(param_grid, X_train, model_type):
    """
    Validasi parameter ranges berdasarkan karakteristik data
    
    Parameters:
    -----------
    param_grid : dict
        Parameter grid yang akan divalidasi
    X_train : pandas.DataFrame
        Data training untuk validasi
    model_type : str
        Tipe model untuk validasi khusus
        
    Returns:
    --------
    dict: Parameter grid yang sudah divalidasi
    """
    if X_train is None or X_train.empty:
        return param_grid
    
    validated_grid = param_grid.copy()
    n_features = X_train.shape[1]
    n_samples = len(X_train)
    
    # Validasi untuk max_depth (Random Forest, Decision Tree, Gradient Boosting)
    if 'max_depth' in validated_grid and model_type in ['Random Forest', 'Decision Tree', 'Gradient Boosting']:
        max_possible_depth = int(np.log2(n_samples)) if n_samples > 1 else 1
        validated_max_depth = []
        for depth in validated_grid['max_depth']:
            if isinstance(depth, (int, float)) and depth > 0:
                if depth <= max_possible_depth:
                    validated_max_depth.append(int(depth))
                else:
                    validated_max_depth.append(max_possible_depth)
            elif depth is None:
                validated_max_depth.append(depth)
        validated_grid['max_depth'] = validated_max_depth
    
    # Validasi untuk max_features (Random Forest, Gradient Boosting)
    if 'max_features' in validated_grid and model_type in ['Random Forest', 'Gradient Boosting']:
        validated_max_features = []
        for feature in validated_grid['max_features']:
            if isinstance(feature, (int, float)) and feature > 0:
                if feature <= n_features:
                    validated_max_features.append(int(feature))
                else:
                    validated_max_features.append(n_features)
            else:
                validated_max_features.append(feature)
        validated_grid['max_features'] = validated_max_features
    
    # Validasi untuk n_neighbors (KNN)
    if 'n_neighbors' in validated_grid and model_type == 'KNN':
        max_neighbors = min(n_samples - 1, 50)  # Batasi maksimal 50 atau n_samples-1
        validated_n_neighbors = []
        for k in validated_grid['n_neighbors']:
            if isinstance(k, (int, float)) and k > 0:
                if k <= max_neighbors:
                    validated_n_neighbors.append(int(k))
                else:
                    validated_n_neighbors.append(max_neighbors)
        validated_grid['n_neighbors'] = validated_n_neighbors
    
    # Validasi untuk min_samples_split dan min_samples_leaf
    if 'min_samples_split' in validated_grid:
        min_samples = 2
        max_samples = max(2, n_samples // 10)  # Maksimal 10% dari data
        validated_min_samples = []
        for samples in validated_grid['min_samples_split']:
            if isinstance(samples, (int, float)) and samples >= 1:
                if samples <= max_samples:
                    validated_min_samples.append(int(samples))
                else:
                    validated_min_samples.append(max_samples)
        validated_grid['min_samples_split'] = validated_min_samples
    
    if 'min_samples_leaf' in validated_grid:
        max_samples = max(1, n_samples // 20)  # Maksimal 5% dari data
        validated_min_samples = []
        for samples in validated_grid['min_samples_leaf']:
            if isinstance(samples, (int, float)) and samples >= 1:
                if samples <= max_samples:
                    validated_min_samples.append(int(samples))
                else:
                    validated_min_samples.append(max_samples)
        validated_grid['min_samples_leaf'] = validated_min_samples
    return validated_grid



def recommend_research_methods(data):
    """Rekomendasikan metode penelitian berdasarkan karakteristik dataset"""
    recommendations = []
    
    # Analisis karakteristik dataset
    n_rows, n_cols = data.shape
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    missing_values = data.isnull().sum().sum()
    missing_percentage = (missing_values / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
    
    # Rekomendasi berdasarkan ukuran dataset
    if n_rows < 100:
        recommendations.append({
            'type': 'warning',
            'title': 'Dataset Kecil' if st.session_state.language == 'id' else 'Small Dataset',
            'description': 'Gunakan Leave-One-Out Cross Validation atau k-fold dengan k yang besar untuk validasi model.' if st.session_state.language == 'id' else 'Use Leave-One-Out Cross Validation or k-fold with large k for model validation.',
            'methods': ['Leave-One-Out CV', 'Stratified K-Fold (k=5-10)', 'Simple models (Logistic/Linear Regression)']
        })
    elif 100 <= n_rows < 1000:
        recommendations.append({
            'type': 'info',
            'title': 'Dataset Sedang' if st.session_state.language == 'id' else 'Medium Dataset',
            'description': 'Gunakan Stratified K-Fold Cross Validation dengan k=5 atau 10 untuk hasil yang stabil.' if st.session_state.language == 'id' else 'Use Stratified K-Fold Cross Validation with k=5 or 10 for stable results.',
            'methods': ['Stratified K-Fold CV', 'Grid Search CV', 'Random Forest, SVM, Neural Networks']
        })
    else:
        recommendations.append({
            'type': 'success',
            'title': 'Dataset Besar' if st.session_state.language == 'id' else 'Large Dataset',
            'description': 'Dataset cukup besar untuk deep learning dan ensemble methods yang kompleks.' if st.session_state.language == 'id' else 'Dataset is large enough for complex deep learning and ensemble methods.',
            'methods': ['K-Fold CV', 'Hold-out Validation', 'Deep Learning, Gradient Boosting, XGBoost']
        })
    
    # Rekomendasi berdasarkan missing values
    if missing_percentage > 20:
        recommendations.append({
            'type': 'warning',
            'title': 'Banyak Missing Values' if st.session_state.language == 'id' else 'High Missing Values',
            'description': f'Terdapat {missing_percentage:.1f}% missing values. Pertimbangkan imputasi atau analisis sensitivitas.' if st.session_state.language == 'id' else f'There are {missing_percentage:.1f}% missing values. Consider imputation or sensitivity analysis.',
            'methods': ['Multiple Imputation', 'KNN Imputation', 'Missing Indicator Features', 'Tree-based Models']
        })
    elif missing_percentage > 5:
        recommendations.append({
            'type': 'info',
            'title': 'Missing Values Moderat' if st.session_state.language == 'id' else 'Moderate Missing Values',
            'description': f'Terdapat {missing_percentage:.1f}% missing values. Gunakan imputasi yang sesuai.' if st.session_state.language == 'id' else f'There are {missing_percentage:.1f}% missing values. Use appropriate imputation.',
            'methods': ['Mean/Median Imputation', 'KNN Imputation', 'MICE', 'Model-based Imputation']
        })
    
    # Rekomendasi berdasarkan jenis data
    if len(categorical_cols) > len(numerical_cols):
        recommendations.append({
            'type': 'info',
            'title': 'Data Dominan Kategorikal' if st.session_state.language == 'id' else 'Categorical Dominant Data',
            'description': 'Dataset memiliki lebih banyak fitur kategorikal. Gunakan encoding yang tepat.' if st.session_state.language == 'id' else 'Dataset has more categorical features. Use appropriate encoding.',
            'methods': ['One-Hot Encoding', 'Target Encoding', 'Ordinal Encoding', 'Tree-based Models']
        })
    elif len(numerical_cols) > len(categorical_cols):
        recommendations.append({
            'type': 'info',
            'title': 'Data Dominan Numerik' if st.session_state.language == 'id' else 'Numerical Dominant Data',
            'description': 'Dataset memiliki lebih banyak fitur numerik. Scaling mungkin diperlukan.' if st.session_state.language == 'id' else 'Dataset has more numerical features. Scaling may be needed.',
            'methods': ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PCA for Dimensionality Reduction']
        })
    
    # Rekomendasi berdasarkan jumlah fitur
    if n_cols > 50:
        recommendations.append({
            'type': 'warning',
            'title': 'Dimensi Tinggi' if st.session_state.language == 'id' else 'High Dimensionality',
            'description': f'Terdapat {n_cols} fitur. Pertimbangkan reduksi dimensi untuk menghindari overfitting.' if st.session_state.language == 'id' else f'There are {n_cols} features. Consider dimensionality reduction to avoid overfitting.',
            'methods': ['PCA', 'Feature Selection (RFE, SelectKBest)', 'L1 Regularization', 'Autoencoders']
        })
    elif n_cols > 20:
        recommendations.append({
            'type': 'info',
            'title': 'Banyak Fitur' if st.session_state.language == 'id' else 'Many Features',
            'description': f'Terdapat {n_cols} fitur. Gunakan feature selection untuk meningkatkan performa.' if st.session_state.language == 'id' else f'There are {n_cols} features. Use feature selection to improve performance.',
            'methods': ['Feature Importance', 'Recursive Feature Elimination', 'Mutual Information', 'Correlation Analysis']
        })
    
    # Rekomendasi umum
    recommendations.append({
        'type': 'success',
        'title': 'Langkah Selanjutnya' if st.session_state.language == 'id' else 'Next Steps',
        'description': 'Ikuti langkah-langkah berikut untuk analisis yang komprehensif.' if st.session_state.language == 'id' else 'Follow these steps for comprehensive analysis.',
        'methods': [
            '1. Exploratory Data Analysis (EDA)',
            '2. Preprocessing & Feature Engineering',
            '3. Model Training & Cross Validation',
            '4. Model Interpretation (SHAP/LIME)',
            '5. Hyperparameter Tuning',
            '6. Final Model Evaluation'
        ]
    })
    
    return recommendations

def analyze_dataset_with_ai(data, analysis_type='comprehensive'):
    """
    Enhanced AI-powered dataset analysis using rule-based agentic system
    that provides intelligent recommendations based on dataset characteristics
    """
    recommendations = []
    
    # Basic dataset characteristics
    n_rows, n_cols = data.shape
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Advanced analysis
    missing_values = data.isnull().sum().sum()
    missing_percentage = (missing_values / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
    duplicate_rows = data.duplicated().sum()
    
    # Statistical analysis
    if len(numerical_cols) > 0:
        numerical_data = data[numerical_cols]
        skewness_values = numerical_data.skew().abs().mean()
        kurtosis_values = numerical_data.kurtosis().abs().mean()
        outlier_percentage = ((numerical_data > numerical_data.quantile(0.95)).any(axis=1).sum() / n_rows) * 100
    else:
        skewness_values = 0
        kurtosis_values = 0
        outlier_percentage = 0
    
    # AI-powered analysis based on dataset patterns
    
    # 1. Dataset Quality Assessment
    quality_score = 100
    quality_issues = []
    
    if missing_percentage > 10:
        quality_score -= 20
        quality_issues.append("High missing values")
    if duplicate_rows > n_rows * 0.05:
        quality_score -= 15
        quality_issues.append("Significant duplicates")
    if outlier_percentage > 10:
        quality_score -= 10
        quality_issues.append("High outlier presence")
    if n_rows < 50:
        quality_score -= 25
        quality_issues.append("Very small dataset")
    
    recommendations.append({
        'type': 'success' if quality_score >= 80 else 'warning' if quality_score >= 60 else 'error',
        'title': 'Dataset Quality Assessment' if st.session_state.language == 'id' else 'Dataset Quality Assessment',
        'description': f"Dataset Quality Score: {quality_score:.1f}/100. {'Excellent quality' if quality_score >= 80 else 'Good quality with minor issues' if quality_score >= 60 else 'Requires significant preprocessing'}.",
        'details': quality_issues,
        'priority': 'high'
    })
    
    # 2. Optimal ML Algorithm Recommendations
    ml_recommendations = []
    
    # Rule-based algorithm selection
    if n_rows < 100:
        ml_recommendations.extend(['Naive Bayes', 'Logistic Regression', 'Decision Tree'])
    elif n_rows < 1000:
        if len(categorical_cols) > len(numerical_cols):
            ml_recommendations.extend(['Random Forest', 'XGBoost', 'LightGBM'])
        else:
            ml_recommendations.extend(['SVM', 'Random Forest', 'Neural Network'])
    else:
        if n_cols > 50:
            ml_recommendations.extend(['XGBoost', 'Deep Learning', 'Ensemble Methods'])
        else:
            ml_recommendations.extend(['Gradient Boosting', 'Neural Networks', 'Stacking Ensemble'])
    
    # Special cases
    if missing_percentage > 30:
        ml_recommendations = ['Random Forest', 'XGBoost', 'KNN Imputation + Any Model']
    
    if len(datetime_cols) > 0:
        ml_recommendations.append('Time Series Models (ARIMA, Prophet, LSTM)')
    
    recommendations.append({
        'type': 'info',
        'title': 'AI-Recommended ML Algorithms' if st.session_state.language == 'id' else 'AI-Recommended ML Algorithms',
        'description': 'Optimal algorithms based on dataset characteristics and proven ML practices.' if st.session_state.language == 'id' else 'Optimal algorithms based on dataset characteristics and proven ML practices.',
        'methods': ml_recommendations,
        'priority': 'high'
    })
    
    # 3. Feature Engineering Recommendations
    feature_recommendations = []
    
    if len(numerical_cols) > 0:
        if skewness_values > 1:
            feature_recommendations.append('Log transformation for skewed features')
        if outlier_percentage > 5:
            feature_recommendations.append('Outlier treatment (Winsorization/IQR)')
        feature_recommendations.append('StandardScaler/MinMaxScaler normalization')
    
    if len(categorical_cols) > 0:
        if len(categorical_cols) < 10:
            feature_recommendations.append('One-Hot Encoding for categorical variables')
        else:
            feature_recommendations.append('Target/Frequency Encoding for high-cardinality categoricals')
    
    if n_rows > 1000 and len(numerical_cols) > 5:
        feature_recommendations.append('Polynomial features for non-linear relationships')
        feature_recommendations.append('Feature interaction terms')
    
    recommendations.append({
        'type': 'info',
        'title': 'Smart Feature Engineering' if st.session_state.language == 'id' else 'Smart Feature Engineering',
        'description': 'Automated feature engineering suggestions based on data patterns.' if st.session_state.language == 'id' else 'Automated feature engineering suggestions based on data patterns.',
        'methods': feature_recommendations,
        'priority': 'medium'
    })
    
    # 4. Cross-Validation Strategy
    cv_recommendations = []
    
    if n_rows < 100:
        cv_recommendations.append('Leave-One-Out Cross-Validation (LOOCV)')
        cv_recommendations.append('Stratified K-Fold with k=5')
    elif n_rows < 1000:
        cv_recommendations.append('Stratified K-Fold with k=5 or 10')
        cv_recommendations.append('Repeated Stratified K-Fold (n_repeats=3)')
    else:
        cv_recommendations.append('Stratified K-Fold with k=5')
        cv_recommendations.append('Hold-out validation (80-20 split)')
        cv_recommendations.append('Time-based split for temporal data')
    
    if missing_percentage > 20:
        cv_recommendations.append('Nested cross-validation for hyperparameter tuning')
    
    recommendations.append({
        'type': 'info',
        'title': 'Optimal Cross-Validation Strategy' if st.session_state.language == 'id' else 'Optimal Cross-Validation Strategy',
        'description': 'Best validation approach based on dataset size and characteristics.' if st.session_state.language == 'id' else 'Best validation approach based on dataset size and characteristics.',
        'methods': cv_recommendations,
        'priority': 'medium'
    })
    
    # 5. Advanced Analytics Recommendations
    advanced_recommendations = []
    
    if len(numerical_cols) > 2:
        advanced_recommendations.append('Principal Component Analysis (PCA)')
        advanced_recommendations.append('Clustering analysis (K-means, DBSCAN)')
    
    if len(categorical_cols) > 1:
        advanced_recommendations.append('Association Rule Mining')
        advanced_recommendations.append('Chi-square test for independence')
    
    if n_rows > 500 and len(numerical_cols) > 3:
        advanced_recommendations.append('SHAP values for model interpretability')
        advanced_recommendations.append('Partial Dependence Plots (PDP)')
    
    if duplicate_rows > 0:
        advanced_recommendations.append('Anomaly detection for data quality')
    
    recommendations.append({
        'type': 'info',
        'title': 'Advanced Analytics' if st.session_state.language == 'id' else 'Advanced Analytics',
        'description': 'Sophisticated analysis techniques for deeper insights.' if st.session_state.language == 'id' else 'Sophisticated analysis techniques for deeper insights.',
        'methods': advanced_recommendations,
        'priority': 'low'
    })
    
    # 6. Problem-Specific Recommendations
    if analysis_type == 'classification':
        class_recommendations = []
        if len(categorical_cols) > 0:
            target_col = categorical_cols[0] if len(categorical_cols) > 0 else None
            if target_col and data[target_col].nunique() == 2:
                class_recommendations.append('Binary classification metrics (Precision, Recall, F1, AUC)')
            elif target_col:
                class_recommendations.append('Multi-class classification metrics (Accuracy, F1-macro, Cohen\'s Kappa)')
        
        recommendations.append({
            'type': 'info',
            'title': 'Classification-Specific Recommendations' if st.session_state.language == 'id' else 'Classification-Specific Recommendations',
            'description': 'Specialized techniques for classification problems.' if st.session_state.language == 'id' else 'Specialized techniques for classification problems.',
            'methods': class_recommendations,
            'priority': 'high'
        })
    
    elif analysis_type == 'regression':
        reg_recommendations = []
        if len(numerical_cols) > 0:
            reg_recommendations.append('Regression metrics (RMSE, MAE, R², Adjusted R²)')
            reg_recommendations.append('Residual analysis and diagnostics')
            reg_recommendations.append('Feature scaling for regularization methods')
        
        recommendations.append({
            'type': 'info',
            'title': 'Regression-Specific Recommendations' if st.session_state.language == 'id' else 'Regression-Specific Recommendations',
            'description': 'Specialized techniques for regression problems.' if st.session_state.language == 'id' else 'Specialized techniques for regression problems.',
            'methods': reg_recommendations,
            'priority': 'high'
        })
    
    return recommendations

def create_agentic_ai_analysis(data, analysis_type='comprehensive', language='id'):
    """
    Create an agentic AI analysis that simulates intelligent reasoning about the dataset
    """
    agent_analysis = {
        'agent_name': 'DataScience-AI-Agent',
        'analysis_timestamp': pd.Timestamp.now(),
        'dataset_summary': {},
        'intelligent_insights': [],
        'actionable_recommendations': [],
        'risk_assessment': [],
        'success_probability': 0
    }
    
    # Dataset summary
    n_rows, n_cols = data.shape
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    agent_analysis['dataset_summary'] = {
        'total_rows': n_rows,
        'total_columns': n_cols,
        'numerical_features': len(numerical_cols),
        'categorical_features': len(categorical_cols),
        'data_density': ((data.size - data.isnull().sum().sum()) / data.size) * 100,
        'complexity_score': min(100, (n_cols * 2) + (n_rows / 1000))
    }
    
    # Intelligent insights
    insights = []
    
    # Language-specific messages
    if language == 'id':
        success_msg = 'Dataset sangat cocok untuk machine learning dengan jumlah sampel dan fitur yang memadai.'
        adequate_msg = 'Dataset cukup untuk tugas ML dasar tetapi dapat diperbaiki dengan rekayasa fitur.'
        small_msg = 'Dataset mungkin terlalu kecil untuk hasil machine learning yang dapat diandalkan.'
        low_complexity_msg = 'Kompleksitas dataset rendah - model sederhana direkomendasikan.'
        moderate_complexity_msg = 'Kompleksitas sedang - metode ensemble direkomendasikan.'
        high_complexity_msg = 'Kompleksitas tinggi - teknik lanjutan diperlukan.'
    else:
        success_msg = 'Dataset is well-suited for machine learning with sufficient samples and features.'
        adequate_msg = 'Dataset is adequate for basic ML tasks but may benefit from feature engineering.'
        small_msg = 'Dataset may be too small for reliable machine learning results.'
        low_complexity_msg = 'Dataset complexity is low - simple models are recommended.'
        moderate_complexity_msg = 'Moderate complexity - ensemble methods are recommended.'
        high_complexity_msg = 'High complexity - advanced techniques required.'
    
    # Insight 1: Dataset readiness
    if n_rows >= 1000 and n_cols >= 5:
        insights.append({
            'type': 'success',
            'insight': success_msg,
            'confidence': 0.95,
            'evidence': f'{n_rows} sampel dan {n_cols} fitur memberikan kekuatan statistik yang baik.' if language == 'id' else f'{n_rows} samples and {n_cols} features provide good statistical power.'
        })
    elif n_rows >= 100 and n_cols >= 3:
        insights.append({
            'type': 'info',
            'insight': adequate_msg,
            'confidence': 0.80,
            'evidence': f'{n_rows} sampel dan {n_cols} fitur cukup untuk model sederhana.' if language == 'id' else f'{n_rows} samples and {n_cols} features are sufficient for simple models.'
        })
    else:
        insights.append({
            'type': 'warning',
            'insight': small_msg,
            'confidence': 0.90,
            'evidence': f'Hanya {n_rows} sampel dan {n_cols} fitur dapat menyebabkan overfitting.' if language == 'id' else f'Only {n_rows} samples and {n_cols} features may lead to overfitting.'
        })
    
    # Insight 2: Feature quality
    if len(numerical_cols) > 0:
        missing_pct = (data[numerical_cols].isnull().sum().sum() / data[numerical_cols].size) * 100
        if missing_pct < 5:
            insights.append({
                'type': 'success',
                'insight': 'Fitur numerik memiliki kelengkapan yang sangat baik.' if language == 'id' else 'Numerical features have excellent completeness.',
                'confidence': 0.90,
                'evidence': f'Hanya {missing_pct:.1f}% nilai yang hilang di kolom numerik.' if language == 'id' else f'Only {missing_pct:.1f}% missing values in numerical columns.'
            })
        elif missing_pct < 20:
            insights.append({
                'type': 'info',
                'insight': 'Fitur numerik memiliki kelengkapan yang dapat diterima dengan nilai hilang yang dapat dikelola.' if language == 'id' else 'Numerical features have acceptable completeness with manageable missing values.',
                'confidence': 0.85,
                'evidence': f'{missing_pct:.1f}% nilai hilang dapat ditangani secara efektif dengan imputasi.' if language == 'id' else f'{missing_pct:.1f}% missing values can be effectively handled with imputation.'
            })
    
    # Insight 3: Model complexity recommendation
    complexity_factors = []
    if n_cols > 20:
        complexity_factors.append('Dimensi tinggi' if language == 'id' else 'High dimensionality')
    if missing_pct > 15:
        complexity_factors.append('Data hilang signifikan' if language == 'id' else 'Significant missing data')
    if n_rows > 10000:
        complexity_factors.append('Ukuran dataset besar' if language == 'id' else 'Large dataset size')
    
    if len(complexity_factors) == 0:
        insights.append({
            'type': 'success',
            'insight': low_complexity_msg,
            'confidence': 0.90,
            'evidence': 'Data bersih dan terstruktur dengan baik, cocok untuk model yang dapat ditafsirkan.' if language == 'id' else 'Clean, well-structured data suitable for interpretable models.'
        })
    elif len(complexity_factors) <= 2:
        insights.append({
            'type': 'info',
            'insight': moderate_complexity_msg,
            'confidence': 0.85,
            'evidence': f'Faktor: {", ".join(complexity_factors)}' if language == 'id' else f'Factors: {", ".join(complexity_factors)}'
        })
    else:
        insights.append({
            'type': 'warning',
            'insight': high_complexity_msg,
            'confidence': 0.80,
            'evidence': f'Beberapa faktor kompleksitas: {", ".join(complexity_factors)}' if language == 'id' else f'Multiple complexity factors: {", ".join(complexity_factors)}'
        })
    
    agent_analysis['intelligent_insights'] = insights
    
    # Actionable recommendations
    recommendations = []
    
    # Language-specific recommendation messages
    if language == 'id':
        imp_action = 'Terapkan strategi imputasi lanjutan'
        scaling_action = 'Terapkan penskalaan dan transformasi fitur'
        ensemble_action = 'Gunakan metode ensemble dengan penyetelan hyperparameter'
        interpretable_action = 'Mulai dengan model yang dapat ditafsirkan'
        
        imp_rationale = f'{missing_pct:.1f}% nilai hilang memerlukan penanganan yang canggih'
        scaling_rationale = 'Beberapa fitur numerik mendapat manfaat dari standarisasi'
        ensemble_rationale = 'Dataset besar dapat mendukung model kompleks dengan validasi yang tepat'
        interpretable_rationale = 'Ukuran dataset sedang cocok untuk pendekatan yang seimbang'
        
        imp_implementation = 'Gunakan imputasi iteratif atau metode imputasi berbasis model'
        scaling_implementation = 'Gunakan StandardScaler dan pertimbangkan transformasi log untuk fitur yang miring'
        ensemble_implementation = 'Terapkan Random Forest atau XGBoost dengan optimasi Optuna'
        interpretable_implementation = 'Gunakan Regresi Logistik atau Pohon Keputusan dengan validasi silang'
        
        imp_impact = 'Tingkatkan performa model sebesar 15-25%'
        scaling_impact = 'Tingkatkan konvergensi dan performa model'
        ensemble_impact = 'Capai akurasi prediksi 85-95%'
        interpretable_impact = 'Capai akurasi prediksi 75-85% dengan dapat ditafsirkan'
    else:
        imp_action = 'Implement advanced imputation strategy'
        scaling_action = 'Apply feature scaling and transformation'
        ensemble_action = 'Use ensemble methods with hyperparameter tuning'
        interpretable_action = 'Start with interpretable models'
        
        imp_rationale = f'{missing_pct:.1f}% missing values require sophisticated handling'
        scaling_rationale = 'Multiple numerical features benefit from standardization'
        ensemble_rationale = 'Large dataset can support complex models with proper validation'
        interpretable_rationale = 'Moderate dataset size suitable for balanced approach'
        
        imp_implementation = 'Use iterative imputation or model-based imputation methods'
        scaling_implementation = 'Use StandardScaler and consider log transformations for skewed features'
        ensemble_implementation = 'Implement Random Forest or XGBoost with Optuna optimization'
        interpretable_implementation = 'Use Logistic Regression or Decision Tree with cross-validation'
        
        imp_impact = 'Improve model performance by 15-25%'
        scaling_impact = 'Improve model convergence and performance'
        ensemble_impact = 'Achieve 85-95% prediction accuracy'
        interpretable_impact = 'Achieve 75-85% prediction accuracy with interpretability'
    
    # Recommendation 1: Preprocessing strategy
    if missing_pct > 10:
        recommendations.append({
            'action': imp_action,
            'priority': 'high',
            'rationale': imp_rationale,
            'implementation': imp_implementation,
            'expected_impact': imp_impact
        })
    
    # Recommendation 2: Feature engineering
    if len(numerical_cols) > 3:
        recommendations.append({
            'action': scaling_action,
            'priority': 'medium',
            'rationale': scaling_rationale,
            'implementation': scaling_implementation,
            'expected_impact': scaling_impact
        })
    
    # Recommendation 3: Model selection
    if n_rows > 1000 and n_cols > 10:
        recommendations.append({
            'action': ensemble_action,
            'priority': 'high',
            'rationale': ensemble_rationale,
            'implementation': ensemble_implementation,
            'expected_impact': ensemble_impact
        })
    elif n_rows > 100:
        recommendations.append({
            'action': interpretable_action,
            'priority': 'medium',
            'rationale': interpretable_rationale,
            'implementation': interpretable_implementation,
            'expected_impact': interpretable_impact
        })
    
    agent_analysis['actionable_recommendations'] = recommendations
    
    # Risk assessment
    risks = []
    
    # Language-specific risk messages
    if language == 'id':
        overfitting_risk = 'Risiko overfitting tinggi'
        missing_bias_risk = 'Model bias karena pola data hilang'
        dimensionality_risk = 'Kutukan dimensionalitas'
        
        overfitting_mitigation = 'Gunakan model sederhana, regularisasi agresif, dan validasi ekstensif'
        missing_bias_mitigation = 'Analisis pola data hilang dan gunakan beberapa strategi imputasi'
        dimensionality_mitigation = 'Terapkan teknik reduksi dimensionalitas sebelum pemodelan'
    else:
        overfitting_risk = 'High overfitting risk'
        missing_bias_risk = 'Biased model due to missing data patterns'
        dimensionality_risk = 'Curse of dimensionality'
        
        overfitting_mitigation = 'Use simple models, aggressive regularization, and extensive validation'
        missing_bias_mitigation = 'Analyze missing data patterns and use multiple imputation strategies'
        dimensionality_mitigation = 'Apply dimensionality reduction techniques before modeling'
    
    if n_rows < 100:
        risks.append({
            'risk': overfitting_risk,
            'severity': 'high',
            'mitigation': overfitting_mitigation,
            'probability': 0.8
        })
    
    if missing_pct > 20:
        risks.append({
            'risk': missing_bias_risk,
            'severity': 'medium',
            'mitigation': missing_bias_mitigation,
            'probability': 0.6
        })
    
    if n_cols > 50:
        risks.append({
            'risk': dimensionality_risk,
            'severity': 'medium',
            'mitigation': dimensionality_mitigation,
            'probability': 0.7
        })
    
    agent_analysis['risk_assessment'] = risks
    
    # Success probability calculation
    success_factors = []
    if n_rows >= 200:
        success_factors.append(0.2)
    if missing_pct < 15:
        success_factors.append(0.2)
    if n_cols >= 3:
        success_factors.append(0.2)
    if len(complexity_factors) <= 1:
        success_factors.append(0.2)
    if len(risks) <= 1:
        success_factors.append(0.2)
    
    agent_analysis['success_probability'] = min(0.95, sum(success_factors))
    agent_analysis['language'] = language
    
    return agent_analysis

def verify_captcha(input_text, correct_text):
    """Verify captcha input"""
    return input_text.upper().strip() == correct_text.upper().strip()

def verify_captcha(input_text, correct_text):
    """Verify captcha input"""
    return input_text.upper().strip() == correct_text.upper().strip()

