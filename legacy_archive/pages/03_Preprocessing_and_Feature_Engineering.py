import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

if "data" not in st.session_state or st.session_state.data is None:
    st.warning("Silakan unggah data terlebih dahulu di halaman Data Upload.")
    st.stop()

data = st.session_state.data
categorical_cols = st.session_state.categorical_columns
numerical_cols = st.session_state.numerical_columns
st.header("🔄 Pemrosesan Data Awal" if st.session_state.language == 'id' else "🔄 Data Preprocessing")

if st.session_state.data is not None:
    data = st.session_state.data.copy()
    
    st.subheader("🎯 Pilih Variabel Target" if st.session_state.language == 'id' else "🎯 Select Target Variable")
    target_column = st.selectbox("Pilih kolom target untuk diprediksi:" if st.session_state.language == 'id' else "Choose the target column for prediction:", data.columns)
    st.session_state.target_column = target_column
    
    # Determine problem type
    # Check if time series data is available
    time_columns = [col for col in data.columns if any(keyword in str(col).lower() for keyword in ['date', 'time', 'tanggal', 'waktu', 'year', 'month', 'day'])]
    
    if data[target_column].dtype in ['int64', 'float64']:
        if len(data[target_column].unique()) <= 10:
            problem_type_options = ["Classification", "Regression", "Forecasting"] if time_columns else ["Classification", "Regression"]
            problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", problem_type_options, index=0)
        else:
            problem_type_options = ["Classification", "Regression", "Forecasting"] if time_columns else ["Classification", "Regression"]
            problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", problem_type_options, index=1)
    else:
        problem_type_options = ["Classification", "Forecasting"] if time_columns else ["Classification"]
        problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", problem_type_options, index=0)
    
    st.session_state.problem_type = problem_type
    
    st.subheader("🧹 Atasi Nilai Hilang" if st.session_state.language == 'id' else "🧹 Handle Missing Values")
    
    # Update original missing counts when data changes
    current_missing = data.isnull().sum().to_dict()
    if 'original_missing_counts' not in st.session_state or st.session_state.get('last_data_shape') != data.shape:
        st.session_state.original_missing_counts = current_missing.copy()
        st.session_state.last_data_shape = data.shape
    
    # Display columns with missing values
    missing_cols = data.columns[data.isnull().any()].tolist()
    
    if missing_cols:
        st.write("Kolom yang memiliki nilai hilang:" if st.session_state.language == 'id' else "Columns with missing values:", ", ".join(missing_cols))
        
        # Advanced missing value analysis
        st.markdown("### 🔍 Analisis Nilai Hilang Canggih" if st.session_state.language == 'id' else "### 🔍 Advanced Missing Value Analysis")
        
        if st.button("Jalankan Analisis Nilai Hilang" if st.session_state.language == 'id' else "Run Missing Value Analysis", key="advanced_missing_analysis"):
            try:
                from utils import advanced_missing_value_analysis
                analysis_result = advanced_missing_value_analysis(data, target_column, st.session_state.language)
                
                if analysis_result.get('success', True):
                    st.success("Analisis nilai hilang selesai" if st.session_state.language == 'id' else "Missing value analysis completed")
                    
                    # Display analysis results
                    st.write(f"**Total missing values:** {analysis_result['missing_summary']['total_missing']}")
                    st.write(f"**Columns affected:** {analysis_result['missing_summary']['columns_with_missing']}")
                    st.write(f"**Total percentage:** {analysis_result['missing_summary']['total_percentage']:.2f}%")
                    st.write(f"**Missing pattern:** {analysis_result.get('missing_pattern', 'Unknown')}")
                    
                    # Display recommendations
                    if 'recommendations' in analysis_result:
                        st.write("**Rekomendasi:**" if st.session_state.language == 'id' else "**Recommendations:**")
                        for rec in analysis_result['recommendations']:
                            st.write(f"- **{rec['column']}** ({rec['missing_percentage']:.1f}%): {', '.join(rec['strategies'])}")
                else:
                    st.error(f"Error dalam analisis: {analysis_result.get('error', 'Unknown error')}")
            except Exception as e:
                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                    error_result = error_handler.handle_error(e, "Advanced Missing Value Analysis")
                    st.error(f"❌ {error_result['message']}")
                    with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                        st.write(error_result.get('details', str(e)))
                else:
                    st.error(f"Error menjalankan analisis: {str(e)}")
        
        # Auto imputation option
        st.markdown("### ⚙️ Imputasi Otomatis" if st.session_state.language == 'id' else "### ⚙️ Automatic Imputation")
        auto_impute = st.checkbox("Gunakan imputasi otomatis canggih" if st.session_state.language == 'id' else "Use advanced automatic imputation", key="auto_impute_advanced")
        
        # Integrate session state management for preprocessing
        if SESSION_MANAGER_AVAILABLE and session_manager is not None:
            try:
                # Initialize preprocessing session state
                session_manager.initialize_tab_state('preprocessing')
                
                # Store current preprocessing settings
                preprocessing_settings = {
                    'auto_impute': auto_impute,
                    'target_column': target_column,
                    'problem_type': problem_type
                }
                session_manager.update_tab_state('preprocessing', preprocessing_settings)
                
                # Display preprocessing progress
                with st.expander("📊 Status Preprocessing" if st.session_state.language == 'id' else "📊 Preprocessing Status"):
                    tab_state = session_manager.get_tab_state('preprocessing')
                    if tab_state:
                        st.write("**Status:**" if st.session_state.language == 'id' else "**Status:**")
                        st.write(f"- Auto imputasi: {'Aktif' if tab_state.get('auto_impute') else 'Non-aktif'}")
                        st.write(f"- Kolom target: {tab_state.get('target_column', 'Belum dipilih')}")
                        st.write(f"- Tipe masalah: {tab_state.get('problem_type', 'Belum ditentukan')}")
                        
            except Exception as e:
                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                    error_result = error_handler.handle_error(e, "Session Management (Preprocessing)")
                    st.warning(f"⚠️ {error_result['message']}")
                else:
                    st.warning(f"⚠️ Gagal mengelola sesi preprocessing: {str(e)}")
        
        if auto_impute:
            imputation_strategy = st.selectbox(
                "Strategi imputasi:" if st.session_state.language == 'id' else "Imputation strategy:",
                ["auto", "simple", "knn", "iterative", "model"],
                key="imputation_strategy"
            )
            
            if st.button("Terapkan Imputasi Otomatis" if st.session_state.language == 'id' else "Apply Automatic Imputation", key="apply_auto_impute"):
                try:
                    from utils import advanced_missing_value_imputation
                    impute_result = advanced_missing_value_imputation(data, imputation_strategy, target_column, st.session_state.language)
                    
                    if impute_result['success']:
                        st.success(f"Imputasi berhasil menggunakan strategi: {impute_result['strategy_used']}")
                        st.write("**Informasi imputasi:**")
                        for col, info in impute_result['imputation_info'].items():
                            st.write(f"- {col}: {info['method']}")
                        
                        # Update data
                        data = impute_result['data'].copy()
                        st.session_state.data = data.copy()
                        st.rerun()
                    else:
                        st.error(f"Error dalam imputasi: {impute_result.get('error', 'Unknown error')}")
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Advanced Missing Value Imputation")
                        st.error(f"❌ {error_result['message']}")
                        with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                            st.write(error_result.get('details', str(e)))
                    else:
                        st.error(f"Error dalam imputasi otomatis: {str(e)}")
        
        # Manual handling section
        st.markdown("### 🔧 Metode Penanganan Nilai Hilang Manual" if st.session_state.language == 'id' else "### 🔧 Manual Missing Value Handling Methods")
        st.info("Penanganan nilai hilang dilakukan manual satu per satu kolom untuk hasil yang lebih akurat" if st.session_state.language == 'id' else "Missing value handling is done manually one column at a time for more accurate results")
        
        # Individual column handling
        for col in missing_cols:
                col_type = "numerical" if data[col].dtype in ['int64', 'float64'] else "categorical"
                
                # Enhanced data type detection
                def get_column_type(col_data):
                    """Enhanced column type detection"""
                    if col_data.dtype in ['int64', 'float64']:
                        # Check if it's actually categorical (few unique values)
                        unique_ratio = col_data.nunique() / len(col_data)
                        if unique_ratio < 0.05 and col_data.nunique() <= 10:  # Less than 5% unique and <= 10 categories
                            return "categorical"
                        return "numerical"
                    elif col_data.dtype in ['object', 'category']:
                        return "categorical"
                    elif 'datetime' in str(col_data.dtype):
                        return "datetime"
                    else:
                        return "categorical"
                
                col_type = get_column_type(data[col])
                
                with st.expander(f"⚙️ {col} ({col_type})", expanded=False):
                    method = st.selectbox(
                        f"Metode untuk {col}:" if st.session_state.language == 'id' else f"Method for {col}:",
                        [
                            "Drop rows",
                            "Mean",
                            "Median", 
                            "Mode",
                            "Zero",
                            "Min",
                            "Max",
                            "Forward Fill",
                            "Backward Fill",
                            "Linear Interpolation",
                            "Polynomial Interpolation",
                            "KNN Imputation",
                            "Iterative Imputation",
                            "Custom Value"
                        ] if col_type == "numerical" else [
                            "Drop rows",
                            "Mode",
                            "New category",
                            "Forward Fill",
                            "Backward Fill",
                            "Custom Value"
                        ],
                        key=f"method_{col}"
                    )
                    
                    if method == "Linear Interpolation" and col_type == "numerical":
                        data[col] = data[col].interpolate(method='linear')
                    elif method == "Polynomial Interpolation" and col_type == "numerical":
                        order = st.slider("Orde polinomial:" if st.session_state.language == 'id' else "Polynomial order:", 2, 5, 2, key=f"poly_order_{col}")
                        data[col] = data[col].interpolate(method='polynomial', order=order)
                    elif method == "Forward Fill":
                        data[col] = data[col].fillna(method='ffill')
                    elif method == "Backward Fill":
                        data[col] = data[col].fillna(method='bfill')
                    elif method == "KNN Imputation" and col_type == "numerical":
                        try:
                            from sklearn.impute import KNNImputer
                            imputer = KNNImputer(n_neighbors=5)
                            data[[col]] = imputer.fit_transform(data[[col]])
                        except ImportError:
                            st.warning("KNN Imputation memerlukan scikit-learn. Menggunakan median sebagai fallback." if st.session_state.language == 'id' else "KNN Imputation requires scikit-learn. Using median as fallback.")
                            data[col] = data[col].fillna(data[col].median())
                    elif method == "Iterative Imputation" and col_type == "numerical":
                        try:
                            from sklearn.impute import IterativeImputer
                            from sklearn.ensemble import RandomForestRegressor
                            imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42)
                            data[[col]] = imputer.fit_transform(data[[col]])
                        except ImportError:
                            st.warning("Iterative Imputation memerlukan scikit-learn. Menggunakan median sebagai fallback." if st.session_state.language == 'id' else "Iterative Imputation requires scikit-learn. Using median as fallback.")
                            data[col] = data[col].fillna(data[col].median())
                    elif method == "Drop rows":
                        data = data.dropna(subset=[col])
                    elif method == "Mean" and col_type == "numerical":
                        data[col] = data[col].fillna(data[col].mean())
                    elif method == "Median" and col_type == "numerical":
                        data[col] = data[col].fillna(data[col].median())
                    elif method == "Mode":
                        # Better mode handling with validation
                        try:
                            mode_values = data[col].mode()
                            if len(mode_values) > 0:
                                mode_val = mode_values[0]
                            else:
                                # Fallback if no mode found
                                mode_val = 0 if col_type == "numerical" else "Unknown"
                            
                            # Additional validation for numerical data
                            if col_type == "numerical" and pd.isna(mode_val):
                                mode_val = data[col].median() if not data[col].empty else 0
                            
                            data[col] = data[col].fillna(mode_val)
                        except Exception as e:
                            st.error(f"Error in mode imputation for {col}: {str(e)}")
                            # Fallback to safe values
                            fallback_val = 0 if col_type == "numerical" else "Unknown"
                            data[col] = data[col].fillna(fallback_val)
                    elif method == "Zero" and col_type == "numerical":
                        data[col] = data[col].fillna(0)
                    elif method == "Min" and col_type == "numerical":
                        data[col] = data[col].fillna(data[col].min())
                    elif method == "Max" and col_type == "numerical":
                        data[col] = data[col].fillna(data[col].max())
                    elif method == "New category" and col_type != "numerical":
                        data[col] = data[col].fillna("Unknown")
                    elif method == "Custom Value":
                        custom_value = st.text_input("Masukkan nilai kustom:" if st.session_state.language == 'id' else "Enter custom value:", key=f"custom_{col}")
                        if custom_value:
                            try:
                                if col_type == "numerical":
                                    custom_value = float(custom_value)
                                data[col] = data[col].fillna(custom_value)
                            except ValueError:
                                st.error("Nilai kustom tidak valid untuk tipe data ini" if st.session_state.language == 'id' else "Invalid custom value for this data type")
                    
                    # Show preview after handling with validation
                    if st.button("Tampilkan Preview" if st.session_state.language == 'id' else "Show Preview", key=f"preview_{col}"):
                        missing_after = data[col].isnull().sum()
                        original_missing = st.session_state.original_missing_counts.get(col, 0)
                        imputed_count = original_missing - missing_after
                        
                        # Calculate success rate
                        success_rate = (imputed_count / original_missing * 100) if original_missing > 0 else 0
                        
                        if missing_after == 0:
                            st.success(f"✅ Semua nilai hilang di {col} berhasil ditangani!" if st.session_state.language == 'id' else f"✅ All missing values in {col} successfully handled!")
                        elif imputed_count > 0:
                            st.info(f"ℹ️ {imputed_count} dari {original_missing} nilai hilang berhasil ditangani ({success_rate:.1f}%)" if st.session_state.language == 'id' else f"ℹ️ {imputed_count} of {original_missing} missing values handled ({success_rate:.1f}%)")
                        else:
                            st.warning(f"⚠️ Tidak ada nilai hilang yang berhasil ditangani di {col}" if st.session_state.language == 'id' else f"⚠️ No missing values were handled in {col}")
                        
                        if missing_after > 0:
                            st.warning(f"Sisa nilai hilang di {col}: {missing_after}" if st.session_state.language == 'id' else f"Remaining missing values in {col}: {missing_after}")
                        st.write(data[col].describe() if col_type == "numerical" else data[col].value_counts())
                        
                        # Visual comparison before and after
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Sebelum:**" if st.session_state.language == 'id' else "**Before:**")
                            if col_type == "numerical":
                                fig, ax = plt.subplots(figsize=(4, 3))
                                data[col].hist(bins=20, ax=ax)
                                ax.set_title(f"{col} - Original")
                                st.pyplot(fig)
                            else:
                                st.write("Original distribution")
                        
                        with col2:
                            st.write("**Sesudah:**" if st.session_state.language == 'id' else "**After:**")
                            if col_type == "numerical":
                                fig, ax = plt.subplots(figsize=(4, 3))
                                data[col].hist(bins=20, ax=ax)
                                ax.set_title(f"{col} - After Imputation")
                                st.pyplot(fig)
                            else:
                                st.write("After imputation")
    else:
        st.success("Tidak ditemukan nilai yang hilang dalam dataset." if st.session_state.language == 'id' else "No missing values found in the dataset.")

    # Summary section
    st.markdown("---")
    st.subheader("📊 Ringkasan Penanganan Nilai Hilang" if st.session_state.language == 'id' else "📊 Missing Value Handling Summary")
    
    # Create missing summary with proper Series alignment
    original_missing = pd.Series(st.session_state.get('original_missing_counts', {}))
    missing_after = data.isnull().sum()
    
    # Calculate percentage imputed correctly - based on original missing values, not total data
    def calc_percentage_imputed(col):
        orig_missing = original_missing.get(col, 0)
        if orig_missing == 0:
            return 0.0
        imputed_count = orig_missing - missing_after[col]
        return (imputed_count / orig_missing) * 100
    
    missing_summary = pd.DataFrame({
        'Column': data.columns,
        'Missing_Before': [original_missing.get(col, 0) for col in data.columns],
        'Missing_After': missing_after.values,
        'Percentage_Imputed': [calc_percentage_imputed(col) for col in data.columns]
    })
    
    missing_summary = missing_summary[missing_summary['Missing_Before'] > 0]
    
    if not missing_summary.empty:
        # Show summary statistics
        total_original_missing = missing_summary['Missing_Before'].sum()
        total_remaining_missing = missing_summary['Missing_After'].sum()
        total_imputed = total_original_missing - total_remaining_missing
        overall_success_rate = (total_imputed / total_original_missing * 100) if total_original_missing > 0 else 0
        
        # Display overall statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Missing Values" if st.session_state.language == 'id' else "Total Missing Values", 
                     f"{total_original_missing:,}")
        with col2:
            st.metric("Successfully Imputed" if st.session_state.language == 'id' else "Successfully Imputed", 
                     f"{total_imputed:,}")
        with col3:
            st.metric("Success Rate" if st.session_state.language == 'id' else "Success Rate", 
                     f"{overall_success_rate:.1f}%")
        
        st.dataframe(missing_summary)
        
        # Download handled data
        csv = data.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Data yang Sudah Diperbaiki" if st.session_state.language == 'id' else "⬇️ Download Cleaned Data",
            data=csv,
            file_name="data_tanpa_missing.csv" if st.session_state.language == 'id' else "cleaned_data.csv",
            mime="text/csv"
        )
    
    # Handle Outliers
    st.subheader("🛡️ Atasi Data Outlier" if st.session_state.language == 'id' else "🛡️ Handle Outliers")
    
    # Only for numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numerical_cols:
        handle_outliers = st.checkbox("Deteksi dan tangani outlier" if st.session_state.language == 'id' else "Detect and handle outliers")
        
        if handle_outliers:
            # Advanced outlier detection
            st.markdown("### 🔍 Deteksi Outlier Canggih" if st.session_state.language == 'id' else "### 🔍 Advanced Outlier Detection")
            
            if st.button("Jalankan Analisis Outlier" if st.session_state.language == 'id' else "Run Outlier Analysis", key="advanced_outlier_analysis"):
                try:
                    from utils import advanced_outlier_detection
                    outlier_result = advanced_outlier_detection(data[numerical_cols], method='auto', language=st.session_state.language)
                    
                    if outlier_result['success']:
                        st.success("Analisis outlier selesai" if st.session_state.language == 'id' else "Outlier analysis completed")
                        
                        # Display analysis results
                        st.write(f"**Metode yang digunakan:** {outlier_result['analysis']['method_used']}")
                        st.write(f"**Total outlier terdeteksi:** {outlier_result['analysis']['total_outliers']}")
                        st.write(f"**Persentase outlier:** {outlier_result['analysis']['total_percentage']:.2f}%")
                        st.write(f"**Kolom yang terdampak:** {', '.join(outlier_result['analysis']['affected_columns'])}")
                        
                        # Display recommendations
                        if 'recommendations' in outlier_result:
                            st.write("**Rekomendasi:**" if st.session_state.language == 'id' else "**Recommendations:**")
                            for rec in outlier_result['recommendations']:
                                st.write(f"- {rec}")
                        
                        # Store outlier information in session state
                        st.session_state.outlier_analysis = outlier_result
                    else:
                        st.error(f"Error dalam analisis: {outlier_result.get('error', 'Unknown error')}")
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Advanced Outlier Detection Analysis")
                        st.error(f"❌ {error_result['message']}")
                        with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                            st.write(error_result.get('details', str(e)))
                    else:
                        st.error(f"Error menjalankan analisis outlier: {str(e)}")
            
            # Manual outlier handling methods
            st.markdown("### ⚙️ Metode Penanganan Outlier Manual" if st.session_state.language == 'id' else "### ⚙️ Manual Outlier Handling Methods")
            outlier_method = st.radio(
                "Metode penanganan outlier:" if st.session_state.language == 'id' else "Outlier handling method:",
                ["IQR (Interquartile Range)", "Z-Score", "Winsorization", "Isolation Forest", "Local Outlier Factor", "DBSCAN"]
            )
            
            if outlier_method == "IQR (Interquartile Range)":
                for col in numerical_cols:
                    # Calculate IQR
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Count outliers
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                        
                        outlier_action = st.radio(
                            f"Tindakan untuk outlier di '{col}':" if st.session_state.language == 'id' else f"Action for outliers in '{col}':",
                            ["Remove", "Cap", "Keep"],
                            key=f"outlier_{col}"
                        )
                        
                        if outlier_action == "Remove":
                            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                            st.success(f"Outlier dihapus dari kolom '{col}'" if st.session_state.language == 'id' else f"Outliers removed from column '{col}'")
                        elif outlier_action == "Cap":
                            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                            st.success(f"Outlier di-cap pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers capped in column '{col}'")
            
            elif outlier_method == "Z-Score":
                z_threshold = st.slider(
                    "Ambang batas Z-Score:" if st.session_state.language == 'id' else "Z-Score threshold:",
                    2.0, 4.0, 3.0, 0.1
                )
                
                for col in numerical_cols:
                    # Calculate Z-scores
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    
                    # Identify outliers
                    outliers = data[z_scores > z_threshold][col]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                        
                        outlier_action = st.radio(
                            f"Tindakan untuk outlier di '{col}':" if st.session_state.language == 'id' else f"Action for outliers in '{col}':",
                            ["Remove", "Cap", "Keep"],
                            key=f"outlier_{col}"
                        )
                        
                        if outlier_action == "Remove":
                            data = data[z_scores <= z_threshold]
                            st.success(f"Outlier dihapus dari kolom '{col}'" if st.session_state.language == 'id' else f"Outliers removed from column '{col}'")
                        elif outlier_action == "Cap":
                            # Calculate bounds
                            mean = data[col].mean()
                            std = data[col].std()
                            lower_bound = mean - z_threshold * std
                            upper_bound = mean + z_threshold * std
                            
                            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                            st.success(f"Outlier di-cap pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers capped in column '{col}'")
            
            elif outlier_method == "Winsorization":
                percentile = st.slider(
                    "Persentil untuk Winsorization:" if st.session_state.language == 'id' else "Percentile for Winsorization:",
                    90, 99, 95, 1
                )
                
                for col in numerical_cols:
                    lower_bound = np.percentile(data[col], 100 - percentile)
                    upper_bound = np.percentile(data[col], percentile)
                    
                    # Count outliers
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                        st.success(f"Outlier di-winsorize pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers winsorized in column '{col}'")
            
            elif outlier_method == "Isolation Forest":
                contamination = st.slider(
                    "Tingkat kontaminasi:" if st.session_state.language == 'id' else "Contamination level:",
                    0.01, 0.5, 0.1, 0.01
                )
                
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    for col in numerical_cols:
                        # Reshape data for Isolation Forest
                        X = data[[col]].values
                        
                        # Fit Isolation Forest
                        iso_forest = IsolationForest(contamination=contamination, random_state=42)
                        outliers = iso_forest.fit_predict(X)
                        
                        # Count outliers (-1 indicates outlier)
                        outlier_count = np.sum(outliers == -1)
                        
                        if outlier_count > 0:
                            st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                            
                            outlier_action = st.radio(
                                f"Tindakan untuk outlier di '{col}':" if st.session_state.language == 'id' else f"Action for outliers in '{col}':",
                                ["Remove", "Cap", "Keep"],
                                key=f"outlier_iso_{col}"
                            )
                            
                            if outlier_action == "Remove":
                                data = data[outliers == 1]
                                st.success(f"Outlier dihapus dari kolom '{col}'" if st.session_state.language == 'id' else f"Outliers removed from column '{col}'")
                            elif outlier_action == "Cap":
                                # Get non-outlier bounds
                                non_outlier_data = data[outliers == 1][col]
                                lower_bound = non_outlier_data.min()
                                upper_bound = non_outlier_data.max()
                                
                                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                                st.success(f"Outlier di-cap pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers capped in column '{col}'")
                except ImportError:
                    st.error("Isolation Forest membutuhkan scikit-learn. Silakan install terlebih dahulu." if st.session_state.language == 'id' else "Isolation Forest requires scikit-learn. Please install it first.")
                except Exception as e:
                    st.error(f"Error menggunakan Isolation Forest: {str(e)}")
            
            elif outlier_method == "Local Outlier Factor":
                n_neighbors = st.slider(
                    "Jumlah tetangga:" if st.session_state.language == 'id' else "Number of neighbors:",
                    5, 50, 20, 1
                )
                contamination = st.slider(
                    "Tingkat kontaminasi:" if st.session_state.language == 'id' else "Contamination level:",
                    0.01, 0.5, 0.1, 0.01
                )
                
                try:
                    from sklearn.neighbors import LocalOutlierFactor
                    
                    for col in numerical_cols:
                        # Reshape data for LOF
                        X = data[[col]].values
                        
                        # Fit Local Outlier Factor
                        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                        outliers = lof.fit_predict(X)
                        
                        # Count outliers (-1 indicates outlier)
                        outlier_count = np.sum(outliers == -1)
                        
                        if outlier_count > 0:
                            st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                            
                            outlier_action = st.radio(
                                f"Tindakan untuk outlier di '{col}':" if st.session_state.language == 'id' else f"Action for outliers in '{col}':",
                                ["Remove", "Cap", "Keep"],
                                key=f"outlier_lof_{col}"
                            )
                            
                            if outlier_action == "Remove":
                                data = data[outliers == 1]
                                st.success(f"Outlier dihapus dari kolom '{col}'" if st.session_state.language == 'id' else f"Outliers removed from column '{col}'")
                            elif outlier_action == "Cap":
                                # Get non-outlier bounds
                                non_outlier_data = data[outliers == 1][col]
                                lower_bound = non_outlier_data.min()
                                upper_bound = non_outlier_data.max()
                                
                                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                                st.success(f"Outlier di-cap pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers capped in column '{col}'")
                except ImportError:
                    st.error("Local Outlier Factor membutuhkan scikit-learn. Silakan install terlebih dahulu." if st.session_state.language == 'id' else "Local Outlier Factor requires scikit-learn. Please install it first.")
                except Exception as e:
                    st.error(f"Error menggunakan Local Outlier Factor: {str(e)}")
            
            elif outlier_method == "DBSCAN":
                eps = st.slider(
                    "Parameter eps (jarak minimum):" if st.session_state.language == 'id' else "eps parameter (minimum distance):",
                    0.1, 5.0, 0.5, 0.1
                )
                min_samples = st.slider(
                    "Jumlah minimum sample:" if st.session_state.language == 'id' else "Minimum number of samples:",
                    3, 20, 5, 1
                )
                
                try:
                    from sklearn.cluster import DBSCAN
                    
                    for col in numerical_cols:
                        # Reshape data for DBSCAN
                        X = data[[col]].values
                        
                        # Fit DBSCAN
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = dbscan.fit_predict(X)
                        
                        # Count outliers (-1 indicates noise/outlier)
                        outlier_count = np.sum(labels == -1)
                        
                        if outlier_count > 0:
                            st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                            
                            outlier_action = st.radio(
                                f"Tindakan untuk outlier di '{col}':" if st.session_state.language == 'id' else f"Action for outliers in '{col}':",
                                ["Remove", "Cap", "Keep"],
                                key=f"outlier_dbscan_{col}"
                            )
                            
                            if outlier_action == "Remove":
                                data = data[labels != -1]
                                st.success(f"Outlier dihapus dari kolom '{col}'" if st.session_state.language == 'id' else f"Outliers removed from column '{col}'")
                            elif outlier_action == "Cap":
                                # Get non-outlier bounds
                                non_outlier_data = data[labels != -1][col]
                                lower_bound = non_outlier_data.min()
                                upper_bound = non_outlier_data.max()
                                
                                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                                st.success(f"Outlier di-cap pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers capped in column '{col}'")
                except ImportError:
                    st.error("DBSCAN membutuhkan scikit-learn. Silakan install terlebih dahulu." if st.session_state.language == 'id' else "DBSCAN requires scikit-learn. Please install it first.")
                except Exception as e:
                    st.error(f"Error menggunakan DBSCAN: {str(e)}")
            
            st.success("Penanganan outlier selesai" if st.session_state.language == 'id' else "Outlier handling completed")
    
    # Handle Duplicate Data
    st.subheader("👯 Penanganan Data Duplikat" if st.session_state.language == 'id' else "👯 Handle Duplicate Data")
    
    # Check for duplicate rows
    duplicate_count = data.duplicated().sum()
    
    if duplicate_count > 0:
        st.warning(f"Ditemukan {duplicate_count} baris duplikat dalam dataset" if st.session_state.language == 'id' else f"Found {duplicate_count} duplicate rows in the dataset")
        
        # Show preview of duplicate rows
        duplicate_rows = data[data.duplicated(keep=False)].sort_values(by=data.columns.tolist())
        st.write("Preview baris duplikat:" if st.session_state.language == 'id' else "Preview of duplicate rows:")
        st.dataframe(duplicate_rows.head(10))
        
        # Options for handling duplicates
        handle_duplicates = st.checkbox("Hapus data duplikat" if st.session_state.language == 'id' else "Remove duplicate data", value=True)
        
        if handle_duplicates:
            # Store original data count
            original_count = len(data)
            
            # Remove duplicate rows
            data = data.drop_duplicates()
            
            # Calculate removed duplicates
            removed_count = original_count - len(data)
            
            st.success(f"Berhasil menghapus {removed_count} baris duplikat" if st.session_state.language == 'id' else f"Successfully removed {removed_count} duplicate rows")
            st.info(f"Jumlah data: {original_count} → {len(data)}" if st.session_state.language == 'id' else f"Data count: {original_count} → {len(data)}")
    else:
        st.success("Tidak ditemukan data duplikat dalam dataset" if st.session_state.language == 'id' else "No duplicate data found in the dataset")

    # Feature selection
    st.subheader("🛠️ Rekayasa Data" if st.session_state.language == 'id' else "🛠️ Data Modification")

    # Encoding fitur kategorikal
    categorical_cols = [col for col in data.columns if col in st.session_state.categorical_columns and col != target_column]
    if categorical_cols:
        st.subheader("Lakukan Encoding" if st.session_state.language == 'id' else "Encode Categorical Features")
        
        # Tampilkan fitur kategorikal yang akan diencode
        st.write("**Fitur kategorikal yang akan diubah:**" if st.session_state.language == 'id' else "**Categorical features to be transformed:**")
        for col in categorical_cols:
            unique_values = data[col].nunique()
            st.write(f"- **{col}**: {unique_values} nilai unik" if st.session_state.language == 'id' else f"- **{col}**: {unique_values} unique values")
        
        encoding_method = st.radio("Encoding method:", ["Label Encoding", "One-Hot Encoding"])
        if encoding_method == "Label Encoding":
            encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
            st.session_state.encoders = encoders
            st.success("Encoding label diaplikasikan pada fitur kategorikal." if st.session_state.language == 'id' else "Label encoding applied to categorical features.")
        else:  # One-Hot Encoding
            # Simpan target column
            target_series = data[target_column].copy()
            # One-hot encode data
            data = pd.get_dummies(data.drop(columns=[target_column]), columns=categorical_cols, drop_first=True)
            # Kembalikan target column
            data[target_column] = target_series
            st.success("One-hot encoding diaplikasikan pada fitur kategorikal." if st.session_state.language == 'id' else "One-hot encoding applied to categorical features.")            
        
        # Tampilkan deskripsi fitur setelah encoding
        st.subheader("Deskripsi Fitur Setelah Encoding" if st.session_state.language == 'id' else "Feature Description After Encoding")
        
        # Buat dataframe deskripsi fitur
        feature_desc = pd.DataFrame({
            'Nama Fitur' if st.session_state.language == 'id' else 'Feature Name': data.columns,
            'Tipe Data' if st.session_state.language == 'id' else 'Data Type': data.dtypes.astype(str),
            'Jumlah Non-Null' if st.session_state.language == 'id' else 'Non-Null Count': data.count(),
            'Jumlah Nilai Unik' if st.session_state.language == 'id' else 'Unique Values': data.nunique(),
            'Nilai yang Hilang' if st.session_state.language == 'id' else 'Missing Values': data.isnull().sum()
        })
        
        # Tampilkan sebagai tabel
        st.dataframe(feature_desc)
        
        # Tampilkan ringkasan statistik
        st.write("**Ringkasan Statistik:**" if st.session_state.language == 'id' else "**Statistical Summary:**")
        st.write(f"- Total fitur: {len(data.columns)}")
        st.write(f"- Total baris: {len(data)}")
        st.write(f"- Fitur numerik: {len(data.select_dtypes(include=[np.number]).columns)}")
        st.write(f"- Fitur kategorikal: {len(data.select_dtypes(include=['object', 'category']).columns)}")
 
        # Tampilkan distribusi kelas
        class_counts = data[target_column].value_counts()
        
        if len(class_counts) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            class_counts.plot(kind='bar', ax=ax)
            plt.title('Distribusi Kelas' if st.session_state.language == 'id' else 'Class Distribution')
            plt.ylabel('Jumlah' if st.session_state.language == 'id' else 'Count')
            plt.xlabel('Kelas' if st.session_state.language == 'id' else 'Class')
            st.pyplot(fig)
        else:
            st.warning("Tidak ada data untuk kolom target yang dipilih" if st.session_state.language == 'id' else "No data available for selected target column")
        
    # Update all_columns setelah encoding
    all_columns = [col for col in data.columns if col != target_column]

    # Train-test split
    st.subheader("Lakukan Train-Test Split" if st.session_state.language == 'id' else "Train-Test Split")

    test_size = st.slider("Ukuran set pengujian (persen):" if st.session_state.language == 'id' else "Test set size (%):", 10, 50, 20) / 100
    random_state = st.number_input("Status acak:" if st.session_state.language == 'id' else "Random state:", 0, 100, 42)

    # Prepare data for modeling dengan semua fitur awal
    X = data[all_columns]
    y = data[target_column]
    
    # Ensure np is defined globally
    import numpy as np

    # Validasi jumlah sampel sebelum train test split
    if len(X) == 0:
        st.error("Tidak ada data untuk diproses. Pastikan dataset memiliki minimal 1 baris data." if st.session_state.language == 'id' else "No data to process. Please ensure your dataset has at least 1 row of data.")
        st.stop()
    elif len(X) < 2:
        st.error("Dataset terlalu kecil. Diperlukan minimal 2 sampel untuk train-test split." if st.session_state.language == 'id' else "Dataset too small. At least 2 samples required for train-test split.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Penanganan imbalanced dataset untuk klasifikasi SETELAH train-test split
    if problem_type == "Classification":
        st.subheader("Penanganan Imbalanced Dataset" if st.session_state.language == 'id' else "Imbalanced Dataset Handling")
        
        # Tampilkan distribusi kelas di Training dan Testing Set
        st.write("#### Distribusi Kelas Sebelum Penyeimbangan" if st.session_state.language == 'id' else "#### Class Distribution Before Balancing")
        
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            st.write("**Training Set:**")
            train_counts = y_train.value_counts()
            fig_train, ax_train = plt.subplots(figsize=(8, 4))
            train_counts.plot(kind='bar', ax=ax_train, color='skyblue')
            plt.title('Distribusi Kelas (Training)' if st.session_state.language == 'id' else 'Class Distribution (Training)')
            st.pyplot(fig_train)
            st.info(f"Rasio Imbalance (Train): {train_counts.max()/train_counts.min():.2f}" if len(train_counts) > 1 else "Hanya 1 kelas terdeteksi")
            
        with col_dist2:
            st.write("**Testing Set:**")
            test_counts = y_test.value_counts()
            fig_test, ax_test = plt.subplots(figsize=(8, 4))
            test_counts.plot(kind='bar', ax=ax_test, color='salmon')
            plt.title('Distribusi Kelas (Testing)' if st.session_state.language == 'id' else 'Class Distribution (Testing)')
            st.pyplot(fig_test)
            st.info(f"Rasio Imbalance (Test): {test_counts.max()/test_counts.min():.2f}" if len(test_counts) > 1 else "Hanya 1 kelas terdeteksi")

        # Hitung rasio imbalance untuk training set
        if len(train_counts) > 1:
            # Advanced imbalanced dataset analysis (Hanya pada Training Set)
            st.markdown("### 🔍 Analisis Dataset Imbalanced Canggih (Training Set)" if st.session_state.language == 'id' else "### 🔍 Advanced Imbalanced Dataset Analysis (Training Set)")
            
            if st.button("Jalankan Analisis Imbalanced" if st.session_state.language == 'id' else "Run Imbalanced Analysis", key="advanced_imbalanced_analysis"):
                try:
                    from utils import advanced_imbalanced_data_handling
                    analysis_result = advanced_imbalanced_data_handling(X_train, y_train, method='analyze', language=st.session_state.language)
                    
                    if analysis_result['success']:
                        st.success("Analisis dataset imbalanced selesai" if st.session_state.language == 'id' else "Imbalanced dataset analysis completed")
                        st.write(f"**Tingkat ketidakseimbangan:** {analysis_result['analysis']['imbalance_level']}")
                        st.write(f"**Skor Gini:** {analysis_result['analysis']['gini_coefficient']:.3f}")
                        
                        if 'recommendations' in analysis_result:
                            st.write("**Rekomendasi metode penyeimbangan:**" if st.session_state.language == 'id' else "**Recommended balancing methods:**")
                            for rec in analysis_result['recommendations']:
                                st.write(f"- {rec}")
                    else:
                        st.error(f"Error dalam analisis: {analysis_result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error menjalankan analisis imbalanced: {str(e)}")
            
            # Auto balancing option
            st.markdown("### ⚙️ Penyeimbangan Otomatis" if st.session_state.language == 'id' else "### ⚙️ Automatic Balancing")
            auto_balance = st.checkbox("Gunakan penyeimbangan otomatis canggih" if st.session_state.language == 'id' else "Use advanced automatic balancing", key="auto_balance_advanced")
            
            if auto_balance:
                balance_strategy = st.selectbox(
                    "Strategi penyeimbangan:" if st.session_state.language == 'id' else "Balancing strategy:",
                    ["auto", "over", "under", "combine", "hybrid"],
                    key="balance_strategy"
                )
                
                if st.button("Terapkan Penyeimbangan Otomatis" if st.session_state.language == 'id' else "Apply Automatic Balancing", key="apply_auto_balance"):
                    try:
                        from utils import advanced_imbalanced_data_handling
                        balance_result = advanced_imbalanced_data_handling(X_train, y_train, method=balance_strategy, language=st.session_state.language)
                        
                        if balance_result['success']:
                            st.success(f"Penyeimbangan berhasil menggunakan strategi: {balance_result['analysis']['method_used']}")
                            # Update data training saja
                            X_train = balance_result['X_balanced'].copy()
                            y_train = balance_result['y_balanced']
                            
                            # Tampilkan hasil setelah penyeimbangan
                            new_counts = y_train.value_counts()
                            fig_new, ax_new = plt.subplots(figsize=(8, 4))
                            new_counts.plot(kind='bar', ax=ax_new, color='lightgreen')
                            plt.title('Distribusi Kelas Setelah Penyeimbangan' if st.session_state.language == 'id' else 'Class Distribution After Balancing')
                            st.pyplot(fig_new)
                            
                            st.info(f"Jumlah sampel training: {len(y_train)} (Sebelum: {len(X_train)})")
                        else:
                            st.error(f"Error dalam penyeimbangan: {balance_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error dalam penyeimbangan otomatis: {str(e)}")

    # Tambahkan normalisasi setelah train test split
    st.subheader("Normalisasi Fitur" if st.session_state.language == 'id' else "Feature Normalization")

    normalization_method = st.selectbox(
        "Metode normalisasi:" if st.session_state.language == 'id' else "Normalization method:",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler", "PowerTransformer (Yeo-Johnson)", "QuantileTransformer (Normal)"]
    )

    # Outlier detection before scaling
    detect_outliers = st.checkbox(
        "Deteksi Outlier Sebelum Scaling" if st.session_state.language == 'id' else "Detect Outliers Before Scaling",
        value=False,
        help="Deteksi dan tangani outlier sebelum menerapkan normalisasi" if st.session_state.language == 'id' else "Detect and handle outliers before applying normalization"
    )
    
    if detect_outliers:
        st.subheader("Deteksi Outlier" if st.session_state.language == 'id' else "Outlier Detection")
        
        # Get numeric columns for outlier detection
        numeric_cols_outlier = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols_outlier) > 0:
            # Outlier detection method
            outlier_method = st.selectbox(
                "Metode deteksi outlier:" if st.session_state.language == 'id' else "Outlier detection method:",
                ["IQR Method", "Z-Score Method"]
            )
            
            # Parameters for outlier detection
            if outlier_method == "IQR Method":
                iqr_multiplier = st.slider(
                    "IQR Multiplier:" if st.session_state.language == 'id' else "IQR Multiplier:",
                    1.0, 3.0, 1.5, 0.1,
                    help="Semakin tinggi nilai, semakin sedikit outlier yang terdeteksi" if st.session_state.language == 'id' else "Higher values detect fewer outliers"
                )
            else:
                z_threshold = st.slider(
                    "Z-Score Threshold:" if st.session_state.language == 'id' else "Z-Score Threshold:",
                    2.0, 4.0, 3.0, 0.1,
                    help="Semakin tinggi nilai, semakin sedikit outlier yang terdeteksi" if st.session_state.language == 'id' else "Higher values detect fewer outliers"
                )
            
            # Handle outliers method
            handle_method = st.selectbox(
                "Metode penanganan outlier:" if st.session_state.language == 'id' else "Outlier handling method:",
                ["Hapus Outlier" if st.session_state.language == 'id' else "Remove Outliers",
                 "Ganti dengan Batas IQR" if st.session_state.language == 'id' else "Replace with IQR Bounds",
                 "Ganti dengan Median" if st.session_state.language == 'id' else "Replace with Median"]
            )
            
            if st.button("Deteksi dan Tangani Outlier" if st.session_state.language == 'id' else "Detect and Handle Outliers"):
                try:
                    # Detect outliers
                    outlier_mask = pd.DataFrame(False, index=X_train.index, columns=numeric_cols_outlier)
                    outlier_summary = {}
                    
                    for col in numeric_cols_outlier:
                        if outlier_method == "IQR Method":
                            Q1 = X_train[col].quantile(0.25)
                            Q3 = X_train[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - iqr_multiplier * IQR
                            upper_bound = Q3 + iqr_multiplier * IQR
                            
                            col_outliers = (X_train[col] < lower_bound) | (X_train[col] > upper_bound)
                            outlier_mask[col] = col_outliers
                            
                            outlier_summary[col] = {
                                'count': col_outliers.sum(),
                                'percentage': (col_outliers.sum() / len(X_train)) * 100,
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound
                            }
                        else:  # Z-Score Method
                            z_scores = np.abs((X_train[col] - X_train[col].mean()) / X_train[col].std())
                            col_outliers = z_scores > z_threshold
                            outlier_mask[col] = col_outliers
                            
                            outlier_summary[col] = {
                                'count': col_outliers.sum(),
                                'percentage': (col_outliers.sum() / len(X_train)) * 100,
                                'threshold': z_threshold
                            }
                    
                    # Display outlier summary
                    st.write("**Ringkasan Outlier:**" if st.session_state.language == 'id' else "**Outlier Summary:**")
                    summary_df = pd.DataFrame(outlier_summary).T
                    st.dataframe(summary_df)
                    
                    # Handle outliers
                    total_outliers = outlier_mask.any(axis=1).sum()
                    
                    if handle_method == "Hapus Outlier" or handle_method == "Remove Outliers":
                        # Remove rows with outliers
                        clean_mask = ~outlier_mask.any(axis=1)
                        X_train = X_train[clean_mask]
                        y_train = y_train[clean_mask]
                        st.success(f"Menghapus {total_outliers} baris dengan outlier. Sisa data: {len(X_train)} baris" if st.session_state.language == 'id' else f"Removed {total_outliers} rows with outliers. Remaining data: {len(X_train)} rows")
                        
                    elif handle_method == "Ganti dengan Batas IQR" or handle_method == "Replace with IQR Bounds":
                        # Replace outliers with IQR bounds
                        for col in numeric_cols_outlier:
                            Q1 = X_train[col].quantile(0.25)
                            Q3 = X_train[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - iqr_multiplier * IQR
                            upper_bound = Q3 + iqr_multiplier * IQR
                            
                            X_train.loc[outlier_mask[col], col] = np.clip(
                                X_train.loc[outlier_mask[col], col], 
                                lower_bound, upper_bound
                            )
                        st.success(f"Mengganti {total_outliers} outlier dengan batas IQR" if st.session_state.language == 'id' else f"Replaced {total_outliers} outliers with IQR bounds")
                        
                    else:  # Replace with Median
                        # Replace outliers with median
                        for col in numeric_cols_outlier:
                            median_val = X_train[col].median()
                            X_train.loc[outlier_mask[col], col] = median_val
                        st.success(f"Mengganti {total_outliers} outlier dengan median" if st.session_state.language == 'id' else f"Replaced {total_outliers} outliers with median")
                    
                    # Update X_test with same handling if needed
                    if handle_method != "Hapus Outlier" and handle_method != "Remove Outliers":
                        for col in numeric_cols_outlier:
                            if outlier_method == "IQR Method":
                                Q1 = X_test[col].quantile(0.25)
                                Q3 = X_test[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                
                                test_outliers = (X_test[col] < lower_bound) | (X_test[col] > upper_bound)
                                if handle_method == "Ganti dengan Batas IQR" or handle_method == "Replace with IQR Bounds":
                                    X_test.loc[test_outliers, col] = np.clip(
                                        X_test.loc[test_outliers, col], 
                                        lower_bound, upper_bound
                                    )
                                else:  # Replace with Median
                                    X_test.loc[test_outliers, col] = X_test[col].median()
                            else:  # Z-Score Method
                                z_scores = np.abs((X_test[col] - X_test[col].mean()) / X_test[col].std())
                                test_outliers = z_scores > z_threshold
                                if handle_method == "Ganti dengan Median" or handle_method == "Replace with Median":
                                    X_test.loc[test_outliers, col] = X_test[col].median()
                    
                except Exception as e:
                    st.error(f"Error saat deteksi outlier: {str(e)}" if st.session_state.language == 'id' else f"Error during outlier detection: {str(e)}")
        
        else:
            st.warning("Tidak ada fitur numerik untuk deteksi outlier" if st.session_state.language == 'id' else "No numeric features for outlier detection")

    if normalization_method != "None":
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
        
        if normalization_method == "StandardScaler":
            scaler = StandardScaler()
        elif normalization_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif normalization_method == "RobustScaler":
            scaler = RobustScaler()
        elif normalization_method == "PowerTransformer (Yeo-Johnson)":
            scaler = PowerTransformer(method='yeo-johnson')
            st.info("📊 PowerTransformer menggunakan metode Yeo-Johnson untuk mengubah data menjadi lebih Gaussian-like" if st.session_state.language == 'id' else "📊 PowerTransformer uses Yeo-Johnson method to make data more Gaussian-like")
        elif normalization_method == "QuantileTransformer (Normal)":
            scaler = QuantileTransformer(output_distribution='normal')
            st.info("📊 QuantileTransformer mengubah distribusi menjadi normal menggunakan quantiles" if st.session_state.language == 'id' else "📊 QuantileTransformer transforms distribution to normal using quantiles")
        
        # Simpan scaler ke session state untuk inverse transform
        st.session_state.scaler = scaler
        st.session_state.normalization_method = normalization_method
        st.session_state.numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
        
        # Fit dan transform hanya pada fitur numerik
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
            
            st.success(f"Normalisasi {normalization_method} berhasil diterapkan")
            st.info(f"Fitur numerik yang dinormalisasi: {len(numeric_cols)} fitur")
            
            # Tambahkan tombol untuk inverse transform
            if st.button("Tampilkan Inverse Transform" if st.session_state.language == 'id' else "Show Inverse Transform"):
                try:
                    # Lakukan inverse transform pada data training
                    X_train_inverse = X_train.copy()
                    X_test_inverse = X_test.copy()
                    
                    if len(numeric_cols) > 0:
                        X_train_inverse[numeric_cols] = scaler.inverse_transform(X_train[numeric_cols])
                        X_test_inverse[numeric_cols] = scaler.inverse_transform(X_test[numeric_cols])
                    
                    # Tampilkan perbandingan
                    st.subheader("Perbandingan Data Asli vs Dinormalisasi" if st.session_state.language == 'id' else "Comparison of Original vs Normalized Data")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Data Setelah Normalisasi:**" if st.session_state.language == 'id' else "**Normalized Data:**")
                        st.dataframe(X_train[numeric_cols[:5]].head())
                    with col2:
                        st.write("**Data Asli (Inverse Transform):**" if st.session_state.language == 'id' else "**Original Data (Inverse Transform):**")
                        st.dataframe(X_train_inverse[numeric_cols[:5]].head())
                    
                    # Tampilkan statistik perbandingan
                    st.write("**Statistik Perbandingan:**" if st.session_state.language == 'id' else "**Comparison Statistics:**")
                    comparison_stats = pd.DataFrame({
                        'Fitur': numeric_cols[:5],
                        'Mean_Normalized': X_train[numeric_cols[:5]].mean(),
                        'Mean_Original': X_train_inverse[numeric_cols[:5]].mean(),
                        'Std_Normalized': X_train[numeric_cols[:5]].std(),
                        'Std_Original': X_train_inverse[numeric_cols[:5]].std()
                    })
                    st.dataframe(comparison_stats)
                    
                except Exception as e:
                    st.error(f"Error saat inverse transform: {str(e)}" if st.session_state.language == 'id' else f"Error during inverse transform: {str(e)}")
    
    # Handle class imbalance for training data (classification only)
    if st.session_state.problem_type == "Classification" and IMB_AVAILABLE:
        st.subheader("Penanganan Ketidakseimbangan Dataset" if st.session_state.language == 'id' else "Handle Class Imbalance")
        
        # Check for class imbalance
        train_counts = pd.Series(y_train).value_counts()
        imbalance_ratio = train_counts.max() / train_counts.min()
        
        if imbalance_ratio > 3.0:  # Only show if there's significant imbalance (ratio > 3:1)
            st.warning(f"Terdeteksi ketidakseimbangan kelas dengan rasio {imbalance_ratio:.2f}" if st.session_state.language == 'id' else f"Detected class imbalance with ratio {imbalance_ratio:.2f}")
            
            # Imbalance handling options
            balance_method = st.selectbox(
                "Pilih metode penyeimbangan:" if st.session_state.language == 'id' else "Select balancing method:",
                ["Tidak ada" if st.session_state.language == 'id' else "None",
                 "Random Over Sampling",
                 "Random Under Sampling", 
                 "SMOTE",
                 "SMOTEENN",
                 "SMOTETomek"]
            )
            
            if balance_method != "Tidak ada" and balance_method != "None":
                with st.spinner("Menerapkan penyeimbangan dataset..." if st.session_state.language == 'id' else "Applying dataset balancing..."):
                    try:
                        # Validasi minimum samples untuk SMOTE-based methods
                        if balance_method in ["SMOTE", "SMOTEENN", "SMOTETomek"]:
                            min_samples_per_class = pd.Series(y_train).value_counts().min()
                            if min_samples_per_class < 6:
                                st.error(f"Error: {balance_method} membutuhkan minimal 6 sampel per kelas. Kelas dengan jumlah terkecil memiliki {min_samples_per_class} sampel." if st.session_state.language == 'id' else 
                                        f"Error: {balance_method} requires at least 6 samples per class. Smallest class has {min_samples_per_class} samples.")
                                st.info("Menggunakan Random Over Sampling sebagai alternatif..." if st.session_state.language == 'id' else "Using Random Over Sampling as alternative...")
                                balance_method = "Random Over Sampling"
                        
                        if balance_method == "Random Over Sampling":
                            ros = RandomOverSampler(random_state=random_state)
                            X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
                        elif balance_method == "Random Under Sampling":
                            rus = RandomUnderSampler(random_state=random_state)
                            X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
                        elif balance_method == "SMOTE":
                            smote = SMOTE(random_state=random_state)
                            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                        elif balance_method == "SMOTEENN":
                            smoteenn = SMOTEENN(random_state=random_state)
                            X_train_bal, y_train_bal = smoteenn.fit_resample(X_train, y_train)
                        elif balance_method == "SMOTETomek":
                            smotetomek = SMOTETomek(random_state=random_state)
                            X_train_bal, y_train_bal = smotetomek.fit_resample(X_train, y_train)
                        
                        # Show before/after comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Distribusi Sebelum:**" if st.session_state.language == 'id' else "**Before Distribution:**")
                            st.write(pd.Series(y_train).value_counts().to_dict())
                        with col2:
                            st.write("**Distribusi Setelah:**" if st.session_state.language == 'id' else "**After Distribution:**")
                            st.write(pd.Series(y_train_bal).value_counts().to_dict())
                        
                        # Use balanced data
                        X_train, y_train = X_train_bal, y_train_bal
                        st.success(f"Dataset berhasil diseimbangkan! Ukuran training: {len(y_train)} sampel" if st.session_state.language == 'id' else f"Dataset successfully balanced! Training size: {len(y_train)} samples")
                        
                    except Exception as e:
                        st.error(f"Error saat penyeimbangan: {e}" if st.session_state.language == 'id' else f"Error during balancing: {e}")
                        st.info("Menggunakan data training asli..." if st.session_state.language == 'id' else "Using original training data...")
        else:
            st.info("Dataset seimbang, tidak perlu penanganan khusus" if st.session_state.language == 'id' else "Dataset is balanced, no special handling needed")

    # Feature selection
    st.subheader("Seleksi Fitur" if st.session_state.language == 'id' else "Feature Selection")
    
    # Pilih algoritma seleksi fitur
    feature_selection_method = st.selectbox(
        "Metode seleksi fitur:" if st.session_state.language == 'id' else "Feature selection method:",
        [
            "Manual",
            "SelectKBest (Statistical)",
            "Mutual Information", 
            "Pearson Correlation",
            "Recursive Feature Elimination (RFE)",
            "RFECV (RFE + Cross-Validation)",
            "LASSO",
            "Gradient Boosting Importance",
            "Random Forest Importance",
            "Ensemble Feature Selection",
            "Multi-Stage Feature Selection",
            "Genetic Algorithm (PyGAD)"
        ]
    )

    # Gunakan data training untuk seleksi fitur
    X_train_for_selection = X_train.copy()
    y_train_for_selection = y_train.copy()
    
    # Simpan nama kolom asli untuk referensi
    all_columns_for_selection = X_train_for_selection.columns.tolist()
    selected_features = all_columns_for_selection

    # Setelah feature selection selesai, terapkan pada X_train dan X_test
    final_selected_features = selected_features
    X_train_final = X_train[final_selected_features]
    X_test_final = X_test[final_selected_features]
    
    # Update session state
    st.session_state.X_train = X_train_final
    st.session_state.X_test = X_test_final
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.processed_data = data
    
    st.success(f"Data training memiliki {X_train_final.shape[0]} sampel dan {X_train_final.shape[1]} fitur setelah seleksi" if st.session_state.language == 'id' else f"Training data has {X_train_final.shape[0]} samples and {X_train_final.shape[1]} features after selection")
    st.success(f"Data testing memiliki {X_test_final.shape[0]} sampel dan {X_test_final.shape[1]} fitur" if st.session_state.language == 'id' else f"Testing data has {X_test_final.shape[0]} samples and {X_test_final.shape[1]} features")

    # Display processed data
    st.subheader("Tampilkan Data Terproses" if st.session_state.language == 'id' else "Processed Data Preview")
    st.dataframe(X_train_final.head())

    # Update session state setelah encoding/scaling
    st.session_state.X_train = X_train_final
    st.session_state.X_test = X_test_final
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    # Display class distribution table for classification problems
    if st.session_state.problem_type == "Classification":
        st.subheader("Distribusi Label Target" if st.session_state.language == 'id' else "Target Label Distribution")
        
        # Create distribution table
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        
        # Align the indices and fill missing values with 0
        all_labels = train_counts.index.union(test_counts.index)
        train_aligned = train_counts.reindex(all_labels, fill_value=0)
        test_aligned = test_counts.reindex(all_labels, fill_value=0)
        
        distribution_df = pd.DataFrame({
            'Label': all_labels,
            'Jumlah Data Training': train_aligned.values,
            'Jumlah Data Testing': test_aligned.values,
            'Total': train_aligned.values + test_aligned.values
        })
        
        # Add percentages
        total_samples = len(y_train) + len(y_test)
        distribution_df['Persentase Training (%)'] = (distribution_df['Jumlah Data Training'] / len(y_train) * 100).round(2)
        distribution_df['Persentase Testing (%)'] = (distribution_df['Jumlah Data Testing'] / len(y_test) * 100).round(2)
        distribution_df['Persentase Total (%)'] = (distribution_df['Total'] / total_samples * 100).round(2)
        
        st.dataframe(distribution_df)
        
        # Display summary statistics
        st.write(f"**Total sampel:** {total_samples}")
        st.write(f"**Training set:** {len(y_train)} sampel ({len(y_train)/total_samples*100:.1f}%)")
        st.write(f"**Testing set:** {len(y_test)} sampel ({len(y_test)/total_samples*100:.1f}%)")
        
        # Display class imbalance information
        if len(train_counts) > 1:
            imbalance_ratio = train_counts.max() / train_counts.min()
            st.write(f"**Rasio ketidakseimbangan kelas (training):** {imbalance_ratio:.2f}")

    if feature_selection_method == "Manual":
        selected_features = st.multiselect(
            "Pilih fitur untuk model:" if st.session_state.language == 'id' else "Select features to include in the model:",
            all_columns,
            default=all_columns
        )

    elif feature_selection_method == "SelectKBest (Statistical)":
        st.subheader("SelectKBest Feature Selection" if st.session_state.language == 'id' else "SelectKBest Feature Selection")
        st.info("Menggunakan SelectKBest untuk seleksi fitur berdasarkan uji statistik" if st.session_state.language == 'id' else "Using SelectKBest for feature selection based on statistical tests")
        
        # Choose scoring function based on problem type
        if problem_type == "Regression":
            score_options = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }
            score_func_name = st.selectbox(
                "Fungsi skor untuk regresi:" if st.session_state.language == 'id' else "Scoring function for regression:",
                list(score_options.keys()),
                index=0
            )
            score_func = score_options[score_func_name]
        else:  # Classification
            score_options = {
                "f_classif": f_classif,
                "mutual_info_classif": mutual_info_classif
            }
            score_func_name = st.selectbox(
                "Fungsi skor untuk klasifikasi:" if st.session_state.language == 'id' else "Scoring function for classification:",
                list(score_options.keys()),
                index=0
            )
            score_func = score_options[score_func_name]
        
        # Choose number of features
        max_features = len(all_columns)
        default_k = min(10, max_features)
        
        k_features = st.number_input(
            "Jumlah fitur terbaik yang ingin dipilih:" if st.session_state.language == 'id' else "Number of best features to select:",
            min_value=1,
            max_value=max_features,
            value=default_k,
            step=1
        )
        
        # Cross-validation option
        use_cv = st.checkbox(
            "Gunakan Cross-Validation" if st.session_state.language == 'id' else "Use Cross-Validation",
            value=False,
            help="Gunakan cross-validation untuk evaluasi yang lebih robust" if st.session_state.language == 'id' else "Use cross-validation for more robust evaluation"
        )
        
        cv_folds = 5
        if use_cv:
            cv_folds = st.number_input(
                "Jumlah lipatan CV:" if st.session_state.language == 'id' else "Number of CV folds:",
                min_value=3,
                max_value=10,
                value=5,
                step=1
            )
        
        if st.button("Jalankan SelectKBest" if st.session_state.language == 'id' else "Run SelectKBest"):
            try:
                if use_cv:
                    # Use cross-validation for more robust feature selection
                    from sklearn.model_selection import cross_val_score
                    
                    # Get scores for each feature individually using CV
                    cv_scores = []
                    for i, feature in enumerate(all_columns):
                        X_single = X_train_for_selection[:, i].reshape(-1, 1)
                        if problem_type == "Regression":
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=50, random_state=42)
                            scores = cross_val_score(model, X_single, y_train_for_selection, 
                                                   cv=cv_folds, scoring='neg_mean_squared_error')
                            score = -np.mean(scores)  # Convert to positive (lower is better)
                        else:
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(n_estimators=50, random_state=42)
                            scores = cross_val_score(model, X_single, y_train_for_selection, 
                                                   cv=cv_folds, scoring='accuracy')
                            score = np.mean(scores)  # Higher is better
                        
                        cv_scores.append(score)
                    
                    # Select top k features based on CV scores
                    # For regression: lower MSE is better, for classification: higher accuracy is better
                    if problem_type == "Regression":
                        top_indices = np.argsort(cv_scores)[:k_features]  # Take lowest scores (best MSE)
                    else:
                        top_indices = np.argsort(cv_scores)[-k_features:]  # Take highest scores (best accuracy)
                    
                    selected_mask = np.zeros(len(all_columns), dtype=bool)
                    selected_mask[top_indices] = True
                    selected_features = [all_columns[i] for i in top_indices]
                    
                    # Create feature scores dataframe
                    feature_scores = pd.DataFrame({
                        'Feature': all_columns,
                        'Score': cv_scores,
                        'Selected': selected_mask
                    })
                    
                    if problem_type == "Regression":
                        feature_scores = feature_scores.sort_values('Score', ascending=True)  # Lower MSE is better
                    else:
                        feature_scores = feature_scores.sort_values('Score', ascending=False)  # Higher accuracy is better
                    
                    # Apply selection to training data
                    X_train_selected = X_train_for_selection[:, selected_mask]
                    
                else:
                    # Original SelectKBest without cross-validation
                    selector = SelectKBest(score_func=score_func, k=k_features)
                    
                    # Fit and transform the training data
                    X_train_selected = selector.fit_transform(X_train_for_selection, y_train_for_selection)
                    
                    # Get selected feature names
                    selected_mask = selector.get_support()
                    selected_features = [all_columns[i] for i, selected in enumerate(selected_mask) if selected]
                    
                    # Get feature scores
                    feature_scores = pd.DataFrame({
                        'Feature': all_columns,
                        'Score': selector.scores_,
                        'Selected': selected_mask
                    }).sort_values('Score', ascending=False)
                
                # Display results
                st.success("SelectKBest selesai!" if st.session_state.language == 'id' else "SelectKBest completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Jumlah fitur terpilih" if st.session_state.language == 'id' else "Selected features count", 
                            len(selected_features))
                    st.metric("Total fitur" if st.session_state.language == 'id' else "Total features", 
                            len(all_columns))
                
                with col2:
                    st.metric("Persentase fitur terpilih" if st.session_state.language == 'id' else "Feature selection ratio", 
                            f"{len(selected_features)/len(all_columns)*100:.1f}%")
                
                # Display selected features
                st.write("**Fitur yang dipilih:**" if st.session_state.language == 'id' else "**Selected features:**")
                st.write(selected_features)
                
                # Display feature scores
                st.write("**Skor fitur:**" if st.session_state.language == 'id' else "**Feature scores:**")
                st.dataframe(feature_scores[['Feature', 'Score', 'Selected']])
                
                # Visualize feature scores
                if len(all_columns) > 1:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Top features bar plot
                    top_features = feature_scores.head(min(15, len(feature_scores)))
                    ax1.barh(top_features['Feature'], top_features['Score'])
                    ax1.set_xlabel('Score' if st.session_state.language == 'id' else 'Score')
                    ax1.set_title('Top 15 Fitur Berdasarkan Skor' if st.session_state.language == 'id' else 'Top 15 Features by Score')
                    ax1.invert_yaxis()
                    
                    # Selected vs not selected
                    selection_counts = feature_scores['Selected'].value_counts()
                    labels = ['Terpilih' if st.session_state.language == 'id' else 'Selected', 
                            'Tidak Terpilih' if st.session_state.language == 'id' else 'Not Selected']
                    colors = ['green', 'red']
                    ax2.pie(selection_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
                    ax2.set_title('Distribusi Seleksi Fitur' if st.session_state.language == 'id' else 'Feature Selection Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download feature selection results
                    csv = feature_scores.to_csv(index=False)
                    st.download_button(
                        label="Unduh hasil seleksi fitur (CSV)" if st.session_state.language == 'id' else "Download feature selection results (CSV)",
                        data=csv,
                        file_name="selectkbest_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error saat menjalankan SelectKBest: {str(e)}" if st.session_state.language == 'id' else 
                        f"Error running SelectKBest: {str(e)}")
    
    elif feature_selection_method == "Genetic Algorithm (PyGAD)":
        st.subheader("Genetic Algorithm Feature Selection (PyGAD)" if st.session_state.language == 'id' else "Genetic Algorithm Feature Selection (PyGAD)")
        st.info("Menggunakan algoritma genetik PyGAD untuk seleksi fitur otomatis" if st.session_state.language == 'id' else "Using PyGAD genetic algorithm for automatic feature selection")
        
        # Parameters for PyGAD
        col1, col2 = st.columns(2)
        with col1:
            ga_population_size = st.number_input(
                "Ukuran populasi:" if st.session_state.language == 'id' else "Population size:",
                min_value=10, max_value=200, value=50, step=5,
                help="Jumlah kromosom dalam populasi" if st.session_state.language == 'id' else "Number of chromosomes in population"
            )
            ga_generations = st.number_input(
                "Jumlah generasi:" if st.session_state.language == 'id' else "Number of generations:",
                min_value=10, max_value=500, value=100, step=10,
                help="Maksimum iterasi algoritma genetik" if st.session_state.language == 'id' else "Maximum genetic algorithm iterations"
            )
            ga_mutation_rate = st.slider(
                "Tingkat mutasi:" if st.session_state.language == 'id' else "Mutation rate:",
                0.01, 0.3, 0.1, 0.01,
                help="Probabilitas mutasi gen" if st.session_state.language == 'id' else "Gene mutation probability"
            )
        
        with col2:
            ga_crossover_rate = st.slider(
                "Tingkat crossover:" if st.session_state.language == 'id' else "Crossover rate:",
                0.1, 0.9, 0.7, 0.1,
                help="Probabilitas crossover antar kromosom" if st.session_state.language == 'id' else "Crossover probability between chromosomes"
            )
            ga_elite_size = st.number_input(
                "Ukuran elit:" if st.session_state.language == 'id' else "Elite size:",
                min_value=1, max_value=20, value=5, step=1,
                help="Jumlah kromosom terbaik yang dilestarikan" if st.session_state.language == 'id' else "Number of best chromosomes to preserve"
            )
            target_features = st.number_input(
                "Target jumlah fitur:" if st.session_state.language == 'id' else "Target number of features:",
                min_value=1, max_value=len(all_columns), value=min(10, len(all_columns)), step=1
            )
        
        # Prepare data for PyGAD
        X_ga = data[all_columns].copy()
        y_ga = data[target_column].copy()
        
        # Handle categorical variables
        for col in X_ga.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_ga[col] = le.fit_transform(X_ga[col].astype(str))
        
        # Standardize features
        scaler = StandardScaler()
        X_ga_scaled = scaler.fit_transform(X_ga)
        
        if st.button("Jalankan Algoritma Genetik" if st.session_state.language == 'id' else "Run Genetic Algorithm"):
            try:
                import pygad
                
                # Define fitness function
                def fitness_func(ga_instance, solution, solution_idx):
                    # Get selected features based on binary solution
                    selected_indices = np.where(solution == 1)[0]
                    
                    if len(selected_indices) == 0:
                        return 0.0
                    
                    # Limit to target number of features
                    if len(selected_indices) > target_features:
                        # Select top features based on importance
                        if problem_type == "Regression":
                            from sklearn.ensemble import RandomForestRegressor
                            temp_model = RandomForestRegressor(n_estimators=50, random_state=42)
                            temp_model.fit(X_ga_scaled, y_ga)
                            importances = temp_model.feature_importances_
                            top_indices = np.argsort(importances)[-target_features:]
                            selected_indices = np.intersect1d(selected_indices, top_indices)
                        else:
                            from sklearn.ensemble import RandomForestClassifier
                            temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
                            temp_model.fit(X_ga_scaled, y_ga)
                            importances = temp_model.feature_importances_
                            top_indices = np.argsort(importances)[-target_features:]
                            selected_indices = np.intersect1d(selected_indices, top_indices)
                    
                    if len(selected_indices) == 0:
                        return 0.0
                    
                    # Get selected features
                    X_selected = X_ga_scaled[:, selected_indices]
                    
                    # Use cross-validation to evaluate fitness
                    if problem_type == "Regression":
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        scores = cross_val_score(model, X_selected, y_ga, cv=3, 
                                               scoring='neg_mean_squared_error')
                        fitness = -np.mean(scores)  # Negative MSE, so higher is better
                    else:
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=50, random_state=42)
                        scores = cross_val_score(model, X_selected, y_ga, cv=3, 
                                               scoring='accuracy')
                        fitness = np.mean(scores)
                    
                    # Penalty for too many features
                    penalty = abs(len(selected_indices) - target_features) * 0.01
                    return max(0, fitness - penalty)
                
                # Initialize PyGAD
                gene_space = [0, 1]  # Binary genes
                
                ga_instance = pygad.GA(
                    num_generations=ga_generations,
                    num_parents_mating=ga_population_size // 2,
                    fitness_func=fitness_func,
                    sol_per_pop=ga_population_size,
                    num_genes=len(all_columns),
                    gene_space=gene_space,
                    init_range_low=0,
                    init_range_high=2,
                    parent_selection_type="tournament",
                    K_tournament=3,
                    crossover_type="single_point",
                    crossover_probability=ga_crossover_rate,
                    mutation_type="random",
                    mutation_probability=ga_mutation_rate,
                    keep_elitism=ga_elite_size,
                    random_seed=42,
                    suppress_warnings=True
                )
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def on_generation(ga_instance):
                    generation = ga_instance.generations_completed
                    max_generations = ga_instance.num_generations
                    progress = generation / max_generations
                    progress_bar.progress(progress)
                    
                    best_fitness = ga_instance.best_solution()[1]
                    status_text.text(
                        f"Generasi {generation}/{max_generations} - Fitness terbaik: {best_fitness:.4f}" 
                        if st.session_state.language == 'id' 
                        else f"Generation {generation}/{max_generations} - Best fitness: {best_fitness:.4f}"
                    )
                
                ga_instance.on_generation = on_generation
                
                # Run genetic algorithm
                with st.spinner("Menjalankan algoritma genetik..." if st.session_state.language == 'id' else "Running genetic algorithm..."):
                    ga_instance.run()
                
                # Get results
                solution, solution_fitness, solution_idx = ga_instance.best_solution()
                selected_indices = np.where(solution == 1)[0]
                selected_features = [all_columns[i] for i in selected_indices]
                
                # Display results
                st.success(f"Algoritma genetik selesai!" if st.session_state.language == 'id' else "Genetic algorithm completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Jumlah fitur terpilih" if st.session_state.language == 'id' else "Selected features count", 
                            len(selected_features))
                    st.metric("Fitness terbaik" if st.session_state.language == 'id' else "Best fitness", 
                            f"{solution_fitness:.4f}")
                
                with col2:
                    st.metric("Total fitur" if st.session_state.language == 'id' else "Total features", 
                            len(all_columns))
                    st.metric("Persentase fitur terpilih" if st.session_state.language == 'id' else "Feature selection ratio", 
                            f"{len(selected_features)/len(all_columns)*100:.1f}%")
                
                # Display selected features
                st.write("**Fitur yang dipilih algoritma genetik:**" if st.session_state.language == 'id' else "**Features selected by genetic algorithm:**")
                st.write(selected_features)
                
                # Feature importance visualization
                if len(selected_features) > 0:
                    st.write("**Visualisasi seleksi fitur:**" if st.session_state.language == 'id' else "**Feature selection visualization:**")
                    
                    # Create a dataframe with selection status
                    selection_df = pd.DataFrame({
                        'Feature': all_columns,
                        'Selected': [1 if i in selected_indices else 0 for i in range(len(all_columns))]
                    })
                    
                    # Plot selection status
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Bar plot of selected vs not selected
                    selection_counts = selection_df['Selected'].value_counts()
                    colors = ['#ff9999', '#66b3ff']
                    ax1.pie(selection_counts.values, labels=['Not Selected', 'Selected'], 
                           colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Distribusi Seleksi Fitur' if st.session_state.language == 'id' else 'Feature Selection Distribution')
                    
                    # Fitness evolution plot
                    ax2.plot(ga_instance.best_solutions_fitness, 'b-', linewidth=2)
                    ax2.set_xlabel('Generasi' if st.session_state.language == 'id' else 'Generation')
                    ax2.set_ylabel('Fitness' if st.session_state.language == 'id' else 'Fitness')
                    ax2.set_title('Evolusi Fitness Algoritma Genetik' if st.session_state.language == 'id' else 'Genetic Algorithm Fitness Evolution')
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                # Clean up
                progress_bar.empty()
                status_text.empty()
                
            except ImportError:
                st.error("PyGAD tidak terinstal. Silakan install dengan: pip install pygad" if st.session_state.language == 'id' else 
                        "PyGAD not installed. Please install with: pip install pygad")
                selected_features = all_columns

    elif feature_selection_method == "Mutual Information":
                if problem_type == "Regression":
                    mi = mutual_info_regression(data[all_columns], data[target_column])
                else:
                    mi = mutual_info_classif(data[all_columns], data[target_column])
                mi_df = pd.DataFrame({"Feature": all_columns, "Mutual Information": mi})
                mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                
                # Tambahan: Slider untuk ambang batas minimum
                min_threshold = st.slider("Ambang batas minimum Mutual Information:", 0.0, 1.0, 0.25, 0.01, 
                                         help="Fitur dengan nilai Mutual Information di bawah ambang ini akan dihilangkan")
                
                # Filter berdasarkan ambang batas
                filtered_df = mi_df[mi_df["Mutual Information"] >= min_threshold]
                
                st.dataframe(mi_df)
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = mi_df.head(30)  # Tampilkan 15 fitur teratas
                ax.barh(top_features['Feature'], top_features['Mutual Information'])
                ax.set_xlabel('Mutual Information Score')
                ax.set_title('Top 30 Features by Mutual Information')
                ax.invert_yaxis()  # Fitur dengan score tertinggi di atas
                st.pyplot(fig)
                
                # Pilih fitur berdasarkan ambang batas atau top N
                use_threshold = st.checkbox("Gunakan ambang batas", value=True)
                if use_threshold:
                    selected_features = filtered_df["Feature"].tolist()
                    st.info(f"{len(selected_features)} fitur terpilih dengan ambang batas {min_threshold}")
                else:
                    top_n = st.slider("Top N fitur:", 1, len(all_columns), min(10, len(all_columns)))
                    selected_features = mi_df.head(top_n)["Feature"].tolist()
    elif feature_selection_method == "Pearson Correlation":
        numeric_columns = data[all_columns].select_dtypes(include=[np.number]).columns.tolist()
        if data[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
            st.error("Target kolom harus numerik untuk Pearson Correlation.")
            corr = pd.Series([np.nan]*len(numeric_columns), index=numeric_columns)
        else:
            corr = data[numeric_columns].corrwith(data[target_column]).abs()
        corr_df = pd.DataFrame({"Feature": numeric_columns, "Correlation": corr})
        corr_df = corr_df.sort_values("Correlation", ascending=False)
        st.dataframe(corr_df)
        top_n = st.slider("Top N features:", 1, len(all_columns), min(10, len(all_columns)))
        selected_features = corr_df.head(top_n)["Feature"].tolist()
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = corr_df.head(30)
        ax.barh(top_features['Feature'], top_features['Correlation'])
        ax.set_xlabel('Absolute Correlation')
        ax.set_title('Top 30 Features by Pearson Correlation')
        ax.invert_yaxis()
        st.pyplot(fig)
    elif feature_selection_method == "Recursive Feature Elimination (RFE)":
        from sklearn.feature_selection import RFE
        X_rfe = data[all_columns].copy()
        for col in X_rfe.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_rfe[col] = le.fit_transform(X_rfe[col].astype(str))
        if problem_type == "Regression":
            estimator = LinearRegression()
        else:
            estimator = LogisticRegression(max_iter=500)
        rfe = RFE(estimator, n_features_to_select=min(10, len(all_columns)))
        rfe.fit(X_rfe, data[target_column])
        rfe_df = pd.DataFrame({"Feature": all_columns, "Selected": rfe.support_})
        st.dataframe(rfe_df)
        selected_features = rfe_df[rfe_df["Selected"]]["Feature"].tolist()
        selected_count = rfe_df['Selected'].sum()
        fig, ax = plt.subplots(figsize=(8, 6))
        value_counts = rfe_df['Selected'].value_counts()
        value_counts.plot(kind='bar', ax=ax)
         # Fix: dynamically set labels based on actual values
        labels = ['Not Selected' if val == False else 'Selected' for val in value_counts.index]
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel('Count')
        ax.set_title(f'RFE Selection Results ({selected_count} features selected)')
        st.pyplot(fig)
        
    elif feature_selection_method == "RFECV (RFE + Cross-Validation)":
        from sklearn.feature_selection import RFECV
        X_rfecv = data[all_columns].copy()
        for col in X_rfecv.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_rfecv[col] = le.fit_transform(X_rfecv[col].astype(str))
        
        # Pilih estimator berdasarkan problem type
        if problem_type == "Regression":
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'r2'
        else:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        
        # RFECV dengan cross-validation
        cv_folds = st.slider("Jumlah fold untuk CV:" if st.session_state.language == 'id' else "Number of CV folds:", 3, 10, 5)
        step_size = st.slider("Step size (fitur yang dihapus per iterasi):" if st.session_state.language == 'id' else "Step size (features removed per iteration):", 1, 5, 1)
        
        with st.spinner("Menjalankan RFECV..." if st.session_state.language == 'id' else "Running RFECV..."):
            rfecv = RFECV(
                estimator=estimator,
                step=step_size,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                min_features_to_select=1
            )
            rfecv.fit(X_rfecv, data[target_column])
            
            # Hasil seleksi
            rfecv_df = pd.DataFrame({
                "Feature": all_columns,
                "Selected": rfecv.support_,
                "Ranking": rfecv.ranking_
            }).sort_values("Ranking")
            
            st.dataframe(rfecv_df)
            
            selected_features = rfecv_df[rfecv_df["Selected"]]["Feature"].tolist()
            optimal_count = rfecv.n_features_
            
            st.success(f"Jumlah fitur optimal: {optimal_count}" if st.session_state.language == 'id' else f"Optimal number of features: {optimal_count}")
            
            # Plot hasil CV
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
                   rfecv.cv_results_['mean_test_score'])
            ax.axvline(x=optimal_count, color='r', linestyle='--', 
                      label=f'Optimal: {optimal_count}')
            ax.set_xlabel('Number of Features' if st.session_state.language == 'id' else 'Jumlah Fitur')
            ax.set_ylabel(f'{scoring.upper()} Score' if st.session_state.language == 'id' else f'Skor {scoring.upper()}')
            ax.set_title('RFECV: Score vs Number of Features' if st.session_state.language == 'id' else 'RFECV: Skor vs Jumlah Fitur')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Plot feature rankings
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            top_features = rfecv_df[rfecv_df["Ranking"] <= 20]  # Top 20 rankings
            colors = ['green' if sel else 'gray' for sel in top_features["Selected"]]
            ax2.barh(top_features['Feature'], top_features['Ranking'], color=colors)
            ax2.set_xlabel('Ranking (1 = Best)' if st.session_state.language == 'id' else 'Ranking (1 = Terbaik)')
            ax2.set_title('Feature Rankings (Green = Selected)' if st.session_state.language == 'id' else 'Ranking Fitur (Hijau = Terpilih)')
            ax2.invert_yaxis()
            st.pyplot(fig2)

    elif feature_selection_method == "LASSO":
        from sklearn.linear_model import Lasso, LogisticRegression
        if problem_type == "Regression":
            lasso = Lasso(alpha=0.01, max_iter=1000)
        else:
            lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
        lasso.fit(data[all_columns], data[target_column])
        coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
        if coef.ndim > 1:
            coef = coef[0]
        lasso_df = pd.DataFrame({"Feature": all_columns, "Coefficient": coef})
        lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
        st.dataframe(lasso_df)
        selected_features = lasso_df["Feature"].tolist()
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = lasso_df.head(15)
        ax.barh(top_features['Feature'], abs(top_features['Coefficient']))
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title('Top 15 Features by LASSO Coefficient')
        ax.invert_yaxis()
        st.pyplot(fig)
    elif feature_selection_method == "Gradient Boosting Importance":
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        if problem_type == "Regression":
            model = GradientBoostingRegressor(random_state=42)
        else:
            model = GradientBoostingClassifier(random_state=42)
        model.fit(data[all_columns], data[target_column])
        importances = model.feature_importances_
        gb_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
        gb_df = gb_df.sort_values("Importance", ascending=False)
        st.dataframe(gb_df)
        top_n = st.slider("Top N features:", 1, len(all_columns), min(10, len(all_columns)))
        selected_features = gb_df.head(top_n)["Feature"].tolist()
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = gb_df.head(30)
        ax.barh(top_features['Feature'], top_features['Importance'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 30 Features by Gradient Boosting Importance')
        ax.invert_yaxis()
        st.pyplot(fig)
    elif feature_selection_method == "Random Forest Importance":
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # Tambahan: Input untuk jumlah pohon
                n_estimators = st.number_input("Jumlah pohon Random Forest:", min_value=10, max_value=1000, value=100, step=10,
                                               help="Semakin banyak pohon, semakin akurat tetapi lebih lambat")
                
                if problem_type == "Regression":
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                
                model.fit(data[all_columns], data[target_column])
                importances = model.feature_importances_
                rf_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
                rf_df = rf_df.sort_values("Importance", ascending=False)
                
                # Tambahan: Slider untuk ambang batas minimum
                min_threshold = st.slider("Ambang batas minimum Importance:", 0.0, 1.0, 0.2, 0.01,
                                         help="Fitur dengan nilai Importance di bawah ambang ini akan dihilangkan")
                
                # Filter berdasarkan ambang batas
                filtered_df = rf_df[rf_df["Importance"] >= min_threshold]
                
                st.dataframe(rf_df)
                
                # Pilih fitur berdasarkan ambang batas atau top N
                use_threshold = st.checkbox("Gunakan ambang batas", value=True)
                if use_threshold:
                    selected_features = filtered_df["Feature"].tolist()
                    st.info(f"{len(selected_features)} fitur terpilih dengan ambang batas {min_threshold}")
                else:
                    top_n = st.slider("Top N fitur:", 1, len(all_columns), min(10, len(all_columns)))
                    selected_features = rf_df.head(top_n)["Feature"].tolist()

                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = rf_df.head(30)
                ax.barh(top_features['Feature'], top_features['Importance'])
                ax.set_xlabel('Importance Score')
                ax.set_title('Top 30 Features by Random Forest Importance')
                ax.invert_yaxis()
                st.pyplot(fig)

    elif feature_selection_method == "Ensemble Feature Selection":
        st.info("Pilih dua metode seleksi fitur untuk digabungkan." if st.session_state.language == 'id' else "Select two feature selection methods to combine.")
        method1 = st.selectbox("Metode pertama:" if st.session_state.language == 'id' else "First method:", [
            "Mutual Information",
            "Pearson Correlation",
            "Recursive Feature Elimination (RFE)",
            "LASSO",
            "Gradient Boosting Importance",
            "Random Forest Importance"
        ], key="ensemble_method1")
        method2 = st.selectbox("Metode kedua:" if st.session_state.language == 'id' else "Second method:", [
            "Random Forest Importance",
            "Mutual Information",
            "Pearson Correlation",
            "Recursive Feature Elimination (RFE)",
            "LASSO",
            "Gradient Boosting Importance"
            
        ], key="ensemble_method2")

        combine_type = st.radio("Gabungkan hasil dengan:" if st.session_state.language == 'id' else "Combine results with:", ["Intersection", "Union"], index=0)

        def get_features_by_method(method):
            if method == "Mutual Information":
                if problem_type == "Regression":
                    mi = mutual_info_regression(data[all_columns], data[target_column])
                else:
                    mi = mutual_info_classif(data[all_columns], data[target_column])
                mi_df = pd.DataFrame({"Feature": all_columns, "Mutual Information": mi})
                mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                
                # Tambahan: Ambang batas untuk ensemble
                min_threshold = st.slider(f"Ambang batas minimum {method}:", 0.0, 1.0, 0.25, 0.01, 
                                        key=f"threshold_{method}")
                filtered_df = mi_df[mi_df["Mutual Information"] >= min_threshold]
                return set(filtered_df["Feature"].tolist())
            elif method == "Pearson Correlation":
                corr = data[all_columns].corrwith(data[target_column]).abs()
                corr_df = pd.DataFrame({"Feature": all_columns, "Correlation": corr})
                corr_df = corr_df.sort_values("Correlation", ascending=False)
                top_n = st.slider(f"Top N fitur ({method}):", 1, len(all_columns), min(10, len(all_columns)), key=f"topn_{method}")
                return set(corr_df.head(top_n)["Feature"].tolist())
            elif method == "Recursive Feature Elimination (RFE)":
                from sklearn.feature_selection import RFE
                from sklearn.linear_model import LinearRegression, LogisticRegression  # Tambahkan import ini
                if problem_type == "Regression":
                    estimator = LinearRegression()
                else:
                    estimator = LogisticRegression(max_iter=500)
                rfe = RFE(estimator, n_features_to_select=min(10, len(all_columns)))
                rfe.fit(data[all_columns], data[target_column])
                rfe_df = pd.DataFrame({"Feature": all_columns, "Selected": rfe.support_})
                return set(rfe_df[rfe_df["Selected"]]["Feature"].tolist())
            elif method == "LASSO":
                from sklearn.linear_model import Lasso, LogisticRegression
                if problem_type == "Regression":
                    lasso = Lasso(alpha=0.01, max_iter=1000)
                else:
                    lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                lasso.fit(data[all_columns], data[target_column])
                coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
                if coef.ndim > 1:
                    coef = coef[0]
                lasso_df = pd.DataFrame({"Feature": all_columns, "Coefficient": coef})
                lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
                return set(lasso_df["Feature"].tolist())
            elif method == "Gradient Boosting Importance":
                from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                if problem_type == "Regression":
                    model = GradientBoostingRegressor(random_state=42)
                else:
                    model = GradientBoostingClassifier(random_state=42)
                model.fit(data[all_columns], data[target_column])
                importances = model.feature_importances_
                gb_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
                gb_df = gb_df.sort_values("Importance", ascending=False)
                top_n = st.slider(f"Top N fitur ({method}):", 1, len(all_columns), min(10, len(all_columns)), key=f"topn_{method}")
                return set(gb_df.head(top_n)["Feature"].tolist())
            elif method == "Random Forest Importance":
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # Tambahan: Jumlah pohon untuk ensemble
                n_estimators = st.number_input(f"Jumlah pohon {method}:", 10, 1000, 100, 10,
                                            key=f"trees_{method}")
                
                if problem_type == "Regression":
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                
                model.fit(data[all_columns], data[target_column])
                importances = model.feature_importances_
                rf_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
                rf_df = rf_df.sort_values("Importance", ascending=False)
                
                # Tambahan: Ambang batas untuk ensemble
                min_threshold = st.slider(f"Ambang batas minimum {method}:", 0.0, 1.0, 0.2, 0.01,
                                        key=f"threshold_{method}")
                filtered_df = rf_df[rf_df["Importance"] >= min_threshold]
                return set(filtered_df["Feature"].tolist())
            else:
                return set(all_columns)

        features1 = get_features_by_method(method1)
        features2 = get_features_by_method(method2)

        if combine_type == "Intersection":
            selected_features = list(features1 & features2)
        else:
            selected_features = list(features1 | features2)

        st.write(f"Fitur hasil gabungan: {selected_features}" if st.session_state.language == 'id' else f"Combined features: {selected_features}")

    elif feature_selection_method == "Multi-Stage Feature Selection":
        st.subheader("Multi-Stage Feature Selection" if st.session_state.language == 'id' else "Multi-Stage Feature Selection")
        st.info("Metode ini menggunakan pendekatan 3 tahap: Information Gain → Random Forest Feature Importance → RFE" if st.session_state.language == 'id' else 
               "This method uses a 3-stage approach: Information Gain → Random Forest Feature Importance → RFE")
        
        from sklearn.feature_selection import RFE, SelectKBest
        from sklearn.ensemble import RandomForestClassifier
        
        # Persiapkan data untuk feature selection
        X_fs = data[all_columns].copy()
        for col in X_fs.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_fs[col] = le.fit_transform(X_fs[col].astype(str))
        
        # Tampilkan parameter untuk setiap tahap
        st.write("Tahap 1: Information Gain" if st.session_state.language == 'id' else "Stage 1: Information Gain")
        ig_percent = st.slider("Persentase fitur yang dipertahankan setelah Information Gain (%)" if st.session_state.language == 'id' else 
                              "Percentage of features to keep after Information Gain (%)", 10, 90, 40)
        
        st.write("Tahap 2: Random Forest Feature Importance" if st.session_state.language == 'id' else "Stage 2: Random Forest Feature Importance")
        rf_percent = st.slider("Persentase fitur yang dipertahankan setelah Random Forest (%)" if st.session_state.language == 'id' else 
                              "Percentage of features to keep after Random Forest (%)", 10, 90, 50)
        
        st.write("Tahap 3: Recursive Feature Elimination" if st.session_state.language == 'id' else "Stage 3: Recursive Feature Elimination")
        final_features = st.slider("Jumlah fitur akhir" if st.session_state.language == 'id' else "Final number of features", 
                                  1, min(20, len(all_columns)), min(10, len(all_columns)))
        
        # Tahap 1: Seleksi Fitur dengan Information Gain (SelectKBest + mutual_info_classif)
        n_features_after_ig = max(1, int(X_fs.shape[1] * ig_percent / 100))
        
        if problem_type == "Regression":
            selector_ig = SelectKBest(score_func=mutual_info_regression, k=n_features_after_ig)
        else:
            selector_ig = SelectKBest(score_func=mutual_info_classif, k=n_features_after_ig)
            
        X_train_ig = selector_ig.fit_transform(X_fs, data[target_column])
        
        # Dapatkan nama fitur yang terpilih
        selected_features_ig_mask = selector_ig.get_support()
        selected_features_ig_names = X_fs.columns[selected_features_ig_mask]
        
        # Tampilkan hasil tahap 1
        st.write(f"Fitur terpilih setelah Information Gain ({n_features_after_ig}):" if st.session_state.language == 'id' else 
                f"Selected features after Information Gain ({n_features_after_ig}):")
        st.write(", ".join(selected_features_ig_names))
        
        # Tahap 2: Seleksi Fitur dengan Feature Importance dari Random Forest
        n_features_after_rf_fi = max(1, int(len(selected_features_ig_names) * rf_percent / 100))
        
        # Latih Random Forest pada data yang sudah difilter IG
        if problem_type == "Regression":
            rf_model_for_importance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf_model_for_importance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
        rf_model_for_importance.fit(X_fs[selected_features_ig_names], data[target_column])
        
        # Dapatkan feature importance
        feature_importances_rf = pd.Series(rf_model_for_importance.feature_importances_, index=selected_features_ig_names)
        sorted_importances_rf = feature_importances_rf.sort_values(ascending=False)
        
        # Pilih fitur-fitur teratas berdasarkan importance
        top_features_rf_names = sorted_importances_rf.head(n_features_after_rf_fi).index.tolist()
        
        # Tampilkan hasil tahap 2
        st.write(f"Fitur terpilih setelah Random Forest Feature Importance ({n_features_after_rf_fi}):" if st.session_state.language == 'id' else 
                f"Selected features after Random Forest Feature Importance ({n_features_after_rf_fi}):")
        st.write(", ".join(top_features_rf_names))
        
        # Tahap 3: Seleksi Fitur dengan Recursive Feature Elimination (RFE) + Random Forest
        n_features_final = min(final_features, len(top_features_rf_names))
        
        # Pastikan jumlah fitur akhir minimal 2 untuk RFE
        if len(top_features_rf_names) < 2:
            # Jika fitur kurang dari 2, gunakan semua fitur yang tersisa tanpa RFE
            final_selected_features_names = top_features_rf_names
            st.warning("Jumlah fitur setelah tahap 2 kurang dari 2. RFE membutuhkan minimal 2 fitur. Menggunakan semua fitur dari tahap 2." if st.session_state.language == 'id' else 
                      "Number of features after stage 2 is less than 2. RFE requires at least 2 features. Using all features from stage 2.")
        else:
            # Gunakan Random Forest Classifier/Regressor sebagai estimator untuk RFE
            if problem_type == "Regression":
                estimator_rfe = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                estimator_rfe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Pastikan n_features_final minimal 2
            n_features_final = max(2, n_features_final)
            
            selector_rfe = RFE(estimator=estimator_rfe, n_features_to_select=n_features_final, step=1, verbose=0)
            
            # Lakukan RFE pada data yang sudah difilter oleh RF Feature Importance
            selector_rfe.fit(X_fs[top_features_rf_names], data[target_column])
            
            # Dapatkan nama fitur akhir yang terpilih
            selected_features_rfe_mask = selector_rfe.get_support()
            final_selected_features_names = np.array(top_features_rf_names)[selected_features_rfe_mask].tolist()
        
        # Tampilkan hasil akhir
        st.write(f"Fitur akhir terpilih setelah RFE ({n_features_final}):" if st.session_state.language == 'id' else 
                f"Final selected features after RFE ({n_features_final}):")
        st.write(", ".join(final_selected_features_names))
        
        # Tampilkan perbandingan jumlah fitur di setiap tahap
        st.subheader("Ringkasan Seleksi Fitur" if st.session_state.language == 'id' else "Feature Selection Summary")
        summary_data = {
            "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal" if st.session_state.language == 'id' else "Initial", 
                                                                       "Information Gain", 
                                                                       "Random Forest", 
                                                                       "RFE"],
            "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns), 
                                                                                         n_features_after_ig, 
                                                                                         n_features_after_rf_fi, 
                                                                                         n_features_final]
        }
        st.table(pd.DataFrame(summary_data))
        
        # Set fitur yang terpilih untuk digunakan dalam model
        selected_features = final_selected_features_names    

    if not selected_features:
        st.warning("Silahkan pilih fitur terlebih dahulu." if st.session_state.language == 'id' else "Please select at least one feature.")
    else:
        # Tampilkan hasil tahap pertama
        st.success(f"Tahap 1 selesai: {len(selected_features)} fitur terpilih" if st.session_state.language == 'id' else f"Stage 1 completed: {selected_features} features selected")
        st.write(f"Fitur terpilih tahap 1: {', '.join(selected_features)}" if st.session_state.language == 'id' else f"Stage 1 selected features: {', '.join(selected_features)}")
        
        # TAHAP KEDUA FEATURE SELECTION
        st.subheader("Tahap 2: Seleksi Fitur Lanjutan" if st.session_state.language == 'id' else "Stage 2: Advanced Feature Selection")
        
        # Checkbox untuk mengaktifkan tahap kedua
        enable_second_stage = st.checkbox("Aktifkan tahap kedua seleksi fitur" if st.session_state.language == 'id' else "Enable second stage feature selection", value=False)
        
        if enable_second_stage:
            # Gunakan hasil tahap pertama sebagai input tahap kedua
            all_columns_stage2 = selected_features
            
            # Pilih algoritma seleksi fitur tahap kedua
            feature_selection_method_stage2 = st.selectbox(
                "Metode seleksi fitur tahap 2:" if st.session_state.language == 'id' else "Feature selection method stage 2:",
                [
                    "Manual",
                    "Mutual Information",
                    "Pearson Correlation",
                    "Recursive Feature Elimination (RFE)",
                    "LASSO",
                    "Gradient Boosting Importance",
                    "Random Forest Importance",
                    "Ensemble Feature Selection",
                    "Multi-Stage Feature Selection"
                ],
                key="feature_selection_stage2"
            )

            selected_features_stage2 = all_columns_stage2  # Default

            if feature_selection_method_stage2 == "Manual":
                selected_features_stage2 = st.multiselect(
                    "Pilih fitur untuk model (tahap 2):" if st.session_state.language == 'id' else "Select features to include in the model (stage 2):",
                    all_columns_stage2,
                    default=all_columns_stage2,
                    key="manual_selection_stage2"
                )
            elif feature_selection_method_stage2 == "Mutual Information":
                if problem_type == "Regression":
                    mi = mutual_info_regression(data[all_columns_stage2], data[target_column])
                else:
                    mi = mutual_info_classif(data[all_columns_stage2], data[target_column])
                mi_df = pd.DataFrame({"Feature": all_columns_stage2, "Mutual Information": mi})
                mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                st.dataframe(mi_df)
                top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_mi_stage2")
                selected_features_stage2 = mi_df.head(top_n)["Feature"].tolist()
            elif feature_selection_method_stage2 == "Pearson Correlation":
                numeric_columns = data[all_columns_stage2].select_dtypes(include=[np.number]).columns.tolist()
                if data[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                    st.error("Target kolom harus numerik untuk Pearson Correlation.")
                    corr = pd.Series([np.nan]*len(numeric_columns), index=numeric_columns)
                else:
                    corr = data[numeric_columns].corrwith(data[target_column]).abs()
                corr_df = pd.DataFrame({"Feature": numeric_columns, "Correlation": corr})
                corr_df = corr_df.sort_values("Correlation", ascending=False)
                st.dataframe(corr_df)
                top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_corr_stage2")
                selected_features_stage2 = corr_df.head(top_n)["Feature"].tolist()
            elif feature_selection_method_stage2 == "Recursive Feature Elimination (RFE)":
                from sklearn.feature_selection import RFE
                # --- Tambahkan encoding untuk fitur kategorikal sebelum RFE ---
                X_rfe = data[all_columns_stage2].copy()
                for col in X_rfe.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X_rfe[col] = le.fit_transform(X_rfe[col].astype(str))
                if problem_type == "Regression":
                    estimator = LinearRegression()
                else:
                    estimator = LogisticRegression(max_iter=500)
                n_features_rfe = st.slider("Jumlah fitur RFE (tahap 2):" if st.session_state.language == 'id' else "Number of RFE features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="rfe_features_stage2")
                rfe = RFE(estimator, n_features_to_select=n_features_rfe)
                rfe.fit(X_rfe, data[target_column])
                rfe_df = pd.DataFrame({"Feature": all_columns_stage2, "Selected": rfe.support_})
                st.dataframe(rfe_df)
                selected_features_stage2 = rfe_df[rfe_df["Selected"]]["Feature"].tolist()
            elif feature_selection_method_stage2 == "LASSO":
                from sklearn.linear_model import Lasso, LogisticRegression
                alpha_lasso = st.slider("Alpha LASSO (tahap 2):" if st.session_state.language == 'id' else "LASSO Alpha (stage 2):", 0.001, 1.0, 0.01, key="alpha_lasso_stage2")
                if problem_type == "Regression":
                    lasso = Lasso(alpha=alpha_lasso, max_iter=1000)
                else:
                    lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, C=1/alpha_lasso)
                lasso.fit(data[all_columns_stage2], data[target_column])
                coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
                if coef.ndim > 1:
                    coef = coef[0]
                lasso_df = pd.DataFrame({"Feature": all_columns_stage2, "Coefficient": coef})
                lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
                st.dataframe(lasso_df)
                selected_features_stage2 = lasso_df["Feature"].tolist()
            elif feature_selection_method_stage2 == "Gradient Boosting Importance":
                from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                if problem_type == "Regression":
                    model = GradientBoostingRegressor(random_state=42)
                else:
                    model = GradientBoostingClassifier(random_state=42)
                model.fit(data[all_columns_stage2], data[target_column])
                importances = model.feature_importances_
                gb_df = pd.DataFrame({"Feature": all_columns_stage2, "Importance": importances})
                gb_df = gb_df.sort_values("Importance", ascending=False)
                st.dataframe(gb_df)
                top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_gb_stage2")
                selected_features_stage2 = gb_df.head(top_n)["Feature"].tolist()
            elif feature_selection_method_stage2 == "Random Forest Importance":
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                if problem_type == "Regression":
                    model = RandomForestRegressor(random_state=42)
                else:
                    model = RandomForestClassifier(random_state=42)
                model.fit(data[all_columns_stage2], data[target_column])
                importances = model.feature_importances_
                rf_df = pd.DataFrame({"Feature": all_columns_stage2, "Importance": importances})
                rf_df = rf_df.sort_values("Importance", ascending=False)
                st.dataframe(rf_df)
                top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_rf_stage2")
                selected_features_stage2 = rf_df.head(top_n)["Feature"].tolist()

            elif feature_selection_method_stage2 == "Ensemble Feature Selection":
                st.info("Pilih dua metode seleksi fitur untuk digabungkan (tahap 2)." if st.session_state.language == 'id' else "Select two feature selection methods to combine (stage 2).")
                method1_stage2 = st.selectbox("Metode pertama (tahap 2):" if st.session_state.language == 'id' else "First method (stage 2):", [
                    "Mutual Information",
                    "Pearson Correlation",
                    "Recursive Feature Elimination (RFE)",
                    "LASSO",
                    "Gradient Boosting Importance",
                    "Random Forest Importance"
                ], key="ensemble_method1_stage2")
                method2_stage2 = st.selectbox("Metode kedua (tahap 2):" if st.session_state.language == 'id' else "Second method (stage 2):", [
                    "Random Forest Importance",
                    "Mutual Information",
                    "Pearson Correlation",
                    "Recursive Feature Elimination (RFE)",
                    "LASSO",
                    "Gradient Boosting Importance"
                ], key="ensemble_method2_stage2")

                combine_type_stage2 = st.radio("Gabungkan hasil dengan (tahap 2):" if st.session_state.language == 'id' else "Combine results with (stage 2):", ["Intersection", "Union"], index=0, key="combine_type_stage2")

                def get_features_by_method_stage2(method, features_list):
                    if method == "Mutual Information":
                        if problem_type == "Regression":
                            mi = mutual_info_regression(data[features_list], data[target_column])
                        else:
                            mi = mutual_info_classif(data[features_list], data[target_column])
                        mi_df = pd.DataFrame({"Feature": features_list, "Mutual Information": mi})
                        mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                        top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                        return set(mi_df.head(top_n)["Feature"].tolist())
                    elif method == "Pearson Correlation":
                        numeric_columns = data[features_list].select_dtypes(include=[np.number]).columns.tolist()
                        if data[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                            corr = pd.Series([np.nan]*len(numeric_columns), index=numeric_columns)
                        else:
                            corr = data[numeric_columns].corrwith(data[target_column]).abs()
                        corr_df = pd.DataFrame({"Feature": numeric_columns, "Correlation": corr})
                        corr_df = corr_df.sort_values("Correlation", ascending=False)
                        top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                        return set(corr_df.head(top_n)["Feature"].tolist())
                    elif method == "Recursive Feature Elimination (RFE)":
                        from sklearn.feature_selection import RFE
                        from sklearn.linear_model import LinearRegression, LogisticRegression
                        X_rfe = data[features_list].copy()
                        for col in X_rfe.select_dtypes(include=['object', 'category']).columns:
                            le = LabelEncoder()
                            X_rfe[col] = le.fit_transform(X_rfe[col].astype(str))
                        if problem_type == "Regression":
                            estimator = LinearRegression()
                        else:
                            estimator = LogisticRegression(max_iter=500)
                        n_features_rfe = st.slider(f"Jumlah fitur RFE ({method}, tahap 2):" if st.session_state.language == 'id' else f"Number of RFE features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"rfe_{method}_stage2")
                        rfe = RFE(estimator, n_features_to_select=n_features_rfe)
                        rfe.fit(X_rfe, data[target_column])
                        rfe_df = pd.DataFrame({"Feature": features_list, "Selected": rfe.support_})
                        return set(rfe_df[rfe_df["Selected"]]["Feature"].tolist())
                    elif method == "LASSO":
                        from sklearn.linear_model import Lasso, LogisticRegression
                        alpha_lasso = st.slider(f"Alpha LASSO ({method}, tahap 2):" if st.session_state.language == 'id' else f"LASSO Alpha ({method}, stage 2):", 0.001, 1.0, 0.01, key=f"alpha_{method}_stage2")
                        if problem_type == "Regression":
                            lasso = Lasso(alpha=alpha_lasso, max_iter=1000)
                        else:
                            lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, C=1/alpha_lasso)
                        lasso.fit(data[features_list], data[target_column])
                        coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
                        if coef.ndim > 1:
                            coef = coef[0]
                        lasso_df = pd.DataFrame({"Feature": features_list, "Coefficient": coef})
                        lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
                        return set(lasso_df["Feature"].tolist())
                    elif method == "Gradient Boosting Importance":
                        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                        if problem_type == "Regression":
                            model = GradientBoostingRegressor(random_state=42)
                        else:
                            model = GradientBoostingClassifier(random_state=42)
                        model.fit(data[features_list], data[target_column])
                        importances = model.feature_importances_
                        gb_df = pd.DataFrame({"Feature": features_list, "Importance": importances})
                        gb_df = gb_df.sort_values("Importance", ascending=False)
                        top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                        return set(gb_df.head(top_n)["Feature"].tolist())
                    elif method == "Random Forest Importance":
                        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                        if problem_type == "Regression":
                            model = RandomForestRegressor(random_state=42)
                        else:
                            model = RandomForestClassifier(random_state=42)
                        model.fit(data[features_list], data[target_column])
                        importances = model.feature_importances_
                        rf_df = pd.DataFrame({"Feature": features_list, "Importance": importances})
                        rf_df = rf_df.sort_values("Importance", ascending=False)
                        top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                        return set(rf_df.head(top_n)["Feature"].tolist())
                    else:
                        return set(features_list)

                features1_stage2 = get_features_by_method_stage2(method1_stage2, all_columns_stage2)
                features2_stage2 = get_features_by_method_stage2(method2_stage2, all_columns_stage2)

                if combine_type_stage2 == "Intersection":
                    selected_features_stage2 = list(features1_stage2 & features2_stage2)
                else:
                    selected_features_stage2 = list(features1_stage2 | features2_stage2)

                st.write(f"Fitur hasil gabungan tahap 2: {selected_features_stage2}" if st.session_state.language == 'id' else f"Combined features stage 2: {selected_features_stage2}")

            elif feature_selection_method_stage2 == "Multi-Stage Feature Selection":
                st.subheader("Multi-Stage Feature Selection (Tahap 2)" if st.session_state.language == 'id' else "Multi-Stage Feature Selection (Stage 2)")
                st.info("Metode ini menggunakan pendekatan 3 tahap: Information Gain → Random Forest Feature Importance → RFE (pada hasil tahap 1)" if st.session_state.language == 'id' else 
                       "This method uses a 3-stage approach: Information Gain → Random Forest Feature Importance → RFE (on stage 1 results)")
                
                from sklearn.feature_selection import RFE, SelectKBest
                from sklearn.ensemble import RandomForestClassifier
                
                # Persiapkan data untuk feature selection tahap 2
                X_fs_stage2 = data[all_columns_stage2].copy()
                for col in X_fs_stage2.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X_fs_stage2[col] = le.fit_transform(X_fs_stage2[col].astype(str))
                
                # Tampilkan parameter untuk setiap tahap
                st.write("Tahap 1: Information Gain (pada hasil tahap 1)" if st.session_state.language == 'id' else "Stage 1: Information Gain (on stage 1 results)")
                ig_percent_stage2 = st.slider("Persentase fitur yang dipertahankan setelah Information Gain (%, tahap 2)" if st.session_state.language == 'id' else 
                                      "Percentage of features to keep after Information Gain (%, stage 2)", 10, 90, 40, key="ig_percent_stage2")
                
                st.write("Tahap 2: Random Forest Feature Importance (tahap 2)" if st.session_state.language == 'id' else "Stage 2: Random Forest Feature Importance (stage 2)")
                rf_percent_stage2 = st.slider("Persentase fitur yang dipertahankan setelah Random Forest (%, tahap 2)" if st.session_state.language == 'id' else 
                                      "Percentage of features to keep after Random Forest (%, stage 2)", 10, 90, 50, key="rf_percent_stage2")
                
                st.write("Tahap 3: Recursive Feature Elimination (tahap 2)" if st.session_state.language == 'id' else "Stage 3: Recursive Feature Elimination (stage 2)")
                final_features_stage2 = st.slider("Jumlah fitur akhir (tahap 2)" if st.session_state.language == 'id' else "Final number of features (stage 2)", 
                                          1, min(10, len(all_columns_stage2)), min(5, len(all_columns_stage2)), key="final_features_stage2")
                
                # Tahap 1: Seleksi Fitur dengan Information Gain (SelectKBest + mutual_info_classif)
                n_features_after_ig_stage2 = max(1, int(X_fs_stage2.shape[1] * ig_percent_stage2 / 100))
                
                if problem_type == "Regression":
                    selector_ig_stage2 = SelectKBest(score_func=mutual_info_regression, k=n_features_after_ig_stage2)
                else:
                    selector_ig_stage2 = SelectKBest(score_func=mutual_info_classif, k=n_features_after_ig_stage2)
                    
                X_train_ig_stage2 = selector_ig_stage2.fit_transform(X_fs_stage2, data[target_column])
                
                # Dapatkan nama fitur yang terpilih
                selected_features_ig_mask_stage2 = selector_ig_stage2.get_support()
                selected_features_ig_names_stage2 = X_fs_stage2.columns[selected_features_ig_mask_stage2]
                
                # Tampilkan hasil tahap 1
                st.write(f"Fitur terpilih setelah Information Gain tahap 2 ({n_features_after_ig_stage2}):" if st.session_state.language == 'id' else 
                        f"Selected features after Information Gain stage 2 ({n_features_after_ig_stage2}):")
                st.write(", ".join(selected_features_ig_names_stage2))
                
                # Tahap 2: Seleksi Fitur dengan Feature Importance dari Random Forest
                n_features_after_rf_fi_stage2 = max(1, int(len(selected_features_ig_names_stage2) * rf_percent_stage2 / 100))
                
                # Latih Random Forest pada data yang sudah difilter IG
                if problem_type == "Regression":
                    rf_model_for_importance_stage2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    rf_model_for_importance_stage2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    
                rf_model_for_importance_stage2.fit(X_fs_stage2[selected_features_ig_names_stage2], data[target_column])
                
                # Dapatkan feature importance
                feature_importances_rf_stage2 = pd.Series(rf_model_for_importance_stage2.feature_importances_, index=selected_features_ig_names_stage2)
                sorted_importances_rf_stage2 = feature_importances_rf_stage2.sort_values(ascending=False)
                
                # Pilih fitur-fitur teratas berdasarkan importance
                top_features_rf_names_stage2 = sorted_importances_rf_stage2.head(n_features_after_rf_fi_stage2).index.tolist()
                
                # Tampilkan hasil tahap 2
                st.write(f"Fitur terpilih setelah Random Forest Feature Importance tahap 2 ({n_features_after_rf_fi_stage2}):" if st.session_state.language == 'id' else 
                        f"Selected features after Random Forest Feature Importance stage 2 ({n_features_after_rf_fi_stage2}):")
                st.write(", ".join(top_features_rf_names_stage2))
                
                # Tahap 3: Seleksi Fitur dengan Recursive Feature Elimination (RFE) + Random Forest
                n_features_final_stage2 = min(final_features_stage2, len(top_features_rf_names_stage2))
                
                # Pastikan jumlah fitur akhir minimal 2 untuk RFE
                if len(top_features_rf_names_stage2) < 2:
                    # Jika fitur kurang dari 2, gunakan semua fitur yang tersisa tanpa RFE
                    final_selected_features_names_stage2 = top_features_rf_names_stage2
                    st.warning("Jumlah fitur setelah tahap 2 kurang dari 2. RFE membutuhkan minimal 2 fitur. Menggunakan semua fitur dari tahap 2." if st.session_state.language == 'id' else 
                              "Number of features after stage 2 is less than 2. RFE requires at least 2 features. Using all features from stage 2.")
                else:
                    # Gunakan Random Forest Classifier/Regressor sebagai estimator untuk RFE
                    if problem_type == "Regression":
                        estimator_rfe_stage2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        estimator_rfe_stage2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    
                    # Pastikan n_features_final minimal 2
                    n_features_final_stage2 = max(2, n_features_final_stage2)
                    
                    selector_rfe_stage2 = RFE(estimator=estimator_rfe_stage2, n_features_to_select=n_features_final_stage2, step=1, verbose=0)
                    
                    # Lakukan RFE pada data yang sudah difilter oleh RF Feature Importance
                    selector_rfe_stage2.fit(X_fs_stage2[top_features_rf_names_stage2], data[target_column])
                    
                    # Dapatkan nama fitur akhir yang terpilih
                    selected_features_rfe_mask_stage2 = selector_rfe_stage2.get_support()
                    final_selected_features_names_stage2 = np.array(top_features_rf_names_stage2)[selected_features_rfe_mask_stage2].tolist()
                
                # Tampilkan hasil akhir
                st.write(f"Fitur akhir terpilih setelah RFE tahap 2 ({n_features_final_stage2}):" if st.session_state.language == 'id' else 
                        f"Final selected features after RFE stage 2 ({n_features_final_stage2}):")
                st.write(", ".join(final_selected_features_names_stage2))
                
                # Tampilkan perbandingan jumlah fitur di setiap tahap
                st.subheader("Ringkasan Seleksi Fitur Tahap 2" if st.session_state.language == 'id' else "Feature Selection Summary Stage 2")
                summary_data_stage2 = {
                    "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal (dari tahap 1)" if st.session_state.language == 'id' else "Initial (from stage 1)", 
                                                                           "Information Gain", 
                                                                           "Random Forest", 
                                                                           "RFE"],
                    "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns_stage2), 
                                                                                             n_features_after_ig_stage2, 
                                                                                             n_features_after_rf_fi_stage2, 
                                                                                             n_features_final_stage2]
                }
                st.table(pd.DataFrame(summary_data_stage2))
                
                # Set fitur yang terpilih untuk digunakan dalam model
                selected_features_stage2 = final_selected_features_names_stage2
            
            # Tampilkan hasil tahap kedua
            if selected_features_stage2:
                st.success(f"Tahap 2 selesai: {len(selected_features_stage2)} fitur terpilih" if st.session_state.language == 'id' else f"Stage 2 completed: {len(selected_features_stage2)} features selected")
                st.write(f"Fitur terpilih tahap 2: {', '.join(selected_features_stage2)}" if st.session_state.language == 'id' else f"Stage 2 selected features: {', '.join(selected_features_stage2)}")
                
                # Gunakan hasil tahap kedua sebagai fitur final
                final_selected_features = selected_features_stage2
            else:
                st.warning("Tidak ada fitur yang terpilih di tahap 2. Menggunakan hasil tahap 1." if st.session_state.language == 'id' else "No features selected in stage 2. Using stage 1 results.")
                final_selected_features = selected_features
        else:
            # Jika tahap kedua tidak diaktifkan, gunakan hasil tahap pertama
            final_selected_features = selected_features
        
        # Tampilkan ringkasan akhir
        st.subheader("Ringkasan Seleksi Fitur Akhir" if st.session_state.language == 'id' else "Final Feature Selection Summary")
        if enable_second_stage:
            comparison_data = {
                "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal" if st.session_state.language == 'id' else "Initial", 
                                                                       "Tahap 1" if st.session_state.language == 'id' else "Stage 1", 
                                                                       "Tahap 2" if st.session_state.language == 'id' else "Stage 2"],
                "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns), 
                                                                                         len(selected_features), 
                                                                                         len(final_selected_features)]
            }
        else:
            comparison_data = {
                "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal" if st.session_state.language == 'id' else "Initial", 
                                                                       "Tahap 1" if st.session_state.language == 'id' else "Stage 1"],
                "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns), 
                                                                                         len(selected_features)]
            }
            final_selected_features = selected_features
        
        st.table(pd.DataFrame(comparison_data))
        st.write(f"Fitur akhir yang akan digunakan: {', '.join(final_selected_features)}" if st.session_state.language == 'id' else f"Final features to be used: {', '.join(final_selected_features)}")
        
        # Prepare data for modeling dengan fitur akhir
        X = data[final_selected_features]
        y = data[target_column]
                    
        
        # Display processed data
        st.subheader("Tampilkan Data Terproses" if st.session_state.language == 'id' else "Processed Data Preview")
        st.dataframe(X.head())

        # Update session_state setelah encoding/scaling
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        
else:
    st.info("Silahkan unggah dataset di tab 'Data Upload' terlebih dahulu." if st.session_state.language == 'id' else "Please upload a dataset in the 'Data Upload' tab first.")

# (wizard nav below)


# --- Wizard Navigation ---
st.markdown("---")
st.markdown("### ⏩ Langkah Selanjutnya" if st.session_state.language == 'id' else "### ⏩ Next Step")
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅️ Kembali ke EDA" if st.session_state.language == 'id' else "⬅️ Back to EDA", use_container_width=True):
        st.switch_page("pages/02_Exploratory_Data_Analytic.py")
with col_next:
    if st.button("Lanjutkan ke Model Training ➡️" if st.session_state.language == 'id' else "Continue to Model Training ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/04_Cross_Validation_and_Model_Training.py")
