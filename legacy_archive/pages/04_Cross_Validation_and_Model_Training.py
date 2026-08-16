import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from ml_engine.evaluation import calculate_vif, breusch_pagan_test
from ml_engine.tuning import create_optuna_study, parse_custom_range
from ml_engine.ui_helpers import merge_custom_param_ranges, validate_param_ranges

if "X_train" not in st.session_state or st.session_state.X_train is None:
    st.warning("Silakan lakukan preprocessing data terlebih dahulu.")
    st.stop()
st.header("🧠 Pelatihan dan Evaluasi Model" if st.session_state.language == 'id' else "🧠 Model Training and Evaluation")

# Ensure np is defined globally for this scope
import numpy as np

if (st.session_state.X_train is not None and 
    st.session_state.y_train is not None and 
    st.session_state.problem_type is not None):
    
    problem_type = st.session_state.problem_type
    
    # Check if data might be time series
    is_timeseries = False
    date_columns = []
    
    # Try to identify date/time columns
    if st.session_state.data is not None:
        for col in st.session_state.data.columns:
            # Check if column name contains date-related keywords
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day', 'tanggal', 'waktu', 'tahun', 'bulan', 'hari']):
                try:
                    # Try to convert to datetime
                    pd.to_datetime(st.session_state.data[col])
                    date_columns.append(col)
                except:
                    pass
    
    # If date columns found, ask user if this is time series data
    if date_columns:
        is_timeseries = st.checkbox("Data ini adalah data deret waktu (time series)", value=False)
    
    # Cross Validation Options
    st.subheader("⚙️ Pilihan Validasi Silang" if st.session_state.language == 'id' else "⚙️ Cross Validation Options")
    
    cv_options = [
        "None (Holdout Validation)",
        "K-Fold Cross Validation", 
        "Stratified K-Fold Cross Validation",
        "Leave-One-Out Cross Validation",
        "Leave-P-Out Cross Validation",
        "Time Series Split"
    ]
    
    cv_method = st.selectbox(
        "Pilih metode validasi silang:" if st.session_state.language == 'id' else "Select cross validation method:",
        cv_options
    )
    
    cv_params = {}
    
    if cv_method == "K-Fold Cross Validation":
        from sklearn.model_selection import KFold, cross_val_score
        
        n_splits = st.slider("Jumlah fold (K):" if st.session_state.language == 'id' else "Number of folds (K):", 2, 10, 5)
        cv_params['cv'] = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_params['name'] = f"K-Fold (K={n_splits})"
        
    elif cv_method == "Stratified K-Fold Cross Validation":
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        
        n_splits = st.slider("Jumlah fold (K):" if st.session_state.language == 'id' else "Number of folds (K):", 2, 10, 5)
        cv_params['cv'] = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_params['name'] = f"Stratified K-Fold (K={n_splits})"
        
    elif cv_method == "Leave-One-Out Cross Validation":
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        
        cv_params['cv'] = LeaveOneOut()
        cv_params['name'] = "Leave-One-Out"
        
    elif cv_method == "Leave-P-Out Cross Validation":
        from sklearn.model_selection import LeavePOut, cross_val_score
        
        max_p = min(5, len(X) - 1)
        p_value = st.slider("Nilai P:" if st.session_state.language == 'id' else "P value:", 1, max_p, 2)
        cv_params['cv'] = LeavePOut(p=p_value)
        cv_params['name'] = f"Leave-{p_value}-Out"
        
    elif cv_method == "Time Series Split":
        st.info("⚠️ Time Series Split untuk data temporal. Tidak ada shuffle pada data." if st.session_state.language == 'id' else "⚠️ Time Series Split for temporal data. No data shuffling.")
        
        n_splits = st.slider("Jumlah fold:" if st.session_state.language == 'id' else "Number of folds:", 2, 10, 5)
        test_size = st.slider("Test size (per fold):" if st.session_state.language == 'id' else "Test size (per fold):", 0.1, 0.5, 0.2, 0.05)
        
        # Time Series Split tidak menggunakan shuffle
        cv_params['cv'] = TimeSeriesSplit(n_splits=n_splits)
        cv_params['name'] = f"Time Series Split (n={n_splits})"
        cv_params['gap'] = 0
        
        st.markdown("""
        **Time Series Split Info:**
        - Fold 1: [train] -> [test]
        - Fold 2: [train train] -> [test]
        - Fold 3: [train train train] -> [test]
        - Dan seterusnya...
        
        **Catatan:** Data dijaga urutannya (tidak di-shuffle) untuk menjaga dependensi temporal.
        """)
        
    else:  # None (Holdout)
        cv_params['cv'] = None
        cv_params['name'] = "Holdout Validation"
    
    # Select evaluation metric
    if cv_params['cv'] is not None:
        st.subheader("📊 Pengaturan Evaluasi" if st.session_state.language == 'id' else "📊 Evaluation Settings")
        
        if problem_type == "Classification":
            cv_scoring = st.selectbox(
                "Metrik evaluasi:" if st.session_state.language == 'id' else "Evaluation metric:",
                ["accuracy", "precision", "recall", "f1", "roc_auc"]
            )
            cv_params['scoring'] = cv_scoring
        else:  # Regression
            cv_scoring = st.selectbox(
                "Metrik evaluasi:" if st.session_state.language == 'id' else "Evaluation metric:",
                ["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
            )
            cv_params['scoring'] = cv_scoring
            
        # Display data distribution for classification with stratified k-fold
        if problem_type == "Classification" and cv_method == "Stratified K-Fold Cross Validation":
            st.write("**Distribusi Data per Fold:**" if st.session_state.language == 'id' else "**Data Distribution per Fold:**")
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            fold_info = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                y_fold_train = y.iloc[train_idx]
                y_fold_val = y.iloc[val_idx]
                
                fold_counts = pd.Series(y_fold_val).value_counts().sort_index()
                fold_info.append({
                    'Fold': f'Fold {fold_idx + 1}',
                    'Training Samples': len(train_idx),
                    'Validation Samples': len(val_idx),
                    **{f'Class {label}': count for label, count in fold_counts.items()}
                })
            
            fold_df = pd.DataFrame(fold_info)
            st.dataframe(fold_df)
    
    if is_timeseries:
        st.subheader("📈 Pelatihan Model Forecasting" if st.session_state.language == 'id' else "📈 Forecasting Model Training")
        
        # Select date column
        date_column = st.selectbox("Pilih kolom tanggal/waktu:" if st.session_state.language == 'id' else "Select date column:", date_columns)
        
        # Select target column
        target_column = st.selectbox("Pilih kolom target untuk diprediksi:" if st.session_state.language == 'id' else "Select target column for prediction:", 
                                    [col for col in st.session_state.data.columns 
                                     if col != date_column and col in st.session_state.numerical_columns])
        
        # Select frequency
        freq = st.selectbox("Frekuensi data:", ["Harian (D)", "Mingguan (W)", "Bulanan (M)", "Tahunan (Y)", "Lainnya"] if st.session_state.language == 'id' else ["Daily (D)", "Weekly (W)", "Monthly (M)", "Yearly (Y)", "Other"])
        freq_map = {"Harian (D)": "D", "Mingguan (W)": "W", "Bulanan (M)": "M", "Tahunan (Y)": "Y", "Lainnya": None if st.session_state.language == 'id' else "Other"}
        selected_freq = freq_map[freq]
        
        # Number of periods to forecast
        forecast_periods = st.slider("Jumlah periode untuk prediksi ke depan:" if st.session_state.language == 'id' else "Number of periods to forecast:", 1, 100, 10)
        
        # Select forecasting model
        model_type = st.selectbox("Pilih model forecasting:" if st.session_state.language == 'id' else "Select forecasting model:", 
                                 ["ARIMA", "SARIMA", "Exponential Smoothing", "Prophet", "Random Forest", 
                                  "Gradient Boosting", "Linear Regression", "SARIMAX", "Holt-Winters", "LSTM", "DLinear"])
        
        # Import required modules
        try:
            from forecasting_utils import (
                train_arima_model, train_exponential_smoothing, 
                train_ml_forecaster, forecast_future, 
                evaluate_forecast_model, plot_forecast_results
            )
            from utils import prepare_timeseries_data, check_stationarity, plot_timeseries_analysis
            
            FORECASTING_MODULES_AVAILABLE = True
        except ImportError:
            st.error("Modul forecasting tidak tersedia. Pastikan file utils.py dan forecasting_utils.py ada di direktori yang sama." if st.session_state.language == 'id' else "Forecasting modules not available. Ensure utils.py and forecasting_utils.py are in the same directory.")
            FORECASTING_MODULES_AVAILABLE = False
        
        if FORECASTING_MODULES_AVAILABLE:
            # Import statsmodels conditionally
            try:
                import statsmodels.api as sm
                from statsmodels.tsa.arima.model import ARIMA
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                STATSMODELS_AVAILABLE = True
            except ImportError:
                STATSMODELS_AVAILABLE = False
                if model_type in ["ARIMA", "Exponential Smoothing"]:
                    st.warning("Statsmodels tidak terinstal. Silakan instal dengan 'pip install statsmodels'." if st.session_state.language == 'id' else "Statsmodels not installed. Please install with 'pip install statsmodels'.")
            
            # Import Prophet conditionally
            try:
                from prophet import Prophet
                PROPHET_AVAILABLE = True
            except ImportError:
                PROPHET_AVAILABLE = False
                if model_type == "Prophet":
                    st.warning("Prophet tidak terinstal. Silakan instal dengan 'pip install prophet'." if st.session_state.language == 'id' else "Prophet not installed. Please install with 'pip install prophet'.")
            
            # Prepare time series data
            if st.button("Proses Data Time Series" if st.session_state.language == 'id' else "Process Time Series Data"):
                with st.spinner("Memproses data time series..." if st.session_state.language == 'id' else "Processing time series data..."):
                    try:
                        log_feature('timeseries_process')
                    except Exception:
                        pass
                    # Prepare data
                    ts_data = prepare_timeseries_data(
                        st.session_state.data, 
                        date_column, 
                        target_column, 
                        freq=selected_freq
                    )
                    
                    # Check stationarity
                    stationarity_result = check_stationarity(ts_data[target_column])
                    st.write("Hasil Uji Stasioneritas:" if st.session_state.language == 'id' else "Stationarity Test Results:")
                    
                    if stationarity_result['Message']:
                        st.warning(stationarity_result['Message'])
                    
                    if stationarity_result['Test Statistic'] is not None:
                        st.write(f"- Test Statistic: {stationarity_result['Test Statistic']:.4f}")
                    if stationarity_result['p-value'] is not None:
                        st.write(f"- p-value: {stationarity_result['p-value']:.4f}")
                    st.write(f"- Data {'stasioner' if stationarity_result['Stationary'] else 'tidak stasioner'}")
                    
                    # Plot time series analysis
                    st.write("Analisis Time Series:" if st.session_state.language == 'id' else "Time Series Analysis:")
                    fig = plot_timeseries_analysis(ts_data[target_column])
                    st.pyplot(fig)
                    
                    # Split data for training and testing
                    train_size = int(len(ts_data) * 0.8)
                    train_data = ts_data.iloc[:train_size]
                    test_data = ts_data.iloc[train_size:]
                    
                    st.write(f"Data dibagi menjadi {len(train_data)} sampel training dan {len(test_data)} sampel testing" if st.session_state.language == 'id' else f"Data split into {len(train_data)} training samples and {len(test_data)} testing samples.")
                    
                    # Import all required forecasting functions
                    try:
                        from forecasting_utils import (
                            train_arima_model, train_exponential_smoothing, train_sarima_model,
                            train_sarimax_model, train_holt_winters, train_lstm_model,
                            train_ml_forecaster, forecast_future, evaluate_forecast_model, 
                            plot_forecast_results
                        )
                    except ImportError as e:
                        st.error(f"Error importing forecasting functions: {str(e)}")
                    
                    # Train model based on selection
                    if model_type == "ARIMA" and STATSMODELS_AVAILABLE:
                        p = st.slider("Parameter p (AR):", 0, 5, 1)
                        d = st.slider("Parameter d (differencing):", 0, 2, 1)
                        q = st.slider("Parameter q (MA):", 0, 5, 1)
                        
                        # Session management for ARIMA parameters
                        if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                            try:
                                session_manager.initialize_tab_state('time_series')
                                arima_params = {
                                    'model_type': 'ARIMA',
                                    'p': p,
                                    'd': d,
                                    'q': q,
                                    'target_column': target_column
                                }
                                session_manager.update_tab_state('time_series', arima_params)
                            except Exception as e:
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Session Management (ARIMA)")
                                    st.warning(f"⚠️ {error_result['message']}")
                        
                        with st.spinner("Melatih model ARIMA..." if st.session_state.language == 'id' else "Training ARIMA model..."):
                            try:
                                model = train_arima_model(train_data, target_column, order=(p, d, q))
                                st.session_state.model = model
                                st.success("Model ARIMA berhasil dilatih!" if st.session_state.language == 'id' else "ARIMA model trained successfully!")
                                
                                # Update session state with training success
                                if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                                    try:
                                        session_manager.update_tab_state('time_series', {'training_success': True, 'model_trained': 'ARIMA'})
                                    except Exception as e:
                                        if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                            error_result = error_handler.handle_error(e, "Session Update (ARIMA Success)")
                                            st.warning(f"⚠️ {error_result['message']}")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    try:
                                        model_bytes = pickle.dumps(st.session_state.model)
                                        st.download_button(
                                            label="📥 Unduh Model ARIMA (.pkl)" if st.session_state.language == 'id' else "📥 Download ARIMA Model (.pkl)",
                                            data=model_bytes,
                                            file_name=f"arima_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                    except Exception as pickle_error:
                                        st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                            except Exception as e:
                                # Enhanced error handling for ARIMA training
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "ARIMA Model Training")
                                    st.error(f"❌ {error_result['message']}")
                                    if 'details' in error_result:
                                        with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                                            st.write(error_result['details'])
                                    if 'recommendations' in error_result:
                                        with st.expander("💡 Rekomendasi" if st.session_state.language == 'id' else "💡 Recommendations"):
                                            for rec in error_result['recommendations']:
                                                st.write(f"• {rec}")
                                else:
                                    st.error(f"Error training ARIMA: {str(e)}")
                    
                    elif model_type == "SARIMA" and STATSMODELS_AVAILABLE:
                        p = st.slider("Parameter p (AR):", 0, 5, 1)
                        d = st.slider("Parameter d (differencing):", 0, 2, 1)
                        q = st.slider("Parameter q (MA):", 0, 5, 1)
                        P = st.slider("Parameter P (Seasonal AR):", 0, 2, 1)
                        D = st.slider("Parameter D (Seasonal differencing):", 0, 2, 1)
                        Q = st.slider("Parameter Q (Seasonal MA):", 0, 2, 1)
                        s = st.slider("Periode musiman (s):", 1, 52, 12)
                        
                        # Session management for SARIMA parameters
                        if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                            try:
                                session_manager.initialize_tab_state('time_series')
                                sarima_params = {
                                    'model_type': 'SARIMA',
                                    'p': p,
                                    'd': d,
                                    'q': q,
                                    'P': P,
                                    'D': D,
                                    'Q': Q,
                                    's': s,
                                    'target_column': target_column
                                }
                                session_manager.update_tab_state('time_series', sarima_params)
                            except Exception as e:
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Session Management (SARIMA)")
                                    st.warning(f"⚠️ {error_result['message']}")
                        
                        with st.spinner("Melatih model SARIMA..." if st.session_state.language == 'id' else "Training SARIMA model..."):
                            try:
                                model = train_sarima_model(
                                    train_data, 
                                    target_column, 
                                    order=(p, d, q), 
                                    seasonal_order=(P, D, Q, s)
                                )
                                st.session_state.model = model
                                st.success("Model SARIMA berhasil dilatih!" if st.session_state.language == 'id' else "SARIMA model trained successfully!")
                                
                                # Update session state with training success
                                if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                                    try:
                                        session_manager.update_tab_state('time_series', {'training_success': True, 'model_trained': 'SARIMA'})
                                    except Exception as e:
                                        if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                            error_result = error_handler.handle_error(e, "Session Update (SARIMA Success)")
                                            st.warning(f"⚠️ {error_result['message']}")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    try:
                                        model_bytes = pickle.dumps(st.session_state.model)
                                        st.download_button(
                                            label="📥 Unduh Model SARIMA (.pkl)" if st.session_state.language == 'id' else "📥 Download SARIMA Model (.pkl)",
                                            data=model_bytes,
                                            file_name=f"sarima_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                    except Exception as pickle_error:
                                        st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                            except Exception as e:
                                # Enhanced error handling for SARIMA training
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "SARIMA Model Training")
                                    st.error(f"❌ {error_result['message']}")
                                    if 'details' in error_result:
                                        with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                                            st.write(error_result['details'])
                                    if 'recommendations' in error_result:
                                        with st.expander("💡 Rekomendasi" if st.session_state.language == 'id' else "💡 Recommendations"):
                                            for rec in error_result['recommendations']:
                                                st.write(f"• {rec}")
                                else:
                                    st.error(f"Error training SARIMA: {str(e)}")
                    
                    elif model_type == "SARIMAX" and STATSMODELS_AVAILABLE:
                        p = st.slider("Parameter p (AR):", 0, 5, 1)
                        d = st.slider("Parameter d (differencing):", 0, 2, 1)
                        q = st.slider("Parameter q (MA):", 0, 5, 1)
                        P = st.slider("Parameter P (Seasonal AR):", 0, 2, 1)
                        D = st.slider("Parameter D (Seasonal differencing):", 0, 2, 1)
                        Q = st.slider("Parameter Q (Seasonal MA):", 0, 2, 1)
                        s = st.slider("Periode musiman (s):", 1, 52, 12)
                        
                        # Session management for SARIMAX parameters
                        if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                            try:
                                session_manager.initialize_tab_state('time_series')
                                sarimax_params = {
                                    'model_type': 'SARIMAX',
                                    'p': p,
                                    'd': d,
                                    'q': q,
                                    'P': P,
                                    'D': D,
                                    'Q': Q,
                                    's': s,
                                    'target_column': target_column
                                }
                                session_manager.update_tab_state('time_series', sarimax_params)
                            except Exception as e:
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Session Management (SARIMAX)")
                                    st.warning(f"⚠️ {error_result['message']}")
                        
                        with st.spinner("Melatih model SARIMAX..." if st.session_state.language == 'id' else "Training SARIMAX model..."):
                            try:
                                model = train_sarimax_model(
                                    train_data, 
                                    target_column, 
                                    order=(p, d, q), 
                                    seasonal_order=(P, D, Q, s)
                                )
                                st.session_state.model = model
                                st.success("Model SARIMAX berhasil dilatih!" if st.session_state.language == 'id' else "SARIMAX model trained successfully!")
                                
                                # Update session state with training success
                                if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                                    try:
                                        session_manager.update_tab_state('time_series', {'training_success': True, 'model_trained': 'SARIMAX'})
                                    except Exception as e:
                                        if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                            error_result = error_handler.handle_error(e, "Session Update (SARIMAX Success)")
                                            st.warning(f"⚠️ {error_result['message']}")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    try:
                                        model_bytes = pickle.dumps(st.session_state.model)
                                        st.download_button(
                                            label="📥 Unduh Model SARIMAX (.pkl)" if st.session_state.language == 'id' else "📥 Download SARIMAX Model (.pkl)",
                                            data=model_bytes,
                                            file_name=f"sarimax_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                    except Exception as pickle_error:
                                        st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                            except Exception as e:
                                # Enhanced error handling for SARIMAX training
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "SARIMAX Model Training")
                                    st.error(f"❌ {error_result['message']}")
                                    if 'details' in error_result:
                                        with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                                            st.write(error_result['details'])
                                    if 'recommendations' in error_result:
                                        with st.expander("💡 Rekomendasi" if st.session_state.language == 'id' else "💡 Recommendations"):
                                            for rec in error_result['recommendations']:
                                                st.write(f"• {rec}")
                                else:
                                    st.error(f"Error training SARIMAX: {str(e)}")
                    
                    elif model_type == "Exponential Smoothing" and STATSMODELS_AVAILABLE:
                        trend = st.selectbox("Tipe trend:", ["add", "mul", None])
                        seasonal = st.selectbox("Tipe seasonal:", ["add", "mul", None])
                        seasonal_periods = st.slider("Periode seasonal:", 0, 52, 12)
                        
                        # Session management for Exponential Smoothing parameters
                        if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                            try:
                                session_manager.initialize_tab_state('time_series')
                                exp_smooth_params = {
                                    'model_type': 'Exponential Smoothing',
                                    'trend': trend,
                                    'seasonal': seasonal,
                                    'seasonal_periods': seasonal_periods,
                                    'target_column': target_column
                                }
                                session_manager.update_tab_state('time_series', exp_smooth_params)
                            except Exception as e:
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Session Management (Exponential Smoothing)")
                                    st.warning(f"⚠️ {error_result['message']}")
                        
                        with st.spinner("Melatih model Exponential Smoothing..." if st.session_state.language == 'id' else "Training Exponential Smoothing model..."):
                            try:
                                model = train_exponential_smoothing(
                                    train_data, 
                                    target_column, 
                                    trend=trend, 
                                    seasonal=seasonal, 
                                    seasonal_periods=seasonal_periods
                                )
                                st.session_state.model = model
                                st.success("Model Exponential Smoothing berhasil dilatih!" if st.session_state.language == 'id' else "Exponential Smoothing model trained successfully!")
                                
                                # Update session state with training success
                                if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                                    try:
                                        session_manager.update_tab_state('time_series', {'training_success': True, 'model_trained': 'Exponential Smoothing'})
                                    except Exception as e:
                                        if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                            error_result = error_handler.handle_error(e, "Session Update (Exponential Smoothing Success)")
                                            st.warning(f"⚠️ {error_result['message']}")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    try:
                                        model_bytes = pickle.dumps(st.session_state.model)
                                        st.download_button(
                                            label="📥 Unduh Model Exponential Smoothing (.pkl)" if st.session_state.language == 'id' else "📥 Download Exponential Smoothing Model (.pkl)",
                                            data=model_bytes,
                                            file_name=f"exp_smoothing_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                    except Exception as pickle_error:
                                        st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                            except Exception as e:
                                # Enhanced error handling for Exponential Smoothing training
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Exponential Smoothing Model Training")
                                    st.error(f"❌ {error_result['message']}")
                                    if 'details' in error_result:
                                        with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                                            st.write(error_result['details'])
                                    if 'recommendations' in error_result:
                                        with st.expander("💡 Rekomendasi" if st.session_state.language == 'id' else "💡 Recommendations"):
                                            for rec in error_result['recommendations']:
                                                st.write(f"• {rec}")
                                else:
                                    st.error(f"Error training Exponential Smoothing: {str(e)}")
                    
                    elif model_type == "Holt-Winters" and STATSMODELS_AVAILABLE:
                        trend = st.selectbox("Tipe trend:", ["add", "mul"])
                        seasonal = st.selectbox("Tipe seasonal:", ["add", "mul"])
                        seasonal_periods = st.slider("Periode seasonal:", 1, 52, 12)
                        
                        # Session management for Holt-Winters parameters
                        if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                            try:
                                session_manager.initialize_tab_state('time_series')
                                holt_winters_params = {
                                    'model_type': 'Holt-Winters',
                                    'trend': trend,
                                    'seasonal': seasonal,
                                    'seasonal_periods': seasonal_periods,
                                    'target_column': target_column
                                }
                                session_manager.update_tab_state('time_series', holt_winters_params)
                            except Exception as e:
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Session Management (Holt-Winters)")
                                    st.warning(f"⚠️ {error_result['message']}")
                        
                        with st.spinner("Melatih model Holt-Winters..." if st.session_state.language == 'id' else "Training Holt-Winters model..."):
                            try:
                                model = train_holt_winters(
                                    train_data, 
                                    target_column, 
                                    trend=trend, 
                                    seasonal=seasonal, 
                                    seasonal_periods=seasonal_periods
                                )
                                st.session_state.model = model
                                st.success("Model Holt-Winters berhasil dilatih!" if st.session_state.language == 'id' else "Holt-Winters model trained successfully!")
                                
                                # Update session state with training success
                                if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                                    try:
                                        session_manager.update_tab_state('time_series', {'training_success': True, 'model_trained': 'Holt-Winters'})
                                    except Exception as e:
                                        if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                            error_result = error_handler.handle_error(e, "Session Update (Holt-Winters Success)")
                                            st.warning(f"⚠️ {error_result['message']}")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    try:
                                        model_bytes = pickle.dumps(st.session_state.model)
                                        st.download_button(
                                            label="📥 Unduh Model Holt-Winters (.pkl)" if st.session_state.language == 'id' else "📥 Download Holt-Winters Model (.pkl)",
                                            data=model_bytes,
                                            file_name=f"holt_winters_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                    except Exception as pickle_error:
                                        st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                            except Exception as e:
                                # Enhanced error handling for Holt-Winters training
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Holt-Winters Model Training")
                                    st.error(f"❌ {error_result['message']}")
                                    if 'details' in error_result:
                                        with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                                            st.write(error_result['details'])
                                    if 'recommendations' in error_result:
                                        with st.expander("💡 Rekomendasi" if st.session_state.language == 'id' else "💡 Recommendations"):
                                            for rec in error_result['recommendations']:
                                                st.write(f"• {rec}")
                                else:
                                    st.error(f"Error training Holt-Winters: {str(e)}")
                    
                    elif model_type == "Prophet" and PROPHET_AVAILABLE:
                        yearly_seasonality = st.selectbox("Seasonality tahunan:" if st.session_state.language == 'id' else "Yearly seasonality:", ["auto", True, False])
                        weekly_seasonality = st.selectbox("Seasonality mingguan:" if st.session_state.language == 'id' else "Weekly seasonality:", ["auto", True, False])
                        daily_seasonality = st.selectbox("Seasonality harian:" if st.session_state.language == 'id' else "Daily seasonality:", ["auto", True, False])
                        
                        # Implementasi Prophet akan dilakukan di forecasting_utils.py
                        st.info("Implementasi Prophet akan menggunakan forecasting_utils.py" if st.session_state.language == 'id' else "Prophet implementation will use forecasting_utils.py")
                    
                    elif model_type == "LSTM":
                        look_back = st.slider("Jumlah time steps untuk look back:", 10, 100, 60)
                        epochs = st.slider("Jumlah epochs:", 10, 200, 100)
                        batch_size = st.slider("Batch size:", 16, 128, 32)
                        
                        # Parameter arsitektur lanjutan dengan expander
                        with st.expander("Parameter Arsitektur Lanjutan" if st.session_state.language == 'id' else "Advanced Architecture Parameters"):
                            lstm_units = st.slider("Unit LSTM per layer:" if st.session_state.language == 'id' else "LSTM units per layer:", 10, 200, 50)
                            num_layers = st.slider("Jumlah layer LSTM:" if st.session_state.language == 'id' else "Number of LSTM layers:", 1, 5, 2)
                            dropout = st.slider("Dropout rate:" if st.session_state.language == 'id' else "Dropout rate:", 0.0, 0.5, 0.2, 0.05)
                            recurrent_dropout = st.slider("Recurrent dropout rate:" if st.session_state.language == 'id' else "Recurrent dropout rate:", 0.0, 0.5, 0.2, 0.05)
                            bidirectional = st.checkbox("Gunakan LSTM bidirectional:" if st.session_state.language == 'id' else "Use bidirectional LSTM:", value=False)
                            learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.0001, 0.01, 0.001, 0.0001)
                            optimizer = st.selectbox("Optimizer:" if st.session_state.language == 'id' else "Optimizer:", ["adam", "sgd", "rmsprop"], index=0)
                        
                        with st.spinner("Melatih model LSTM..." if st.session_state.language == 'id' else "Training LSTM model..."):
                            try:
                                model = train_lstm_model(
                                    train_data, 
                                    target_column, 
                                    look_back=look_back, 
                                    epochs=epochs, 
                                    batch_size=batch_size,
                                    lstm_units=lstm_units,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout,
                                    bidirectional=bidirectional,
                                    learning_rate=learning_rate,
                                    optimizer=optimizer
                                )
                                st.session_state.model = model
                                st.success("Model LSTM berhasil dilatih!" if st.session_state.language == 'id' else "LSTM model trained successfully!")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    try:
                                        model_bytes = pickle.dumps(st.session_state.model)
                                        st.download_button(
                                            label="📥 Unduh Model LSTM (.pkl)" if st.session_state.language == 'id' else "📥 Download LSTM Model (.pkl)",
                                            data=model_bytes,
                                            file_name=f"lstm_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                    except Exception as pickle_error:
                                        st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                            except ImportError as e:
                                st.error(f"TensorFlow tidak tersedia: {str(e)}")
                            except Exception as e:
                                st.error(f"Error training LSTM: {str(e)}")
                    
                    elif model_type == "DLinear":
                        seq_len = st.slider("Look back window (seq_len):", 3, 50, 10)
                        
                        with st.spinner("Melatih model DLinear (Deep Learning)..." if st.session_state.language == 'id' else "Training DLinear model..."):
                            try:
                                from advanced_ml import train_dlinear_forecaster
                                model_info, train_err = train_dlinear_forecaster(
                                    train_data,
                                    target_column=target_column,
                                    seq_len=seq_len,
                                    pred_len=forecast_periods
                                )
                                if model_info is None:
                                    raise ValueError(train_err)
                                    
                                st.session_state.model = model_info
                                st.success("Model DLinear berhasil dilatih!" if st.session_state.language == 'id' else "DLinear model trained successfully!")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    model_bytes = pickle.dumps(st.session_state.model)
                                    st.download_button(
                                        label="📥 Unduh Model DLinear (.pkl)" if st.session_state.language == 'id' else "📥 Download DLinear Model (.pkl)",
                                        data=model_bytes,
                                        file_name=f"dlinear_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                        mime="application/octet-stream"
                                    )
                            except Exception as e:
                                st.error(f"Error training DLinear: {str(e)}")
                    
                    elif model_type in ["Random Forest", "Gradient Boosting", "Linear Regression"]:
                        if model_type == "Random Forest":
                            n_estimators = st.slider("Jumlah trees:" if st.session_state.language == 'id' else "Number of trees:", 10, 500, 100)
                            max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                            
                            # Parameter lanjutan dengan expander
                            with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                                min_samples_split = st.slider("Jumlah sampel minimum untuk membagi:" if st.session_state.language == 'id' else "Minimum samples to split:", 2, 20, 2)
                                min_samples_leaf = st.slider("Jumlah sampel minimum di leaf:" if st.session_state.language == 'id' else "Minimum samples in leaf:", 1, 10, 1)
                                max_features = st.selectbox("Fitur maksimum:" if st.session_state.language == 'id' else "Max features:", ["sqrt", "log2", "None"], index=0)
                                bootstrap = st.checkbox("Bootstrap sampel:" if st.session_state.language == 'id' else "Bootstrap samples:", value=True)
                                
                                # Konversi max_features dari string ke None jika diperlukan
                                max_features_value = None if max_features == "None" else max_features
                            
                            model_params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'max_features': max_features_value,
                                'bootstrap': bootstrap,
                                'random_state': 42
                            }
                        elif model_type == "Gradient Boosting":
                            n_estimators = st.slider("Jumlah trees:" if st.session_state.language == 'id' else "Number of trees:", 10, 500, 100)
                            learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
                            max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                            model_params = {
                                'n_estimators': n_estimators,
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'random_state': 42
                            }
                        else:  # Linear Regression
                            model_params = {'random_state': 42}
                        
                        with st.spinner(f"Melatih model {model_type}..." if st.session_state.language == 'id' else f"Training {model_type} model..."):
                            try:
                                model_info = train_ml_forecaster(
                                    st.session_state.data,
                                    date_column,
                                    target_column,
                                    model_type=model_type.lower().replace(" ", "_"),
                                    **model_params
                                )
                                st.session_state.model = model_info
                                st.success(f"Model {model_type} berhasil dilatih!" if st.session_state.language == 'id' else f"{model_type} model trained successfully!")
                                
                                # Add download button for trained model
                                if st.session_state.model is not None:
                                    try:
                                        model_bytes = pickle.dumps(st.session_state.model)
                                        model_type_clean = model_type.lower().replace(" ", "_")
                                        st.download_button(
                                            label=f"📥 Unduh Model {model_type} (.pkl)" if st.session_state.language == 'id' else f"📥 Download {model_type} Model (.pkl)",
                                            data=model_bytes,
                                            file_name=f"{model_type_clean}_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                            mime="application/octet-stream"
                                        )
                                    except Exception as pickle_error:
                                        st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                            except Exception as e:
                                st.error(f"Error training {model_type}: {str(e)}")
                    
                    # Evaluate model if available
                    if st.session_state.model is not None:
                        with st.spinner("Mengevaluasi model..." if st.session_state.language == 'id' else "Evaluating model..."):
                            try:
                                eval_results = evaluate_forecast_model(st.session_state.model, test_data, target_column)
                                
                                # Tampilkan hasil evaluasi dengan penanganan nilai None
                                st.write("Hasil Evaluasi Model:" if st.session_state.language == 'id' else "Model Evaluation Results:")
                                
                                # Buat tabel evaluasi yang lebih rapi
                                eval_df = pd.DataFrame()
                                
                                import math
                                
                                def format_metric(value, format_str="{:.4f}"):
                                    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                                        return "N/A"
                                    try:
                                        return format_str.format(value)
                                    except:
                                        return "N/A"
                                
                                eval_df['MAE'] = [format_metric(eval_results.get('MAE'))]
                                eval_df['MSE'] = [format_metric(eval_results.get('MSE'))]
                                eval_df['RMSE'] = [format_metric(eval_results.get('RMSE'))]
                                eval_df['MAPE (%)'] = [format_metric(eval_results.get('MAPE'), "{:.2f}%")]
                                eval_df['R²'] = [format_metric(eval_results.get('R2'))]
                                    
                                st.dataframe(eval_df, use_container_width=True)
                                
                                # Tampilkan pesan error jika ada
                                if 'error' in eval_results:
                                    st.warning(f"⚠️ {eval_results['error']}" if st.session_state.language == 'id' else f"⚠️ {eval_results['error']}")
                                
                                # Generate forecast
                                try:
                                    forecast_data = forecast_future(st.session_state.model, periods=forecast_periods)
                                    
                                    # Validasi dan perbaikan data forecast
                                    if forecast_data is not None and not forecast_data.empty:
                                        # Pastikan kolom tanggal dalam format yang benar
                                        if 'date' in forecast_data.columns:
                                            try:
                                                forecast_data['date'] = pd.to_datetime(forecast_data['date'], errors='coerce')
                                                forecast_data = forecast_data.dropna(subset=['date'])
                                            except Exception:
                                                # Jika gagal, biarkan sebagai string
                                                pass
                                        
                                        # Pastikan kolom forecast ada dan valid
                                        if 'forecast' not in forecast_data.columns:
                                            st.warning("Data forecast tidak memiliki kolom 'forecast'")
                                            forecast_data = None
                                        else:
                                            # Hapus nilai forecast yang tidak valid
                                            forecast_data = forecast_data.dropna(subset=['forecast'])
                                            if forecast_data.empty:
                                                st.warning("Data forecast kosong setelah validasi")
                                                forecast_data = None
                                    else:
                                        st.warning("Data forecast kosong atau tidak valid")
                                        forecast_data = None
                                        
                                except Exception as e:
                                    st.error(f"Error saat membuat forecast: {str(e)}" if st.session_state.language == 'id' else f"Error generating forecast: {str(e)}")
                                    forecast_data = None

                                # Plot results dengan penanganan error yang lebih baik
                                if forecast_data is not None and not forecast_data.empty:
                                    try:
                                        fig = plot_forecast_results(train_data, test_data, forecast_data, target_column)
                                        if fig is not None:
                                            st.pyplot(fig)
                                            plt.close(fig)  # Tutup figure untuk menghemat memory
                                        else:
                                            st.warning("Gagal membuat plot forecast")
                                        
                                        # Store data for visualization
                                        st.session_state.forecast_data = forecast_data
                                        st.session_state.train_data = train_data
                                        st.session_state.test_data = test_data
                                        st.session_state.target_column = target_column
                                        st.session_state.eval_results = eval_results
                                        
                                    except Exception as e:
                                        st.error(f"Error saat memplot hasil: {str(e)}" if st.session_state.language == 'id' else f"Error plotting results: {str(e)}")
                                        st.info("Menampilkan data forecast dalam bentuk tabel..." if st.session_state.language == 'id' else "Displaying forecast data in table format...")

                                # Show forecast data dengan preview
                                if forecast_data is not None and not forecast_data.empty:
                                    st.write("Data Hasil Forecasting:" if st.session_state.language == 'id' else "Forecast Data:")
                                    st.dataframe(forecast_data.head(50))  # Tampilkan maksimal 50 baris
                                    
                                    # Download button untuk forecast data
                                    csv = forecast_data.to_csv(index=False)
                                    st.download_button(
                                        label="Download Forecast Data (CSV)" if st.session_state.language == 'id' else "Download Forecast Data (CSV)",
                                        data=csv,
                                        file_name=f"forecast_{model_type.lower().replace(' ', '_')}.csv",
                                        mime="text/csv"
                                    )

                            except Exception as e:
                                st.error(f"Error saat evaluasi model: {str(e)}" if st.session_state.language == 'id' else f"Error evaluating model: {str(e)}")
                                st.info("Pastikan model telah dilatih dengan benar dan data yang digunakan sesuai." if st.session_state.language == 'id' else "Please ensure the model is properly trained and data is appropriate.")

                    # Button for detailed visualization
                    if hasattr(st.session_state, 'forecast_data') and st.session_state.forecast_data is not None:
                        if st.button("Tampilkan Visualisasi Forecasting Lengkap" if st.session_state.language == 'id' else "Show Complete Forecasting Visualization"):
                            try:
                                log_feature('forecasting_visualization_full')
                            except Exception:
                                pass
                            display_forecast_summary(
                                st.session_state.train_data,
                                st.session_state.test_data,
                                st.session_state.forecast_data,
                                st.session_state.target_column,
                                st.session_state.eval_results
                            )

    else:
        # Non-time series data - Classification or Regression
        st.subheader(f"Melatih Model {problem_type}" if st.session_state.language == 'id' else f"Training a {problem_type} Model")
        
        # Clustering insights integration
        clustering_insights_available = False
        clustering_results = None
        
        if SESSION_MANAGER_AVAILABLE and session_manager is not None:
            try:
                clustering_results = session_manager.get_tab_state('clustering')
                if clustering_results and 'algorithm' in clustering_results:
                    clustering_insights_available = True
            except Exception as e:
                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                    error_result = error_handler.handle_error(e, "Clustering Insights Retrieval")
                    st.warning(f"⚠️ {error_result['message']}")
        else:
            # Fallback to direct session state
            clustering_results = st.session_state.get('clustering_results', {})
            if clustering_results and 'algorithm' in clustering_results:
                clustering_insights_available = True
        
        if clustering_insights_available and clustering_results:
            with st.expander("🔍 Wawasan Clustering dari Tab EDA" if st.session_state.language == 'id' else "🔍 Clustering Insights from EDA Tab"):
                st.write(f"**Algoritma Clustering:** {clustering_results['algorithm']}")
                st.write(f"**Jumlah Cluster:** {clustering_results['n_clusters']}")
                
                if clustering_results.get('silhouette_score', 0) > 0:
                    st.write(f"**Silhouette Score:** {clustering_results['silhouette_score']:.3f}")
                
                if clustering_results.get('calinski_harabasz_score', 0) > 0:
                    st.write(f"**Calinski-Harabasz Score:** {clustering_results['calinski_harabasz_score']:.1f}")
                
                # Recommendations based on clustering results
                if clustering_results['n_clusters'] <= 3:
                    st.info("💡 **Rekomendasi:** Jumlah cluster yang sedikit menunjukkan data yang relatif homogen. Model seperti Random Forest atau Gradient Boosting cocok untuk data ini." if st.session_state.language == 'id' else "💡 **Recommendation:** Few clusters indicate relatively homogeneous data. Models like Random Forest or Gradient Boosting are suitable for this data.")
                elif clustering_results['n_clusters'] >= 5:
                    st.info("💡 **Rekomendasi:** Banyak cluster menunjukkan data yang heterogen. Pertimbangkan ensemble methods atau neural networks untuk menangani kompleksitas." if st.session_state.language == 'id' else "💡 **Recommendation:** Many clusters indicate heterogeneous data. Consider ensemble methods or neural networks to handle complexity.")
                
                if clustering_results.get('silhouette_score', 0) > 0.5:
                    st.success("✅ Clustering yang baik (Silhouette > 0.5) menunjukkan struktur data yang jelas. Model supervised learning akan berkinerja baik." if st.session_state.language == 'id' else "✅ Good clustering (Silhouette > 0.5) indicates clear data structure. Supervised learning models will perform well.")
                elif clustering_results.get('silhouette_score', 0) < 0.2:
                    st.warning("⚠️ Clustering yang buruk (Silhouette < 0.2) menunjukkan struktur data yang tidak jelas. Pertimbangkan feature engineering tambahan atau model ensemble." if st.session_state.language == 'id' else "⚠️ Poor clustering (Silhouette < 0.2) indicates unclear data structure. Consider additional feature engineering or ensemble models.")
        
        # Opsi untuk hyperparameter optimization
        optimization_method = st.selectbox(
            "Metode Hyperparameter Optimization:" if st.session_state.language == 'id' else "Hyperparameter Optimization Method:",
            ["None", "GridSearchCV", "RandomizedSearchCV", "Bayesian Optimization (Optuna)"]
        )
        
        # Opsi rentang parameter kustom
        use_custom_ranges = False
        custom_param_ranges = {}
        
        if optimization_method in ["GridSearchCV", "RandomizedSearchCV"]:
            use_custom_ranges = st.checkbox(
                "Gunakan rentang parameter kustom (Klasifikasi)" if st.session_state.language == 'id' else "Use custom parameter ranges (Classification)",
                value=False,
                help="Aktifkan untuk menentukan rentang parameter sendiri untuk model klasifikasi" if st.session_state.language == 'id' else "Enable to specify custom parameter ranges for classification models",
                key="classification_custom_ranges"
            )
            
            if use_custom_ranges:
                st.info("💡 Gunakan format: min:max:step untuk numerik, atau val1,val2,val3 untuk kategorikal" if st.session_state.language == 'id' else "💡 Use format: min:max:step for numeric, or val1,val2,val3 for categorical")
                st.info("⚠️ Kosongkan field untuk menggunakan rentang default" if st.session_state.language == 'id' else "⚠️ Leave field empty to use default ranges")
        
        # Model selection
        if problem_type == "Classification":
            # Define available classification models
            classification_models = ["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree", "Naive Bayes", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Voting Classifier", "Stacking Classifier", "MLP (Neural Network)"]
                               
            model_type = st.selectbox("Select a classification model:" if st.session_state.language == 'id' else "Pilih model klasifikasi:", classification_models)
            st.session_state.model_type = model_type
            
            # Data type detection integration for supervised ML
            if DATA_TYPE_DETECTOR_AVAILABLE and data_type_detector is not None:
                try:
                    with st.expander("🔍 Deteksi Tipe Data Otomatis" if st.session_state.language == 'id' else "🔍 Automatic Data Type Detection"):
                        # Use detect_column_types instead of non-existent detect_data_types
                        data_info = data_type_detector.detect_column_types(st.session_state.X_train)
                        st.write("**Tipe Data yang Terdeteksi:**" if st.session_state.language == 'id' else "**Detected Data Types:**")
                        
                        numeric_features = []
                        categorical_features = []
                        
                        for feature_name, analysis in data_info.items():
                            detected_type = analysis['detected_type']
                            confidence = analysis['confidence']
                            
                            if detected_type == 'numeric':
                                numeric_features.append(feature_name)
                            elif detected_type == 'object' or detected_type == 'categorical':
                                categorical_features.append(feature_name)
                            
                            st.write(f"- {feature_name}: {detected_type} (confidence: {confidence:.2f})")
                        
                        # Model recommendations based on data types
                        if len(categorical_features) > len(numeric_features):
                            st.info("💡 **Rekomendasi Model:** Random Forest, Gradient Boosting, atau XGBoost cocok untuk data dengan banyak fitur kategorikal" if st.session_state.language == 'id' else "💡 **Model Recommendation:** Random Forest, Gradient Boosting, or XGBoost are suitable for data with many categorical features")
                        elif len(numeric_features) > len(categorical_features):
                            st.info("💡 **Rekomendasi Model:** SVM, Logistic Regression, atau Neural Network cocok untuk data dengan banyak fitur numerik" if st.session_state.language == 'id' else "💡 **Model Recommendation:** SVM, Logistic Regression, or Neural Network are suitable for data with many numeric features")
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Data Type Detection (Classification)")
                        st.warning(f"⚠️ {error_result['message']}")
                    else:
                        st.warning(f"⚠️ Gagal mendeteksi tipe data: {str(e)}")
            
            if model_type == "Random Forest":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                n_estimators = st.slider("Number of trees:" if st.session_state.language == 'id' else "Jumlah pohon:", 10, 500, 100)
                max_depth = st.slider("Maximum depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 50, 10)
                
                # Parameter lanjutan dengan expander
                with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                    min_samples_split = st.slider("Minimum samples to split:" if st.session_state.language == 'id' else "Jumlah sampel minimum untuk membagi:", 2, 20, 2)
                    min_samples_leaf = st.slider("Minimum samples in leaf:" if st.session_state.language == 'id' else "Jumlah sampel minimum di leaf:", 1, 10, 1)
                    max_features = st.selectbox("Max features:" if st.session_state.language == 'id' else "Fitur maksimum:", ["sqrt", "log2", "None"], index=0)
                    bootstrap = st.checkbox("Bootstrap samples:" if st.session_state.language == 'id' else "Bootstrap sampel:", value=True)
                    
                    # Konversi max_features dari string ke None jika diperlukan
                    max_features_value = None if max_features == "None" else max_features
                
                base_model = RandomForestClassifier(random_state=42)
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                        'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'n_estimators': list(range(50, 301, 25)),
                        'max_depth': list(range(3, 21)) + [None],
                        'min_samples_split': [2, 5, 10, 15, 20],
                        'min_samples_leaf': [1, 2, 4, 8, 16],
                        'max_features': ['sqrt', 'log2', None]
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='accuracy', n_jobs=-1, n_iter=50, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "Random Forest", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = RandomForestClassifier(**best_params, random_state=42)
                else:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features_value,
                        bootstrap=bootstrap,
                        random_state=42
                    )
                    
            elif model_type == "Logistic Regression" :
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.01, 10.0, 1.0)
                max_iter = st.slider("Maximum iterations:" if st.session_state.language == 'id' else "Iterasi maksimum:", 100, 1000, 100)
                
                # Parameter lanjutan dalam expander
                with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                    class_weight = st.selectbox("Class weight:" if st.session_state.language == 'id' else "Bobot kelas:", ["None", "balanced", "balanced_subsample"], index=0)
                    solver = st.selectbox("Solver algorithm:" if st.session_state.language == 'id' else "Algoritma solver:", ["lbfgs", "liblinear", "saga", "newton-cg", "sag"], index=0)
                    penalty = st.selectbox("Penalty type:" if st.session_state.language == 'id' else "Jenis penalti:", ["l2", "l1", "elasticnet"], index=0)
                    
                    # Konversi class_weight dari string ke None jika diperlukan
                    class_weight_value = None if class_weight == "None" else class_weight
                    
                    # Elastic net memerlukan parameter l1_ratio
                    l1_ratio = None
                    if penalty == "elasticnet":
                        l1_ratio = st.slider("L1 ratio (for elasticnet):" if st.session_state.language == 'id' else "Rasio L1 (untuk elasticnet):", 0.0, 1.0, 0.5)
                
                base_model = LogisticRegression(random_state=42)
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.01, C/2), C, min(10.0, C*2)],
                        'solver': ['liblinear', 'lbfgs', 'saga'],
                        'max_iter': [100, 500, 1000]
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'solver': ['liblinear', 'lbfgs', 'saga'],
                        'max_iter': [100, 500, 1000],
                        'penalty': ['l1', 'l2', 'elasticnet']
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "Logistic Regression", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = LogisticRegression(**best_params, random_state=42)
                else:
                    model = LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight_value,
                        solver=solver,
                        penalty=penalty,
                        l1_ratio=l1_ratio,
                        random_state=42
                    )
                    
            elif model_type == "SVM":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.1, 10.0, 1.0)
                kernel = st.selectbox("Kernel:" if st.session_state.language == 'id' else "Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                gamma = st.selectbox("Gamma (kernel coefficient):" if st.session_state.language == 'id' else "Gamma (koefisien kernel):", ["scale", "auto"])
                
                # Parameter lanjutan dalam expander
                with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                    coef0 = st.slider("Coefficient for polynomial kernel (coef0):" if st.session_state.language == 'id' else "Koefisien untuk kernel polinomial (coef0):", 0.0, 1.0, 0.0)
                    shrinking = st.checkbox("Use shrinking heuristic:" if st.session_state.language == 'id' else "Gunakan heuristik shrinking:", value=True)
                    probability = st.checkbox("Enable probability estimates:" if st.session_state.language == 'id' else "Aktifkan estimasi probabilitas:", value=True)
                    
                    # Parameter degree hanya untuk kernel poly
                    degree = 3  # default
                    if kernel == "poly":
                        degree = st.slider("Degree for polynomial kernel:" if st.session_state.language == 'id' else "Derajat untuk kernel polinomial:", 2, 10, 3)
                
                base_model = SVC(probability=probability, random_state=42)
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.1, C/2), C, min(10.0, C*2)],
                        'kernel': [kernel] if kernel != "rbf" else ['linear', 'rbf'],
                        'gamma': [gamma] if gamma != "scale" else ['scale', 'auto'],
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'C': [0.01, 0.1, 1, 10, 100],
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                        'degree': [2, 3, 4, 5]  # untuk kernel poly
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "SVM", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = SVC(**best_params, probability=True, random_state=42)
                else:
                    model = SVC(
                        C=C,
                        kernel=kernel,
                        gamma=gamma,
                        coef0=coef0,
                        shrinking=shrinking,
                        probability=probability,
                        degree=degree if kernel == "poly" else 3,
                        random_state=42
                    )
                    
            elif model_type == "KNN":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                n_neighbors = st.slider("Number of neighbors (K):" if st.session_state.language == 'id' else "Jumlah tetangga (K):", 1, 20, 5)
                weights = st.selectbox("Weight function:" if st.session_state.language == 'id' else "Fungsi bobot:", ["uniform", "distance"])
                algorithm = st.selectbox("Algorithm:" if st.session_state.language == 'id' else "Algoritma:", ["auto", "ball_tree", "kd_tree", "brute"])
                
                # Parameter lanjutan dalam expander
                with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                    metric = st.selectbox("Distance metric:" if st.session_state.language == 'id' else "Metrik jarak:", ["minkowski", "euclidean", "manhattan", "chebyshev", "wminkowski", "seuclidean", "mahalanobis"], index=0)
                    p_value = st.slider("Power parameter for Minkowski metric:" if st.session_state.language == 'id' else "Parameter daya untuk metrik Minkowski:", 1, 5, 2)
                    leaf_size = st.slider("Leaf size:" if st.session_state.language == 'id' else "Ukuran daun:", 10, 50, 30)
                    
                    # Metric parameters untuk metrik tertentu
                    metric_params = None
                    if metric == "wminkowski":
                        w = st.text_input("Weight vector for wminkowski (comma-separated):" if st.session_state.language == 'id' else "Vektor bobot untuk wminkowski (pisahkan koma):", "1,1,1")
                        try:
                            metric_params = {'w': [float(x.strip()) for x in w.split(",")]}
                        except:
                            metric_params = None
                    elif metric == "mahalanobis":
                        VI = st.text_input("Inverse covariance matrix for mahalanobis (optional):" if st.session_state.language == 'id' else "Matriks kovarian terbalik untuk mahalanobis (opsional):", "")
                        if VI:
                            try:
                                # Parse matrix dari string (sederhana)
                                metric_params = {'VI': np.eye(10)}  # Default identity matrix
                            except:
                                metric_params = None
                
                base_model = KNeighborsClassifier()
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'n_neighbors': [3, 5, 7] if n_neighbors == 5 else [max(1, n_neighbors-2), n_neighbors, min(20, n_neighbors+2)],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'p': [1, 2]  # Manhattan or Euclidean distance
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'n_neighbors': list(range(3, 21)),
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'p': [1, 2, 3, 4, 5]
                    }
                    # Gabungkan dengan custom parameter ranges
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=20, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "KNN", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = KNeighborsClassifier(**best_params)
                else:
                    model = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        algorithm=algorithm,
                        metric=metric,
                        p=p_value,
                        leaf_size=leaf_size,
                        metric_params=metric_params
                    )
                    
            elif model_type == "Decision Tree":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                max_depth = st.slider("Maximum depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 50, 10)
                min_samples_split = st.slider("Minimum samples to split:" if st.session_state.language == 'id' else "Jumlah sampel untuk membagi:", 2, 20, 2)
                criterion = st.selectbox("Split criterion:" if st.session_state.language == 'id' else "Kriteria membagi:", ["gini", "entropy"])
                
                # Parameter lanjutan untuk Decision Tree
                with st.expander("Parameter Lanjutan Decision Tree" if st.session_state.language == 'id' else "Advanced Decision Tree Parameters"):
                    min_samples_leaf = st.slider("Minimum samples per leaf:" if st.session_state.language == 'id' else "Jumlah sampel minimum per leaf:", 1, 20, 1)
                    max_features_options = ["None", "sqrt", "log2", "auto"]
                    max_features = st.selectbox("Max features for split:" if st.session_state.language == 'id' else "Fitur maksimum untuk pembelahan:", max_features_options)
                    max_features_value = None if max_features == "None" else max_features
                    
                    class_weight_options = ["None", "balanced"]
                    class_weight = st.selectbox("Class weight:" if st.session_state.language == 'id' else "Bobot kelas:", class_weight_options)
                    class_weight_value = None if class_weight == "None" else class_weight
                    
                    ccp_alpha = st.slider("Cost complexity pruning alpha:" if st.session_state.language == 'id' else "Alpha pemangkasan kompleksitas biaya:", 0.0, 0.1, 0.0, 0.001)
                
                base_model = DecisionTreeClassifier(random_state=42)
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'criterion': ['gini', 'entropy']
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'max_depth': list(range(1, 21)) + [None],
                        'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
                        'min_samples_leaf': [1, 2, 4, 8, 16, 32],
                        'criterion': ['gini', 'entropy', 'log_loss']
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "Decision Tree", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = DecisionTreeClassifier(**best_params, random_state=42)
                else:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        criterion=criterion,
                        max_features=max_features_value,
                        class_weight=class_weight_value,
                        ccp_alpha=ccp_alpha,
                        random_state=42
                    )
                    
            elif model_type == "Naive Bayes":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                var_smoothing = st.slider("Variance smoothing:" if st.session_state.language == 'id' else "Penyesuaian varian:", 1e-10, 1e-8, 1e-9, format="%.1e")
                
                base_model = GaussianNB()
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'var_smoothing': [1e-10, 1e-9, 1e-8]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "Naive Bayes", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=10, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = GaussianNB(**best_params)
                else:
                    model = GaussianNB(
                        var_smoothing=var_smoothing
                    )
                    
            elif model_type == "Gradient Boosting":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                n_estimators = st.slider("Number of boosting stages:" if st.session_state.language == 'id' else "Jumlah boosting stages:", 10, 500, 100)
                learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 10, 3)
                
                # Parameter lanjutan untuk Gradient Boosting
                with st.expander("Parameter Lanjutan Gradient Boosting" if st.session_state.language == 'id' else "Advanced Gradient Boosting Parameters"):
                    subsample = st.slider("Subsample ratio:" if st.session_state.language == 'id' else "Rasio subsample:", 0.5, 1.0, 1.0, 0.1)
                    min_samples_split = st.slider("Minimum samples to split:" if st.session_state.language == 'id' else "Jumlah sampel minimum untuk membagi:", 2, 20, 2)
                    min_samples_leaf = st.slider("Minimum samples per leaf:" if st.session_state.language == 'id' else "Jumlah sampel minimum per leaf:", 1, 20, 1)
                    max_features_options = ["None", "sqrt", "log2", "auto"]
                    max_features = st.selectbox("Max features for split:" if st.session_state.language == 'id' else "Fitur maksimum untuk pembelahan:", max_features_options)
                    max_features_value = None if max_features == "None" else max_features
                
                base_model = GradientBoostingClassifier(random_state=42)
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                        'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                        'max_depth': [3, 6, 9] if max_depth == 3 else [max(1, max_depth-3), max_depth, min(10, max_depth+3)],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'n_estimators': list(range(50, 301, 25)),
                        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                        'max_depth': list(range(3, 16)),
                        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_samples_split': [2, 5, 10, 15, 20],
                        'min_samples_leaf': [1, 2, 4, 8, 16]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=40, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "Gradient Boosting", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = GradientBoostingClassifier(**best_params, random_state=42)
                else:
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features_value,
                        random_state=42
                    )
                    
            elif model_type == "XGBoost":
                if not XGBOOST_AVAILABLE:
                    st.error("XGBoost tidak terinstal. Silakan install dengan: pip install xgboost")
                    model = None
                else:
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Number of boosting rounds:" if st.session_state.language == 'id' else "Jumlah boosting rounds:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Max depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 15, 6)
                    
                    with st.expander("Parameter Lanjutan XGBoost" if st.session_state.language == 'id' else "Advanced XGBoost Parameters"):
                        subsample = st.slider("Subsample ratio:" if st.session_state.language == 'id' else "Rasio subsample:", 0.5, 1.0, 1.0, 0.1)
                        colsample_bytree = st.slider("Column sample by tree:" if st.session_state.language == 'id' else "Sample kolom per tree:", 0.5, 1.0, 1.0, 0.1)
                        min_child_weight = st.slider("Min child weight:" if st.session_state.language == 'id' else "Bobot minimum anak:", 1, 10, 1)
                        gamma = st.slider("Gamma (min loss reduction):" if st.session_state.language == 'id' else "Gamma (min pengurangan loss):", 0.0, 5.0, 0.0, 0.1)
                    
                    base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [3, 6, 9] if max_depth == 6 else [max(1, max_depth-3), max_depth, min(15, max_depth+3)]
                        }
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'max_depth': list(range(3, 16)),
                            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                            'colsample_bytree': [0.6, 0.8, 1.0],
                            'min_child_weight': [1, 3, 5, 7]
                        }
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "XGBoost", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
                    else:
                        model = XGBClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            min_child_weight=min_child_weight,
                            gamma=gamma,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='logloss'
                        )
                    
            elif model_type == "LightGBM":
                if not LIGHTGBM_AVAILABLE:
                    st.error("LightGBM tidak terinstal. Silakan install dengan: pip install lightgbm")
                    model = None
                else:
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Number of boosting rounds:" if st.session_state.language == 'id' else "Jumlah boosting rounds:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Max depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", -1, 15, -1, help="-1 means no limit")
                    
                    with st.expander("Parameter Lanjutan LightGBM" if st.session_state.language == 'id' else "Advanced LightGBM Parameters"):
                        num_leaves = st.slider("Number of leaves:" if st.session_state.language == 'id' else "Jumlah daun:", 2, 256, 31)
                        subsample = st.slider("Subsample ratio:" if st.session_state.language == 'id' else "Rasio subsample:", 0.5, 1.0, 1.0, 0.1)
                        colsample_bytree = st.slider("Column sample by tree:" if st.session_state.language == 'id' else "Sample kolom per tree:", 0.5, 1.0, 1.0, 0.1)
                        min_child_samples = st.slider("Min child samples:" if st.session_state.language == 'id' else "Min sampel anak:", 1, 50, 20)
                    
                    base_model = LGBMClassifier(random_state=42, verbose=-1)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [5, 10, 15] if max_depth == -1 else [max(1, max_depth-3), max_depth, min(15, max_depth+3)]
                        }
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'max_depth': [-1, 5, 10, 15, 20],
                            'num_leaves': [20, 31, 50, 100, 150],
                            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                            'colsample_bytree': [0.6, 0.8, 1.0]
                        }
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "LightGBM", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
                    else:
                        model = LGBMClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            num_leaves=num_leaves,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            min_child_samples=min_child_samples,
                            random_state=42,
                            verbose=-1
                        )
                    
            elif model_type == "CatBoost":
                if not CATBOOST_AVAILABLE:
                    st.error("CatBoost tidak terinstal. Silakan install dengan: pip install catboost")
                    model = None
                else:
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    iterations = st.slider("Number of iterations:" if st.session_state.language == 'id' else "Jumlah iterasi:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    depth = st.slider("Tree depth:" if st.session_state.language == 'id' else "Kedalaman tree:", 2, 16, 6)
                    
                    with st.expander("Parameter Lanjutan CatBoost" if st.session_state.language == 'id' else "Advanced CatBoost Parameters"):
                        l2_leaf_reg = st.slider("L2 leaf regularization:" if st.session_state.language == 'id' else "Regularisasi L2 leaf:", 1.0, 10.0, 3.0, 0.5)
                        border_count = st.slider("Border count:" if st.session_state.language == 'id' else "Jumlah border:", 1, 255, 128)
                        random_strength = st.slider("Random strength:" if st.session_state.language == 'id' else "Kekuatan random:", 0.0, 2.0, 1.0, 0.1)
                    
                    base_model = CatBoostClassifier(random_state=42, verbose=False)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'iterations': [50, 100, 200] if iterations == 100 else [max(10, iterations-50), iterations, min(500, iterations+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'depth': [4, 6, 8] if depth == 6 else [max(2, depth-2), depth, min(16, depth+2)]
                        }
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'iterations': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'depth': [4, 6, 8, 10, 12],
                            'l2_leaf_reg': [1.0, 3.0, 5.0, 7.0, 9.0],
                            'border_count': [32, 64, 128, 254]
                        }
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "CatBoost", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = CatBoostClassifier(**best_params, random_state=42, verbose=False)
                    else:
                        model = CatBoostClassifier(
                            iterations=iterations,
                            learning_rate=learning_rate,
                            depth=depth,
                            l2_leaf_reg=l2_leaf_reg,
                            border_count=border_count,
                            random_strength=random_strength,
                            random_state=42,
                            verbose=False
                        )
                    
            elif model_type == "Voting Classifier":
                # Pilih base estimators untuk VotingClassifier
                base_estimators = []
                if st.checkbox("Gunakan Random Forest", value=True, key="vote_clf_rf"):
                    base_estimators.append(('rf', RandomForestClassifier(n_estimators=50, random_state=42)))
                if st.checkbox("Gunakan Logistic Regression", value=True, key="vote_clf_lr"):
                    base_estimators.append(('lr', LogisticRegression(max_iter=500)))
                if st.checkbox("Gunakan Gradient Boosting", value=False, key="vote_clf_gb"):
                    base_estimators.append(('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)))
                if st.checkbox("Gunakan KNN", value=False, key="vote_clf_knn"):
                    base_estimators.append(('knn', KNeighborsClassifier()))
                if st.checkbox("Gunakan Naive Bayes", value=False, key="vote_clf_nb"):
                    base_estimators.append(('nb', GaussianNB()))
                if st.checkbox("Gunakan SVM", value=False, key="vote_clf_svm"):
                    base_estimators.append(('svm', SVC(probability=True)))
                if st.checkbox("Gunakan XGBoost", value=False, key="vote_clf_xgb"):
                    if XGBOOST_AVAILABLE:
                        base_estimators.append(('xgb', XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')))
                    else:
                        st.warning("XGBoost tidak tersedia. Silakan install dengan: pip install xgboost")
                if len(base_estimators) < 2:
                    st.warning("Pilih minimal dua base estimator untuk Voting Classifier." if st.session_state.language == 'id' else "Select at least two base estimators for Voting Classifier.")
                    model = None
                else:
                    voting_type = st.selectbox("Voting type:", ["hard", "soft"], help="Hard: majority voting | Soft: probability-based voting")
                    model = VotingClassifier(estimators=base_estimators, voting=voting_type)
                    
            elif model_type == "Stacking Classifier":
                # Simple stacking with 2-3 base models and a final classifier
                base_estimators = []
                if st.checkbox("Gunakan Random Forest (Stacking)", value=True, key="stack_clf_rf"):
                    base_estimators.append(('rf', RandomForestClassifier(n_estimators=50, random_state=42)))
                if st.checkbox("Gunakan Logistic Regression (Stacking)", value=True, key="stack_clf_lr"):
                    base_estimators.append(('lr', LogisticRegression(max_iter=500)))
                if st.checkbox("Gunakan Gradient Boosting (Stacking)", value=False, key="stack_clf_gb"):
                    base_estimators.append(('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)))
                if st.checkbox("Gunakan SVM (Stacking)", value=False, key="stack_clf_svm"):
                    base_estimators.append(('svm', SVC(probability=True)))
                if st.checkbox("Gunakan XGBoost (Stacking)", value=False, key="stack_clf_xgb"):
                    if XGBOOST_AVAILABLE:
                        base_estimators.append(('xgb', XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')))
                    else:
                        st.warning("XGBoost tidak tersedia. Silakan install dengan: pip install xgboost")
                
                final_estimator = st.selectbox("Final estimator:" if st.session_state.language == 'id' else "Final estimator:", ["Logistic Regression", "Random Forest"], key="stack_clf_final")
                if final_estimator == "Logistic Regression":
                    final = LogisticRegression(max_iter=500)
                else:
                    final = RandomForestClassifier(n_estimators=20, random_state=42)
                if len(base_estimators) < 2:
                    st.warning("Pilih minimal dua base estimator untuk Stacking Classifier." if st.session_state.language == 'id' else "Select at least two base estimators for Stacking Classifier.")
                    model = None
                else:
                    model = StackingClassifier(
                        estimators=base_estimators,
                        final_estimator=final,
                        passthrough=True
                    )
                    
            elif model_type == "MLP (Neural Network)":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                st.subheader("Konfigurasi Neural Network Klasifikasi Lengkap" if st.session_state.language == 'id' else "Complete Neural Network Classification Configuration")
                
                # Tambahkan penjelasan teori di bagian atas
                with st.expander("📚 Penjelasan Teori Neural Network Klasifikasi" if st.session_state.language == 'id' else "📚 Neural Network Classification Theory Explanation"):
                    st.markdown("""
                    ### 🧠 **Fungsi Aktivasi**
                    - **ReLU**: f(x) = max(0,x) - Cocok untuk hidden layers, mengatasi vanishing gradient
                    - **Sigmoid**: f(x) = 1/(1+e^(-x)) - Cocok untuk output binary classification
                    - **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - Range [-1,1], lebih stabil dari sigmoid
                    
                    ### 🏗️ **Arsitektur Jaringan**
                    - **Feedforward Neural Network (FNN)**: Informasi mengalir satu arah (input → hidden → output)
                    - **Parameter yang dikonfigurasi**: Hidden layers, neurons per layer, activation function
                    
                    ### ⚡ **Optimizer & Learning**
                    - **Adam**: Kombinasi momentum dan adaptive learning rate, efisien untuk data besar
                    - **SGD**: Stochastic Gradient Descent dengan momentum untuk konvergensi lebih stabil
                    - **L-BFGS**: Optimizer kuasi-Newton untuk dataset kecil/medium
                    
                    ### 🛡️ **Regularization & Overfitting Prevention**
                    - **L2 Regularization (alpha)**: Menghindari overfitting dengan menjaga bobot tetap kecil
                    - **Early Stopping**: Menghentikan training saat validasi tidak membaik
                    - **Dropout**: (Tidak tersedia di MLPClassifier scikit-learn)
                    
                    ### 📊 **Hyperparameter Penting**
                    - **Learning Rate**: Kontrol kecepatan pembelajaran (0.001-0.1)
                    - **Batch Size**: Jumlah sampel per update (16-512)
                    - **Epochs**: Jumlah iterasi seluruh dataset (max_iter)
                    - **Hidden Layers**: Kompleksitas model (1-5 layers)
                    """)
                
                # Mode konfigurasi parameter
                config_mode = st.radio(
                    "Mode Konfigurasi Parameter:" if st.session_state.language == 'id' else "Parameter Configuration Mode:",
                    ["Quick Setup", "Advanced Settings"],
                    horizontal=True,
                    help="Quick Setup: Parameter dasar | Advanced Settings: Kontrol penuh semua parameter"
                )
                
                if config_mode == "Quick Setup":
                    # Architecture Configuration - Quick Setup
                    st.write("**Arsitektur Jaringan Klasifikasi:**" if st.session_state.language == 'id' else "**Classification Network Architecture:**")
                    
                    # Hidden layers configuration
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        num_hidden_layers = st.slider("Jumlah hidden layers:", 1, 5, 2)
                    with col2:
                        neurons_per_layer = st.text_input("Neurons per layer:", "128,128")
                    try:
                        neurons_list = [int(x.strip()) for x in neurons_per_layer.split(",")]
                        if len(neurons_list) < num_hidden_layers:
                            neurons_list.extend([neurons_list[-1]] * (num_hidden_layers - len(neurons_list)))
                        elif len(neurons_list) > num_hidden_layers:
                            neurons_list = neurons_list[:num_hidden_layers]
                        hidden_layer_sizes = tuple(neurons_list)
                    except:
                        hidden_layer_sizes = (128, 128)
                with col3:
                    activation = st.selectbox("Activation function:", 
                                            ["relu", "tanh", "logistic", "identity"],
                                            help="ReLU: max(0,x) | Sigmoid: 1/(1+e^-x) | Tanh: (e^x-e^-x)/(e^x+e^-x) | Identity: x")
                
                # Advanced parameters
                with st.expander("Advanced Parameters"):
                    col4, col5 = st.columns(2)
                    with col4:
                        solver = st.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"])
                        
                        if solver == "adam":
                            beta_1 = st.slider("Beta 1:", 0.8, 0.999, 0.9, format="%.3f")
                            beta_2 = st.slider("Beta 2:", 0.9, 0.9999, 0.999, format="%.4f")
                            epsilon = st.slider("Epsilon:", 1e-8, 1e-3, 1e-8, format="%.1e")
                        elif solver == "sgd":
                            momentum = st.slider("Momentum:", 0.0, 0.9, 0.9)
                            power_t = st.slider("Power t:", 0.1, 0.9, 0.5)
                            
                    with col5:
                        learning_rate_init = st.slider("Initial learning rate:", 0.0001, 0.001, 0.0003, format="%.4f")
                        learning_rate = st.selectbox("Learning rate schedule:", ["constant", "invscaling", "adaptive"])
                        
                    col6, col7 = st.columns(2)
                    with col6:
                        alpha = st.slider("L2 regularization (alpha):", 0.00001, 0.1, 0.0001, format="%.5f")
                        batch_size = st.selectbox("Batch size:", ["auto", 16, 32, 64, 128, 256])
                        if batch_size == "auto":
                            actual_batch_size = min(200, len(st.session_state.X_train))
                        else:
                            actual_batch_size = batch_size
                            
                    with col7:
                        max_iter = st.slider("Maximum iterations:", 100, 2000, 200)
                        tol = st.slider("Tolerance:", 1e-6, 1e-2, 1e-4, format="%.1e")
                
                # Parameter validation
                validation_errors = []
                
                # Validate hidden layer sizes
                if any(size <= 0 for size in hidden_layer_sizes):
                    validation_errors.append("Hidden layer sizes must be positive integers")
                
                # Validate learning rate
                if learning_rate_init <= 0 or learning_rate_init > 1:
                    validation_errors.append("Learning rate must be between 0 and 1")
                
                # Validate regularization
                if alpha < 0:
                    validation_errors.append("Alpha (regularization) must be non-negative")
                
                # Validate max iterations
                if max_iter <= 0:
                    validation_errors.append("Max iterations must be positive")
                
                # Validate tolerance
                if tol <= 0:
                    validation_errors.append("Tolerance must be positive")
                
                # Validate batch size
                if isinstance(actual_batch_size, int) and actual_batch_size <= 0:
                    validation_errors.append("Batch size must be positive")
                
                # Validate solver-specific parameters
                if solver == "adam":
                    if not (0 < beta_1 < 1):
                        validation_errors.append("Beta 1 must be between 0 and 1")
                    if not (0 < beta_2 < 1):
                        validation_errors.append("Beta 2 must be between 0 and 1")
                    if epsilon <= 0:
                        validation_errors.append("Epsilon must be positive")
                elif solver == "sgd":
                    if not (0 <= momentum <= 1):
                        validation_errors.append("Momentum must be between 0 and 1")
                    if power_t <= 0:
                        validation_errors.append("Power t must be positive")
                
                # Validate early stopping parameters
                if early_stopping:
                    if not (0 < validation_fraction < 1):
                        validation_errors.append("Validation fraction must be between 0 and 1")
                    if n_iter_no_change <= 0:
                        validation_errors.append("Iterations no change must be positive")
                
                # Show validation errors if any
                if validation_errors:
                    st.error("**Parameter Validation Errors:**")
                    for error in validation_errors:
                        st.error(f"• {error}")
                    st.stop()
                
                # Create comprehensive parameters
                mlp_params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': activation,
                    'solver': solver,
                    'alpha': alpha,
                    'learning_rate_init': learning_rate_init,
                    'learning_rate': learning_rate,
                    'max_iter': max_iter,
                    'tol': tol,
                    'batch_size': actual_batch_size,
                    'random_state': 42
                }
                
                # Add solver-specific parameters
                if solver == "adam":
                    mlp_params.update({
                        'beta_1': beta_1,
                        'beta_2': beta_2,
                        'epsilon': epsilon
                    })
                elif solver == "sgd":
                    mlp_params.update({
                        'momentum': momentum,
                        'power_t': power_t if learning_rate == "invscaling" else 0.5
                    })
                
                # Add early stopping parameters if enabled
                if early_stopping:
                    mlp_params.update({
                        'early_stopping': early_stopping,
                        'validation_fraction': validation_fraction,
                        'n_iter_no_change': n_iter_no_change
                    })
                
                # Add shuffle parameter
                mlp_params['shuffle'] = shuffle
                
                base_model = MLPClassifier()
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'hidden_layer_sizes': [
                            (50,), (100,), (200,),
                            (50, 50), (100, 50), (100, 100),
                            (100, 50, 25)
                        ],
                        'activation': ['relu', 'tanh', 'logistic'],
                        'solver': ['adam', 'sgd'],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate_init': [0.001, 0.01, 0.1],
                        'max_iter': [200, 500, 1000]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (100,50), (150,100), (200,100), (100,50,25), (200,100,50)],
                        'activation': ['relu', 'tanh', 'logistic', 'identity'],
                        'solver': ['adam', 'sgd', 'lbfgs'],
                        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                        'max_iter': [200, 500, 1000, 1500]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Classification", "MLP", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = MLPClassifier(**best_params)
                else:
                    model = MLPClassifier(**mlp_params)

        else:  # Regression
            # Regular regression models (non-time series)
            
            # Opsi rentang parameter kustom untuk regresi
            use_custom_ranges = False
            custom_param_ranges = {}
            
            if optimization_method in ["GridSearchCV", "RandomizedSearchCV"]:
                use_custom_ranges = st.checkbox(
                    "Gunakan rentang parameter kustom (Regresi)" if st.session_state.language == 'id' else "Use custom parameter ranges (Regression)",
                    value=False,
                    help="Aktifkan untuk menentukan rentang parameter sendiri untuk model regresi" if st.session_state.language == 'id' else "Enable to specify custom parameter ranges for regression models",
                    key="regression_custom_ranges"
                )
                
                if use_custom_ranges:
                    st.info("💡 Gunakan format: min:max:step untuk numerik, atau val1,val2,val3 untuk kategorikal" if st.session_state.language == 'id' else "💡 Use format: min:max:step for numeric, or val1,val2,val3 for categorical")
                    st.info("⚠️ Kosongkan field untuk menggunakan rentang default" if st.session_state.language == 'id' else "⚠️ Leave field empty to use default ranges")
            
            model_type = st.selectbox("Pilih model regresi:" if st.session_state.language == 'id' else "Select a regression model:", 
                                     ["Random Forest", "Linear Regression", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "SVR", "Bagging Regressor", "Voting Regressor", "Stacking Regressor", "KNN Regressor", "MLP Regressor"])
            
            # Clustering insights integration for regression
            clustering_insights_available = False
            clustering_results = None
            
            if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                try:
                    clustering_results = session_manager.get_tab_state('clustering')
                    if clustering_results and 'algorithm' in clustering_results:
                        clustering_insights_available = True
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Clustering Insights Retrieval (Regression)")
                        st.warning(f"⚠️ {error_result['message']}")
            else:
                # Fallback to direct session state
                clustering_results = st.session_state.get('clustering_results', {})
                if clustering_results and 'algorithm' in clustering_results:
                    clustering_insights_available = True
            
            if clustering_insights_available and clustering_results:
                with st.expander("🔍 Wawasan Clustering dari Tab EDA (Regresi)" if st.session_state.language == 'id' else "🔍 Clustering Insights from EDA Tab (Regression)"):
                    st.write(f"**Algoritma Clustering:** {clustering_results['algorithm']}")
                    st.write(f"**Jumlah Cluster:** {clustering_results['n_clusters']}")
                    
                    if clustering_results.get('silhouette_score', 0) > 0:
                        st.write(f"**Silhouette Score:** {clustering_results['silhouette_score']:.3f}")
                    
                    if clustering_results.get('calinski_harabasz_score', 0) > 0:
                        st.write(f"**Calinski-Harabasz Score:** {clustering_results['calinski_harabasz_score']:.1f}")
                    
                    # Recommendations for regression based on clustering results
                    if clustering_results['n_clusters'] <= 3:
                        st.info("💡 **Rekomendasi (Regresi):** Jumlah cluster yang sedikit menunjukkan hubungan linear yang kuat. Linear Regression atau Random Forest cocok untuk data ini." if st.session_state.language == 'id' else "💡 **Recommendation (Regression):** Few clusters indicate strong linear relationships. Linear Regression or Random Forest are suitable for this data.")
                    elif clustering_results['n_clusters'] >= 5:
                        st.info("💡 **Rekomendasi (Regresi):** Banyak cluster menunjukkan hubungan non-linear yang kompleks. Pertimbangkan Gradient Boosting, SVR, atau MLP Regressor." if st.session_state.language == 'id' else "💡 **Recommendation (Regression):** Many clusters indicate complex non-linear relationships. Consider Gradient Boosting, SVR, or MLP Regressor.")
                    
                    if clustering_results.get('silhouette_score', 0) > 0.5:
                        st.success("✅ Clustering yang baik (Silhouette > 0.5) menunjukkan struktur data yang jelas. Model regresi akan berkinerja baik." if st.session_state.language == 'id' else "✅ Good clustering (Silhouette > 0.5) indicates clear data structure. Regression models will perform well.")
                    elif clustering_results.get('silhouette_score', 0) < 0.2:
                        st.warning("⚠️ Clustering yang buruk (Silhouette < 0.2) menunjukkan struktur data yang tidak jelas. Pertimbangkan feature engineering tambahan atau ensemble methods." if st.session_state.language == 'id' else "⚠️ Poor clustering (Silhouette < 0.2) indicates unclear data structure. Consider additional feature engineering or ensemble methods.")
            
            # Data type detection integration for regression
            if DATA_TYPE_DETECTOR_AVAILABLE and data_type_detector is not None:
                try:
                    with st.expander("🔍 Deteksi Tipe Data Otomatis (Regresi)" if st.session_state.language == 'id' else "🔍 Automatic Data Type Detection (Regression)"):
                        # Use detect_column_types instead of non-existent detect_data_types
                        data_info = data_type_detector.detect_column_types(st.session_state.X_train)
                        st.write("**Tipe Data yang Terdeteksi:**" if st.session_state.language == 'id' else "**Detected Data Types:**")
                        
                        numeric_features = []
                        categorical_features = []
                        
                        for feature_name, analysis in data_info.items():
                            detected_type = analysis['detected_type']
                            confidence = analysis['confidence']
                            
                            if detected_type == 'numeric':
                                numeric_features.append(feature_name)
                            elif detected_type == 'object' or detected_type == 'categorical':
                                categorical_features.append(feature_name)
                            
                            st.write(f"- {feature_name}: {detected_type} (confidence: {confidence:.2f})")
                        
                        # Model recommendations based on data types for regression
                        if len(categorical_features) > len(numeric_features):
                            st.info("💡 **Rekomendasi Model:** Random Forest, Gradient Boosting, atau Bagging Regressor cocok untuk data dengan banyak fitur kategorikal" if st.session_state.language == 'id' else "💡 **Model Recommendation:** Random Forest, Gradient Boosting, or Bagging Regressor are suitable for data with many categorical features")
                        elif len(numeric_features) > len(categorical_features):
                            st.info("💡 **Rekomendasi Model:** Linear Regression, SVR, atau Neural Network cocok untuk data dengan banyak fitur numerik" if st.session_state.language == 'id' else "💡 **Model Recommendation:** Linear Regression, SVR, or Neural Network are suitable for data with many numeric features")
                        
                        # Check for multicollinearity issues
                        if len(numeric_features) > 10:
                            st.warning("⚠️ **Perhatian:** Banyak fitur numerik dapat menyebabkan multikolinearitas. Pertimbangkan untuk menggunakan Regularized Regression (Ridge/Lasso) atau Random Forest" if st.session_state.language == 'id' else "⚠️ **Warning:** Many numeric features can cause multicollinearity. Consider using Regularized Regression (Ridge/Lasso) or Random Forest")
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Data Type Detection (Regression)")
                        st.warning(f"⚠️ {error_result['message']}")
                    else:
                        st.warning(f"⚠️ Gagal mendeteksi tipe data: {str(e)}")
            
            if model_type == "Random Forest":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                n_estimators = st.slider("Jumlah pepohonan:" if st.session_state.language == 'id' else "Number of Trees:", 10, 500, 100)
                max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                
                # Parameter lanjutan dengan expander
                with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                    min_samples_split = st.slider("Jumlah sampel minimum untuk membagi:" if st.session_state.language == 'id' else "Minimum samples to split:", 2, 20, 2)
                    min_samples_leaf = st.slider("Jumlah sampel minimum di leaf:" if st.session_state.language == 'id' else "Minimum samples in leaf:", 1, 10, 1)
                    max_features = st.selectbox("Fitur maksimum:" if st.session_state.language == 'id' else "Max features:", ["sqrt", "log2", "None"], index=0)
                    bootstrap = st.checkbox("Bootstrap sampel:" if st.session_state.language == 'id' else "Bootstrap samples:", value=True)
                    
                    # Konversi max_features dari string ke None jika diperlukan
                    max_features_value = None if max_features == "None" else max_features
                
                base_model = RandomForestRegressor(random_state=42)
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                        'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='r2', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'n_estimators': list(range(50, 301, 25)),
                        'max_depth': list(range(3, 21)) + [None],
                        'min_samples_split': [2, 5, 10, 15, 20],
                        'min_samples_leaf': [1, 2, 4, 8, 16],
                        'max_features': ['sqrt', 'log2', None]
                    }
                    cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                    model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='r2', n_jobs=-1, n_iter=50, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Regression", "Random Forest", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = RandomForestRegressor(**best_params, random_state=42)
                else:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features_value,
                        bootstrap=bootstrap,
                        random_state=42
                    )
                    
            elif model_type == "Gradient Boosting":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                n_estimators = st.slider("Jumlah boosting stages:" if st.session_state.language == 'id' else "Number of boosting stages:", 10, 500, 100)
                learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 10, 3)
                
                base_model = GradientBoostingRegressor(random_state=42)
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                        'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                        'max_depth': [2, 3, 5] if max_depth == 3 else [max(1, max_depth-1), max_depth, min(10, max_depth+2)],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'n_estimators': list(range(50, 301, 25)),
                        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                        'max_depth': list(range(2, 16)),
                        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_samples_split': [2, 5, 10, 15, 20],
                        'min_samples_leaf': [1, 2, 4, 8, 16]
                    }
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=40, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Regression", "Gradient Boosting", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = GradientBoostingRegressor(**best_params, random_state=42)
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42
                    )
                    
            elif model_type == "XGBoost":
                if not XGBOOST_AVAILABLE:
                    st.error("XGBoost tidak terinstal. Silakan install dengan: pip install xgboost")
                    model = None
                else:
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Number of boosting rounds:" if st.session_state.language == 'id' else "Jumlah boosting rounds:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Max depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 15, 6)
                    
                    with st.expander("Parameter Lanjutan XGBoost" if st.session_state.language == 'id' else "Advanced XGBoost Parameters"):
                        subsample = st.slider("Subsample ratio:" if st.session_state.language == 'id' else "Rasio subsample:", 0.5, 1.0, 1.0, 0.1)
                        colsample_bytree = st.slider("Column sample by tree:" if st.session_state.language == 'id' else "Sample kolom per tree:", 0.5, 1.0, 1.0, 0.1)
                        min_child_weight = st.slider("Min child weight:" if st.session_state.language == 'id' else "Bobot minimum anak:", 1, 10, 1)
                    
                    base_model = XGBRegressor(random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [3, 6, 9] if max_depth == 6 else [max(1, max_depth-3), max_depth, min(15, max_depth+3)]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'max_depth': list(range(3, 16)),
                            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                            'colsample_bytree': [0.6, 0.8, 1.0],
                            'min_child_weight': [1, 3, 5, 7]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='r2', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "XGBoost", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = XGBRegressor(**best_params, random_state=42)
                    else:
                        model = XGBRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            min_child_weight=min_child_weight,
                            random_state=42
                        )
                    
            elif model_type == "LightGBM":
                if not LIGHTGBM_AVAILABLE:
                    st.error("LightGBM tidak terinstal. Silakan install dengan: pip install lightgbm")
                    model = None
                else:
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Number of boosting rounds:" if st.session_state.language == 'id' else "Jumlah boosting rounds:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Max depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", -1, 15, -1, help="-1 means no limit")
                    
                    with st.expander("Parameter Lanjutan LightGBM" if st.session_state.language == 'id' else "Advanced LightGBM Parameters"):
                        num_leaves = st.slider("Number of leaves:" if st.session_state.language == 'id' else "Jumlah daun:", 2, 256, 31)
                        subsample = st.slider("Subsample ratio:" if st.session_state.language == 'id' else "Rasio subsample:", 0.5, 1.0, 1.0, 0.1)
                        colsample_bytree = st.slider("Column sample by tree:" if st.session_state.language == 'id' else "Sample kolom per tree:", 0.5, 1.0, 1.0, 0.1)
                    
                    base_model = LGBMRegressor(random_state=42, verbose=-1)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [5, 10, 15] if max_depth == -1 else [max(1, max_depth-3), max_depth, min(15, max_depth+3)]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'max_depth': [-1, 5, 10, 15, 20],
                            'num_leaves': [20, 31, 50, 100, 150],
                            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                            'colsample_bytree': [0.6, 0.8, 1.0]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='r2', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "LightGBM", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
                    else:
                        model = LGBMRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            num_leaves=num_leaves,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            random_state=42,
                            verbose=-1
                        )
                    
            elif model_type == "CatBoost":
                if not CATBOOST_AVAILABLE:
                    st.error("CatBoost tidak terinstal. Silakan install dengan: pip install catboost")
                    model = None
                else:
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    iterations = st.slider("Number of iterations:" if st.session_state.language == 'id' else "Jumlah iterasi:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    depth = st.slider("Tree depth:" if st.session_state.language == 'id' else "Kedalaman tree:", 2, 16, 6)
                    
                    with st.expander("Parameter Lanjutan CatBoost" if st.session_state.language == 'id' else "Advanced CatBoost Parameters"):
                        l2_leaf_reg = st.slider("L2 leaf regularization:" if st.session_state.language == 'id' else "Regularisasi L2 leaf:", 1.0, 10.0, 3.0, 0.5)
                        border_count = st.slider("Border count:" if st.session_state.language == 'id' else "Jumlah border:", 1, 255, 128)
                    
                    base_model = CatBoostRegressor(random_state=42, verbose=False)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'iterations': [50, 100, 200] if iterations == 100 else [max(10, iterations-50), iterations, min(500, iterations+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'depth': [4, 6, 8] if depth == 6 else [max(2, depth-2), depth, min(16, depth+2)]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'iterations': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'depth': [4, 6, 8, 10, 12],
                            'l2_leaf_reg': [1.0, 3.0, 5.0, 7.0, 9.0],
                            'border_count': [32, 64, 128, 254]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='r2', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "CatBoost", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = CatBoostRegressor(**best_params, random_state=42, verbose=False)
                    else:
                        model = CatBoostRegressor(
                            iterations=iterations,
                            learning_rate=learning_rate,
                            depth=depth,
                            l2_leaf_reg=l2_leaf_reg,
                            border_count=border_count,
                            random_state=42,
                            verbose=False
                        )
                    
            elif model_type == "Linear Regression":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                fit_intercept = st.checkbox("Fit intercept" if st.session_state.language == 'id' else "Fit intercept", value=True)
                
                # Parameter lanjutan untuk Linear Regression
                with st.expander("Parameter Lanjutan Linear Regression" if st.session_state.language == 'id' else "Advanced Linear Regression Parameters"):
                    positive = st.checkbox("Force positive coefficients" if st.session_state.language == 'id' else "Paksa koefisien positif", value=False)
                    copy_X = st.checkbox("Copy X" if st.session_state.language == 'id' else "Salin X", value=True)
                
                base_model = LinearRegression()
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'fit_intercept': [True, False],
                        'positive': [True, False],
                        'copy_X': [True, False]
                    }
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'fit_intercept': [True, False],
                        'positive': [True, False],
                        'copy_X': [True, False]
                    }
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=8, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Regression", "Linear Regression", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=8, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = LinearRegression(**best_params)
                else:
                    model = LinearRegression(
                        fit_intercept=fit_intercept,
                        positive=positive,
                        copy_X=copy_X
                    )
                    
            elif model_type == "SVR":
                # Dapatkan rentang parameter kustom jika diaktifkan
                custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                
                C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.1, 10.0, 1.0)
                kernel = st.selectbox("Kernel:" if st.session_state.language == 'id' else "Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                gamma = st.selectbox("Gamma (kernel coefficient):" if st.session_state.language == 'id' else "Gamma (koefisien kernel):", ["scale", "auto"])
                epsilon = st.slider("Epsilon:" if st.session_state.language == 'id' else "Epsilon:", 0.01, 0.5, 0.1)

                base_model = SVR()

                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.1, C/2), C, min(10.0, C*2)],
                        'kernel': [kernel] if kernel != "rbf" else ['linear', 'rbf'],
                        'gamma': [gamma] if gamma != "scale" else ['scale', 'auto'],
                        'epsilon': [epsilon]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                        'epsilon': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                        'degree': [2, 3, 4, 5]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=25, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Regression", "SVR", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=25, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = SVR(**best_params)
                else:
                    model = SVR(
                        C=C,
                        kernel=kernel,
                        gamma=gamma,
                        epsilon=epsilon
                    )

            elif model_type == "Voting Regressor":
                from sklearn.ensemble import VotingRegressor
                from sklearn.neighbors import KNeighborsRegressor
                # Pilih base estimators untuk VotingRegressor
                base_estimators = []
                if st.checkbox("Gunakan Random Forest", value=True, key="vote_rf"):
                    base_estimators.append(('rf', RandomForestRegressor(n_estimators=50, random_state=42)))
                if st.checkbox("Gunakan Linear Regression", value=True, key="vote_lr"):
                    base_estimators.append(('lr', LinearRegression()))
                if st.checkbox("Gunakan Gradient Boosting", value=False, key="vote_gb"):
                    base_estimators.append(('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)))
                if st.checkbox("Gunakan KNN Regressor", value=False, key="vote_knn"):
                    base_estimators.append(('knn', KNeighborsRegressor()))
                if len(base_estimators) < 2:
                    st.warning("Pilih minimal dua base estimator untuk Voting Regressor." if st.session_state.language == 'id' else "Select at least two base estimators for Voting Regressor.")
                    model = None
                else:
                    model = VotingRegressor(estimators=base_estimators)
                    
            elif model_type == "Stacking Regressor":
                # Simple stacking with 2-3 base models and a final regressor
                base_estimators = []
                if st.checkbox("Gunakan Random Forest (Stacking)" if st.session_state.language == 'id' else "Use Random Forest (Stacking)", value=True, key="stack_rf"):
                    base_estimators.append(('rf', RandomForestRegressor(n_estimators=50, random_state=42)))
                if st.checkbox("Gunakan Linear Regression (Stacking)" if st.session_state.language == 'id' else "Use Linear Regression (Stacking)", value=True, key="stack_lr"):
                    base_estimators.append(('lr', LinearRegression()))
                if st.checkbox("Gunakan Gradient Boosting (Stacking)" if st.session_state.language == 'id' else "Use Gradient Boosting (Stacking)", value=False, key="stack_gb"):
                    base_estimators.append(('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)))
                final_estimator = st.selectbox("Final estimator:" if st.session_state.language == 'id' else "Final estimator:", ["Linear Regression", "Random Forest"], key="stack_final")
                if final_estimator == "Linear Regression":
                    final = LinearRegression()
                else:
                    final = RandomForestRegressor(n_estimators=20, random_state=42)
                if len(base_estimators) < 2:
                    st.warning("Pilih minimal dua base estimator untuk Stacking Regressor." if st.session_state.language == 'id' else "Select at least two base estimators for Stacking Regressor.")
                    model = None
                else:
                    model = StackingRegressor(
                        estimators=base_estimators,
                        final_estimator=final,
                        passthrough=True
                    )
            elif model_type == "KNN Regressor":
                from sklearn.neighbors import KNeighborsRegressor
                n_neighbors = st.slider("Number of neighbors (K):" if st.session_state.language == 'id' else "Jumlah tetangga (K):", 1, 20, 5)
                weights = st.selectbox("Weight function:" if st.session_state.language == 'id' else "Fungsi bobot:", ["uniform", "distance"])
                algorithm = st.selectbox("Algorithm:" if st.session_state.language == 'id' else "Algoritma:", ["auto", "ball_tree", "kd_tree", "brute"])
                base_model = KNeighborsRegressor()
                custom_param_ranges = get_custom_param_inputs("KNN Regressor", use_custom_ranges, st.session_state)
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'n_neighbors': [3, 5, 7] if n_neighbors == 5 else [max(1, n_neighbors-2), n_neighbors, min(20, n_neighbors+2)],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'p': [1, 2]  # Manhattan or Euclidean distance
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'n_neighbors': list(range(1, 31)),
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'p': [1, 2, 3],  # Manhattan, Euclidean, or Minkowski distance
                        'leaf_size': list(range(10, 51, 5))
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=20, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Regression", "KNN", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    model = KNeighborsRegressor(**best_params)
                else:
                    model = KNeighborsRegressor(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        algorithm=algorithm
                    )
            elif model_type == "MLP Regressor":
                custom_param_ranges = get_custom_param_inputs("MLP Regressor", use_custom_ranges, st.session_state)
                st.subheader("Konfigurasi Neural Network Regresi Lengkap" if st.session_state.language == 'id' else "Complete Neural Network Regression Configuration")
                
                # Tambahkan penjelasan teori di bagian atas
                with st.expander("📚 Penjelasan Teori Neural Network" if st.session_state.language == 'id' else "📚 Neural Network Theory Explanation"):
                    st.markdown("""
                    ### 🧠 **Fungsi Aktivasi**
                    - **ReLU**: f(x) = max(0,x) - Cocok untuk hidden layers, mengatasi vanishing gradient
                    - **Sigmoid**: f(x) = 1/(1+e^(-x)) - Cocok untuk output binary classification
                    - **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - Range [-1,1], lebih stabil dari sigmoid
                    - **Identity**: f(x) = x - Untuk output regression
                    
                    ### 🏗️ **Arsitektur Jaringan**
                    - **Feedforward Neural Network (FNN)**: Informasi mengalir satu arah (input → hidden → output)
                    - **Parameter yang dikonfigurasi**: Hidden layers, neurons per layer, activation function
                    
                    ### ⚡ **Optimizer & Learning**
                    - **Adam**: Kombinasi momentum dan adaptive learning rate, efisien untuk data besar
                    - **SGD**: Stochastic Gradient Descent dengan momentum untuk konvergensi lebih stabil
                    - **L-BFGS**: Optimizer kuasi-Newton untuk dataset kecil/medium
                    
                    ### 🛡️ **Regularization & Overfitting Prevention**
                    - **L2 Regularization (alpha)**: Menghindari overfitting dengan menjaga bobot tetap kecil
                    - **Early Stopping**: Menghentikan training saat validasi tidak membaik
                    - **Dropout**: (Tidak tersedia di MLPClassifier scikit-learn)
                    
                    ### 📊 **Hyperparameter Penting**
                    - **Learning Rate**: Kontrol kecepatan pembelajaran (0.001-0.1)
                    - **Batch Size**: Jumlah sampel per update (16-512)
                    - **Epochs**: Jumlah iterasi seluruh dataset (max_iter)
                    - **Hidden Layers**: Kompleksitas model (1-5 layers)
                    """)
                
                # Parameter configuration mode
                param_config_mode = st.radio(
                    "Mode konfigurasi parameter:" if st.session_state.language == 'id' else "Parameter configuration mode:",
                    ["Quick Setup", "Advanced Settings"],
                    horizontal=True,
                    help="Pilih mode konfigurasi: Quick Setup untuk pengaturan cepat, Advanced Settings untuk kontrol penuh"
                )
                
                # Architecture Configuration
                st.write("**Arsitektur Jaringan Regresi:**" if st.session_state.language == 'id' else "**Regression Network Architecture:**")
                
                if param_config_mode == "Quick Setup":
                    # Quick Setup - simplified interface
                    col1, col2 = st.columns(2)
                    with col1:
                        num_hidden_layers = st.slider("Jumlah hidden layers:", 1, 5, 2)
                        neurons_per_layer = st.text_input("Neurons per layer:", "100,50")
                        try:
                            neurons_list = [int(x.strip()) for x in neurons_per_layer.split(",")]
                            if len(neurons_list) < num_hidden_layers:
                                neurons_list.extend([neurons_list[-1]] * (num_hidden_layers - len(neurons_list)))
                            elif len(neurons_list) > num_hidden_layers:
                                neurons_list = neurons_list[:num_hidden_layers]
                            hidden_layer_sizes = tuple(neurons_list)
                        except:
                            hidden_layer_sizes = (100, 50)
                    with col2:
                        activation = st.selectbox("Activation function:", 
                                                ["relu", "tanh", "logistic", "identity"],
                                                help="ReLU: max(0,x) | Sigmoid: 1/(1+e^-x) | Tanh: (e^x-e^-x)/(e^x+e^-x) | Identity: x")
                    
                    # Basic parameters
                    col3, col4 = st.columns(2)
                    with col3:
                        solver = st.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"])
                        learning_rate_init = st.slider("Initial learning rate:", 0.0001, 0.1, 0.001, format="%.4f")
                    with col4:
                        max_iter = st.slider("Maximum iterations:", 100, 2000, 200)
                        alpha = st.slider("L2 regularization (alpha):", 0.00001, 0.01, 0.0001, format="%.5f")
                    
                    # Set default values for other parameters
                    learning_rate = "constant"
                    batch_size = "auto"
                    actual_batch_size = min(200, len(st.session_state.X_train))
                    tol = 1e-4
                    beta_1 = 0.9
                    beta_2 = 0.999
                    epsilon = 1e-8
                    momentum = 0.9
                    power_t = 0.5
                    shuffle = True
                    early_stopping = False
                    validation_fraction = 0.1
                    n_iter_no_change = 10
                    
                else:  # Advanced Settings
                    # Advanced Settings - full control
                    st.write("**Arsitektur Jaringan Regresi - Advanced:**" if st.session_state.language == 'id' else "**Regression Network Architecture - Advanced:**")
                    
                    # Hidden layers configuration
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        num_hidden_layers = st.slider("Jumlah hidden layers:", 1, 10, 2, help="Jumlah lapisan tersembunyi dalam jaringan")
                    with col2:
                        neurons_per_layer = st.text_input("Neurons per layer:", "100,50", help="Jumlah neuron di setiap lapisan, pisahkan dengan koma")
                        try:
                            neurons_list = [int(x.strip()) for x in neurons_per_layer.split(",")]
                            if len(neurons_list) < num_hidden_layers:
                                neurons_list.extend([neurons_list[-1]] * (num_hidden_layers - len(neurons_list)))
                            elif len(neurons_list) > num_hidden_layers:
                                neurons_list = neurons_list[:num_hidden_layers]
                            hidden_layer_sizes = tuple(neurons_list)
                        except:
                            hidden_layer_sizes = (100, 50)
                            st.error("Format neurons per layer tidak valid. Gunakan format: 100,50,25")
                    with col3:
                        activation = st.selectbox("Activation function:", 
                                                ["relu", "tanh", "logistic", "identity"],
                                                help="ReLU: max(0,x) | Sigmoid: 1/(1+e^-x) | Tanh: (e^x-e^-x)/(e^x+e^-x) | Identity: x")
                    
                    # Advanced parameters
                    with st.expander("🔧 Advanced Parameters", expanded=True):
                        col4, col5 = st.columns(2)
                        with col4:
                            solver = st.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"], 
                                                help="Algoritma optimasi untuk training")
                            
                            if solver == "adam":
                                st.write("**Adam Optimizer Parameters:**")
                                beta_1 = st.slider("Beta 1 (exponential decay rate):", 0.8, 0.999, 0.9, format="%.3f",
                                                   help="Faktor decay untuk estimasi pertama (default: 0.9)")
                                beta_2 = st.slider("Beta 2 (exponential decay rate):", 0.9, 0.9999, 0.999, format="%.4f",
                                                   help="Faktor decay untuk estimasi kedua (default: 0.999)")
                                epsilon = st.slider("Epsilon (numerical stability):", 1e-8, 1e-3, 1e-8, format="%.1e",
                                                    help="Nilai kecil untuk menghindari division by zero")
                            elif solver == "sgd":
                                st.write("**SGD Optimizer Parameters:**")
                                momentum = st.slider("Momentum:", 0.0, 0.9, 0.9,
                                                     help="Faktor momentum untuk mempercepat konvergensi")
                                power_t = st.slider("Power t (inverse scaling exponent):", 0.1, 0.9, 0.5,
                                                    help="Eksponen untuk inverse scaling learning rate")
                            else:  # lbfgs
                                st.info("L-BFGS tidak memiliki parameter tambahan")
                                beta_1 = 0.9
                                beta_2 = 0.999
                                epsilon = 1e-8
                                momentum = 0.9
                                power_t = 0.5
                                
                        with col5:
                            learning_rate_init = st.slider("Initial learning rate:", 0.00001, 0.1, 0.001, format="%.5f",
                                                           help="Learning rate awal untuk optimasi")
                            learning_rate = st.selectbox("Learning rate schedule:", 
                                                       ["constant", "invscaling", "adaptive"],
                                                       help="Strategi penyesuaian learning rate selama training")
                            
                            col6, col7 = st.columns(2)
                            with col6:
                                alpha = st.slider("L2 regularization (alpha):", 0.000001, 0.1, 0.0001, format="%.6f",
                                                  help="Regularisasi L2 untuk mencegah overfitting")
                                batch_size_options = ["auto", 8, 16, 32, 64, 128, 256, 512, 1024]
                                batch_size = st.selectbox("Batch size:", batch_size_options,
                                                        help="Jumlah sampel per update weight")
                                if batch_size == "auto":
                                    actual_batch_size = min(200, len(st.session_state.X_train))
                                else:
                                    actual_batch_size = batch_size
                                    
                            with col7:
                                max_iter = st.slider("Maximum iterations:", 100, 5000, 200,
                                                     help="Maksimum iterasi training")
                                tol = st.slider("Tolerance (convergence threshold):", 1e-6, 1e-2, 1e-4, format="%.1e",
                                                help="Threshold untuk menghentikan training")
                                
                            # Additional advanced parameters
                            st.write("**Additional Parameters:**")
                            col8, col9 = st.columns(2)
                            with col8:
                                shuffle = st.checkbox("Shuffle samples", value=True,
                                                    help="Mengacak sampel di setiap iterasi")
                                early_stopping = st.checkbox("Early stopping", value=False,
                                                           help="Menghentikan training jika validasi tidak membaik")
                            with col9:
                                validation_fraction = st.slider("Validation fraction:", 0.05, 0.5, 0.1, format="%.2f",
                                                              help="Fraksi data untuk validasi (jika early stopping=True)")
                                n_iter_no_change = st.slider("Iterations no change:", 5, 50, 10,
                                                             help="Jumlah iterasi tanpa perbaikan sebelum stopping")
                
                # Create comprehensive parameters
                mlp_params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': activation,
                    'solver': solver,
                    'alpha': alpha,
                    'learning_rate_init': learning_rate_init,
                    'learning_rate': learning_rate,
                    'max_iter': max_iter,
                    'tol': tol,
                    'batch_size': actual_batch_size,
                    'random_state': 42
                }
                
                # Add solver-specific parameters
                if solver == "adam":
                    mlp_params.update({
                        'beta_1': beta_1,
                        'beta_2': beta_2,
                        'epsilon': epsilon
                    })
                elif solver == "sgd":
                    mlp_params.update({
                        'momentum': momentum,
                        'power_t': power_t if learning_rate == "invscaling" else 0.5
                    })
                
                base_model = MLPRegressor()
                
                if optimization_method == "GridSearchCV":
                    param_grid = {
                        'hidden_layer_sizes': [
                            (50,), (100,), (200,),
                            (50, 50), (100, 50), (100, 100),
                            (100, 50, 25)
                        ],
                        'activation': ['relu', 'tanh', 'logistic'],
                        'solver': ['adam', 'sgd'],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate_init': [0.001, 0.01, 0.1],
                        'max_iter': [200, 500, 1000]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                    model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                elif optimization_method == "RandomizedSearchCV":
                    param_dist = {
                        'hidden_layer_sizes': [
                            (50,), (100,), (150,), (200,), (250,),
                            (50, 50), (100, 50), (100, 100), (150, 100),
                            (100, 50, 25), (150, 100, 50)
                        ],
                        'activation': ['relu', 'tanh', 'logistic', 'identity'],
                        'solver': ['adam', 'sgd', 'lbfgs'],
                        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                        'max_iter': [200, 500, 1000, 1500]
                    }
                    # Gabungkan dengan rentang parameter kustom
                    param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                    # Validasi parameter berdasarkan data
                    param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                    model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=20, random_state=42)
                elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    objective = create_optuna_study("Regression", "MLP", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)
                    best_params = study.best_params
                    # Convert hidden_layer_sizes back to tuple
                    if 'hidden_layer_sizes' in best_params:
                        best_params['hidden_layer_sizes'] = tuple(best_params['hidden_layer_sizes'])
                    model = MLPRegressor(**best_params)
                else:
                    model = MLPRegressor(**mlp_params)
            else:
                st.error("Silahkan pilih model regresi." if st.session_state.language == 'id' else "Please select a valid regression model.")
                model = None
        
        model_custom_name = st.text_input("Nama model (bebas, gunakan huruf/angka/underscore):" if st.session_state.language == 'id' else "Nama model (bebas, gunakan huruf/angka/underscore):", value=f"")
        st.session_state.model_type = model_type

        # Train model button
        if model is not None and st.button("Train Model"):
            try:
                log_feature('train_model')
            except Exception:
                pass
            
            # Integrate workflow validation before training
            if WORKFLOW_VALIDATOR_AVAILABLE and workflow_validator is not None:
                try:
                    # Validate model training readiness
                    ml_readiness = workflow_validator.validate_ml_training_readiness(
                        st.session_state.X_train, 
                        st.session_state.y_train, 
                        st.session_state.problem_type,
                        model_type
                    )
                    
                    with st.expander("✅ Validasi Kesiapan Training Model" if st.session_state.language == 'id' else "✅ Model Training Readiness Validation"):
                        st.info("Validasi kesiapan data dan parameter untuk training model:" if st.session_state.language == 'id' else "Validation of data and parameter readiness for model training:")
                        
                        # Display validation results
                        for result in ml_readiness:
                            if result['status'] == 'success':
                                st.success(f"✅ {result['message']}")
                            elif result['status'] == 'warning':
                                st.warning(f"⚠️ {result['message']}")
                            elif result['status'] == 'error':
                                st.error(f"❌ {result['message']}")
                        
                        # Store validation results in session state
                        st.session_state.ml_training_readiness = ml_readiness
                        
                        # Check if ready for training
                        training_ready = all(result['status'] != 'error' for result in ml_readiness)
                        if not training_ready:
                            st.error("❌ Training model dibatalkan: Masalah kritis terdeteksi" if st.session_state.language == 'id' else "❌ Model training cancelled: Critical issues detected")
                            st.stop()
                            
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "ML Training Readiness Validation")
                        st.warning(f"⚠️ {error_result['message']}")
                    else:
                        st.warning(f"⚠️ Gagal melakukan validasi kesiapan training: {str(e)}")
            
            with st.spinner(f"Melatih model {model_type}..." if st.session_state.language == 'id' else f"Training {model_type} model..."):
                try:
                    # Preprocessing: Remove NaN values from training data
                    # This is crucial for models like GradientBoostingRegressor that don't handle NaN natively
                    X_train_clean = st.session_state.X_train.dropna()
                    y_train_clean = st.session_state.y_train.loc[X_train_clean.index]
                    
                    # Check if we have enough data after cleaning
                    if len(X_train_clean) == 0:
                        raise ValueError("Tidak ada data yang valid untuk training setelah menghapus nilai NaN. Silakan periksa data Anda." if st.session_state.language == 'id' else "No valid data available for training after removing NaN values. Please check your data.")
                    
                    if len(X_train_clean) < 10:
                        st.warning(f"Hanya {len(X_train_clean)} sampel tersedia untuk training. Hasil mungkin tidak optimal." if st.session_state.language == 'id' else f"Only {len(X_train_clean)} samples available for training. Results may not be optimal.")
                    
                    start_time = time.time()
                    model.fit(X_train_clean, y_train_clean)
                    training_time = time.time() - start_time
                    
                    # Tambahkan validasi sebelum prediksi
                    try:
                        # Test prediksi dengan data dummy untuk memastikan model berfungsi
                        dummy_data = pd.DataFrame(np.zeros((1, len(st.session_state.X_train.columns))), 
                                                columns=st.session_state.X_train.columns)
                        model.predict(dummy_data)  # Use local 'model' instead of st.session_state.model
                        st.success("Model siap digunakan untuk prediksi" if st.session_state.language == 'id' else "Model ready for prediction")
                    except Exception as e:
                        st.error(f"Model error: {str(e)}. Silakan latih ulang model." if st.session_state.language == 'id' else f"Model error: {str(e)}. Please retrain the model.")

                    # Jika menggunakan optimasi hyperparameter, tampilkan parameter terbaik
                    if optimization_method != "None" and hasattr(model, "best_params_"):
                        st.success(f"Pelatihan model selesai dalam {training_time:.2f} detik dengan {optimization_method}. Parameter terbaik: {model.best_params_}" if st.session_state.language == 'id' else f"Model training completed in {training_time:.2f} seconds with {optimization_method}!")
                        st.subheader("Parameter Terbaik" if st.session_state.language == 'id' else "Best Parameters:")
                        st.write(model.best_params_)
                        st.write(f"Skor terbaik (CV): {model.best_score_:.4f}" if st.session_state.language == 'id' else f"Best Score (CV): {model.best_score_:.4f}")

                        # Gunakan model terbaik untuk prediksi (dengan handling NaN)
                        X_test_clean = st.session_state.X_test.dropna()
                        y_test_clean = st.session_state.y_test.loc[X_test_clean.index]
                        y_pred = model.best_estimator_.predict(X_test_clean)
                        st.session_state.model = model.best_estimator_
                        # Update y_test untuk evaluasi
                        st.session_state.y_test_eval = y_test_clean
                    else:
                        st.success(f"Model selesai dilatih dalam {training_time:.2f} detik" if st.session_state.language == 'id' else f"Model training completed in {training_time:.2f} seconds!")
                        # Gunakan model terbaik untuk prediksi (dengan handling NaN)
                        X_test_clean = st.session_state.X_test.dropna()
                        y_test_clean = st.session_state.y_test.loc[X_test_clean.index]
                        y_pred = model.predict(X_test_clean)
                        st.session_state.model = model
                        # Update y_test untuk evaluasi
                        st.session_state.y_test_eval = y_test_clean
                    
                    # Add download button for trained model (always show after training)
                    if st.session_state.model is not None:
                        try:
                            model_bytes = pickle.dumps(st.session_state.model)
                            model_type_clean = model_type.lower().replace(" ", "_")
                            st.download_button(
                                label=f"📥 Unduh Model {model_type} (.pkl)" if st.session_state.language == 'id' else f"📥 Download {model_type} Model (.pkl)",
                                data=model_bytes,
                                file_name=f"{model_type_clean}_model_{target_column}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                mime="application/octet-stream"
                            )
                        except Exception as pickle_error:
                            st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                    
                    # Cross-validation evaluation
                    if cv_params['cv'] is not None:
                        st.subheader("Hasil Cross-Validation" if st.session_state.language == 'id' else "Cross-Validation Results")
                        
                        with st.spinner("Menghitung cross-validation..." if st.session_state.language == 'id' else "Calculating cross-validation..."):
                            try:
                                # Get the actual model (best estimator if using optimization)
                                eval_model = model.best_estimator_ if optimization_method != "None" else model
                                
                                # Perform cross-validation using cleaned data (without NaN values)
                                cv_scores = cross_val_score(
                                    eval_model, 
                                    X_train_clean, 
                                    y_train_clean,
                                    cv=cv_params['cv'],
                                    scoring=cv_params['scoring'],
                                    n_jobs=-1
                                )
                                
                                # Display results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Rata-rata Skor CV" if st.session_state.language == 'id' else "Mean CV Score",
                                        f"{cv_scores.mean():.4f}"
                                    )
                                with col2:
                                    st.metric(
                                        "Standar Deviasi" if st.session_state.language == 'id' else "Std Deviation",
                                        f"{cv_scores.std():.4f}"
                                    )
                                with col3:
                                    st.metric(
                                        "Metode Validasi" if st.session_state.language == 'id' else "Validation Method",
                                        cv_params['name']
                                    )
                                
                                # Plot cross-validation scores
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.boxplot(cv_scores)
                                ax.set_title(f"Cross-Validation Scores - {cv_params['name']}" if st.session_state.language == 'id' else f"Cross-Validation Scores - {cv_params['name']}")
                                ax.set_ylabel("Score")
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                
                                # Detailed scores
                                st.write("**Detail Skor per Fold:**" if st.session_state.language == 'id' else "**Detailed Scores per Fold:**")
                                fold_df = pd.DataFrame({
                                    'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                                    'Score': cv_scores
                                })
                                st.dataframe(fold_df)
                                
                            except Exception as e:
                                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                    error_result = error_handler.handle_error(e, "Model Cross-Validation")
                                    st.error(f"❌ {error_result['message']}")
                                    with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                                        st.write(error_result.get('details', str(e)))
                                else:
                                    st.error(f"Error dalam cross-validation: {str(e)}" if st.session_state.language == 'id' else f"Error in cross-validation: {str(e)}")
                    
                    # Save model dengan nama custom
                    os.makedirs("models", exist_ok=True)
                    # Bersihkan nama agar hanya huruf/angka/underscore
                    safe_name = "".join([c if c.isalnum() or c == "_" else "_" for c in model_custom_name])
                    model_filename = f"models/{safe_name}.pkl"
                    with open(model_filename, 'wb') as f:
                        pickle.dump(st.session_state.model, f)
                    st.success(f"Model telah disimpan sebagai '{model_filename}'" if st.session_state.language == 'id' else "Model saved as '{model_filename}'")
                    
                    # Evaluasi model
                    if problem_type == "Classification":
                        accuracy = accuracy_score(st.session_state.y_test_eval, y_pred)
                        st.write(f"Accuracy: {accuracy:.4f}")
                        
                        # Confusion Matrix
                        cm = confusion_matrix(st.session_state.y_test_eval, y_pred)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        plt.title('Confusion Matrix')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        st.pyplot(fig)
                        
                        # Classification Report
                        report = classification_report(st.session_state.y_test_eval, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.write("Label Report" if st.session_state.language == 'id' else "Classification Report:")
                        st.dataframe(report_df)
                        
                        # ROC Curve dan AUC Score
                        st.subheader("ROC Curve dan AUC Score" if st.session_state.language == 'id' else "ROC Curve and AUC Score")
                        
                        # Cek apakah model mendukung predict_proba
                        if hasattr(model, 'predict_proba'):
                            # Untuk klasifikasi biner
                            if len(np.unique(st.session_state.y_test_eval)) == 2:
                                y_prob = model.predict_proba(X_test_clean)[:, 1]
                                # Menangani kasus ketika y_test berisi nilai kategorikal seperti '<20', '>20'
                                if isinstance(st.session_state.y_test_eval.iloc[0], str):
                                    # Konversi nilai kategorikal ke numerik (0 dan 1)
                                    unique_values = sorted(np.unique(st.session_state.y_test_eval))
                                    pos_label = unique_values[1]  # Nilai kedua sebagai pos_label
                                    fpr, tpr, thresholds = roc_curve(st.session_state.y_test_eval, y_prob, pos_label=pos_label)
                                else:
                                    fpr, tpr, thresholds = roc_curve(st.session_state.y_test_eval, y_prob)
                                roc_auc = auc(fpr, tpr)
                                
                                # Plot ROC Curve
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                ax.plot([0, 1], [0, 1], 'k--')
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel('False Positive Rate')
                                ax.set_ylabel('True Positive Rate')
                                ax.set_title('Receiver Operating Characteristic (ROC)')
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                                
                                st.write(f"AUC Score: {roc_auc:.4f}")
                            
                            # Untuk klasifikasi multi-kelas
                            else:
                                try:
                                    y_prob = model.predict_proba(X_test_clean)
                                    
                                    # Buat label biner untuk setiap kelas
                                    y_test_bin = pd.get_dummies(st.session_state.y_test_eval).values
                                    
                                    # Pastikan jumlah kelas dalam y_prob dan y_test_bin sama
                                    n_classes = min(y_prob.shape[1], y_test_bin.shape[1])
                                    
                                    if n_classes > 0:
                                        # One-vs-Rest ROC
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        
                                        for i in range(n_classes):
                                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                                            roc_auc = auc(fpr, tpr)
                                            ax.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
                                        
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlim([0.0, 1.0])
                                        ax.set_ylim([0.0, 1.05])
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('Multi-class ROC Curve (One-vs-Rest)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig)
                                        
                                        # Hitung dan tampilkan AUC Score untuk setiap kelas
                                        st.write("AUC Scores per class:")
                                        for i in range(n_classes):
                                            class_auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
                                            st.write(f"Class {i}: {class_auc:.4f}")
                                        
                                        # Hitung rata-rata AUC (macro) hanya jika jumlah kelas sama
                                        if y_prob.shape[1] == y_test_bin.shape[1]:
                                            macro_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
                                            st.write(f"Macro Average AUC: {macro_auc:.4f}")
                                        else:
                                            st.warning("Tidak dapat menghitung Macro Average AUC karena jumlah kelas berbeda antara prediksi dan aktual." 
                                                      if st.session_state.language == 'id' else 
                                                      "Cannot calculate Macro Average AUC because the number of classes differs between prediction and actual.")
                                    else:
                                        st.warning("Tidak ada kelas yang dapat digunakan untuk kurva ROC." 
                                                  if st.session_state.language == 'id' else 
                                                  "No classes available for ROC curve.")
                                except Exception as e:
                                    st.error(f"Error saat membuat kurva ROC: {str(e)}" 
                                            if st.session_state.language == 'id' else 
                                            f"Error creating ROC curve: {str(e)}")
                                    st.warning("Pastikan data pengujian memiliki semua kelas yang ada dalam data pelatihan." 
                                              if st.session_state.language == 'id' else 
                                              "Make sure the test data contains all classes present in the training data.")
                                    # Tampilkan informasi tambahan untuk debugging
                                    st.write(f"Jumlah kelas unik dalam y_test: {len(np.unique(st.session_state.y_test_eval))}")
                                    if hasattr(model, 'classes_'):
                                        st.write(f"Kelas dalam model: {model.classes_}")
                                        st.write(f"Jumlah kelas dalam model: {len(model.classes_)}")
                                    if 'y_prob' in locals():
                                        st.write(f"Dimensi y_prob: {y_prob.shape}")
                                    if 'y_test_bin' in locals():
                                        st.write(f"Dimensi y_test_bin: {y_test_bin.shape}")
                        else:
                            st.warning("Model ini tidak mendukung prediksi probabilitas, sehingga kurva ROC tidak dapat ditampilkan." 
                                      if st.session_state.language == 'id' else 
                                      "This model doesn't support probability prediction, so ROC curve cannot be displayed.")
                        
                    else:  # Regression
                        mse = mean_squared_error(st.session_state.y_test_eval, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(st.session_state.y_test_eval, y_pred)
                        # Tambahan: Adjusted R²
                        n = X_test_clean.shape[0]
                        k = X_test_clean.shape[1]
                        adj_r2 = adjusted_r2_score(r2, n, k)
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"Root Mean Squared Error: {rmse:.4f}")
                        st.write(f"R² Score: {r2:.4f}")
                        st.write(f"Adjusted R² Score: {adj_r2:.4f}")

                        # Tambahan: Uji Multikolinearitas (VIF) - hanya untuk Linear Regression
                        if st.session_state.model_type == "Linear Regression":
                            st.subheader("Uji Multikolinearitas (VIF)" if st.session_state.language == 'id' else "Multicollinearity Test (VIF)")
                            vif_df = calculate_vif(st.session_state.X_train)
                            st.dataframe(vif_df)

                            # Tambahan: Uji Heteroskedastisitas (Breusch-Pagan) - hanya untuk Linear Regression
                            st.subheader("Uji Heteroskedastisitas (Breusch-Pagan)" if st.session_state.language == 'id' else "Heteroskedasticity Test (Breusch-Pagan)")
                            bp_result = breusch_pagan_test(st.session_state.y_test_eval, y_pred, X_test_clean)
                            st.write(f"Lagrange multiplier statistic: {bp_result['Lagrange multiplier statistic']:.4f}")
                            st.write(f"p-value: {bp_result['p-value']:.4f}")
                            st.write(f"f-value: {bp_result['f-value']:.4f}")
                            st.write(f"f p-value: {bp_result['f p-value']:.4f}")
                            
                            # Add assumptions check for linear regression
                            st.subheader("Asumsi Regresi Linear" if st.session_state.language == 'id' else "Linear Regression Assumptions")
                            
                            # Check VIF values for multicollinearity
                            high_vif = vif_df[vif_df['VIF'] > 10]
                            if len(high_vif) > 0:
                                st.warning(f"⚠️ Multikolinearitas terdeteksi! {len(high_vif)} fitur memiliki VIF > 10" 
                                        if st.session_state.language == 'id' else 
                                        f"⚠️ Multicollinearity detected! {len(high_vif)} features have VIF > 10")
                                st.dataframe(high_vif)
                            else:
                                st.success("✅ Tidak ada multikolinearitas yang signifikan (semua VIF ≤ 10)" 
                                        if st.session_state.language == 'id' else 
                                        "✅ No significant multicollinearity detected (all VIF ≤ 10)")
                            
                            # Check heteroskedasticity
                            if bp_result['p-value'] < 0.05:
                                st.warning("⚠️ Heteroskedastisitas terdeteksi (p-value < 0.05)" 
                                        if st.session_state.language == 'id' else 
                                        "⚠️ Heteroskedasticity detected (p-value < 0.05)")
                            else:
                                st.success("✅ Tidak ada heteroskedastisitas yang signifikan (p-value ≥ 0.05)" 
                                        if st.session_state.language == 'id' else 
                                        "✅ No significant heteroskedasticity detected (p-value ≥ 0.05)")
                        
                        # Plot actual vs predicted
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(st.session_state.y_test_eval, y_pred, alpha=0.5)
                        ax.plot([st.session_state.y_test_eval.min(), st.session_state.y_test_eval.max()], 
                               [st.session_state.y_test_eval.min(), st.session_state.y_test_eval.max()], 
                               'r--')
                        plt.title('Actual vs Predicted')
                        plt.xlabel('Actual')
                        plt.ylabel('Predicted')
                        st.pyplot(fig)
                        
                        # Residual plot
                        residuals = st.session_state.y_test_eval - y_pred
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y_pred, residuals, alpha=0.5)
                        ax.axhline(y=0, color='r', linestyle='--')
                        plt.title('Residual Plot')
                        plt.xlabel('Predicted')
                        plt.ylabel('Residuals')
                        st.pyplot(fig)
                        
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Model Training and Evaluation")
                        st.error(f"❌ {error_result['message']}")
                        
                        # Display additional error details if available
                        if 'details' in error_result:
                            with st.expander("🔍 Detail Error" if st.session_state.language == 'id' else "🔍 Error Details"):
                                st.write(error_result['details'])
                        
                        # Display recommendations if available
                        if 'recommendations' in error_result:
                            with st.expander("💡 Rekomendasi" if st.session_state.language == 'id' else "💡 Recommendations"):
                                for rec in error_result['recommendations']:
                                    st.write(f"• {rec}")
                    else:
                        st.error(f"❌ Error saat evaluasi model: {str(e)}" if st.session_state.language == 'id' else f"❌ Error during model training: {str(e)}")
                
                # Simpan hasil evaluasi untuk perbandingan
                if 'model_results' not in st.session_state:
                    st.session_state.model_results = []
                
                model_name = type(st.session_state.model).__name__
                
                # Pastikan y_pred didefinisikan berdasarkan problem type
                X_test_eval = st.session_state.X_test.dropna()
                y_pred = st.session_state.model.predict(X_test_eval)
                
                result = {
                    'model_name': model_name,
                    'model': st.session_state.model,
                    'y_test': st.session_state.y_test,
                    'y_pred': y_pred,
                    'problem_type': problem_type
                }
                
                if problem_type == "Classification":
                    result.update({
                        'accuracy': accuracy,
                        'confusion_matrix': cm,
                        'classification_report': report
                    })
                else:  # Regression
                    result.update({
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'adj_r2': adj_r2
                    })
                
                st.session_state.model_results.append(result)
        
        # Tampilkan perbandingan model jika ada lebih dari satu model
        if len(st.session_state.model_results) > 1:
            st.header("Perbandingan Model" if st.session_state.language == 'id' else "Model Comparison")
            
            # Buat tabs untuk berbagai jenis perbandingan
            comparison_tabs = st.tabs([
                "Confusion Matrix Comparison" if st.session_state.language == 'id' else "Confusion Matrix Comparison",
                "Performance Metrics" if st.session_state.language == 'id' else "Performance Metrics",
                "Model Rankings" if st.session_state.language == 'id' else "Model Rankings"
            ])
            
            with comparison_tabs[0]:
                if problem_type == "Classification":
                    st.subheader("Perbandingan Confusion Matrix" if st.session_state.language == 'id' else "Confusion Matrix Comparison")
                    
                    # Hitung jumlah model dan baris/kolom yang dibutuhkan
                    n_models = len(st.session_state.model_results)
                    n_cols = min(3, n_models)  # Maksimal 3 kolom per baris
                    n_rows = (n_models + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                    axes = axes.flatten() if n_models > 1 else [axes]
                    
                    for idx, result in enumerate(st.session_state.model_results):
                        if result['problem_type'] == "Classification":
                            cm = result['confusion_matrix']
                            model_name = result['model_name']
                            
                            # Buat heatmap untuk confusion matrix
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                      ax=axes[idx], cbar=False)
                            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}')
                            axes[idx].set_xlabel('Predicted')
                            axes[idx].set_ylabel('Actual')
                    
                    # Sembunyikan subplot kosong
                    for idx in range(n_models, len(axes)):
                        axes[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Tampilkan ringkasan performa
                    st.subheader("Ringkasan Performa Klasifikasi" if st.session_state.language == 'id' else "Classification Performance Summary")
                    comparison_df = pd.DataFrame([
                        {
                            'Model': r['model_name'],
                            'Accuracy': f"{r['accuracy']:.4f}",
                            'Precision': f"{r['classification_report']['weighted avg']['precision']:.4f}",
                            'Recall': f"{r['classification_report']['weighted avg']['recall']:.4f}",
                            'F1-Score': f"{r['classification_report']['weighted avg']['f1-score']:.4f}"
                        }
                        for r in st.session_state.model_results 
                        if r['problem_type'] == "Classification"
                    ])
                    st.dataframe(comparison_df)
                    
                    # Visualisasi perbandingan metrik
                    fig, ax = plt.subplots(figsize=(12, 6))
                    metrics_data = []
                    for r in st.session_state.model_results:
                        if r['problem_type'] == "Classification":
                            metrics_data.append({
                                'Model': r['model_name'],
                                'Accuracy': r['accuracy'],
                                'Precision': r['classification_report']['weighted avg']['precision'],
                                'Recall': r['classification_report']['weighted avg']['recall'],
                                'F1-Score': r['classification_report']['weighted avg']['f1-score']
                            })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df.set_index('Model').plot(kind='bar', ax=ax)
                        plt.title('Perbandingan Metrik Klasifikasi' if st.session_state.language == 'id' else 'Classification Metrics Comparison')
                        plt.ylabel('Score')
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.info("Perbandingan confusion matrix hanya tersedia untuk masalah klasifikasi." if st.session_state.language == 'id' else "Confusion matrix comparison is only available for classification problems.")
            
            with comparison_tabs[1]:
                st.subheader("Metrik Performa" if st.session_state.language == 'id' else "Performance Metrics")
                
                if problem_type == "Classification":
                    # Tampilkan semua metrik klasifikasi
                    metrics_summary = []
                    for result in st.session_state.model_results:
                        if result['problem_type'] == "Classification":
                            report = result['classification_report']
                            metrics_summary.append({
                                'Model': result['model_name'],
                                'Accuracy': result['accuracy'],
                                'Macro Precision': report['macro avg']['precision'],
                                'Macro Recall': report['macro avg']['recall'],
                                'Macro F1-Score': report['macro avg']['f1-score'],
                                'Weighted Precision': report['weighted avg']['precision'],
                                'Weighted Recall': report['weighted avg']['recall'],
                                'Weighted F1-Score': report['weighted avg']['f1-score']
                            })
                    
                    if metrics_summary:
                        metrics_df = pd.DataFrame(metrics_summary)
                        st.dataframe(metrics_df)
                        
                        # Heatmap perbandingan
                        fig, ax = plt.subplots(figsize=(10, 6))
                        comparison_matrix = metrics_df.set_index('Model').T
                        sns.heatmap(comparison_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
                        plt.title('Heatmap Perbandingan Metrik' if st.session_state.language == 'id' else 'Metrics Comparison Heatmap')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                else:  # Regression
                    # Tampilkan metrik regresi
                    metrics_summary = []
                    for result in st.session_state.model_results:
                        if result['problem_type'] == "Regression":
                            metrics_summary.append({
                                'Model': result['model_name'],
                                'MSE': result['mse'],
                                'RMSE': result['rmse'],
                                'R²': result['r2'],
                                'Adjusted R²': result['adj_r2']
                            })
                    
                    if metrics_summary:
                        metrics_df = pd.DataFrame(metrics_summary)
                        st.dataframe(metrics_df)
                        
                        # Visualisasi perbandingan
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Plot untuk error metrics (semakin rendah semakin baik)
                        error_df = metrics_df[['Model', 'RMSE', 'MSE']].set_index('Model')
                        error_df.plot(kind='bar', ax=ax1, color=['red', 'orange'])
                        ax1.set_title('Perbandingan Error Metrics' if st.session_state.language == 'id' else 'Error Metrics Comparison')
                        ax1.set_ylabel('Error Value')
                        ax1.legend(['RMSE', 'MSE'])
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # Plot untuk R² metrics (semakin tinggi semakin baik)
                        r2_df = metrics_df[['Model', 'R²', 'Adjusted R²']].set_index('Model')
                        r2_df.plot(kind='bar', ax=ax2, color=['green', 'blue'])
                        ax2.set_title('Perbandingan R² Metrics' if st.session_state.language == 'id' else 'R² Metrics Comparison')
                        ax2.set_ylabel('Score')
                        ax2.legend(['R²', 'Adjusted R²'])
                        ax2.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            with comparison_tabs[2]:
                st.subheader("Peringkat Model" if st.session_state.language == 'id' else "Model Rankings")
                
                if problem_type == "Classification":
                    # Ranking berdasarkan F1-Score
                    ranking_data = []
                    for result in st.session_state.model_results:
                        if result['problem_type'] == "Classification":
                            ranking_data.append({
                                'Model': result['model_name'],
                                'Accuracy': result['accuracy'],
                                'F1-Score': result['classification_report']['weighted avg']['f1-score'],
                                'Precision': result['classification_report']['weighted avg']['precision'],
                                'Recall': result['classification_report']['weighted avg']['recall']
                            })
                    
                    if ranking_data:
                        ranking_df = pd.DataFrame(ranking_data)
                        ranking_df['Rank'] = ranking_df['F1-Score'].rank(ascending=False)
                        ranking_df = ranking_df.sort_values('Rank')
                        
                        st.write("**Ranking berdasarkan F1-Score (terbaik → terburuk):**" if st.session_state.language == 'id' else "**Ranking by F1-Score (best → worst):**")
                        st.dataframe(ranking_df[['Rank', 'Model', 'F1-Score', 'Accuracy', 'Precision', 'Recall']])
                        
                        # Visualisasi ranking
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=ranking_df, x='Model', y='F1-Score', ax=ax)
                        ax.set_title('Ranking Model berdasarkan F1-Score' if st.session_state.language == 'id' else 'Model Ranking by F1-Score')
                        ax.set_ylabel('F1-Score')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                
                else:  # Regression
                    # Ranking berdasarkan R² (semakin tinggi semakin baik)
                    ranking_data = []
                    for result in st.session_state.model_results:
                        if result['problem_type'] == "Regression":
                            ranking_data.append({
                                'Model': result['model_name'],
                                'RMSE': result['rmse'],
                                'R²': result['r2'],
                                'Adjusted R²': result['adj_r2']
                            })
                    
                    if ranking_data:
                        ranking_df = pd.DataFrame(ranking_data)
                        ranking_df['Rank'] = ranking_df['R²'].rank(ascending=False)
                        ranking_df = ranking_df.sort_values('Rank')
                        
                        st.write("**Ranking berdasarkan R² Score (terbaik → terburuk):**" if st.session_state.language == 'id' else "**Ranking by R² Score (best → worst):**")
                        st.dataframe(ranking_df[['Rank', 'Model', 'R²', 'Adjusted R²', 'RMSE']])
                        
                        # Visualisasi ranking
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=ranking_df, x='Model', y='R²', ax=ax)
                        ax.set_title('Ranking Model berdasarkan R² Score' if st.session_state.language == 'id' else 'Model Ranking by R² Score')
                        ax.set_ylabel('R² Score')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
        
        # Tombol untuk reset hasil perbandingan
        if st.session_state.model_results:
            if st.button("Reset Hasil Perbandingan" if st.session_state.language == 'id' else "Reset Comparison Results"):
                st.session_state.model_results = []
                st.rerun() 

        # Tambahkan bagian untuk prediksi data baru
        if st.session_state.model is not None:
            st.subheader("Prediksi Data Baru" if st.session_state.language == 'id' else "Predict New Data")
            
            # Import library untuk PDF
            from fpdf import FPDF
            from datetime import datetime
            import json

            # Fungsi untuk membuat laporan PDF
            def create_prediction_report(input_data, predictions, model_info, problem_type):
                pdf = FPDF()
                pdf.add_page()
                
                # Header
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, 'Laporan Hasil Prediksi' if st.session_state.language == 'id' else 'Prediction Report', 0, 1, 'C')
                pdf.ln(10)
                
                # Informasi Umum
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Informasi Umum:' if st.session_state.language == 'id' else 'General Information:', 0, 1)
                pdf.set_font('Arial', '', 12)

                date_format = "%Y-%m-%d %H:%M:%S"
                pdf.cell(0, 10, f'Tanggal: {datetime.now().strftime(date_format)}' if st.session_state.language == 'id' else f'Date: {datetime.now().strftime(date_format)}', 0, 1)
                pdf.cell(0, 10, f'Jenis Model: {type(st.session_state.model).__name__}' if st.session_state.language == 'id' else f'Model Type: {type(st.session_state.model).__name__}', 0, 1)
                pdf.cell(0, 10, f'Metode: {problem_type}' if st.session_state.language == 'id' else f'Method: {problem_type}', 0, 1)
                
                # Parameter Model
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Parameter Model:' if st.session_state.language == 'id' else 'Model Parameters:', 0, 1)
                pdf.set_font('Arial', '', 10)
                try:
                    for param, value in st.session_state.model.get_params().items():
                        # Konversi value ke string dan batasi panjangnya
                        value_str = str(value)
                        if len(value_str) > 50:  # Batasi panjang nilai
                            value_str = value_str[:47] + '...'
                        pdf.multi_cell(0, 10, f'{param}: {value_str}')
                except Exception as e:
                    pdf.cell(0, 10, 'Parameter model tidak tersedia' if st.session_state.language == 'id' else 'Model parameters not available', 0, 1)
                
                # Metrik Evaluasi
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Metrik Evaluasi Model:' if st.session_state.language == 'id' else 'Model Evaluation Metrics:', 0, 1)
                pdf.set_font('Arial', '', 12)
                
                # Tambahkan perhitungan metrik evaluasi
                if problem_type == "Regression" and hasattr(st.session_state, 'y_test'):
                    y_pred = st.session_state.model.predict(st.session_state.X_test)
                    mse = mean_squared_error(st.session_state.y_test_eval, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(st.session_state.y_test_eval, y_pred)
                    
                    pdf.cell(0, 10, f'Mean Squared Error (MSE): {mse:.4f}', 0, 1)
                    pdf.cell(0, 10, f'Root Mean Squared Error (RMSE): {rmse:.4f}', 0, 1)
                    pdf.cell(0, 10, f'R² Score: {r2:.4f}', 0, 1)
                else:
                    pdf.cell(0, 10, 'Metrik evaluasi tidak tersedia' if st.session_state.language == 'id' else 'Evaluation metrics not available', 0, 1)
                
                # Hasil Prediksi
                pdf.add_page()
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Hasil Prediksi:' if st.session_state.language == 'id' else 'Prediction Results:', 0, 1)
                pdf.set_font('Arial', '', 10)
                
                # Tabel hasil prediksi
                # Hitung lebar kolom yang sesuai
                n_columns = len(input_data.columns) + 1  # +1 untuk kolom prediksi
                col_width = min(pdf.w / n_columns - 2, 35)  # Maksimal 35 pt per kolom
                row_height = 8
                
                # Header tabel
                pdf.set_font('Arial', 'B', 10)
                for col in input_data.columns:
                    pdf.cell(col_width, row_height, str(col)[:15], 1)
                pdf.cell(col_width, row_height, 'Prediksi' if st.session_state.language == 'id' else 'Prediction', 1)
                pdf.ln()
                
                # Isi tabel
                pdf.set_font('Arial', '', 10)
                for i in range(len(input_data)):
                    if i > 0 and i % 40 == 0:  # Tambah halaman baru setiap 40 baris
                        pdf.add_page()
                        # Cetak header lagi
                        pdf.set_font('Arial', 'B', 10)
                        for col in input_data.columns:
                            pdf.cell(col_width, row_height, str(col)[:15], 1)
                        pdf.cell(col_width, row_height, 'Prediksi' if st.session_state.language == 'id' else 'Prediction', 1)
                        pdf.ln()
                        pdf.set_font('Arial', '', 10)
                    
                    for col in input_data.columns:
                        value = input_data.iloc[i][col]
                        if isinstance(value, (int, float)):
                            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                        else:
                            value_str = str(value)
                        pdf.cell(col_width, row_height, value_str[:15], 1)
                    
                    pred_value = predictions[i] if isinstance(predictions, (list, np.ndarray)) else predictions
                    if isinstance(pred_value, (int, float)):
                        pred_str = f"{pred_value:.2f}" if isinstance(pred_value, float) else str(pred_value)
                    else:
                        pred_str = str(pred_value)
                    pdf.cell(col_width, row_height, pred_str[:15], 1)
                    pdf.ln()
                
                # Penanggung Jawab
                pdf.add_page()
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Penanggung Jawab:' if st.session_state.language == 'id' else 'Responsible:', 0, 0, 1)
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, 'Nama: ____________________' if st.session_state.language == 'id' else 'Name: ____________________', 0, 1)
                pdf.cell(0, 10, 'Jabatan: ____________________' if st.session_state.language == 'id' else 'Position: ____________________', 0, 1)
                pdf.cell(0, 10, 'Tanggal: ____________________' if st.session_state.language == 'id' else 'Date: ____________________', 0, 1)
                pdf.cell(0, 20, 'Tanda Tangan:'if st.session_state.language == 'id' else 'Signature:', 0, 1)
                pdf.cell(0, 20, '_____________________', 0, 1)
                
                return pdf
            
            # Pilih metode input data
            input_method = st.radio("Pilih metode input data:" if st.session_state.language == 'id' else "Select input method:", ["Input Manual", "Upload CSV"])
            
            if input_method == "Input Manual":
                # Buat form input untuk setiap fitur
                st.write("Masukkan nilai untuk setiap fitur:" if st.session_state.language == 'id' else "Enter values for each feature:")
                
                input_data = {}
                
                for feature in st.session_state.X_train.columns:
                    # Cek apakah fitur adalah kategorikal atau numerikal
                    if feature in st.session_state.categorical_columns:
                        # Jika ada encoder untuk fitur ini, tampilkan opsi yang tersedia
                        if feature in st.session_state.encoders:
                            options = list(st.session_state.encoders[feature].classes_)
                            input_data[feature] = st.selectbox(f"{feature}:", options)
                        else:
                            input_data[feature] = st.text_input(f"{feature}:")
                    else:
                        # Untuk fitur numerikal, gunakan number_input
                        input_data[feature] = st.number_input(f"{feature}:", format="%.4f")
                
                if st.button("Prediksi"):
                    try:
                        # Konversi input menjadi DataFrame
                        input_df = pd.DataFrame([input_data])
                        
                        # Terapkan preprocessing yang sama seperti data training
                        # Encoding untuk fitur kategorikal
                        for col in [c for c in input_df.columns if c in st.session_state.categorical_columns]:
                            if col in st.session_state.encoders:
                                input_df[col] = st.session_state.encoders[col].transform(input_df[col].astype(str))
                        
                        # Scaling untuk fitur numerikal
                        num_cols = [c for c in input_df.columns if c in st.session_state.numerical_columns]
                        if st.session_state.scaler is not None and num_cols:
                            input_df[num_cols] = st.session_state.scaler.transform(input_df[num_cols])

                        # Pastikan urutan kolom sama dengan saat training
                        input_df = input_df[st.session_state.X_train.columns]

                        # Lakukan prediksi
                        prediction = st.session_state.model.predict(input_df)
                        
                        # Tentukan jenis model
                        model_type = get_model_type(st.session_state.model)
                        
                        # Tampilkan hasil prediksi
                        st.subheader("Hasil Prediksi" if st.session_state.language == 'id' else "Prediction Result")
                        
                        if model_type == "Classification":
                            st.write(f"Kelas yang diprediksi: {prediction[0]}" if st.session_state.language == 'id' else f"Predicted Class: {prediction[0]}")
                            
                            # Jika model memiliki predict_proba, tampilkan probabilitas
                            if hasattr(st.session_state.model, 'predict_proba'):
                                try:
                                    proba = st.session_state.model.predict_proba(input_df)
                                    proba_df = pd.DataFrame(proba, columns=st.session_state.model.classes_)
                                    st.write("Probabilitas untuk setiap kelas:" if st.session_state.language == 'id' else "Probabilities for each class:")
                                    st.dataframe(proba_df)
                                except Exception as e:
                                    st.warning(f"Tidak dapat menghitung probabilitas: {str(e)}" if st.session_state.language == 'id' else f"Cannot calculate probabilities: {str(e)}")
                        else:  # Regression
                            st.write(f"Nilai yang diprediksi: {prediction[0]:.4f}" if st.session_state.language == 'id' else f"Predicted Value: {prediction[0]:.4f}")
                        
                        # Buat laporan PDF
                        try:
                            model_type = get_model_type(st.session_state.model)
                            pdf = create_prediction_report(input_df, prediction, st.session_state.model, model_type)
                            pdf_output = pdf.output(dest='S').encode('latin1')
                            st.download_button(
                                label="Download Laporan PDF" if st.session_state.language == 'id' else "Download PDF Report",
                                data=pdf_output,
                                file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Error saat membuat laporan PDF: {str(e)}" if st.session_state.language == 'id' else f"Error creating PDF report: {str(e)}")
                        
                    except Exception as e:
                        st.error(f"Error saat melakukan prediksi: {str(e)}" if st.session_state.language == 'id' else f"Error during prediction: {str(e)}")
            
            else:  # Upload CSV / Excel
                st.write("Upload file CSV atau Excel dengan data yang ingin diprediksi:" if st.session_state.language == 'id' else "Upload CSV or Excel file with data to predict:")
                uploaded_file = st.file_uploader("Pilih file CSV atau Excel / Choose CSV or Excel file", type=["csv", "xlsx", "xls"], key="prediction_file")
                
                if uploaded_file is not None:
                    
                        # Baca file CSV atau Excel
                        fname = getattr(uploaded_file, 'name', '')
                        if str(fname).lower().endswith(('.xlsx', '.xls')):
                            pred_data = pd.read_excel(uploaded_file)
                        else:
                            pred_data = pd.read_csv(uploaded_file)
                        
                        # Tampilkan preview data
                        st.write("Data Preview:" if st.session_state.language == 'id' else "Preview data:" )
                        st.dataframe(pred_data.head())
                        
                        # Periksa apakah semua fitur yang diperlukan ada
                        missing_features = [f for f in st.session_state.X_train.columns if f not in pred_data.columns]
                        
                        if missing_features:
                            st.error(f"Data tidak memiliki fitur yang diperlukan: {', '.join(missing_features)}" if st.session_state.language == 'id' else f"Data is missing required features: {', '.join(missing_features)}")
                        
                        # Add detailed debugging information
                        st.write("**🔍 Informasi Debug Detail:**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Fitur yang diharapkan model:**")
                            expected_features = list(st.session_state.X_train.columns)
                            st.write(expected_features)
                            
                        with col2:
                            st.write("**Fitur yang tersedia dalam data prediksi:**")
                            available_features = list(pred_data.columns)
                            st.write(available_features)
                        
                        # Show comparison table
                        comparison_df = pd.DataFrame({
                            'Expected Features': expected_features,
                            'Available in CSV': ['✅ Ada' if f in pred_data.columns else '❌ Tidak Ada' for f in expected_features]
                        })
                        st.write("**Perbandingan Fitur:**")
                        st.dataframe(comparison_df)
                        
                        # Provide guidance
                        st.info("""**Cara memperbaiki:**
                        1. Pastikan file CSV prediksi memiliki semua kolom yang digunakan saat training
                        2. Periksa nama kolom (case-sensitive)
                        3. Kolom yang hilang harus ditambahkan ke file CSV prediksi
                        4. Jika kolom tidak tersedia, pertimbangkan untuk melatih ulang model tanpa kolom tersebut""" if st.session_state.language == 'id' else 
                        """**How to fix:**
                        1. Ensure your prediction CSV has all columns used during training
                        2. Check column names (case-sensitive)
                        3. Missing columns must be added to the prediction CSV
                        4. If columns are unavailable, consider retraining the model without these columns""")
                    
                        if not st.session_state.model:
                            # Validasi fitur sebelum prediksi
                            st.write("**📊 Validasi Fitur:**")
                            
                            # Check for missing features
                            missing_features = [f for f in st.session_state.X_train.columns if f not in pred_data.columns]
                            
                            if missing_features:
                                st.error(f"Data tidak memiliki fitur yang diperlukan: {', '.join(missing_features)}")
                                
                                # Add detailed debugging information
                                st.write("**🔍 Informasi Debug Detail:**")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Fitur yang diharapkan model:**")
                                    expected_features = list(st.session_state.X_train.columns)
                                    st.write(expected_features)
                                    
                                with col2:
                                    st.write("**Fitur yang tersedia dalam data prediksi:**")
                                    available_features = list(pred_data.columns)
                                    st.write(available_features)
                                
                                # Show comparison table
                                comparison_df = pd.DataFrame({
                                    'Expected Features': expected_features,
                                    'Available in CSV': ['✅ Ada' if f in pred_data.columns else '❌ Tidak Ada' for f in expected_features]
                                })
                                st.write("**Perbandingan Fitur:**")
                                st.dataframe(comparison_df)
                                
                                # Provide guidance
                                st.info("""**Cara memperbaiki:**
                                1. Pastikan file CSV prediksi memiliki semua kolom yang digunakan saat training
                                2. Periksa nama kolom (case-sensitive)
                                3. Kolom yang hilang harus ditambahkan ke file CSV prediksi
                                4. Jika kolom tidak tersedia, pertimbangkan untuk melatih ulang model tanpa kolom tersebut""")
                            else:
                                # Validasi tipe data
                                type_issues = []
                                for col in st.session_state.X_train.columns:
                                    if col in pred_data.columns:
                                        expected_dtype = st.session_state.X_train[col].dtype
                                        actual_dtype = pred_data[col].dtype
                                        if expected_dtype != actual_dtype:
                                            type_issues.append({
                                                'Column': col,
                                                'Expected Type': str(expected_dtype),
                                                'Actual Type': str(actual_dtype)
                                            })

                                if type_issues:
                                    st.warning("**Peringatan Tipe Data:** Beberapa kolom memiliki tipe data yang berbeda")
                                    st.dataframe(pd.DataFrame(type_issues))
                                    
                                    # Konversi otomatis tipe data
                                    for issue in type_issues:
                                        col = issue['Column']
                                        try:
                                            pred_data[col] = pred_data[col].astype(st.session_state.X_train[col].dtype)
                                            st.success(f"Berhasil mengkonversi {col} ke tipe data yang sesuai")
                                        except Exception as e:
                                            st.error(f"Gagal mengkonversi {col}: {str(e)}")
                                
                                # Lanjutkan dengan preprocessing
                                pred_data = pred_data[st.session_state.X_train.columns]
                                
                                # Encoding untuk fitur kategorikal
                                for col in [c for c in pred_data.columns if c in st.session_state.categorical_columns]:
                                    if col in st.session_state.encoders:
                                        try:
                                            pred_data[col] = st.session_state.encoders[col].transform(pred_data[col].astype(str))
                                        except ValueError as e:
                                            st.error(f"Error encoding {col}: {str(e)}")
                                            st.write(f"Nilai unik dalam data: {pred_data[col].unique()}")
                                            st.write(f"Nilai yang diharapkan encoder: {list(st.session_state.encoders[col].classes_)}")
                                
                                # Scaling untuk fitur numerikal
                                num_cols = [c for c in pred_data.columns if c in st.session_state.numerical_columns]
                                if st.session_state.scaler is not None and num_cols:
                                    pred_data[num_cols] = st.session_state.scaler.transform(pred_data[num_cols])
                                
                                if st.button("Prediksi Batch", key="batch_prediction_btn"):
                                    try:
                                        # Lakukan prediksi
                                        predictions = st.session_state.model.predict(pred_data)
                                        
                                        # Tentukan jenis model
                                        model_type = get_model_type(st.session_state.model)
                                        
                                        # Tambahkan hasil prediksi ke DataFrame
                                        result_df = pred_data.copy()
                                        
                                        if model_type == "Classification":
                                            result_df['Predicted_Class'] = predictions
                                            
                                            # Jika model memiliki predict_proba, tambahkan probabilitas untuk setiap kelas
                                            if hasattr(st.session_state.model, 'predict_proba'):
                                                try:
                                                    proba = st.session_state.model.predict_proba(pred_data)
                                                    for i, class_name in enumerate(st.session_state.model.classes_):
                                                        result_df[f'Probability_{class_name}'] = proba[:, i]
                                                except Exception as e:
                                                    st.warning(f"Tidak dapat menghitung probabilitas: {str(e)}" if st.session_state.language == 'id' else f"Cannot calculate probabilities: {str(e)}")
                                        else:  # Regression
                                            result_df['Predicted_Value'] = predictions
                                        
                                        # Tampilkan hasil
                                        st.subheader("Hasil Prediksi")
                                        st.dataframe(result_df)
                                        
                                        # Download hasil
                                        csv = result_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Hasil Prediksi (CSV)" if st.session_state.language == 'id' else "Download Prediction Results (CSV)",
                                            data=csv,
                                            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv"
                                        )
                                        
                                    except Exception as e:
                                        st.error(f"Error saat melakukan prediksi: {str(e)}" if st.session_state.language == 'id' else f"Error during prediction: {str(e)}")
                    
           
            # Tambahkan bagian untuk memuat model yang sudah disimpan
            st.subheader("Muat Model yang Sudah Disimpan" if st.session_state.language == 'id' else "Load Saved Model")
            
            # Cek apakah folder models ada
            if os.path.exists("models"):
                model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
                
                if model_files:
                    selected_model_file = st.selectbox("Pilih model yang akan dimuat:" if st.session_state.language == 'id' else "Select a model to load:", model_files)
                    
                    if st.button("Muat Model" if st.session_state.language == 'id' else "Load Model"):
                        try:
                            with open(os.path.join("models", selected_model_file), 'rb') as f:
                                loaded_model = pickle.load(f)
                            
                            st.session_state.model = loaded_model
                            st.success(f"Model {selected_model_file} berhasil dimuat!" if st.session_state.language == 'id' else f"Model {selected_model_file} loaded successfully!")
                            
                            # Add download button for loaded model
                            if st.session_state.model is not None:
                                try:
                                    model_bytes = pickle.dumps(st.session_state.model)
                                    st.download_button(
                                        label=f"📥 Unduh Model {selected_model_file} (.pkl)" if st.session_state.language == 'id' else f"📥 Download {selected_model_file} (.pkl)",
                                        data=model_bytes,
                                        file_name=f"loaded_{selected_model_file}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                        mime="application/octet-stream"
                                    )
                                except Exception as pickle_error:
                                    st.warning(f"⚠️ Tidak dapat membuat download model: {str(pickle_error)}" if st.session_state.language == 'id' else f"⚠️ Cannot create model download: {str(pickle_error)}")
                        except Exception as e:
                            st.error(f"Error saat memuat model: {str(e)}")
                else:
                    st.info("Tidak ada model tersimpan di folder 'models'." if st.session_state.language == 'id' else "No saved models found in the 'models' folder.")
            else:
                st.info("Folder 'models' belum dibuat. Latih dan simpan model terlebih dahulu." if st.session_state.language == 'id' else "Folder 'models' does not exist. Train and save models first.")
        else:
            st.info("Silakan latih model terlebih dahulu sebelum melakukan prediksi." if st.session_state.language == 'id' else "Please train a model first before making predictions.")
else:
    st.info("Please complete the preprocessing steps in the previous tab first." if st.session_state.language == 'id' else "Please complete the preprocessing steps in the previous tab first.")

# (wizard nav below)


# --- Wizard Navigation ---
st.markdown("---")
st.markdown("### ⏩ Langkah Selanjutnya" if st.session_state.language == 'id' else "### ⏩ Next Step")
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅️ Kembali ke Preprocessing" if st.session_state.language == 'id' else "⬅️ Back to Preprocessing", use_container_width=True):
        st.switch_page("pages/03_Preprocessing_and_Feature_Engineering.py")
with col_next:
    if st.button("Lanjutkan ke Interpretasi SHAP ➡️" if st.session_state.language == 'id' else "Continue to SHAP Interpretation ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/05_SHAP_Model_Interpretation.py")
