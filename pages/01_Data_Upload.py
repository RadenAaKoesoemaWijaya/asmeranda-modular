import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import io
from datetime import datetime
from ml_engine.evaluation import load_and_predict_model
from ml_engine.ui_helpers import recommend_research_methods, analyze_dataset_with_ai

st.header("Unggah Dataset Anda" if st.session_state.language == 'id' else "Upload Your Dataset")

# Model Prediction Section
st.subheader("🔮 Prediksi Data Baru" if st.session_state.language == 'id' else "🔮 Predict New Data")

col_model, col_data = st.columns(2)

with col_model:
    model_file = st.file_uploader(
        "Unggah model .pkl" if st.session_state.language == 'id' else "Upload .pkl model file",
        type=['pkl'],
        key="model_uploader"
    )

with col_data:
    prediction_data_file = st.file_uploader(
        "Unggah data untuk prediksi (CSV)" if st.session_state.language == 'id' else "Upload data for prediction (CSV)",
        type=['csv'],
        key="prediction_data_uploader"
    )

if model_file and prediction_data_file:
    try:
        # Load prediction data
        prediction_df = pd.read_csv(prediction_data_file)
        
        # Load model and make predictions
        prediction_result = load_and_predict_model(model_file, prediction_df)
        
        if prediction_result['success']:
            st.success("Prediksi berhasil!" if st.session_state.language == 'id' else "Prediction successful!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"Model: {prediction_result['model_type']}")
                st.info(f"Jumlah sampel: {prediction_result['n_samples']}" if st.session_state.language == 'id' else f"Number of samples: {prediction_result['n_samples']}")
                st.info(f"Fitur digunakan: {len(prediction_result['features_used'])}")
            
            with col2:
                # Display prediction statistics
                predictions = prediction_result['predictions']
                if len(np.unique(predictions)) <= 10:  # Classification
                    unique, counts = np.unique(predictions, return_counts=True)
                    pred_counts = dict(zip(unique, counts))
                    st.write("**Distribusi Prediksi:**" if st.session_state.language == 'id' else "**Prediction Distribution:**")
                    for pred, count in pred_counts.items():
                        st.write(f"- {pred}: {count} ({count/len(predictions)*100:.1f}%)")
                else:  # Regression
                    st.write("**Statistik Prediksi:**" if st.session_state.language == 'id' else "**Prediction Statistics:**")
                    st.write(f"Mean: {np.mean(predictions):.2f}")
                    st.write(f"Std: {np.std(predictions):.2f}")
                    st.write(f"Min: {np.min(predictions):.2f}")
                    st.write(f"Max: {np.max(predictions):.2f}")
            
            # Display predictions table
            st.write("**Hasil Prediksi:**" if st.session_state.language == 'id' else "**Prediction Results:**")
            result_df = prediction_df.copy()
            result_df['Prediksi'] = predictions
            
            # Add probabilities if available
            if prediction_result['probabilities'] is not None:
                prob_df = pd.DataFrame(prediction_result['probabilities'])
                prob_cols = [f'Prob_Kelas_{i}' for i in range(prob_df.shape[1])]
                prob_df.columns = prob_cols
                result_df = pd.concat([result_df, prob_df], axis=1)
            
            st.dataframe(result_df.head(100))  # Show first 100 rows
            
            # Download results
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Unduh Hasil Prediksi (CSV)" if st.session_state.language == 'id' else "📥 Download Prediction Results (CSV)",
                data=csv,
                file_name=f"prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.error(f"Gagal melakukan prediksi: {prediction_result['error']}" if st.session_state.language == 'id' else f"Prediction failed: {prediction_result['error']}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Single Prediction Section
if model_file:
    st.markdown("---")
    st.subheader("🔮 Prediksi Manual Satu Data" if st.session_state.language == 'id' else "🔮 Manual Single Prediction")
    
    # Load model to get feature information
    try:
        model = pickle.load(model_file)
        model_file.seek(0)  # Reset file pointer for future use
        
        # Get feature names if available
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            st.info(f"Model membutuhkan {len(feature_names)} fitur: {', '.join(feature_names)}" if st.session_state.language == 'id' else f"Model requires {len(feature_names)} features: {', '.join(feature_names)}")
            
            # Create input form for each feature
            with st.form("single_prediction_form"):
                st.write("**Masukkan nilai fitur:**" if st.session_state.language == 'id' else "**Enter feature values:**")
                
                input_data = {}
                cols = st.columns(2)
                for i, feature in enumerate(feature_names):
                    with cols[i % 2]:
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            step=0.01,
                            key=f"feature_{feature}"
                        )
                
                col_predict, col_clear = st.columns(2)
                with col_predict:
                    predict_button = st.form_submit_button(
                        "🔮 Lakukan Prediksi" if st.session_state.language == 'id' else "🔮 Make Prediction",
                        type="primary"
                    )
                with col_clear:
                    clear_button = st.form_submit_button(
                        "🗑️ Bersihkan" if st.session_state.language == 'id' else "🗑️ Clear"
                    )
            
            if predict_button:
                try:
                    # Create DataFrame from input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Display result
                    st.success("Prediksi Berhasil!" if st.session_state.language == 'id' else "Prediction Successful!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Hasil Prediksi" if st.session_state.language == 'id' else "Prediction Result",
                            f"{prediction:.4f}" if isinstance(prediction, (int, float)) else str(prediction)
                        )
                    
                    with col2:
                        # Show probabilities if classification
                        if hasattr(model, 'predict_proba'):
                            try:
                                probabilities = model.predict_proba(input_df)[0]
                                st.write("**Probabilitas Kelas:**" if st.session_state.language == 'id' else "**Class Probabilities:**")
                                for i, prob in enumerate(probabilities):
                                    st.write(f"Kelas {i}: {prob:.4f} ({prob*100:.2f}%)" if st.session_state.language == 'id' else f"Class {i}: {prob:.4f} ({prob*100:.2f}%)")
                            except:
                                pass
                    
                    # Show input summary
                    with st.expander("📋 Ringkasan Input" if st.session_state.language == 'id' else "📋 Input Summary"):
                        st.write("**Nilai Fitur yang Dimasukkan:**" if st.session_state.language == 'id' else "**Entered Feature Values:**")
                        for feature, value in input_data.items():
                            st.write(f"- {feature}: {value}")
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi: {str(e)}" if st.session_state.language == 'id' else f"Error in prediction: {str(e)}")
            
        else:
            st.warning("Model tidak memiliki informasi fitur. Pastikan data yang dimasukkan sesuai dengan training data." if st.session_state.language == 'id' else "Model doesn't have feature information. Make sure input data matches training data.")
            
            # Simple text area for manual input
            manual_input = st.text_area(
                "Masukkan data (format: fitur1,nilai1;fitur2,nilai2)" if st.session_state.language == 'id' else "Enter data (format: feature1,value1;feature2,value2)",
                placeholder="contoh: age,25;income,50000;score,85" if st.session_state.language == 'id' else "example: age,25;income,50000;score,85"
            )
            
            if st.button("🔮 Lakukan Prediksi" if st.session_state.language == 'id' else "🔮 Make Prediction"):
                try:
                    # Parse manual input
                    data_pairs = manual_input.split(';')
                    input_dict = {}
                    for pair in data_pairs:
                        if ',' in pair:
                            feature, value = pair.split(',')
                            input_dict[feature.strip()] = float(value.strip())
                    
                    # Create DataFrame
                    input_df = pd.DataFrame([input_dict])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    st.success("Prediksi Berhasil!" if st.session_state.language == 'id' else "Prediction Successful!")
                    st.metric(
                        "Hasil Prediksi" if st.session_state.language == 'id' else "Prediction Result",
                        f"{prediction:.4f}" if isinstance(prediction, (int, float)) else str(prediction)
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}. Pastikan format input benar." if st.session_state.language == 'id' else f"Error: {str(e)}. Make sure input format is correct.")
                    
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}" if st.session_state.language == 'id' else f"Error loading model: {str(e)}")

st.markdown("---")

# Section title for new dataset training
st.subheader("📚 Latih Dataset Baru" if st.session_state.language == 'id' else "📚 Train New Dataset")

uploaded_file = st.file_uploader(
    "Pilih file CSV, Excel (.xlsx/.xls), atau ZIP berisi folder train/test" if st.session_state.language == 'id' else "Choose a CSV, Excel (.xlsx/.xls), or ZIP file with train/test folders",
    type=["csv", "xlsx", "xls", "zip"]
)

if uploaded_file is not None:
    import zipfile
    import tempfile
    import os

    def read_tabular_file(path_or_buffer, is_path=True):
        """Membaca file CSV atau Excel secara otomatis berdasarkan ekstensi."""
        name = path_or_buffer if is_path else getattr(path_or_buffer, 'name', '')
        if str(name).lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(path_or_buffer)
        else:
            return pd.read_csv(path_or_buffer)

    if uploaded_file.name.lower().endswith('.zip'):
        # Proses ZIP: cari file train/test dalam folder atau file langsung
        with tempfile.TemporaryDirectory() as tmpdir:
            zf = zipfile.ZipFile(uploaded_file)
            zf.extractall(tmpdir)
            # Cari file train dan test
            train_path, test_path = None, None
            train_files, test_files = [], []
            
            TABULAR_EXTS = ('.csv', '.xlsx', '.xls')
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.lower().endswith(TABULAR_EXTS):
                        full_path = os.path.join(root, f)
                        # Cek apakah file ada di folder training atau testing
                        if 'train' in root.lower() or 'train' in f.lower():
                            train_files.append(full_path)
                        elif 'test' in root.lower() or 'test' in f.lower():
                            test_files.append(full_path)
                        # Fallback: cek nama file saja
                        elif 'train' in f.lower():
                            train_files.append(full_path)
                        elif 'test' in f.lower():
                            test_files.append(full_path)
            
            # Ambil file pertama dari masing-masing kategori
            if train_files:
                train_path = train_files[0]
            if test_files:
                test_path = test_files[0]
            
            if train_path and test_path:
                try:
                    train_data = read_tabular_file(train_path)
                    test_data = read_tabular_file(test_path)
                    
                    # Gabungkan dataset training dan testing
                    combined_data = pd.concat([train_data, test_data], ignore_index=True)
                    
                    # Simpan ke session state
                    st.session_state.data = combined_data
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    
                    st.success(f"Berhasil mendeteksi dan memuat data training ({train_data.shape[0]} baris) dan testing ({test_data.shape[0]} baris) dari ZIP." if st.session_state.language == 'id' else f"Successfully loaded training ({train_data.shape[0]} rows) and testing ({test_data.shape[0]} rows) from ZIP.")
                    st.info(f"Dataset gabungan: {combined_data.shape[0]} baris dan {combined_data.shape[1]} kolom." if st.session_state.language == 'id' else f"Combined dataset: {combined_data.shape[0]} rows and {combined_data.shape[1]} columns.")
                    
                    # Tampilkan preview dataset gabungan
                    st.subheader("Preview Dataset Gabungan" if st.session_state.language == 'id' else "Combined Dataset Preview")
                    st.dataframe(combined_data.head())
                    
                    # Integrate modular data type detection for ZIP data
                    if DATA_TYPE_DETECTOR_AVAILABLE and data_type_detector is not None:
                        try:
                            # Detect data types for each column in combined data
                            data_types_info = {}
                            for column in combined_data.columns:
                                analysis = data_type_detector.analyze_series(combined_data[column], column)
                                data_types_info[column] = analysis
                            
                            # Store data type information in session state
                            st.session_state.data_types_info = data_types_info
                            
                            # Display data type detection results
                            with st.expander("🔍 Deteksi Tipe Data Otomatis (ZIP)" if st.session_state.language == 'id' else "🔍 Automatic Data Type Detection (ZIP)"):
                                st.info("Hasil deteksi tipe data otomatis untuk dataset gabungan:" if st.session_state.language == 'id' else "Automatic data type detection results for combined dataset:")
                                
                                detection_df = pd.DataFrame({
                                    'Kolom': [col for col in data_types_info.keys()],
                                    'Tipe Terdeteksi': [info.get('detected_type', 'Unknown') for info in data_types_info.values()],
                                    'Confidence': [f"{info.get('confidence', 0):.2f}" for info in data_types_info.values()],
                                    'Nilai Unik': [info.get('unique_count', 0) for info in data_types_info.values()],
                                    'Nilai Hilang': [info.get('null_count', 0) for info in data_types_info.values()]
                                })
                                st.dataframe(detection_df)
                                
                        except Exception as e:
                            if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                error_result = error_handler.handle_error(e, "Data Type Detection (ZIP)")
                                st.warning(f"⚠️ {error_result['message']}")
                            else:
                                st.warning(f"⚠️ Gagal melakukan deteksi tipe data untuk ZIP: {str(e)}")
                    
                    # Initialize session manager for ZIP data tracking
                    if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                        try:
                            session_manager.initialize_data_session(combined_data)
                            st.session_state.data_session_initialized = True
                        except Exception as e:
                            if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                error_result = error_handler.handle_error(e, "Session Management (ZIP)")
                                st.warning(f"⚠️ {error_result['message']}")
                            else:
                                st.warning(f"⚠️ Gagal menginisialisasi session manager untuk ZIP: {str(e)}")
                    
                    # Tampilkan informasi dataset terpisah
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Dataset Training" if st.session_state.language == 'id' else "Training Dataset")
                        st.dataframe(train_data.head())
                    with col2:
                        st.subheader("Dataset Testing" if st.session_state.language == 'id' else "Testing Dataset")
                        st.dataframe(test_data.head())
                        
                except Exception as e:
                    st.error(f"Error saat membaca file: {e}" if st.session_state.language == 'id' else f"Error reading files: {e}")
                    
            elif train_path or test_path:
                # Jika hanya ada satu file, gunakan sebagai dataset utama
                single_path = train_path or test_path
                try:
                    data = read_tabular_file(single_path)
                    st.session_state.data = data
                    st.session_state.train_data = None
                    st.session_state.test_data = None
                    st.success(f"Dataset berhasil dimuat dengan {data.shape[0]} baris dan {data.shape[1]} kolom." if st.session_state.language == 'id' else f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
                    st.dataframe(data.head())
                    
                    # Integrate modular data type detection for single ZIP file
                    if DATA_TYPE_DETECTOR_AVAILABLE and data_type_detector is not None:
                        try:
                            # Detect data types for each column
                            data_types_info = {}
                            for column in data.columns:
                                analysis = data_type_detector.analyze_series(data[column], column)
                                data_types_info[column] = analysis
                            
                            # Store data type information in session state
                            st.session_state.data_types_info = data_types_info
                            
                            # Display data type detection results
                            with st.expander("🔍 Deteksi Tipe Data Otomatis (ZIP Single)" if st.session_state.language == 'id' else "🔍 Automatic Data Type Detection (ZIP Single)"):
                                st.info("Hasil deteksi tipe data otomatis:" if st.session_state.language == 'id' else "Automatic data type detection results:")
                                
                                detection_df = pd.DataFrame({
                                    'Kolom': [col for col in data_types_info.keys()],
                                    'Tipe Terdeteksi': [info.get('detected_type', 'Unknown') for info in data_types_info.values()],
                                    'Confidence': [f"{info.get('confidence', 0):.2f}" for info in data_types_info.values()],
                                    'Nilai Unik': [info.get('unique_count', 0) for info in data_types_info.values()],
                                    'Nilai Hilang': [info.get('null_count', 0) for info in data_types_info.values()]
                                })
                                st.dataframe(detection_df)
                                
                        except Exception as e:
                            if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                error_result = error_handler.handle_error(e, "Data Type Detection (ZIP Single)")
                                st.warning(f"⚠️ {error_result['message']}")
                            else:
                                st.warning(f"⚠️ Gagal melakukan deteksi tipe data untuk ZIP single: {str(e)}")
                    
                    # Initialize session manager for single ZIP data tracking
                    if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                        try:
                            session_manager.initialize_data_session(data)
                            st.session_state.data_session_initialized = True
                        except Exception as e:
                            if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                                error_result = error_handler.handle_error(e, "Session Management (ZIP Single)")
                                st.warning(f"⚠️ {error_result['message']}")
                            else:
                                st.warning(f"⚠️ Gagal menginisialisasi session manager untuk ZIP single: {str(e)}")
                except Exception as e:
                    st.error(f"Error saat membaca file: {e}" if st.session_state.language == 'id' else f"Error reading files: {e}")
            else:
                st.error("ZIP tidak berisi file train/test yang valid (CSV/Excel)." if st.session_state.language == 'id' else "ZIP does not contain valid train/test files (CSV/Excel).")
                st.info("Pastikan ZIP berisi folder 'training' dan 'testing' dengan file CSV/Excel, atau file dengan nama yang mengandung 'train' dan 'test'." if st.session_state.language == 'id' else "Make sure ZIP contains 'training' and 'testing' folders with CSV/Excel files, or files with names containing 'train' and 'test'.")
    else:
        # Proses single CSV atau Excel
        try:
            data = read_tabular_file(uploaded_file, is_path=False)
            st.session_state.data = data
            st.session_state.train_data = None
            st.session_state.test_data = None
            st.success(f"Dataset berhasil dimuat dengan {data.shape[0]} baris dan {data.shape[1]} kolom." if st.session_state.language == 'id' else f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
            st.dataframe(data.head())
            
            # Integrate modular data type detection
            if DATA_TYPE_DETECTOR_AVAILABLE and data_type_detector is not None:
                try:
                    # Detect data types for each column
                    data_types_info = {}
                    for column in data.columns:
                        analysis = data_type_detector.analyze_series(data[column], column)
                        data_types_info[column] = analysis
                    
                    # Store data type information in session state
                    st.session_state.data_types_info = data_types_info
                    
                    # Display data type detection results
                    with st.expander("🔍 Deteksi Tipe Data Otomatis" if st.session_state.language == 'id' else "🔍 Automatic Data Type Detection"):
                        st.info("Hasil deteksi tipe data otomatis:" if st.session_state.language == 'id' else "Automatic data type detection results:")
                        
                        detection_df = pd.DataFrame({
                            'Kolom': [col for col in data_types_info.keys()],
                            'Tipe Terdeteksi': [info.get('detected_type', 'Unknown') for info in data_types_info.values()],
                            'Confidence': [f"{info.get('confidence', 0):.2f}" for info in data_types_info.values()],
                            'Nilai Unik': [info.get('unique_count', 0) for info in data_types_info.values()],
                            'Nilai Hilang': [info.get('null_count', 0) for info in data_types_info.values()]
                        })
                        st.dataframe(detection_df)
                        
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Data Type Detection")
                        st.warning(f"⚠️ {error_result['message']}")
                    else:
                        st.warning(f"⚠️ Gagal melakukan deteksi tipe data: {str(e)}")
            
            # Initialize session manager for data tracking
            if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                try:
                    session_manager.initialize_data_session(data)
                    st.session_state.data_session_initialized = True
                except Exception as e:
                    if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                        error_result = error_handler.handle_error(e, "Session Management")
                        st.warning(f"⚠️ {error_result['message']}")
                    else:
                        st.warning(f"⚠️ Gagal menginisialisasi session manager: {str(e)}")
        except Exception as e:
            st.error(f"Error: {e}")
        
        st.subheader("Informasi Data" if st.session_state.language == 'id' else "Data Information")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        # Tambahkan pemilihan fitur untuk dibuang
        st.subheader("Pemilihan Fitur" if st.session_state.language == 'id' else "Feature Selection")
        all_columns = data.columns.tolist()
        
        # Gunakan session state untuk menyimpan fitur yang dipilih untuk dibuang
        if 'columns_to_drop' not in st.session_state:
            st.session_state.columns_to_drop = []
        
        columns_to_drop = st.multiselect(
            "Pilih kolom yang ingin dibuang:" if st.session_state.language == 'id' else "Select columns to drop:",
            all_columns,
            default=st.session_state.columns_to_drop
        )
        
        if columns_to_drop:
            st.warning(f"Kolom yang akan dibuang: {', '.join(columns_to_drop)}")
            
            # Update dataset dengan menghapus kolom yang dipilih
            data = data.drop(columns=columns_to_drop)
            st.session_state.data = data
            st.session_state.columns_to_drop = columns_to_drop
            
            st.success(f"Dataset telah diperbarui. Ukuran baru: {data.shape[0]} baris × {data.shape[1]} kolom")
            st.dataframe(data.head())
        
        st.subheader("Statistik Data" if st.session_state.language == 'id' else "Data Statistics")
        st.dataframe(data.describe())
        
        # Time Series Detection
        st.subheader("Deteksi Dataset Time Series" if st.session_state.language == 'id' else "Time Series Dataset Detection")
        
        # Check if this is a time series dataset
        is_time_series = st.checkbox(
            "Apakah ini dataset time series?" if st.session_state.language == 'id' else "Is this a time series dataset?",
            value=st.session_state.is_time_series,
            help="Centang jika dataset ini berisi data time series untuk forecasting" if st.session_state.language == 'id' else "Check if this dataset contains time series data for forecasting"
        )
        
        st.session_state.is_time_series = is_time_series
        
        if is_time_series:
            st.info("Dataset akan diproses sebagai time series untuk analisis forecasting" if st.session_state.language == 'id' else "Dataset will be processed as time series for forecasting analysis")
            
            # Select time column
            date_columns = data.select_dtypes(include=['object', 'datetime64']).columns.tolist()
            numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                time_column = st.selectbox(
                    "Pilih kolom waktu/date:" if st.session_state.language == 'id' else "Select time/date column:",
                    [""] + date_columns + numeric_columns,
                    index=0 if st.session_state.time_column is None else ([""] + date_columns + numeric_columns).index(st.session_state.time_column)
                )
                st.session_state.time_column = time_column if time_column else None
            
            with col2:
                if numeric_columns:
                    # Validasi target_column yang tersimpan di session state
                    default_index = 0
                    if st.session_state.target_column is not None:
                        available_columns = [""] + numeric_columns
                        if st.session_state.target_column in available_columns:
                            default_index = available_columns.index(st.session_state.target_column)
                        else:
                            # Reset jika kolom target tidak valid
                            st.session_state.target_column = None
                    
                    target_column = st.selectbox(
                        "Pilih kolom target untuk forecasting:" if st.session_state.language == 'id' else "Select target column for forecasting:",
                        [""] + numeric_columns,
                        index=default_index
                    )
                    st.session_state.target_column = target_column if target_column else None
                else:
                    st.warning("Tidak ada kolom numerik untuk forecasting" if st.session_state.language == 'id' else "No numeric columns for forecasting")
                    
            if time_column and target_column:
                # Validate time column
                try:
                    if data[time_column].dtype == 'object':
                        data[time_column] = pd.to_datetime(data[time_column])
                    
                    # Check if time column is monotonic
                    is_monotonic = data[time_column].is_monotonic_increasing
                    
                    if not is_monotonic:
                        st.warning("Kolom waktu tidak berurutan. Data akan diurutkan berdasarkan waktu." if st.session_state.language == 'id' else "Time column is not sequential. Data will be sorted by time.")
                        data = data.sort_values(by=time_column)
                        st.session_state.data = data
                    
                    st.success(f"Dataset time series terdeteksi: {len(data)} observasi dari {data[time_column].min()} hingga {data[time_column].max()}")
                    
                    # Display time series preview
                    st.write("Preview data time series:" if st.session_state.language == 'id' else "Time series data preview:")
                    time_series_preview = data[[time_column, target_column]].head(10)
                    st.dataframe(time_series_preview)
                    
                except Exception as e:
                    st.error(f"Error processing time column: {e}")
                    st.session_state.is_time_series = False
        
        # Identify numerical and categorical columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        st.session_state.numerical_columns = numerical_cols
        st.session_state.categorical_columns = categorical_cols
        
        st.write(f"Kolom numerik: {', '.join(numerical_cols)}" if st.session_state.language == 'id' else f"Numerical columns: {', '.join(numerical_cols)}")
        st.write(f"Kolom kategorikal: {', '.join(categorical_cols)}" if st.session_state.language == 'id' else f"Categorical columns: {', '.join(categorical_cols)}")
        
        # Integrate workflow validation for data readiness
        if WORKFLOW_VALIDATOR_AVAILABLE and workflow_validator is not None:
            try:
                # Validate data readiness for different ML workflows
                validation_results = workflow_validator.validate_data_readiness(data, numerical_cols, categorical_cols)
                
                with st.expander("✅ Validasi Kesiapan Data" if st.session_state.language == 'id' else "✅ Data Readiness Validation"):
                    st.info("Hasil validasi kesiapan data untuk machine learning:" if st.session_state.language == 'id' else "Data readiness validation results for machine learning:")
                    
                    # Display validation results
                    for result in validation_results:
                        if result['status'] == 'success':
                            st.success(f"✅ {result['message']}")
                        elif result['status'] == 'warning':
                            st.warning(f"⚠️ {result['message']}")
                        elif result['status'] == 'error':
                            st.error(f"❌ {result['message']}")
                    
                    # Store validation results in session state
                    st.session_state.data_validation_results = validation_results
                    
                    # Check if data is ready for ML workflows
                    ml_readiness = workflow_validator.check_ml_readiness(validation_results)
                    st.session_state.ml_readiness = ml_readiness
                    
                    if ml_readiness['ready']:
                        st.success(f"🎯 {ml_readiness['message']}")
                        st.write("**Workflow yang tersedia:**" if st.session_state.language == 'id' else "**Available workflows:**")
                        for workflow in ml_readiness['available_workflows']:
                            st.write(f"• {workflow}")
                    else:
                        st.warning(f"⚠️ {ml_readiness['message']}")
                        if 'recommendations' in ml_readiness:
                            st.write("**Rekomendasi:**" if st.session_state.language == 'id' else "**Recommendations:**")
                            for rec in ml_readiness['recommendations']:
                                st.write(f"• {rec}")
                                
            except Exception as e:
                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                    error_result = error_handler.handle_error(e, "Workflow Validation")
                    st.warning(f"⚠️ {error_result['message']}")
                else:
                    st.warning(f"⚠️ Gagal melakukan validasi workflow: {str(e)}")
        
        # Rekomendasi Metode Penelitian
        st.subheader("🎯 Rekomendasi Metode Penelitian" if st.session_state.language == 'id' else "🎯 Research Method Recommendations")
        recommendations = recommend_research_methods(data)
        
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.warning(f"**{rec['title']}**\n\n{rec['description']}")
            elif rec['type'] == 'info':
                st.info(f"**{rec['title']}**\n\n{rec['description']}")
            elif rec['type'] == 'success':
                st.success(f"**{rec['title']}**\n\n{rec['description']}")
            
            if 'methods' in rec:
                st.write(f"**Metode yang disarankan:**" if st.session_state.language == 'id' else f"**Recommended methods:**")
                for method in rec['methods']:
                    st.write(f"• {method}")
                st.write("")
        

        
        # Enhanced AI-Powered Analysis Section
        st.subheader("🤖 AI-Powered Dataset Analysis" if st.session_state.language == 'id' else "🤖 AI-Powered Dataset Analysis")
        
        # Add toggle for AI analysis
        if 'show_ai_analysis' not in st.session_state:
            st.session_state.show_ai_analysis = False
        
        # Tombol AI Analysis - Hanya satu tombol utama
        if st.button("🧠 Generate Analisis Dataset" if st.session_state.language == 'id' else "🧠 Generate Dataset Analysis", 
                   type="primary", key="generate_ai_analysis"):
            st.session_state.show_ai_analysis = True
            try:
                log_feature('ai_analysis_generate')
            except Exception:
                pass
            st.rerun()
        
        # Display AI Analysis - Fokus pada Potensi Keberhasilan
        if st.session_state.get('show_ai_analysis', False):
            with st.expander("📊 Potensi Keberhasilan Dataset untuk Penelitian" if st.session_state.language == 'id' else "📊 Dataset Success Potential for Research", expanded=True):
                with st.spinner("Menganalisis potensi keberhasilan dataset..." if st.session_state.language == 'id' else "Analyzing dataset success potential..."):
                    ai_recommendations = analyze_dataset_with_ai(data)
                    
                    # Filter hanya rekomendasi yang relevan untuk keberhasilan penelitian
                    success_recommendations = []
                    for rec in ai_recommendations:
                        if rec['type'] in ['success', 'info'] and any(keyword in rec['title'].lower() or rec['description'].lower() 
                                                                      for keyword in ['siap', 'optimal', 'baik', 'success', 'good', 'optimal', 'ready']):
                            success_recommendations.append(rec)
                    
                    # Tampilkan ringkasan keberhasilan
                    if success_recommendations:
                        st.success("✅ **Dataset ini memiliki potensi keberhasilan yang baik untuk penelitian!**" if st.session_state.language == 'id' else "✅ **This dataset has good success potential for research!**")
                        
                        for rec in success_recommendations[:3]:  # Batasi hingga 3 rekomendasi utama
                            st.write(f"**{rec['title']}**")
                            st.write(f"{rec['description']}")
                            
                            if 'details' in rec and rec['details']:
                                st.write("**Poin-poin utama:**" if st.session_state.language == 'id' else "**Key points:**")
                                for detail in rec['details'][:2]:  # Batasi detail
                                    st.write(f"• {detail}")
                            st.write("")
                    else:
                        # Jika tidak ada rekomendasi sukses, tampilkan yang paling relevan
                        relevant_rec = [rec for rec in ai_recommendations if rec['type'] != 'error'][:2]
                        if relevant_rec:
                            for rec in relevant_rec:
                                if rec['type'] == 'warning':
                                    st.warning(f"**⚠️ {rec['title']}**")
                                else:
                                    st.info(f"**ℹ️ {rec['title']}**")
                                st.write(f"{rec['description']}")
                                st.write("")
                        
                        st.info("💡 **Saran:** Dataset ini memerlukan preprocessing tambahan untuk optimal dalam penelitian." if st.session_state.language == 'id' else "💡 **Suggestion:** This dataset requires additional preprocessing to be optimal for research.")
     

    
    # Tambahkan informasi kolom untuk dataset gabungan dari ZIP
    if uploaded_file.name.endswith('.zip') and st.session_state.data is not None:
        data = st.session_state.data
        st.subheader("Informasi Dataset Gabungan" if st.session_state.language == 'id' else "Combined Dataset Information")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        # Tambahkan pemilihan fitur untuk dibuang
        st.subheader("Pemilihan Fitur" if st.session_state.language == 'id' else "Feature Selection")
        all_columns = data.columns.tolist()
        
        # Gunakan session state untuk menyimpan fitur yang dipilih untuk dibuang
        if 'columns_to_drop' not in st.session_state:
            st.session_state.columns_to_drop = []
        
        columns_to_drop = st.multiselect(
            "Pilih kolom yang ingin dibuang:" if st.session_state.language == 'id' else "Select columns to drop:",
            all_columns,
            default=st.session_state.columns_to_drop
        )
        
        if columns_to_drop:
            st.warning(f"Kolom yang akan dibuang: {', '.join(columns_to_drop)}")
            
            # Update dataset dengan menghapus kolom yang dipilih
            data = data.drop(columns=columns_to_drop)
            st.session_state.data = data
            st.session_state.columns_to_drop = columns_to_drop
            
            st.success(f"Dataset telah diperbarui. Ukuran baru: {data.shape[0]} baris × {data.shape[1]} kolom")
            st.dataframe(data.head())
        
        st.subheader("Statistik Dataset Gabungan" if st.session_state.language == 'id' else "Combined Dataset Statistics")
        st.dataframe(data.describe())
        
        # Identify numerical and categorical columns untuk dataset gabungan
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        st.session_state.numerical_columns = numerical_cols
        st.session_state.categorical_columns = categorical_cols
        
        st.write(f"Kolom numerik: {', '.join(numerical_cols)}" if st.session_state.language == 'id' else f"Numerical columns: {', '.join(numerical_cols)}")
        st.write(f"Kolom kategorikal: {', '.join(categorical_cols)}" if st.session_state.language == 'id' else f"Categorical columns: {', '.join(categorical_cols)}")
        
        # Integrate workflow validation for ZIP dataset
        if WORKFLOW_VALIDATOR_AVAILABLE and workflow_validator is not None:
            try:
                # Validate data readiness for different ML workflows
                validation_results = workflow_validator.validate_data_readiness(data, numerical_cols, categorical_cols)
                
                with st.expander("✅ Validasi Kesiapan Data (ZIP)" if st.session_state.language == 'id' else "✅ Data Readiness Validation (ZIP)"):
                    st.info("Hasil validasi kesiapan data untuk machine learning:" if st.session_state.language == 'id' else "Data readiness validation results for machine learning:")
                    
                    # Display validation results
                    for result in validation_results:
                        if result['status'] == 'success':
                            st.success(f"✅ {result['message']}")
                        elif result['status'] == 'warning':
                            st.warning(f"⚠️ {result['message']}")
                        elif result['status'] == 'error':
                            st.error(f"❌ {result['message']}")
                    
                    # Store validation results in session state
                    st.session_state.data_validation_results = validation_results
                    
                    # Check if data is ready for ML workflows
                    ml_readiness = workflow_validator.check_ml_readiness(validation_results)
                    st.session_state.ml_readiness = ml_readiness
                    
                    if ml_readiness['ready']:
                        st.success(f"🎯 {ml_readiness['message']}")
                        st.write("**Workflow yang tersedia:**" if st.session_state.language == 'id' else "**Available workflows:**")
                        for workflow in ml_readiness['available_workflows']:
                            st.write(f"• {workflow}")
                    else:
                        st.warning(f"⚠️ {ml_readiness['message']}")
                        if 'recommendations' in ml_readiness:
                            st.write("**Rekomendasi:**" if st.session_state.language == 'id' else "**Recommendations:**")
                            for rec in ml_readiness['recommendations']:
                                st.write(f"• {rec}")
                                
            except Exception as e:
                if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                    error_result = error_handler.handle_error(e, "Workflow Validation (ZIP)")
                    st.warning(f"⚠️ {error_result['message']}")
                else:
                    st.warning(f"⚠️ Gagal melakukan validasi workflow untuk ZIP: {str(e)}")
        
        # Rekomendasi Metode Penelitian untuk dataset gabungan
        st.subheader("🎯 Rekomendasi Metode Penelitian" if st.session_state.language == 'id' else "🎯 Research Method Recommendations")
        recommendations = recommend_research_methods(data)
        
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.warning(f"**{rec['title']}**\n\n{rec['description']}")
            elif rec['type'] == 'info':
                st.info(f"**{rec['title']}**\n\n{rec['description']}")
            elif rec['type'] == 'success':
                st.success(f"**{rec['title']}**\n\n{rec['description']}")
            
            if 'methods' in rec:
                st.write(f"**Metode yang disarankan:**" if st.session_state.language == 'id' else f"**Recommended methods:**")
                for method in rec['methods']:
                    st.write(f"• {method}")
                st.write("")
        
        # 📡 Sensor Data Feature Extraction (Optional): slope, integral, min-max + normalisasi sebelum EDA
        if st.session_state.data is not None:
            st.markdown("---")
            st.subheader("📡 Ekstraksi Ciri Data Sensor (Opsional)" if st.session_state.language == 'id' else "📡 Sensor Data Feature Extraction (Optional)")
            
            is_sensor_data = st.checkbox(
                "Apakah ini data log sensor mentah yang perlu ekstraksi ciri?" if st.session_state.language == 'id' else "Is this raw sensor log data that needs feature extraction?",
                key="is_sensor_data_checkbox"
            )
            
            if is_sensor_data:
                st.info("Ciri yang diekstrak: **slope** (kemiringan vs waktu), **integral** (luas di bawah kurva), **min** dan **max** dalam range waktu. Setelah ekstraksi, data dinormalisasi sebelum lanjut ke Exploratory Data Analysis." if st.session_state.language == 'id' else "Extracted features: **slope** (vs time), **integral** (area under curve), **min** and **max** in time range. After extraction, data is normalized before proceeding to Exploratory Data Analysis.")
                
                data = st.session_state.data
                all_cols = data.columns.tolist()
                num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                
                # Kolom waktu (wajib untuk slope/integral/min-max per range waktu)
                time_col = st.selectbox(
                    "Kolom waktu (untuk slope & integral):" if st.session_state.language == 'id' else "Time column (for slope & integral):",
                    options=all_cols,
                    key="sensor_time_col"
                )
                
                col_group, col_target = st.columns(2)
                
                with col_group:
                    group_cols = st.multiselect(
                        "Pilih kolom untuk pengelompokan (misal: ID Sensor):" if st.session_state.language == 'id' else "Select columns for grouping (e.g., Sensor ID):",
                        all_cols,
                        key="sensor_group_cols"
                    )
                    
                with col_target:
                    target_cols = st.multiselect(
                        "Pilih kolom sensor untuk ekstraksi ciri (slope, integral, min, max):" if st.session_state.language == 'id' else "Select sensor columns for feature extraction (slope, integral, min, max):",
                        num_cols,
                        default=[c for c in num_cols if c != time_col and c not in (group_cols or [])],
                        key="sensor_target_cols"
                    )
                
                window_size = st.number_input(
                    "Ukuran jendela waktu (jumlah baris per window). Kosongkan atau 0 = satu window per kelompok:" if st.session_state.language == 'id' else "Time window size (rows per window). Leave 0 for one window per group:",
                    min_value=0,
                    value=0,
                    step=1,
                    key="sensor_window_size"
                )
                normalization_method = st.selectbox(
                    "Normalisasi setelah ekstraksi (sebelum EDA):" if st.session_state.language == 'id' else "Normalization after extraction (before EDA):",
                    ["MinMaxScaler (0-1)", "StandardScaler (z-score)"],
                    key="sensor_norm_method"
                )
                
                if st.button("🚀 Ekstrak Ciri Sensor & Normalisasi" if st.session_state.language == 'id' else "🚀 Extract Sensor Features & Normalize", type="primary"):
                    if not target_cols:
                        st.warning("Mohon pilih setidaknya satu kolom target untuk diekstrak cirinya." if st.session_state.language == 'id' else "Please select at least one target column to extract features from.")
                    else:
                        try:
                            from utils import extract_sensor_features_slope_integral_minmax, normalize_sensor_features
                            window = int(window_size) if window_size and window_size > 0 else None
                            extracted_df = extract_sensor_features_slope_integral_minmax(
                                data,
                                time_column=time_col,
                                group_cols=group_cols if group_cols else None,
                                target_cols=target_cols,
                                window_size=window
                            )
                            
                            if extracted_df is not None and not extracted_df.empty:
                                # Normalisasi sebelum EDA
                                norm_method = 'minmax' if 'MinMax' in normalization_method else 'standard'
                                feature_cols = extracted_df.select_dtypes(include=[np.number]).columns.tolist()
                                extracted_df = normalize_sensor_features(extracted_df, feature_columns=feature_cols, method=norm_method)
                                
                                st.session_state.data = extracted_df
                                st.session_state.numerical_columns = extracted_df.select_dtypes(include=[np.number]).columns.tolist()
                                st.session_state.categorical_columns = extracted_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                                
                                if SESSION_MANAGER_AVAILABLE and session_manager is not None:
                                    try:
                                        session_manager.initialize_data_session(extracted_df)
                                    except:
                                        pass
                                
                                st.success(f"Berhasil mengekstrak ciri (slope, integral, min-max) dan menormalisasi. Ukuran data: {extracted_df.shape[0]} baris × {extracted_df.shape[1]} kolom. Lanjut ke tab Exploratory Data Analytic." if st.session_state.language == 'id' else f"Slope, integral, min-max extracted and normalized. Data size: {extracted_df.shape[0]} rows × {extracted_df.shape[1]} columns. Proceed to Exploratory Data Analysis tab.")
                                st.dataframe(extracted_df.head())
                                
                                try:
                                    auth_db.record_activity(st.session_state.current_username, 'extract_sensor_features')
                                except:
                                    pass
                                    
                                safe_rerun()
                            else:
                                st.error("Ekstraksi tidak menghasilkan data. Periksa kolom waktu dan target." if st.session_state.language == 'id' else "Extraction produced no data. Check time and target columns.")
                        except Exception as e:
                            st.error(f"Error saat ekstraksi ciri: {str(e)}" if st.session_state.language == 'id' else f"Error during feature extraction: {str(e)}")

# Tab 2: Exploratory Data Analysis
