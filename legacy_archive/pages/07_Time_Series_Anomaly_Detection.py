import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if "data" not in st.session_state or st.session_state.data is None:
    st.warning("Silakan unggah data terlebih dahulu.")
    st.stop()
st.header("Deteksi Anomali Time Series" if st.session_state.language == 'id' else "Time Series Anomaly Detection")
from ml_engine.timeseries_utils import calculate_forecast_metrics, plot_forecast_visualization, display_forecast_summary

# Ensure np is defined
import numpy as np

st.info("""
🔍 **Fitur Deteksi Anomali Time Series**

Tab ini menyediakan algoritma deteksi anomali **state-of-the-art** untuk data time series dengan beberapa opsi:
- **Isolation Forest**: Deteksi berbasis isolasi dengan ensemble trees
- **One-Class SVM**: Deteksi berbasis margin hyperplane
- **Statistical**: Deteksi berbasis statistik rolling window (Z-Score)
- **Ensemble**: Kombinasi multiple methods

**Catatan**: Fitur ini khusus untuk data time series dengan kolom tanggal/waktu.
""" if st.session_state.language == 'id' else """
🔍 **Time Series Anomaly Detection Features**

This tab provides **state-of-the-art** anomaly detection algorithms for time series data with multiple options:
- **Isolation Forest**: Isolation-based detection with ensemble trees
- **One-Class SVM**: Margin-based hyperplane detection
- **Statistical**: Rolling window statistics-based detection (Z-Score)
- **Ensemble**: Combination of multiple methods

**Note**: This feature is specifically for time series data with date/time columns.
""")

if st.session_state.data is None:
    st.warning("Silakan unggah dataset di tab 'Data Upload' terlebih dahulu." if st.session_state.language == 'id' else "Please upload a dataset in the 'Data Upload' tab first.")
else:
    # Check for time series data
    date_columns = []
    for col in st.session_state.data.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day', 'tanggal', 'waktu', 'tahun', 'bulan', 'hari']):
            try:
                pd.to_datetime(st.session_state.data[col])
                date_columns.append(col)
            except:
                pass
    
    if not date_columns:
        st.warning("Tidak ditemukan kolom tanggal/waktu dalam dataset. Pastikan ada kolom dengan nama yang mengandung kata kunci tanggal/waktu." if st.session_state.language == 'id' else "No date/time column found in the dataset. Ensure there is a column with date/time keywords in the name.")
    else:
        # Select date column
        date_column = st.selectbox(
            "Pilih kolom tanggal/waktu:" if st.session_state.language == 'id' else "Select date/time column:",
            date_columns,
            key="ts_date_column"
        )
        
        # Select target column for anomaly detection
        numerical_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        target_column = st.selectbox(
            "Pilih kolom target untuk deteksi anomali:" if st.session_state.language == 'id' else "Select target column for anomaly detection:",
            [col for col in numerical_columns if col != date_column],
            key="ts_target_column"
        )
        
        # Data preparation
        st.subheader("📊 Persiapan Data" if st.session_state.language == 'id' else "Data Preparation")
        
        # Prepare time series data
        preview_data = st.session_state.data[[date_column, target_column]].copy()
        preview_data[date_column] = pd.to_datetime(preview_data[date_column])
        preview_data = preview_data.sort_values(date_column)
        
        # Handle missing values
        missing_count = preview_data[target_column].isnull().sum()
        if missing_count > 0:
            st.warning(f"Terdapat {missing_count} nilai missing. Nilai missing akan dihapus." if st.session_state.language == 'id' else f"There are {missing_count} missing values. Missing values will be removed.")
            preview_data = preview_data.dropna()
        
        ts_data = preview_data.set_index(date_column)[target_column]
        
        st.write(f"**Jumlah data:** {len(ts_data)}")
        st.write(f"**Rentang waktu:** {ts_data.index.min()} sampai {ts_data.index.max()}")
        
        # Basic visualization
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts_data.index, ts_data.values, color='blue', alpha=0.7)
        ax.set_title(f'Time Series: {target_column}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Select anomaly detection methods
        st.subheader("🎯 Pilih Metode Deteksi Anomali" if st.session_state.language == 'id' else "Select Anomaly Detection Methods")
        
        available_methods = {
            'isolation_forest': 'Isolation Forest',
            'one_class_svm': 'One-Class SVM',
            'statistical': 'Statistical (Z-Score)',
            'ensemble': 'Ensemble Method'
        }
        
        # Check for sklearn availability
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.svm import OneClassSVM
            from scipy import stats
            SKLEARN_AVAILABLE = True
        except ImportError:
            SKLEARN_AVAILABLE = False
            st.error("Scikit-learn atau scipy tidak tersedia." if st.session_state.language == 'id' else "Scikit-learn or scipy not available.")
        
        if SKLEARN_AVAILABLE:
            selected_methods = st.multiselect(
                "Pilih metode deteksi anomali:" if st.session_state.language == 'id' else "Select anomaly detection methods:",
                options=list(available_methods.keys()),
                format_func=lambda x: available_methods[x],
                default=['isolation_forest', 'statistical']
            )
            
            if selected_methods:
                # Parameters configuration
                st.subheader("⚙️ Konfigurasi Parameter" if st.session_state.language == 'id' else "Parameter Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    contamination = st.slider(
                        "Tingkat kontaminasi (proporsi anomali):" if st.session_state.language == 'id' else "Contamination level (anomaly proportion):",
                        0.01, 0.3, 0.05, 0.01
                    )
                
                with col2:
                    z_threshold = st.slider(
                        "Z-score threshold untuk Statistical method:" if st.session_state.language == 'id' else "Z-score threshold for Statistical method:",
                        1.0, 5.0, 3.0, 0.5
                    )
                
                # Run anomaly detection
                if st.button("🚀 Jalankan Deteksi Anomali" if st.session_state.language == 'id' else "Run Anomaly Detection", type="primary"):
                    with st.spinner("Menjalankan deteksi anomali..." if st.session_state.language == 'id' else "Running anomaly detection..."):
                        try:
                            log_feature('anomaly_detection_run')
                        except Exception:
                            pass
                        try:
                            # Validasi data
                            if len(ts_data) < 10:
                                st.error("Dataset terlalu pendek. Minimal 10 data points diperlukan." if st.session_state.language == 'id' else "Dataset too short. Minimum 10 data points required.")
                            elif ts_data.std() == 0:
                                st.error("Data memiliki nilai konstan. Deteksi anomali tidak dapat dilakukan." if st.session_state.language == 'id' else "Data has constant values. Anomaly detection cannot be performed.")
                            else:
                                # Gunakan fungsi dari anomaly_detection_utils
                                from anomaly_detection_utils import detect_and_visualize_anomalies
                                
                                results = {}
                                
                                # Siapkan DataFrame untuk fungsi deteksi
                                df_for_detection = pd.DataFrame({
                                    date_column: ts_data.index,
                                    target_column: ts_data.values
                                })
                                
                                # Jalankan deteksi untuk setiap metode
                                for method in selected_methods:
                                    try:
                                        detection_results = detect_and_visualize_anomalies(
                                            data=df_for_detection,
                                            target_column=target_column,
                                            date_column=date_column,
                                            methods=[method],
                                            contamination=contamination
                                        )
                                        
                                        if method in detection_results and 'result' in detection_results[method]:
                                            result_data = detection_results[method]['result']
                                            summary_data = detection_results[method]['summary']
                                            
                                            results[method] = {
                                                'anomalies': result_data['anomalies'],
                                                'anomaly_count': summary_data['anomaly_count'],
                                                'anomaly_percentage': summary_data['anomaly_percentage'],
                                                'anomaly_indices': ts_data.index[result_data['anomalies']],
                                                'anomaly_values': ts_data.values[result_data['anomalies']],
                                                'summary': summary_data
                                            }
                                        else:
                                            st.error(f"Error pada metode {method}: Hasil deteksi tidak valid")
                                            
                                    except Exception as e:
                                        st.error(f"Error pada metode {method}: {str(e)}")
                                
                                # Display results
                                st.subheader("📋 Hasil Deteksi Anomali" if st.session_state.language == 'id' else "Anomaly Detection Results")
                            
                            # Summary table
                            if results:
                                summary_data = []
                                for method, data in results.items():
                                    summary_data.append({
                                        'Method': available_methods[method],
                                        'Total Points': len(ts_data),
                                        'Anomalies Detected': data['anomaly_count'],
                                        'Anomaly Percentage (%)': round(data['anomaly_percentage'], 2)
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df)
                                
                                # Detailed results for each method
                                for method, data in results.items():
                                    st.subheader(f"🔍 {available_methods[method]} Results")
                                    
                                    # Create visualization
                                    fig, ax = plt.subplots(figsize=(15, 6))
                                    
                                    # Plot normal data
                                    ax.plot(ts_data.index, ts_data.values, color='blue', alpha=0.7, label='Normal')
                                    
                                    # Plot anomalies
                                    if data['anomaly_count'] > 0:
                                        ax.scatter(data['anomaly_indices'], data['anomaly_values'], 
                                                 color='red', s=50, alpha=0.8, label='Anomalies')
                                    
                                    ax.set_title(f'{available_methods[method]} - Anomaly Detection')
                                    ax.set_xlabel('Date')
                                    ax.set_ylabel('Value')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Show anomaly details
                                    if data['anomaly_count'] > 0:
                                        st.write(f"**{data['anomaly_count']} anomalies detected ({data['anomaly_percentage']:.2f}%)**")
                                        
                                        # Show first 10 anomalies
                                        anomaly_df = pd.DataFrame({
                                            'Date': data['anomaly_indices'][:10],
                                            'Value': data['anomaly_values'][:10]
                                        })
                                        st.dataframe(anomaly_df)
                                
                                # Combined visualization
                                if len(results) > 1:
                                    st.subheader("📊 Analisis Perbandingan" if st.session_state.language == 'id' else "Comparative Analysis")
                                    
                                    fig, ax = plt.subplots(figsize=(15, 8))
                                    ax.plot(ts_data.index, ts_data.values, color='blue', alpha=0.7, label='Normal Data')
                                    
                                    colors = ['red', 'green', 'orange', 'purple']
                                    for i, (method, data) in enumerate(results.items()):
                                        if data['anomaly_count'] > 0:
                                            ax.scatter(data['anomaly_indices'], data['anomaly_values'], 
                                                     color=colors[i % len(colors)], s=50, alpha=0.8, 
                                                     label=f'{available_methods[method]} Anomalies')
                                    
                                    ax.set_title('Anomaly Detection Comparison - All Methods' if st.session_state.language == 'id' else 'Perbandingan Deteksi Anomali - Semua Metode')
                                    ax.set_xlabel('Date')
                                    ax.set_ylabel('Value')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Download results
                                st.subheader("📥 Download Hasil" if st.session_state.language == 'id' else "Download Results")
                                
                                # Prepare download data
                                download_data = preview_data.copy()
                                download_data.set_index(date_column, inplace=True)
                                
                                for method, data in results.items():
                                    download_data[f'{available_methods[method]}_Anomaly'] = data['anomalies'].astype(int)
                                
                                csv = download_data.to_csv()
                                st.download_button(
                                    label="📥 Download Hasil Deteksi Anomali (CSV)" if st.session_state.language == 'id' else "Download Anomaly Detection Results (CSV)",
                                    data=csv,
                                    file_name=f'anomaly_detection_{target_column}.csv',
                                    mime='text/csv'
                                )
                                
                                st.success("Deteksi anomali selesai!" if st.session_state.language == 'id' else "Anomaly detection completed successfully!")
                                
                        except Exception as e:
                            st.error(f"Error saat menjalankan deteksi anomali: {str(e)}")
                            st.error(f"Detail error: {str(e)}")


# Footer Copyright - Muncul di setiap halaman
st.markdown("---")
st.markdown(
"""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #f8f9fa;
    color: #6c757d;
    text-align: center;
    padding: 10px 0;
    font-size: 12px;
    border-top: 1px solid #dee2e6;
    z-index: 1000;
}
</style>
<div class="footer">
    Copyright@2025 PT. ASMER SAHABAT SUKSES. All rights reserved.
</div>
""",
unsafe_allow_html=True
)

# Tambahkan margin bottom untuk konten agar tidak tertutup footer
st.markdown("<style>.main { padding-bottom: 60px; }</style>", unsafe_allow_html=True)


# --- Wizard Navigation ---
st.markdown("---")
st.markdown("### ✅ Alur Kerja Selesai!" if st.session_state.language == 'id' else "### ✅ Workflow Complete!")
st.success("🎉 Anda telah menyelesaikan seluruh tahapan analisis Machine Learning!" if st.session_state.language == 'id' else "🎉 You have completed all Machine Learning analysis stages!")
col_prev, col_restart = st.columns(2)
with col_prev:
    if st.button("⬅️ Kembali ke Interpretasi LIME" if st.session_state.language == 'id' else "⬅️ Back to LIME Interpretation", use_container_width=True):
        st.switch_page("pages/06_LIME_Model_Interpretation.py")
with col_restart:
    if st.button("🔄 Mulai Ulang dari Awal" if st.session_state.language == 'id' else "🔄 Start Over from Beginning", use_container_width=True):
        st.switch_page("pages/01_Data_Upload.py")
