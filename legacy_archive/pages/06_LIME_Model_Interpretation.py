import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if "model" not in st.session_state or st.session_state.model is None:
    st.warning("Silakan latih model terlebih dahulu.")
    st.stop()
st.info("⚠️ **Notifikasi:** Fitur interpretasi LIME sekarang mendukung model **regresi**, **klasifikasi**, dan **forecasting**." if st.session_state.language == 'id' else "⚠️ **Notification:** LIME interpretation now supports **regression**, **classification**, and **forecasting** models.")
if st.session_state.problem_type not in ['Regression', 'Classification', 'Forecasting']:
    st.info("Fitur interpretasi LIME hanya tersedia untuk model regresi, klasifikasi, dan forecasting." if st.session_state.language == 'id' else "LIME interpretation is only available for regression, classification, and forecasting models.")
else:

    if not LIME_AVAILABLE:
        st.error("LIME tidak terinstal. Silakan instal dengan 'pip install lime'." if st.session_state.language == 'id' else "LIME is not installed. Please install it with 'pip install lime'.")
    elif (
        st.session_state.model is not None
        and st.session_state.problem_type in ["Regression", "Classification", "Forecasting"]
        and not ('is_timeseries' in locals() and is_timeseries)
    ):
        st.write("""
        LIME (Local Interpretable Model-agnostic Explanations) adalah teknik untuk menjelaskan prediksi model machine learning.
        Tidak seperti SHAP yang memberikan nilai kontribusi global, LIME fokus pada penjelasan prediksi individual dengan membuat model lokal yang dapat diinterpretasi.
        """ if st.session_state.language == 'id' else """
        LIME (Local Interpretable Model-agnostic Explanations) is a technique for explaining machine learning model predictions.
        Unlike SHAP which provides global contribution values, LIME focuses on individual prediction explanations by creating a local interpretable model.
        """)

        # Pilih fitur untuk LIME
        st.subheader("Pemilihan Fitur untuk Analisis LIME" if st.session_state.language == 'id' else "Feature Selection for LIME Analysis")
        feature_names = st.session_state.X_train.columns.tolist()
        selected_features = st.multiselect(
            "Pilih fitur untuk analisis LIME:" if st.session_state.language == 'id' else "Select features for LIME analysis:",
            options=feature_names,
            default=feature_names[:min(10, len(feature_names))]
        )

        num_features_show = st.slider(
            "Jumlah fitur yang ditampilkan dalam penjelasan:" if st.session_state.language == 'id' else "Number of features to show in the explanation:",
            3, min(20, len(selected_features)), 5
        )

        if st.button("Generate LIME Explanations" if st.session_state.language == 'id' else "Generate LIME Explanations"):
            try:
                log_feature('generate_lime_explanations')
            except Exception:
                pass
            if not selected_features:
                st.error("Silakan pilih setidaknya satu fitur untuk analisis LIME." if st.session_state.language == 'id' else "Please select at least one feature for LIME analysis.")
            else:
                with st.spinner("Menghitung penjelasan LIME..." if st.session_state.language == 'id' else "Calculating LIME explanations..."):
                    X_train_selected = st.session_state.X_train[selected_features]
                    X_test_selected = st.session_state.X_test[selected_features]

                    st.subheader("Penjelasan Prediksi Individual" if st.session_state.language == 'id' else "Individual Prediction Explanation")
                    sample_idx = st.slider(
                        "Indeks sampel:", 0, len(X_test_selected) - 1, 0,
                        key="lime_sample_idx"
                    )
                    sample = X_test_selected.iloc[sample_idx]
                    st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                    st.dataframe(pd.DataFrame([sample], columns=selected_features))

                    actual = st.session_state.y_test_eval.iloc[sample_idx]
                    original_sample = st.session_state.X_test.iloc[sample_idx:sample_idx+1]
                    predicted = st.session_state.model.predict(original_sample)[0]
                    st.write(f"Nilai aktual: {actual}")
                    st.write(f"Nilai prediksi: {predicted}")

                    # Gunakan fungsi utilitas untuk klasifikasi dan forecasting
                    if st.session_state.problem_type == "Classification":
                        # Cek kompatibilitas model dengan LIME
                        compatibility_result = check_model_compatibility(st.session_state.model, 'lime')
                        
                        if not compatibility_result['compatible']:
                            st.error(f"❌ {compatibility_result['message']}")
                            st.warning(f"ℹ️ {compatibility_result['suggestion']}")
                            
                            # Tampilkan rekomendasi metode interpretasi alternatif
                            recommendations = get_model_interpretation_recommendations(st.session_state.model)
                            st.info(f"💡 {recommendations['recommendation']}")
                            
                            # Tampilkan alasan secara detail
                            with st.expander("Penjelasan Detail" if st.session_state.language == 'id' else "Detailed Explanation"):
                                st.write(compatibility_result['reason'])
                            
                            # Tombol aksi untuk pengguna
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Coba dengan SHAP" if st.session_state.language == 'id' else "Try with SHAP"):
                                    st.session_state.interpretation_method = "SHAP"
                                    st.rerun()
                            with col2:
                                if st.button("Lihat Feature Importance" if st.session_state.language == 'id' else "View Feature Importance"):
                                    # Redirect ke bagian feature importance
                                    st.session_state.page = "feature_importance"
                                    st.rerun()
                            with col3:
                                if st.button("Kembali ke Menu" if st.session_state.language == 'id' else "Back to Menu"):
                                    st.session_state.page = "main"
                                    st.rerun()
                        
                        # Jika model kompatibel, lanjutkan dengan LIME
                        lime_result = implement_lime_classification(
                            st.session_state.model,
                            X_train_selected,
                            st.session_state.y_train,
                            problem_type='classification',
                            class_names=st.session_state.model.classes_ if hasattr(st.session_state.model, 'classes_') else None,
                            feature_names=selected_features,
                            num_features=num_features_show
                        )
                        
                        if lime_result['success']:
                            # Gunakan preprocessing yang lebih baik
                            preprocessing_result = improved_data_preprocessing_for_interpretation(
                                X_test_selected, 
                                model=st.session_state.model, 
                                method='lime'
                            )
                            
                            if preprocessing_result['success']:
                                X_processed = preprocessing_result['X_processed']
                                
                                # Tampilkan informasi preprocessing
                                if preprocessing_result['preprocessing_steps']:
                                    st.subheader("Informasi Preprocessing" if st.session_state.language == 'id' else "Preprocessing Information")
                                    for step in preprocessing_result['preprocessing_steps']:
                                        st.write(f"• {step}")
                                
                                if preprocessing_result['warnings']:
                                    st.warning("Peringatan Preprocessing:" if st.session_state.language == 'id' else "Preprocessing Warnings:")
                                    for warning in preprocessing_result['warnings']:
                                        st.write(f"⚠️ {warning}")
                                
                                # Generate LIME explanations dengan data yang sudah diproses
                                with st.spinner("Menghasilkan penjelasan LIME..." if st.session_state.language == 'id' else "Generating LIME explanations..."):
                                    try:
                                        # Gunakan data yang sudah diproses
                                        lime_result_improved = implement_lime_classification(
                                            st.session_state.model,
                                            X_processed,
                                            st.session_state.y_train,
                                            problem_type='classification',
                                            class_names=st.session_state.model.classes_ if hasattr(st.session_state.model, 'classes_') else None,
                                            feature_names=X_processed.columns.tolist(),
                                            num_features=num_features_show
                                        )
                                        
                                        if lime_result_improved['success']:
                                            explanations = lime_result_improved['explanations']
                                            
                                            # Tampilkan summary
                                            st.subheader("Summary LIME" if st.session_state.language == 'id' else "LIME Summary")
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.metric(
                                                    "Sampel Berhasil" if st.session_state.language == 'id' else "Successful Samples",
                                                    lime_result_improved.get('n_successful', 0)
                                                )
                                                st.metric(
                                                    "Sampel Gagal" if st.session_state.language == 'id' else "Failed Samples", 
                                                    lime_result_improved.get('n_failed', 0)
                                                )
                                            
                                            with col2:
                                                if lime_result_improved.get('n_successful', 0) > 0:
                                                    st.success(f"✅ {lime_result_improved.get('n_successful', 0)} penjelasan berhasil dihasilkan" if st.session_state.language == 'id' else f"✅ {lime_result_improved.get('n_successful', 0)} explanations generated successfully")
                                            
                                            # Buat laporan interpretasi
                                            report = create_interpretation_report(lime_result_improved, 'lime', st.session_state.language)
                                            
                                            # Tampilkan laporan
                                            st.subheader(report['summary'].get('status', ''))
                                            st.write(report['summary'].get('method', ''))
                                            
                                            if 'details' in report and report['details']:
                                                for key, value in report['details'].items():
                                                    if isinstance(value, pd.DataFrame):
                                                        st.write(f"**{key}:**")
                                                        st.dataframe(value)
                                                    else:
                                                        st.write(f"**{key}:** {value}")
                                            
                                            if report['recommendations']:
                                                st.subheader("Rekomendasi" if st.session_state.language == 'id' else "Recommendations")
                                                for rec in report['recommendations']:
                                                    st.write(f"• {rec}")
                                            
                                            # Tampilkan penjelasan detail
                                            if explanations:
                                                st.subheader("Penjelasan Detail" if st.session_state.language == 'id' else "Detailed Explanations")
                                                
                                                # Pilih sampel untuk ditampilkan
                                                sample_options = [f"Sample {i}" for i in range(len(explanations)) if explanations[i].get('explanation') is not None]
                                                if sample_options:
                                                    selected_sample = st.selectbox(
                                                        "Pilih sampel:" if st.session_state.language == 'id' else "Select sample:",
                                                        options=range(len(sample_options)),
                                                        format_func=lambda i: sample_options[i]
                                                    )
                                                    
                                                    # Tampilkan penjelasan untuk sampel yang dipilih
                                                    if explanations[selected_sample].get('explanation') is not None:
                                                        explanation = explanations[selected_sample]['explanation']
                                                        
                                                        st.write(f"**Penjelasan untuk {sample_options[selected_sample]}:**")
                                                        
                                                        # Tampilkan dalam bentuk tabel
                                                        exp_list = explanation.as_list()
                                                        exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution', 'Value'])
                                                        exp_df = exp_df.sort_values('Contribution', ascending=False, key=abs)
                                                        st.dataframe(exp_df)
                                                        
                                                        # Visualisasi
                                                        try:
                                                            fig = explanation.as_pyplot_figure()
                                                            st.pyplot(fig)
                                                        except Exception as e:
                                                            st.warning(f"Gagal membuat visualisasi: {str(e)}" if st.session_state.language == 'id' else f"Failed to create visualization: {str(e)}")
                                                    else:
                                                        st.error("Tidak ada penjelasan yang valid untuk sampel ini" if st.session_state.language == 'id' else "No valid explanation for this sample")
                                        else:
                                            st.error(f"Gagal menghasilkan penjelasan LIME yang diperbaiki: {lime_result_improved.get('error', 'Unknown error')}")
                                    except Exception as e:
                                        st.error(f"Error dalam proses LIME: {str(e)}")
                            else:
                                st.error(f"Error preprocessing data untuk LIME: {preprocessing_result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"❌ Error dalam implementasi LIME: {lime_result.get('error', 'Unknown error')}")
                            st.warning("💡 Model ini mungkin tidak kompatibel dengan LIME. Coba SHAP sebagai alternatif." if st.session_state.language == 'id' else "This model may not be compatible with LIME. Try SHAP as an alternative.")

                            # Tombol aksi alternatif
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Coba dengan SHAP" if st.session_state.language == 'id' else "Try with SHAP"):
                                    st.session_state.interpretation_method = "SHAP"
                                    st.rerun()
                            with col2:
                                if st.button("Kembali ke Menu" if st.session_state.language == 'id' else "Back to Menu"):
                                    st.session_state.page = "main"
                                    st.rerun()
                    elif st.session_state.problem_type == "Forecasting":
                        # Cek kompatibilitas model dengan LIME untuk forecasting
                        compatibility_result = check_model_compatibility(st.session_state.model, 'lime')
                        
                        if not compatibility_result['compatible']:
                            st.error(f"❌ {compatibility_result['message']}")
                            st.warning(f"ℹ️ {compatibility_result['suggestion']}")
                            
                            # Tampilkan rekomendasi metode interpretasi alternatif
                            recommendations = get_model_interpretation_recommendations(st.session_state.model)
                            st.info(f"💡 {recommendations['recommendation']}")
                            
                            # Tampilkan alasan secara detail
                            with st.expander("Penjelasan Detail" if st.session_state.language == 'id' else "Detailed Explanation"):
                                st.write(compatibility_result['reason'])
                            
                            # Tombol aksi untuk pengguna
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Coba dengan SHAP" if st.session_state.language == 'id' else "Try with SHAP"):
                                    st.session_state.interpretation_method = "SHAP"
                                    st.rerun()
                            with col2:
                                if st.button("Lihat Feature Importance" if st.session_state.language == 'id' else "View Feature Importance"):
                                    # Redirect ke bagian feature importance
                                    st.session_state.page = "feature_importance"
                                    st.rerun()
                            with col3:
                                if st.button("Kembali ke Menu" if st.session_state.language == 'id' else "Back to Menu"):
                                    st.session_state.page = "main"
                                    st.rerun()
                        
                        # Forecasting - gunakan pendekatan khusus
                        try:
                            # Siapkan data untuk forecasting
                            forecasting_data = prepare_forecasting_data_for_interpretation(
                                st.session_state.X_train,
                                st.session_state.X_test,
                                selected_features,
                                sample_idx
                            )
                            
                            if forecasting_data is not None:
                                st.subheader("Visualisasi Penjelasan LIME untuk Forecasting" if st.session_state.language == 'id' else "LIME Explanation Visualization for Forecasting")
                                
                                # Gunakan LIME untuk forecasting dengan mode regresi
                                lime_mode = "regression"
                                predict_fn = st.session_state.model.predict

                                explainer = lime_tabular.LimeTabularExplainer(
                                    forecasting_data['X_train'].values,
                                    feature_names=selected_features,
                                    mode=lime_mode,
                                    random_state=42
                                )

                                explanation = explainer.explain_instance(
                                    forecasting_data['sample'].values,
                                    predict_fn,
                                    num_features=num_features_show
                                )

                                fig = plt.figure(figsize=(10, 6))
                                lime_fig = explanation.as_pyplot_figure()
                                plt.tight_layout()
                                st.pyplot(lime_fig)

                                st.subheader("Penjelasan dalam Bentuk Tabel" if st.session_state.language == 'id' else "Explanation in Table Format")
                                explanation_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Kontribusi"])
                                explanation_df = explanation_df.sort_values("Kontribusi", ascending=False)
                                st.dataframe(explanation_df)
                                
                                # Penjelasan khusus untuk forecasting
                                st.subheader("Interpretasi untuk Model Forecasting" if st.session_state.language == 'id' else "Interpretation for Forecasting Model")
                                st.info("""
                                Dalam model forecasting, fitur-fitur penting biasanya meliputi:
                                - **Lag Features**: Nilai historis dari variabel target
                                - **Fitur Tanggal/Waktu**: Seperti hari dalam minggu, bulan, kuartal, dll.
                                - **Fitur Rolling**: Seperti rata-rata bergerak, standar deviasi, dll.
                                
                                Nilai LIME tinggi pada lag features menunjukkan bahwa model sangat bergantung pada pola historis terbaru.
                                """ if st.session_state.language == 'id' else """
                                In forecasting models, important features typically include:
                                - **Lag Features**: Historical values of the target variable
                                - **Date/Time Features**: Such as day of week, month, quarter, etc.
                                - **Rolling Features**: Such as moving averages, standard deviations, etc.
                                
                                High LIME values on lag features indicate that the model heavily relies on recent historical patterns.
                                """)
                            else:
                                st.error("❌ Tidak dapat menyiapkan data untuk interpretasi LIME forecasting." if st.session_state.language == 'id' else "❌ Could not prepare data for LIME forecasting interpretation.")
                                st.warning(f"ℹ️ {'Pastikan data training dan testing tersedia untuk forecasting.' if st.session_state.language == 'id' else 'Ensure training and testing data are available for forecasting.'}")
                                
                                # Tombol aksi alternatif
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Coba dengan SHAP" if st.session_state.language == 'id' else "Try with SHAP"):
                                        st.session_state.interpretation_method = "SHAP"
                                        st.rerun()
                                with col2:
                                    if st.button("Kembali ke Menu" if st.session_state.language == 'id' else "Back to Menu"):
                                        st.session_state.page = "main"
                                        st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error dalam implementasi LIME forecasting: {str(e)}")
                            st.warning(f"ℹ️ {'LIME forecasting gagal. Ini bisa terjadi karena struktur data time series yang kompleks atau model yang tidak umum.' if st.session_state.language == 'id' else 'LIME forecasting failed. This may happen due to complex time series data structure or uncommon model.'}")
                            
                            # Berikan saran spesifik berdasarkan jenis error
                            if "explainer" in str(e).lower():
                                st.info(f"💡 {'Gagal membuat LIME explainer untuk data forecasting. Silakan coba SHAP yang lebih cocok untuk time series.' if st.session_state.language == 'id' else 'Failed to create LIME explainer for forecasting data. Please try SHAP which is more suitable for time series.'}")
                            elif "explain_instance" in str(e).lower():
                                st.info(f"💡 {'Gagal menghasilkan penjelasan. Data time series mungkin memiliki struktur yang tidak cocok untuk LIME. Silakan coba SHAP.' if st.session_state.language == 'id' else 'Failed to generate explanation. Time series data may have structure incompatible with LIME. Please try SHAP.'}")
                            
                            # Tombol aksi alternatif
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Coba dengan SHAP" if st.session_state.language == 'id' else "Try with SHAP"):
                                    st.session_state.interpretation_method = "SHAP"
                                    st.rerun()
                            with col2:
                                if st.button("Kembali ke Menu" if st.session_state.language == 'id' else "Back to Menu"):
                                    st.session_state.page = "main"
                                    st.rerun()
                    else:
                        # Regresi - gunakan logika lama
                        lime_mode = "regression"
                        predict_fn = st.session_state.model.predict

                        explainer = lime_tabular.LimeTabularExplainer(
                            X_train_selected.values,
                            feature_names=selected_features,
                            mode=lime_mode,
                            random_state=42
                        )

                        explanation = explainer.explain_instance(
                            sample.values,
                            predict_fn,
                            num_features=num_features_show
                        )

                        st.subheader("Visualisasi Penjelasan LIME" if st.session_state.language == 'id' else "LIME Explanation Visualization")
                        fig = plt.figure(figsize=(10, 6))
                        lime_fig = explanation.as_pyplot_figure()  # Untuk regresi, JANGAN beri argumen label
                        plt.tight_layout()
                        st.pyplot(lime_fig)

                        st.subheader("Penjelasan dalam Bentuk Tabel" if st.session_state.language == 'id' else "Explanation in Table Format")
                        explanation_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Kontribusi"])
                        explanation_df = explanation_df.sort_values("Kontribusi", ascending=False)
                        st.dataframe(explanation_df)

                    st.subheader("Nilai Fitur untuk Sampel yang Dijelaskan" if st.session_state.language == 'id' else "Feature Values for Explained Sample")
                    feature_values = pd.DataFrame({
                        "Feature": selected_features,
                        "Value": sample.values
                    })
                    st.dataframe(feature_values)

                    st.success("Analisis LIME selesai!" if st.session_state.language == 'id' else "LIME analysis completed successfully!")
                    
    else:
        st.info("Silakan latih model terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a model in the 'Model Training' tab first.")

# (wizard nav below)


# --- Wizard Navigation ---
st.markdown("---")
st.markdown("### ⏩ Langkah Selanjutnya" if st.session_state.language == 'id' else "### ⏩ Next Step")
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅️ Kembali ke Interpretasi SHAP" if st.session_state.language == 'id' else "⬅️ Back to SHAP Interpretation", use_container_width=True):
        st.switch_page("pages/05_SHAP_Model_Interpretation.py")
with col_next:
    if st.button("Lanjutkan ke Deteksi Anomali Time Series ➡️" if st.session_state.language == 'id' else "Continue to Time Series Anomaly Detection ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/07_Time_Series_Anomaly_Detection.py")
