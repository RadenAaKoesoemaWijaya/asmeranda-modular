import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

if "model" not in st.session_state or st.session_state.model is None:
    st.warning("Silakan latih model terlebih dahulu.")
    st.stop()
st.info("✅ **Notifikasi:** Fitur interpretasi SHAP sekarang mendukung algoritma model **regresi** dan **klasifikasi**. Untuk model **forecasting**, analisis masih dalam pengembangan." if st.session_state.language == 'id' else "✅ **Notification:** SHAP interpretation now supports **regression** and **classification** algorithms. Analysis for **forecasting** models is still under development.")

        
if st.session_state.problem_type == 'Forecasting':
    st.header("Interpretasi Model Forecasting dengan SHAP" if st.session_state.language == 'id' else "Forecasting Model Interpretation with SHAP")
    st.info("✅ **Notifikasi:** Fitur interpretasi SHAP sekarang mendukung model **forecasting** dengan pendekatan khusus. Gunakan tombol di bawah untuk mulai menganalisis model forecasting Anda." if st.session_state.language == 'id' else "✅ **Notification:** SHAP interpretation now supports **forecasting** models with a specialized approach. Use the button below to start analyzing your forecasting model.")
    
    if st.session_state.model is not None:
        st.write("""
        SHAP untuk model forecasting memerlukan pendekatan khusus karena struktur data deret waktu.
        Kami menggunakan fungsi interpretasi khusus untuk menangani karakteristik unik model forecasting.
        """ if st.session_state.language == 'id' else """
        SHAP for forecasting models requires a special approach due to the time series data structure.
        We use specialized interpretation functions to handle the unique characteristics of forecasting models.
        """)
        
        # Pilih fitur untuk analisis SHAP
        if hasattr(st.session_state, 'forecast_features') and st.session_state.forecast_features:
            feature_names = st.session_state.forecast_features
            selected_features = st.multiselect(
                "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
                options=feature_names,
                default=feature_names[:min(10, len(feature_names))]
            )
            
            # Jumlah sampel untuk analisis
            sample_size = st.slider(
                "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                min_value=10, max_value=min(100, len(st.session_state.X_test)), value=50
            )
            
            if st.button("Generate SHAP Values untuk Forecasting" if st.session_state.language == 'id' else "Generate SHAP Values for Forecasting"):
                try:
                    log_feature('shap_forecasting_generate')
                except Exception:
                    pass
                if not selected_features:
                    st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
                else:
                    with st.spinner("Menghitung nilai SHAP untuk model forecasting..." if st.session_state.language == 'id' else "Calculating SHAP values for forecasting model..."):
                        try:
                            # Gunakan fungsi interpretasi forecasting baru
                            interpretation_results = interpret_forecasting_model(
                                model=st.session_state.model,
                                X_train=st.session_state.X_train[selected_features],
                                y_train=st.session_state.y_train,
                                X_test=st.session_state.X_test[selected_features],
                                feature_names=selected_features,
                                method='shap',
                                n_samples=sample_size,
                                random_state=42
                            )
                            
                            # Tampilkan dashboard interpretasi
                            st.subheader("Dashboard Interpretasi Forecasting" if st.session_state.language == 'id' else "Forecasting Interpretation Dashboard")
                            
                            # Buat dan tampilkan dashboard
                            dashboard_fig = create_forecasting_interpretation_dashboard(interpretation_results, method='shap')
                            st.pyplot(dashboard_fig)
                            
                            # Tampilkan feature importance sebagai tabel
                            st.subheader("Feature Importance" if st.session_state.language == 'id' else "Feature Importance")
                            importance_df = pd.DataFrame(
                                list(interpretation_results['feature_importance'].items()),
                                columns=['Feature', 'Importance']
                            ).sort_values('Importance', ascending=False)
                            st.dataframe(importance_df)
                            
                            st.success("Analisis SHAP untuk model forecasting berhasil diselesaikan!" if st.session_state.language == 'id' else "SHAP analysis for forecasting model completed successfully!")
                            
                        except Exception as e:
                            st.error(f"Error dalam analisis SHAP forecasting: {str(e)}")
                            
        else:
            st.warning("Tidak dapat menemukan fitur untuk model forecasting. Pastikan model telah dilatih dengan benar." if st.session_state.language == 'id' else 
                    "Could not find features for the forecasting model. Make sure the model has been trained correctly.")
    else:
        st.info("Silakan latih model forecasting terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a forecasting model in the 'Model Training' tab first.")
          
elif st.session_state.problem_type == "Classification":
    st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

    if st.session_state.model is not None:
        st.write("""
        SHAP (SHapley Additive exPlanations) adalah pendekatan teori permainan untuk menjelaskan output dari model machine learning mana pun.
        """ if st.session_state.language == 'id' else """
        SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.
        """)
        
        # Pilih fitur untuk SHAP
        st.subheader("Pemilihan Fitur untuk Analisis SHAP" if st.session_state.language == 'id' else "Feature Selection for SHAP Analysis")
        feature_names = st.session_state.X_train.columns.tolist()
        selected_features = st.multiselect(
            "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
            options=feature_names,
            default=feature_names[:min(10, len(feature_names))]
        )
    
        # Jumlah sampel untuk analisis SHAP
        sample_size = st.slider(
            "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
            min_value=1, 
            max_value=max(2, len(st.session_state.X_test)), 
            value=max(1, min(50, len(st.session_state.X_test)))
        )
        
        # Priority 3: Optimization Options
        with st.expander("⚡ Opsi Optimasi (Priority 3)" if st.session_state.language == 'id' else "⚡ Optimization Options (Priority 3)"):
            use_optimization = st.checkbox(
                "Gunakan optimasi untuk dataset besar" if st.session_state.language == 'id' else "Use optimization for large datasets",
                value=len(st.session_state.X_test) > 1000
            )
            
            if use_optimization:
                max_samples_opt = st.slider(
                    "Maksimal sampel (optimasi):" if st.session_state.language == 'id' else "Max samples (optimization):",
                    min_value=1, 
                    max_value=max(2, len(st.session_state.X_test)), 
                    value=max(1, min(1000, len(st.session_state.X_test)))
                )
                
                background_samples_opt = st.slider(
                    "Background samples:" if st.session_state.language == 'id' else "Background samples:",
                    min_value=1, 
                    max_value=max(2, len(st.session_state.X_train)), 
                    value=max(1, min(100, len(st.session_state.X_train)))
                )
                
                use_cache = st.checkbox(
                    "Gunakan cache untuk hasil" if st.session_state.language == 'id' else "Use cache for results",
                    value=True
                )
                
                use_interactive = st.checkbox(
                    "Visualisasi interaktif" if st.session_state.language == 'id' else "Interactive visualization",
                    value=True
                )
            else:
                max_samples_opt = sample_size
                background_samples_opt = None
                use_cache = False
                use_interactive = False
        
        if st.button("Generate SHAP Values" if st.session_state.language == 'id' else "Generate SHAP Values"):
            try:
                log_feature('generate_shap_values')
            except Exception:
                pass
            if not selected_features:
                st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
            else:
                # Cek kompatibilitas model terlebih dahulu
                compatibility_check = check_model_compatibility(
                    st.session_state.model, 
                    method='shap', 
                    language=st.session_state.language
                )
                if not compatibility_check['compatible']:
                    st.error(f"❌ {compatibility_check['message']}")
                    st.warning(f"💡 {compatibility_check['suggestion']}")
                    
                    # Tampilkan rekomendasi alternatif
                    recommendations = get_model_interpretation_recommendations(
                        st.session_state.model, 
                        language=st.session_state.language
                    )
                    
                    if not compatibility_check['compatible']:
                        st.error(f"❌ {compatibility_check['message']}")
                        st.warning(f"💡 {compatibility_check['suggestion']}")
                        
                        # Tampilkan rekomendasi alternatif
                        recommendations = get_model_interpretation_recommendations(
                            st.session_state.model, 
                            language=st.session_state.language
                        )
                        
                        st.info(f"🎯 {recommendations['explanation']}")
                        
                        if recommendations['alternatives']:
                            st.write("**Alternatif lain:**" if st.session_state.language == 'id' else "**Other alternatives:**")
                            for alt in recommendations['alternatives']:
                                st.write(f"• {alt}")
                        
                        # Berikan opsi untuk menggunakan metode lain
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Coba dengan LIME" if st.session_state.language == 'id' else "Try with LIME"):
                                st.session_state.interpretation_method = 'LIME'
                                st.rerun()
                        with col2:
                            if st.button("Kembali ke menu utama" if st.session_state.language == 'id' else "Back to main menu"):
                                st.session_state.page = 'main'
                                st.rerun()
                        
                        st.stop()
                    
                    # Persiapkan data untuk SHAP
                    X_sample = st.session_state.X_test[selected_features].sample(min(sample_size, len(st.session_state.X_test)), random_state=42)
                    
                    # Identifikasi fitur kategorikal dalam sampel
                    categorical_cols = [col for col in selected_features if col in st.session_state.categorical_columns]
                    
                    # Terapkan One-Hot Encoding jika ada fitur kategorikal
                    if categorical_cols:
                        st.info("Fitur kategorikal terdeteksi. Menerapkan One-Hot Encoding untuk analisis SHAP." if st.session_state.language == 'id' else 
                            "Categorical features detected. Applying One-Hot Encoding for SHAP analysis.")
                        X_sample = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=False)
                    
                    # Pastikan semua nilai dalam X_sample adalah numerik
                    for col in X_sample.columns:
                        try:
                            # Konversi ke numpy array terlebih dahulu
                            X_sample[col] = np.array(X_sample[col]).astype(float)
                        except:
                            try:
                                # Jika gagal, gunakan factorize dan konversi ke float
                                X_sample[col] = pd.factorize(X_sample[col])[0].astype(float)
                            except Exception as e:
                                st.error(f"Error saat mengkonversi kolom {col} ke numerik: {str(e)}")
                    
                    try:
                        # Gunakan fungsi implementasi SHAP yang diperbaiki
                        shap_result = implement_shap_classification(
                            model=st.session_state.model, 
                            X_sample=X_sample, 
                            X_train=st.session_state.X_train[selected_features],
                            language=st.session_state.language
                        )
                    except Exception as e:
                        st.error(f"Error dalam implementasi SHAP klasifikasi: {str(e)}")
                        shap_result = {'success': False, 'error': str(e), 'shap_values': None, 'explainer': None}
                    
                    if shap_result['success'] and shap_result.get('shap_values') is not None:
                        # Gunakan fungsi visualisasi yang diperbaiki
                        viz_result = create_shap_visualization(
                            shap_values=shap_result['shap_values'],
                            X_sample=X_sample,
                            feature_names=shap_result['feature_names'],
                            class_names=shap_result['class_names'],
                            problem_type=shap_result['problem_type'],
                            max_display=15
                        )
                        
                        if viz_result['success']:
                            # Tampilkan feature importance
                            st.subheader("Feature Importance (SHAP)" if st.session_state.language == 'id' else "Feature Importance (SHAP)")

                            importance_df = pd.DataFrame(
                                viz_result['feature_importance'],
                                columns=['Feature', 'Mean |SHAP Value|']
                            )
                            st.dataframe(importance_df)
                            
                            # Tampilkan visualisasi
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Summary Plot (Bar)" if st.session_state.language == 'id' else "Summary Plot (Bar)")
                                st.pyplot(viz_result['figures']['summary_bar'])
                            
                            with col2:
                                st.subheader("Summary Plot (Detailed)" if st.session_state.language == 'id' else "Summary Plot (Detailed)")
                                st.pyplot(viz_result['figures']['summary_detailed'])
                            
                            # Multi-class handling
                            if shap_result['problem_type'] == 'multiclass' and isinstance(shap_result['shap_values'], list):
                                st.subheader("Multi-class Analysis" if st.session_state.language == 'id' else "Multi-class Analysis")
                                
                                # Pilihan metode analisis
                                analysis_method = st.selectbox(
                                    "Pilih metode analisis:" if st.session_state.language == 'id' else "Select analysis method:",
                                    options=['individual', 'average', 'max_importance'],
                                    format_func=lambda x: {
                                        'individual': 'Kelas Individual' if st.session_state.language == 'id' else 'Individual Class',
                                        'average': 'Rata-rata Semua Kelas' if st.session_state.language == 'id' else 'Average All Classes',
                                        'max_importance': 'Kelas Penting Terbesar' if st.session_state.language == 'id' else 'Highest Importance Class'
                                    }[x]
                                )
                                
                                if analysis_method == 'individual':
                                    if shap_result['class_names']:
                                        selected_class = st.selectbox(
                                            "Pilih kelas:" if st.session_state.language == 'id' else "Select class:",
                                            options=range(len(shap_result['class_names'])),
                                            format_func=lambda i: shap_result['class_names'][i]
                                        )
                                    else:
                                        selected_class = st.selectbox(
                                            "Pilih indeks kelas:" if st.session_state.language == 'id' else "Select class index:",
                                            options=range(len(shap_result['shap_values']))
                                        )
                                    
                                    # Buat visualisasi untuk kelas yang dipilih
                                    class_viz = create_shap_visualization(
                                        shap_values=shap_result['shap_values'],
                                        X_sample=X_sample,
                                        feature_names=shap_result['feature_names'],
                                        class_names=shap_result['class_names'],
                                        problem_type=shap_result['problem_type'],
                                        selected_class=selected_class,
                                        max_display=15
                                    )
                                    
                                    if class_viz['success']:
                                        st.success(f"Menampilkan analisis untuk: {class_viz['class_name']}" if st.session_state.language == 'id' else 
                                                f"Showing analysis for: {class_viz['class_name']}")
                                        st.pyplot(class_viz['figures']['summary_bar'])
                                
                                elif analysis_method == 'max_importance':
                                    # Gunakan handle_multiclass_shap untuk mencari kelas terpenting
                                    max_importance_result = handle_multiclass_shap(
                                        shap_result['shap_values'],
                                        method='max_importance',
                                        class_names=shap_result['class_names']
                                    )
                                    
                                    if 'class_name' in max_importance_result:
                                        st.success(f"Kelas dengan importance tertinggi: {max_importance_result['class_name']}" if st.session_state.language == 'id' else 
                                                f"Highest importance class: {max_importance_result['class_name']}")
                                        
                                        # Visualisasi untuk kelas terpenting
                                        max_viz = create_shap_visualization(
                                            shap_values=shap_result['shap_values'],
                                            X_sample=X_sample,
                                            feature_names=shap_result['feature_names'],
                                            class_names=shap_result['class_names'],
                                            problem_type=shap_result['problem_type'],
                                            selected_class=max_importance_result['class_focused'],
                                            max_display=15
                                        )
                                        
                                        if max_viz['success']:
                                            st.pyplot(max_viz['figures']['summary_bar'])
                                
                                elif analysis_method == 'average':
                                    # Gunakan handle_multiclass_shap untuk rata-rata
                                    avg_result = handle_multiclass_shap(
                                        shap_result['shap_values'],
                                        method='average',
                                        class_names=shap_result['class_names']
                                    )
                                    
                                    if 'shap_values_average' in avg_result:
                                        st.success("Menampilkan rata-rata importance untuk semua kelas" if st.session_state.language == 'id' else 
                                                "Showing average importance for all classes")
                                        
                                        # Buat visualisasi untuk rata-rata
                                        avg_viz = create_shap_visualization(
                                            shap_values=avg_result['shap_values_average'],
                                            X_sample=X_sample,
                                            feature_names=shap_result['feature_names'],
                                            class_names=shap_result['class_names'],
                                            problem_type='binary',  # Treat as binary for visualization
                                            max_display=15
                                        )
                                        
                                        if avg_viz['success']:
                                            st.pyplot(avg_viz['figures']['summary_bar'])
                            
                            st.success("Analisis SHAP berhasil diselesaikan!" if st.session_state.language == 'id' else "SHAP analysis completed successfully!")
                        else:
                            st.error(f"Error dalam visualisasi SHAP: {viz_result.get('error', 'Unknown error')}")
                    
                    # Tips untuk interpretasi
                    st.subheader("Tips untuk Interpretasi" if st.session_state.language == 'id' else "Tips for Interpretation")
                    st.info("""
                    - **Summary Plot**: Menunjukkan fitur mana yang paling penting dan bagaimana mereka mempengaruhi prediksi. Warna merah menunjukkan nilai fitur tinggi, biru menunjukkan nilai rendah.
                    - **Feature Importance**: Menampilkan fitur berdasarkan kepentingannya (rata-rata nilai absolut SHAP).
                    - **Dependence Plot**: Menunjukkan bagaimana nilai SHAP berubah berdasarkan nilai fitur, membantu mengidentifikasi interaksi.
                    - **Force Plot**: Menunjukkan kontribusi setiap fitur untuk prediksi sampel individual.
                    - **Waterfall Plot**: Menunjukkan bagaimana setiap fitur berkontribusi pada prediksi akhir dari nilai dasar.
                    
                    Jika menggunakan One-Hot Encoding, fitur kategorikal akan dipecah menjadi beberapa kolom biner.
                    """ if st.session_state.language == 'id' else """
                    - **Summary Plot**: Shows which features are most important and how they affect predictions. Red indicates high feature values, blue indicates low values.
                    - **Feature Importance**: Displays features by importance (average absolute SHAP values).
                    - **Dependence Plot**: Shows how SHAP values change based on feature values, helping identify interactions.
                    - **Force Plot**: Shows the contribution of each feature for an individual sample prediction.
                    - **Waterfall Plot**: Shows how each feature contributes to the final prediction from the base value.
                    
                    If using One-Hot Encoding, categorical features will be split into multiple binary columns.
                    """)
                    
                    # Handle case where shap_result indicates failure but wasn't caught by except block
                    if not shap_result.get('success', False):
                        error_msg = shap_result.get('error', 'Unknown error')
                        st.error(f"❌ Error dalam implementasi SHAP klasifikasi: {error_msg}")
                        
                        # Berikan penjelasan yang lebih detail
                        if st.session_state.language == 'id':
                            st.warning("""
                            **Penjelasan Error:**
                            
                            Model Anda tidak dapat diinterpretasi menggunakan metode SHAP karena:
                            
                            1. **Tipe Model Tidak Didukung**: Model ini mungkin tergolong model yang belum didukung oleh library SHAP
                            2. **Struktur Model Kompleks**: Model dengan struktur yang terlalu kompleks mungkin memerlukan pendekatan khusus
                            3. **Library yang Diperlukan**: Mungkin diperlukan library tambahan untuk interpretasi model ini
                            
                            **Solusi yang Tersedia:**
                            """)
                        else:
                            st.warning("""
                            **Error Explanation:**
                            
                            Your model cannot be interpreted using the SHAP method because:
                            
                            1. **Unsupported Model Type**: This model may be categorized as unsupported by the SHAP library
                            2. **Complex Model Structure**: Models with overly complex structures may require special approaches
                            3. **Required Libraries**: Additional libraries may be needed to interpret this model
                            
                            **Available Solutions:**
                            """)
                        
                        # Tampilkan rekomendasi alternatif
                        recommendations = get_model_interpretation_recommendations(
                            st.session_state.model, 
                            language=st.session_state.language
                        )
                        
                        st.info(f"🎯 {recommendations['explanation']}")
                        
                        # Opsi untuk pengguna
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("🔄 Coba dengan LIME" if st.session_state.language == 'id' else "🔄 Try with LIME"):
                                st.session_state.interpretation_method = 'LIME'
                                st.rerun()
                        with col2:
                            if st.button("📊 Lihat Feature Importance" if st.session_state.language == 'id' else "📊 View Feature Importance"):
                                st.session_state.page = 'model_evaluation'
                                st.rerun()
                        with col3:
                            if st.button("🏠 Kembali ke Menu" if st.session_state.language == 'id' else "🏠 Back to Menu"):
                                st.session_state.page = 'main'
                                st.rerun()
    else:
        st.warning("Model belum tersedia. Silakan latih model terlebih dahulu." if st.session_state.language == 'id' else "Model not available. Please train a model first.")
                    
elif st.session_state.problem_type == "Regression":
    st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

    if st.session_state.model is not None:
        st.subheader("Pemilihan Fitur untuk Analisis SHAP" if st.session_state.language == 'id' else "Feature Selection for SHAP Analysis")
        
        # Pilih fitur untuk analisis SHAP
        feature_names = st.session_state.X_train.columns.tolist()
        selected_features = st.multiselect(
            "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
            options=feature_names,
            default=feature_names[:min(10, len(feature_names))]
        )
        
        # Jumlah sampel untuk analisis SHAP
        sample_size = st.slider(
            "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
            min_value=10, max_value=min(100, len(st.session_state.X_train)), value=50
        )
        
        if st.button("Generate SHAP Values" if st.session_state.language == 'id' else "Generate SHAP Values"):
            if not selected_features:
                st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
            else:
                with st.spinner("Menghitung nilai SHAP..." if st.session_state.language == 'id' else "Calculating SHAP values..."):
                    try:
                        # Persiapkan data untuk SHAP
                        X_sample = st.session_state.X_train[selected_features].sample(min(sample_size, len(st.session_state.X_train)), random_state=42)
                        
                        # Handle categorical features with one-hot encoding
                        categorical_features = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
                        if categorical_features:
                            X_sample_encoded = pd.get_dummies(X_sample, columns=categorical_features)
                            st.info(f"Menggunakan one-hot encoding untuk fitur kategorikal: {categorical_features}" if st.session_state.language == 'id' else f"Using one-hot encoding for categorical features: {categorical_features}")
                        else:
                            X_sample_encoded = X_sample.copy()
                        
                        # Pastikan semua nilai dalam X_sample_encoded adalah numerik
                        for col in X_sample_encoded.columns:
                            try:
                                X_sample_encoded[col] = np.array(X_sample_encoded[col]).astype(float)
                            except:
                                try:
                                    X_sample_encoded[col] = pd.factorize(X_sample_encoded[col])[0].astype(float)
                                except Exception as e:
                                    st.error(f"Error saat mengkonversi kolom {col} ke numerik: {str(e)}")
                        
                        # Pilih explainer berdasarkan jenis model
                        if hasattr(st.session_state.model, 'feature_importances_'):
                            # Gunakan TreeExplainer untuk model berbasis pohon
                            explainer = shap.TreeExplainer(st.session_state.model)
                        else:
                            # Gunakan KernelExplainer untuk model lainnya
                            explainer = shap.KernelExplainer(st.session_state.model.predict, X_sample_encoded)
                        
                        # Hitung nilai SHAP
                        shap_values = explainer.shap_values(X_sample_encoded)
                        
                        st.success("Nilai SHAP berhasil dihitung!" if st.session_state.language == 'id' else "SHAP values calculated successfully!")
                        
                        # Validasi bahwa shap_values tersedia
                        if shap_values is None:
                            st.error("SHAP values tidak tersedia untuk visualisasi" if st.session_state.language == 'id' else "SHAP values not available for visualization")
                            st.stop()
                        
                        # Visualisasi SHAP
                        st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                        
                        # 1. Summary Plot
                        st.write("### Summary Plot")
                        try:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.summary_plot(shap_values, X_sample_encoded, show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                        except Exception as e:
                            st.warning(f"Gagal membuat Summary Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Summary Plot: {str(e)}")
                        
                        # 2. Feature Importance Plot
                        st.write("### Feature Importance Plot")
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values, X_sample_encoded, plot_type="bar", show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                        except Exception as e:
                            st.warning(f"Gagal membuat Feature Importance Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Feature Importance Plot: {str(e)}")
                        
                        # 3. Dependence Plots untuk fitur teratas
                        st.write("### Dependence Plots")
                        
                        # Hitung rata-rata nilai absolut SHAP untuk setiap fitur
                        shap_arr = np.array(shap_values, dtype=float)
                        feature_importance = np.abs(shap_arr).mean(0)
                        
                        # Dapatkan indeks fitur terurut berdasarkan kepentingan
                        top_indices = feature_importance.argsort()[-5:][::-1]
                        
                        # Buat dependence plot untuk 5 fitur teratas
                        for idx in top_indices:
                            if idx < len(X_sample_encoded.columns):  # Pastikan indeks valid
                                try:
                                    feature_name = X_sample_encoded.columns[idx]
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    shap.dependence_plot(idx, shap_values, X_sample_encoded, show=False, ax=ax)
                                    plt.title(f"Dependence Plot for {feature_name}")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.clf()
                                except Exception as e:
                                    st.warning(f"Gagal membuat Dependence Plot untuk {feature_name}: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Dependence Plot for {feature_name}: {str(e)}")
                        
                        # 4. Force Plot untuk sampel individual
                        st.write("### Force Plot untuk Sampel Individual")
                        sample_idx = st.slider(
                            "Pilih indeks sampel:" if st.session_state.language == 'id' else "Select sample index:",
                            0, len(X_sample_encoded) - 1, 0
                        )
                        
                        # Tampilkan data sampel
                        st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                        st.dataframe(X_sample_encoded.iloc[[sample_idx]])
                        
                        # Force plot
                        try:
                            expected_val = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                            force_plot = shap.force_plot(expected_val, 
                                                    shap_values[sample_idx, :], 
                                                    X_sample_encoded.iloc[sample_idx, :], 
                                                    matplotlib=True,
                                                    show=False)
                            st.pyplot(force_plot)
                        except Exception as e:
                            st.warning(f"Gagal membuat Force Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Force Plot: {str(e)}")
                        
                        # 5. Waterfall Plot
                        st.write("### Waterfall Plot")
                        try:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            expected_val = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]
                            shap.plots._waterfall.waterfall_legacy(
                                expected_val,
                                shap_values[sample_idx, :],
                                feature_names=X_sample_encoded.columns,
                                show=False,
                                max_display=10
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                        except Exception as e:
                            st.warning(f"Gagal membuat Waterfall Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Waterfall Plot: {str(e)}")
                        
                        # Tips untuk interpretasi
                        st.subheader("Tips untuk Interpretasi" if st.session_state.language == 'id' else "Tips for Interpretation")
                        st.info("""
                        - **Summary Plot**: Menunjukkan fitur mana yang paling penting dan bagaimana mereka mempengaruhi prediksi. Warna merah menunjukkan nilai fitur tinggi, biru menunjukkan nilai rendah.
                        - **Feature Importance**: Menampilkan fitur berdasarkan kepentingannya (rata-rata nilai absolut SHAP).
                        - **Dependence Plot**: Menunjukkan bagaimana nilai SHAP berubah berdasarkan nilai fitur, membantu mengidentifikasi interaksi.
                        - **Force Plot**: Menunjukkan kontribusi setiap fitur untuk prediksi sampel individual.
                        - **Waterfall Plot**: Menunjukkan bagaimana setiap fitur berkontribusi pada prediksi akhir dari nilai dasar.
                        
                        Jika menggunakan One-Hot Encoding, fitur kategorikal akan dipecah menjadi beberapa kolom biner.
                        """ if st.session_state.language == 'id' else """
                        - **Summary Plot**: Shows which features are most important and how they affect predictions. Red indicates high feature values, blue indicates low values.
                        - **Feature Importance**: Displays features by importance (average absolute SHAP values).
                        - **Dependence Plot**: Shows how SHAP values change based on feature values, helping identify interactions.
                        - **Force Plot**: Shows the contribution of each feature for an individual sample prediction.
                        - **Waterfall Plot**: Shows how each feature contributes to the final prediction from the base value.
                        
                        If using One-Hot Encoding, categorical features will be split into multiple binary columns.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error saat menghitung nilai SHAP: {str(e)}")
    else:
        st.warning("Model belum tersedia. Silakan latih model terlebih dahulu." if st.session_state.language == 'id' else "Model not available. Please train a model first.")

elif st.session_state.problem_type == "Forecasting":
    st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

    if st.session_state.model is not None:
        # Cek kompatibilitas model dengan SHAP
        compatibility_result = check_model_compatibility(st.session_state.model, 'shap')
        
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
                if st.button("Coba dengan LIME" if st.session_state.language == 'id' else "Try with LIME"):
                    st.session_state.interpretation_method = "LIME"
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
        
        # Jika model kompatibel, lanjutkan dengan SHAP
        try:
            # Pilih fitur untuk SHAP
            st.subheader("Konfigurasi SHAP" if st.session_state.language == 'id' else "SHAP Configuration")
            
            # Pilih jumlah sampel untuk SHAP
            n_samples = st.slider(
                "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                min_value=10, max_value=min(1000, len(X)), value=min(100, len(X))
            )
            
            # Ambil sampel secara acak
            if n_samples < len(X):
                sample_indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices] if y is not None else None
            else:
                X_sample = X
                y_sample = y
            
            # Konversi categorical features ke numerik jika diperlukan
            X_sample_processed = X_sample.copy()
            
            # Handle categorical features dengan one-hot encoding
            categorical_features = X_sample_processed.select_dtypes(include=['object', 'category']).columns
            if len(categorical_features) > 0:
                st.info(f"Fitur kategorikal ditemukan: {list(categorical_features)}. Menggunakan one-hot encoding." if st.session_state.language == 'id' else f"Categorical features found: {list(categorical_features)}. Using one-hot encoding.")
                X_sample_processed = pd.get_dummies(X_sample_processed, columns=categorical_features, drop_first=True)
            
            # Konversi semua fitur ke tipe numerik
            for col in X_sample_processed.columns:
                if X_sample_processed[col].dtype == 'object':
                    try:
                        X_sample_processed[col] = pd.to_numeric(X_sample_processed[col], errors='coerce')
                    except:
                        # Jika tidak bisa dikonversi, gunakan label encoding
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        X_sample_processed[col] = le.fit_transform(X_sample_processed[col].astype(str))
            
            # Hapus kolom dengan nilai NaN
            X_sample_processed = X_sample_processed.dropna(axis=1, how='all')  # Hapus kolom yang semua nilainya NaN
            X_sample_processed = X_sample_processed.fillna(X_sample_processed.mean())  # Isi nilai yang tersisa dengan rata-rata
            
            st.write(f"Menggunakan {X_sample_processed.shape[0]} sampel dengan {X_sample_processed.shape[1]} fitur untuk analisis SHAP." if st.session_state.language == 'id' else f"Using {X_sample_processed.shape[0]} samples with {X_sample_processed.shape[1]} features for SHAP analysis.")
            
            # Hitung nilai SHAP
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Menghitung nilai SHAP..." if st.session_state.language == 'id' else "Calculating SHAP values...")
            progress_bar.progress(25)
            
            # Gunakan TreeExplainer untuk model berbasis tree, KernelExplainer untuk model lainnya
            try:
                if hasattr(st.session_state.model, 'estimators_'):  # Tree-based models (Random Forest, Gradient Boosting, etc.)
                    explainer = shap.TreeExplainer(st.session_state.model)
                else:  # Other models (Linear Regression, Neural Networks, etc.)
                    explainer = shap.KernelExplainer(st.session_state.model.predict, X_sample_processed)
                
                shap_values_selected = explainer.shap_values(X_sample_processed)
                progress_bar.progress(75)
                
            except Exception as e:
                st.error(f"❌ Error saat membuat explainer: {str(e)}")
                st.warning(f"ℹ️ {'Model ini mungkin tidak kompatibel dengan SHAP. Silakan coba metode interpretasi lainnya.' if st.session_state.language == 'id' else 'This model may not be compatible with SHAP. Please try other interpretation methods.'}")
                
                # Berikan saran spesifik berdasarkan jenis error
                if "TreeExplainer" in str(e):
                    st.info(f"💡 {'Model ini bukan model berbasis pohon. Silakan gunakan KernelExplainer atau coba dengan LIME.' if st.session_state.language == 'id' else 'This is not a tree-based model. Please use KernelExplainer or try with LIME.'}")
                elif "KernelExplainer" in str(e):
                    st.info(f"💡 {'KernelExplainer gagal. Model ini mungkin terlalu kompleks. Silakan coba dengan LIME.' if st.session_state.language == 'id' else 'KernelExplainer failed. This model may be too complex. Please try with LIME.'}")
                
                # Tombol aksi alternatif
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Coba dengan LIME" if st.session_state.language == 'id' else "Try with LIME"):
                        st.session_state.interpretation_method = "LIME"
                        st.rerun()
                with col2:
                    if st.button("Kembali ke Menu" if st.session_state.language == 'id' else "Back to Menu"):
                        st.session_state.page = "main"
                        st.rerun()
                
                shap_values_selected = None
            
            progress_bar.progress(100)
            status_text.text("Selesai!" if st.session_state.language == 'id' else "Done!")
            
            if shap_values_selected is not None:
                # Visualisasi SHAP
                st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                
                # 1. Summary Plot
                st.write("### Summary Plot")
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values_selected, X_sample_processed, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                except Exception as e:
                    st.warning(f"⚠️ Gagal membuat Summary Plot: {str(e)}" if st.session_state.language == 'id' else f"⚠️ Failed to create Summary Plot: {str(e)}")
                    st.info(f"💡 {'Plot ini mungkin gagal karena format data SHAP yang tidak sesuai. Silakan coba metode interpretasi lainnya.' if st.session_state.language == 'id' else 'This plot may fail due to incompatible SHAP data format. Please try other interpretation methods.'}")
                
                # 2. Feature Importance Plot
                st.write("### Feature Importance Plot")
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values_selected, X_sample_processed, plot_type="bar", show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                except Exception as e:
                    st.warning(f"⚠️ Gagal membuat Feature Importance Plot: {str(e)}" if st.session_state.language == 'id' else f"⚠️ Failed to create Feature Importance Plot: {str(e)}")
                    st.info(f"💡 {'Plot ini memerlukan format data SHAP tertentu. Jika gagal, fitur importance dapat dilihat dari model langsung.' if st.session_state.language == 'id' else 'This plot requires specific SHAP data format. If it fails, feature importance can be viewed from the model directly.'}")
                
                # 3. Dependence Plots untuk fitur teratas
                st.write("### Dependence Plots")
                
                # Hitung rata-rata nilai absolut SHAP untuk setiap fitur
                if isinstance(shap_values_selected, list):
                    # Untuk multi-output, ambil output pertama
                    shap_arr = np.array(shap_values_selected[0], dtype=float)
                    feature_importance = np.abs(shap_arr).mean(0)
                else:
                    shap_arr = np.array(shap_values_selected, dtype=float)
                    feature_importance = np.abs(shap_arr).mean(0)
                
                # Dapatkan indeks fitur terurut berdasarkan kepentingan
                top_indices = feature_importance.argsort()[-5:][::-1]
                
                # Buat dependence plot untuk 5 fitur teratas
                for idx in top_indices:
                    if idx < len(X_sample_processed.columns):  # Pastikan indeks valid
                        try:
                            feature_name = X_sample_processed.columns[idx]
                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.dependence_plot(idx, shap_values_selected, X_sample_processed, show=False, ax=ax)
                            plt.title(f"Dependence Plot for {feature_name}")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                        except Exception as e:
                            st.warning(f"Gagal membuat Dependence Plot untuk {feature_name}: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Dependence Plot for {feature_name}: {str(e)}")
                
                # 4. Force Plot untuk sampel individual
                st.write("### Force Plot untuk Sampel Individual")
                sample_idx = st.slider(
                    "Pilih indeks sampel:" if st.session_state.language == 'id' else "Select sample index:",
                    0, len(X_sample_processed) - 1, 0
                )
                
                # Tampilkan data sampel
                st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                st.dataframe(X_sample_processed.iloc[[sample_idx]])
                
                # Force plot
                try:
                    if isinstance(shap_values_selected, list):
                        # Untuk multi-output, ambil output pertama
                        expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        force_plot = shap.force_plot(expected_val, 
                                                shap_values_selected[0][sample_idx, :], 
                                                X_sample_processed.iloc[sample_idx, :], 
                                                matplotlib=True,
                                                show=False)
                    else:
                        expected_val = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                        force_plot = shap.force_plot(expected_val, 
                                                shap_values_selected[sample_idx, :], 
                                                X_sample_processed.iloc[sample_idx, :], 
                                                matplotlib=True,
                                                show=False)
                    st.pyplot(force_plot)
                except Exception as e:
                    st.warning(f"Gagal membuat Force Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Force Plot: {str(e)}")
                
                # 5. Waterfall Plot
                st.write("### Waterfall Plot")
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))

                    if isinstance(shap_values_selected, list):
                        # Untuk multi-output, ambil output dan expected_value untuk kelas pertama
                        expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                        shap.plots._waterfall.waterfall_legacy(
                            expected_val,
                            shap_values_selected[0][sample_idx, :],
                            feature_names=X_sample_processed.columns,
                            show=False,
                            max_display=10
                        )
                    else:
                        expected_val = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]
                        shap.plots._waterfall.waterfall_legacy(
                            expected_val,
                            shap_values_selected[sample_idx, :],
                            feature_names=X_sample_processed.columns,
                            show=False,
                            max_display=10
                        )

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                except Exception as e:
                    st.warning(f"Gagal membuat Waterfall Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Waterfall Plot: {str(e)}")
                
                # Tips untuk interpretasi
                st.subheader("Tips untuk Interpretasi" if st.session_state.language == 'id' else "Tips for Interpretation")
                st.info("""
                - **Summary Plot**: Menunjukkan fitur mana yang paling penting dan bagaimana mereka mempengaruhi prediksi. Warna merah menunjukkan nilai fitur tinggi, biru menunjukkan nilai rendah.
                - **Feature Importance**: Menampilkan fitur berdasarkan kepentingannya (rata-rata nilai absolut SHAP).
                - **Dependence Plot**: Menunjukkan bagaimana nilai SHAP berubah berdasarkan nilai fitur, membantu mengidentifikasi interaksi.
                - **Force Plot**: Menunjukkan kontribusi setiap fitur untuk prediksi sampel individual.
                - **Waterfall Plot**: Menunjukkan bagaimana setiap fitur berkontribusi pada prediksi akhir dari nilai dasar.
                
                Jika menggunakan One-Hot Encoding, fitur kategorikal akan dipecah menjadi beberapa kolom biner.
                """ if st.session_state.language == 'id' else """
                - **Summary Plot**: Shows which features are most important and how they affect predictions. Red indicates high feature values, blue indicates low values.
                - **Feature Importance**: Displays features by importance (average absolute SHAP values).
                - **Dependence Plot**: Shows how SHAP values change based on feature values, helping identify interactions.
                - **Force Plot**: Shows the contribution of each feature for an individual sample prediction.
                - **Waterfall Plot**: Shows how each feature contributes to the final prediction from the base value.
                
                If using One-Hot Encoding, categorical features will be split into multiple binary columns.
                """)
                
            else:
                st.error("SHAP values tidak tersedia untuk visualisasi" if st.session_state.language == 'id' else "SHAP values not available for visualization")
                
        except Exception as e:
            st.error(f"Error saat menghitung nilai SHAP: {str(e)}")
            
    else:
        st.warning("Model belum tersedia. Silakan latih model terlebih dahulu." if st.session_state.language == 'id' else "Model not available. Please train a model first.")
        
elif st.session_state.problem_type == "Forecasting":
    st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

    if st.session_state.model is not None:
        st.write("""
        SHAP untuk model forecasting memerlukan pendekatan khusus karena struktur data deret waktu.
        Berikut adalah interpretasi model forecasting menggunakan SHAP.
        """ if st.session_state.language == 'id' else """
        SHAP for forecasting models requires a special approach due to the time series data structure.
        Here is the interpretation of the forecasting model using SHAP.
        """)
        
        # Cek apakah model adalah model machine learning atau model statistik
        if hasattr(st.session_state, 'forecast_model_type'):
            model_type = st.session_state.forecast_model_type
            
            if model_type in ['random_forest', 'gradient_boosting', 'linear_regression']:
                # Untuk model ML, kita bisa menggunakan SHAP seperti biasa
                st.subheader("Pemilihan Fitur untuk Analisis SHAP" if st.session_state.language == 'id' else "Feature Selection for SHAP Analysis")
                
                if hasattr(st.session_state, 'forecast_features') and st.session_state.forecast_features:
                    feature_names = st.session_state.forecast_features
                    selected_features = st.multiselect(
                        "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
                        options=feature_names,
                        default=feature_names[:min(10, len(feature_names))]
                    )
                    
                    # Jumlah sampel untuk analisis SHAP
                    sample_size = st.slider(
                        "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                        min_value=10, max_value=min(100, len(st.session_state.forecast_test_data)), value=50
                    )
                    
                    if st.button("Generate SHAP Values" if st.session_state.language == 'id' else "Generate SHAP Values"):
                        if not selected_features:
                            st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
                        else:
                            with st.spinner("Menghitung nilai SHAP..." if st.session_state.language == 'id' else "Calculating SHAP values..."):
                                try:
                                    # Persiapkan data untuk SHAP
                                    X_sample = st.session_state.forecast_test_data[selected_features].sample(min(sample_size, len(st.session_state.forecast_test_data)), random_state=42)
                                    
                                    # Pastikan semua nilai dalam X_sample adalah numerik
                                    for col in X_sample.columns:
                                        try:
                                            # Konversi ke numpy array terlebih dahulu
                                            X_sample[col] = np.array(X_sample[col]).astype(float)
                                        except:
                                            try:
                                                # Jika gagal, gunakan factorize dan konversi ke float
                                                X_sample[col] = pd.factorize(X_sample[col])[0].astype(float)
                                            except Exception as e:
                                                st.error(f"Error saat mengkonversi kolom {col} ke numerik: {str(e)}")
                                    
                                    # Pilih explainer berdasarkan jenis model
                                    if model_type in ['random_forest', 'gradient_boosting']:
                                        # Gunakan TreeExplainer untuk model berbasis pohon
                                        explainer = shap.TreeExplainer(st.session_state.model)
                                    else:
                                        # Gunakan KernelExplainer untuk model lainnya
                                        background = shap.kmeans(st.session_state.forecast_train_data[selected_features].sample(min(50, len(st.session_state.forecast_train_data)), random_state=42), 5)
                                        explainer = shap.KernelExplainer(st.session_state.model.predict, background)
                                    
                                    shap_values = explainer.shap_values(X_sample)
                                    
                                    # Visualisasi SHAP
                                    st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                                    
                                    # 1. Summary Plot
                                    st.write("### Summary Plot")
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    shap.summary_plot(shap_values, X_sample, show=False)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.clf()
                                    
                                    # 2. Feature Importance Plot
                                    st.write("### Feature Importance Plot")
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.clf()
                                    
                                    # Interpretasi khusus untuk forecasting
                                    st.subheader("Interpretasi untuk Model Forecasting" if st.session_state.language == 'id' else "Interpretation for Forecasting Model")
                                    st.info("""
                                    Dalam model forecasting, fitur-fitur penting biasanya meliputi:
                                    - **Lag Features**: Nilai historis dari variabel target
                                    - **Fitur Tanggal/Waktu**: Seperti hari dalam minggu, bulan, kuartal, dll.
                                    - **Fitur Rolling**: Seperti rata-rata bergerak, standar deviasi, dll.
                                    - **Fitur Eksternal**: Variabel lain yang mempengaruhi target
                                    
                                    Nilai SHAP tinggi pada lag features menunjukkan bahwa model sangat bergantung pada pola historis terbaru.
                                    """ if st.session_state.language == 'id' else """
                                    In forecasting models, important features typically include:
                                    - **Lag Features**: Historical values of the target variable
                                    - **Date/Time Features**: Such as day of week, month, quarter, etc.
                                    - **Rolling Features**: Such as moving averages, standard deviations, etc.
                                    - **External Features**: Other variables that influence the target
                                    
                                    High SHAP values on lag features indicate that the model heavily relies on recent historical patterns.
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"Error saat menghitung nilai SHAP: {str(e)}")
                                    
                else:
                    st.warning("Tidak dapat menemukan fitur untuk model forecasting. Pastikan model telah dilatih dengan benar." if st.session_state.language == 'id' else 
                            "Could not find features for the forecasting model. Make sure the model has been trained correctly.")
            else:
                # Untuk model statistik seperti ARIMA, SARIMA, dll.
                try:
                    # Gunakan fungsi interpretasi statistik baru
                    from utils import interpret_statistical_model
                    
                    # Deteksi tipe model statistik
                    model_type = type(model).__name__.lower()
                    if 'arima' in model_type:
                        stat_type = 'arima'
                    elif 'sarima' in model_type:
                        stat_type = 'sarima'
                    elif 'exponential' in model_type or 'ets' in model_type:
                        stat_type = 'exponential_smoothing'
                    else:
                        stat_type = 'arima'  # Default
                    
                    # Dapatkan interpretasi statistik
                    interpretation = interpret_statistical_model(model, stat_type, st.session_state.language)
                    
                    if interpretation['success']:
                        st.success("Interpretasi Model Statistik Berhasil!" if st.session_state.language == 'id' else "Statistical Model Interpretation Successful!")
                        
                        # Tampilkan informasi model
                        st.write(f"**{interpretation['interpretation']['model']}**")
                        st.write(interpretation['interpretation']['description'])
                        
                        # Tampilkan koefisien
                        if 'coefficients' in interpretation and interpretation['coefficients']:
                            st.subheader("Koefisien Model" if st.session_state.language == 'id' else "Model Coefficients")
                            
                            if 'ar_terms' in interpretation['coefficients'] and len(interpretation['coefficients']['ar_terms']) > 0:
                                st.write("**Komponen AutoRegresif (AR):**")
                                st.write(interpretation['interpretation']['ar_component'])
                                st.json(interpretation['coefficients']['params'])
                            
                            if 'ma_terms' in interpretation['coefficients'] and len(interpretation['coefficients']['ma_terms']) > 0:
                                st.write("**Komponen Moving Average (MA):**")
                                st.write(interpretation['interpretation']['ma_component'])
                        
                        # Tampilkan goodness of fit
                        if 'goodness_of_fit' in interpretation:
                            st.subheader("Kesesuaian Model" if st.session_state.language == 'id' else "Model Fit")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("AIC", f"{interpretation['goodness_of_fit']['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{interpretation['goodness_of_fit']['bic']:.2f}")
                            
                            st.write(interpretation['interpretation']['aic_interpretation'])
                            st.write(interpretation['interpretation']['bic_interpretation'])
                        
                        # Tampilkan saran interpretasi tambahan
                        if 'interpretation_tips' in interpretation:
                            st.info(interpretation['interpretation_tips'])
                    
                    else:
                        # Fallback ke pesan lama jika interpretasi gagal
                        st.info("""
                        Model statistik seperti ARIMA, SARIMA, atau Exponential Smoothing tidak mendukung interpretasi SHAP secara langsung.
                        Model-model ini didasarkan pada komponen seperti tren, musiman, dan residual, bukan pada fitur individual.
                        
                        Untuk interpretasi model statistik, pertimbangkan untuk melihat:
                        - Koefisien model (AR, MA, dll.)
                        - Dekomposisi deret waktu (tren, musiman, residual)
                        - Analisis residual
                        """ if st.session_state.language == 'id' else """
                        Statistical models like ARIMA, SARIMA, or Exponential Smoothing do not support SHAP interpretation directly.
                        These models are based on components like trend, seasonality, and residuals, not on individual features.
                        
                        For statistical model interpretation, consider looking at:
                        - Model coefficients (AR, MA, etc.)
                        - Time series decomposition (trend, seasonality, residuals)
                        - Residual analysis
                        """)
                
                except Exception as e:
                    # Fallback ke pesan lama jika ada error
                    st.info("""
                    Model statistik seperti ARIMA, SARIMA, atau Exponential Smoothing tidak mendukung interpretasi SHAP secara langsung.
                    Model-model ini didasarkan pada komponen seperti tren, musiman, dan residual, bukan pada fitur individual.
                    
                    Untuk interpretasi model statistik, pertimbangkan untuk melihat:
                    - Koefisien model (AR, MA, dll.)
                    - Dekomposisi deret waktu (tren, musiman, residual)
                    - Analisis residual
                    """ if st.session_state.language == 'id' else """
                    Statistical models like ARIMA, SARIMA, or Exponential Smoothing do not support SHAP interpretation directly.
                    These models are based on components like trend, seasonality, and residuals, not on individual features.
                    
                    For statistical model interpretation, consider looking at:
                    - Model coefficients (AR, MA, etc.)
                    - Time series decomposition (trend, seasonality, residuals)
                    - Residual analysis
                    """)
        else:
            st.warning("Informasi model forecasting tidak lengkap. Pastikan model telah dilatih dengan benar." if st.session_state.language == 'id' else 
                    "Forecasting model information is incomplete. Make sure the model has been trained correctly.")
    else:
        st.info("Silakan latih model terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a model in the 'Model Training' tab first.")

# (wizard nav below)


# --- Wizard Navigation ---
st.markdown("---")
st.markdown("### ⏩ Langkah Selanjutnya" if st.session_state.language == 'id' else "### ⏩ Next Step")
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅️ Kembali ke Model Training" if st.session_state.language == 'id' else "⬅️ Back to Model Training", use_container_width=True):
        st.switch_page("pages/04_Cross_Validation_and_Model_Training.py")
with col_next:
    if st.button("Lanjutkan ke Interpretasi LIME ➡️" if st.session_state.language == 'id' else "Continue to LIME Interpretation ➡️", type="primary", use_container_width=True):
        st.switch_page("pages/06_LIME_Model_Interpretation.py")
