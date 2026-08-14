import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if "data" not in st.session_state or st.session_state.data is None:
    st.warning("Silakan unggah data terlebih dahulu di halaman Data Upload.")
    st.stop()
    
data = st.session_state.data
categorical_cols = st.session_state.categorical_columns
numerical_cols = st.session_state.numerical_columns
st.header("Analisis Data Eksplorasi" if st.session_state.language == 'id' else "Exploratory Data Analysis")

if st.session_state.data is not None:
    data = st.session_state.data
    
    # Missing values analysis
    st.subheader("Analisis Nilai Hilang" if st.session_state.language == 'id' else "Missing Values Analysis")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percentage
    })
    st.dataframe(missing_df)
    
    # Plot missing values
    if missing_values.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_df[missing_df['Missing Values'] > 0]['Percentage (%)'].sort_values(ascending=False).plot(kind='bar', ax=ax)
        plt.title('Persentase Nilai Hilang' if st.session_state.language == 'id' else 'Missing Values Percentage')
        plt.ylabel('Persentase (%)' if st.session_state.language == 'id' else 'Percentage (%)')
        plt.xlabel('Kolom' if st.session_state.language == 'id' else 'Columns')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("Tidak ditemukan nilai yang hilang dalam dataset." if st.session_state.language == 'id' else "No missing values found in the dataset.")
    
    # Integrate workflow validation for EDA to ML transition
    if WORKFLOW_VALIDATOR_AVAILABLE and workflow_validator is not None:
        try:
            # Validate EDA completeness and readiness for ML workflows
            eda_validation = workflow_validator.validate_eda_completeness(data, st.session_state.numerical_columns, st.session_state.categorical_columns)
            
            with st.expander("🔍 Validasi EDA untuk ML" if st.session_state.language == 'id' else "🔍 EDA Validation for ML"):
                st.info("Validasi kesiapan analisis eksplorasi data untuk transisi ke machine learning:" if st.session_state.language == 'id' else "Validation of exploratory data analysis readiness for machine learning transition:")
                
                # Display EDA validation results
                for result in eda_validation:
                    if result['status'] == 'success':
                        st.success(f"✅ {result['message']}")
                    elif result['status'] == 'warning':
                        st.warning(f"⚠️ {result['message']}")
                    elif result['status'] == 'error':
                        st.error(f"❌ {result['message']}")
                
                # Store EDA validation results in session state
                st.session_state.eda_validation_results = eda_validation
                
                # Check if EDA is ready for ML transition
                eda_readiness = workflow_validator.check_eda_readiness(eda_validation)
                st.session_state.eda_readiness = eda_readiness
                
                if eda_readiness['ready']:
                    st.success(f"🎯 {eda_readiness['message']}")
                    st.write("**Transisi ML yang dapat dilakukan:**" if st.session_state.language == 'id' else "**Possible ML transitions:**")
                    for transition in eda_readiness['available_transitions']:
                        st.write(f"• {transition}")
                else:
                    st.warning(f"⚠️ {eda_readiness['message']}")
                    if 'recommendations' in eda_readiness:
                        st.write("**Rekomendasi EDA:**" if st.session_state.language == 'id' else "**EDA Recommendations:**")
                        for rec in eda_readiness['recommendations']:
                            st.write(f"• {rec}")
                            
        except Exception as e:
            if ERROR_HANDLER_AVAILABLE and error_handler is not None:
                error_result = error_handler.handle_error(e, "EDA Workflow Validation")
                st.warning(f"⚠️ {error_result['message']}")
            else:
                st.warning(f"⚠️ Gagal melakukan validasi workflow EDA: {str(e)}")
    
    # Correlation analysis for numerical columns
    if len(st.session_state.numerical_columns) > 1:
        st.subheader("Analisis Korelasi" if st.session_state.language == 'id' else "Correlation Analysis")
        correlation = data[st.session_state.numerical_columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        plt.title('Matriks Korelasi' if st.session_state.language == 'id' else 'Correlation Matrix')
        st.pyplot(fig)
    
    # Distribution of numerical columns
    if len(st.session_state.numerical_columns) > 0:
        st.subheader("Distribusi Fitur Numerik" if st.session_state.language == 'id' else "Distribution of Numerical Features")
        
        selected_num_col = st.selectbox("Pilih kolom numerik untuk analisis distribusi:" if st.session_state.language == 'id' else "Select a numerical column for distribution analysis:", 
                                       st.session_state.numerical_columns)
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        sns.histplot(data[selected_num_col].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f'Histogram {selected_num_col}')
        
        # Box plot
        sns.boxplot(y=data[selected_num_col].dropna(), ax=ax[1])
        ax[1].set_title(f'Boxplot {selected_num_col}')
        
        st.pyplot(fig)
    
    # Time Series Pattern Analysis (if time series data)
    if st.session_state.get('is_time_series', False) and st.session_state.time_column and st.session_state.target_column:
        st.subheader("🔍 Analisis Pola Time Series" if st.session_state.language == 'id' else "🔍 Time Series Pattern Analysis")
        
        try:
            # Prepare time series data
            ts_data = prepare_timeseries_data(
                data, 
                st.session_state.time_column, 
                st.session_state.target_column
            )
            
            # Analyze patterns
            pattern_analysis = analyze_trend_seasonality_cycle(
                ts_data[st.session_state.target_column]
            )
            
            # Display pattern insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Tren Terdeteksi" if st.session_state.language == 'id' else "Trend Detected",
                    "✅ Ya" if pattern_analysis['trend_detected'] else "❌ Tidak",
                    delta=f"Kekuatan: {pattern_analysis['trend_strength']:.2f}"
                )
            
            with col2:
                st.metric(
                    "Seasonality Terdeteksi" if st.session_state.language == 'id' else "Seasonality Detected",
                    "✅ Ya" if pattern_analysis['seasonality_detected'] else "❌ Tidak",
                    delta=f"Kekuatan: {pattern_analysis['seasonality_strength']:.2f}"
                )
            
            with col3:
                st.metric(
                    "Siklus Terdeteksi" if st.session_state.language == 'id' else "Cycle Detected",
                    "✅ Ya" if pattern_analysis['cycle_detected'] else "❌ Tidak",
                    delta=f"Kekuatan: {pattern_analysis['cycle_strength']:.2f}"
                )
            
            # Detailed pattern information
            if pattern_analysis['trend_detected']:
                st.info(f"📈 **Tren:** Kemiringan tren adalah {pattern_analysis['trend_slope']:.4f} per periode")
            
            if pattern_analysis['seasonality_detected']:
                st.info(f"🌊 **Seasonality:** Terdeteksi dengan kekuatan {pattern_analysis['seasonality_strength']:.2f}")
            
            if pattern_analysis['cycle_detected'] and pattern_analysis['dominant_cycle_period']:
                st.info(f"🔄 **Siklus:** Periode dominan adalah {pattern_analysis['dominant_cycle_period']:.1f} periode")
            
            # Visualize patterns
            st.write("**Visualisasi Pola Time Series:**" if st.session_state.language == 'id' else "**Time Series Pattern Visualization:**")
            pattern_fig = plot_pattern_analysis(ts_data[st.session_state.target_column])
            st.pyplot(pattern_fig)
            
            # Seasonal decomposition insights
            if 'decomposition' in pattern_analysis and pattern_analysis['decomposition'] is not None:
                st.write("**Insight dari Dekomposisi:**" if st.session_state.language == 'id' else "**Decomposition Insights:**")
                
                decomposition = pattern_analysis['decomposition']
                
                # Calculate variance explained by each component
                total_var = np.var(ts_data[st.session_state.target_column])
                trend_var = np.var(decomposition.trend.dropna()) if decomposition.trend is not None else 0
                seasonal_var = np.var(decomposition.seasonal.dropna()) if decomposition.seasonal is not None else 0
                residual_var = np.var(decomposition.resid.dropna()) if decomposition.resid is not None else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Varians Tren", f"{trend_var/total_var:.1%}")
                with col2:
                    st.metric("Varians Seasonal", f"{seasonal_var/total_var:.1%}")
                with col3:
                    st.metric("Varians Residual", f"{residual_var/total_var:.1%}")
            
        except Exception as e:
            st.error(f"Error dalam analisis pola: {str(e)}")
    
    # Distribution of categorical columns
    if len(st.session_state.categorical_columns) > 0:
        st.subheader("Distribusi Fitur Kategorikal" if st.session_state.language == 'id' else "Distribution of Categorical Features")
        
        selected_cat_col = st.selectbox("Pilih kolom kategorikal untuk analisis distribusi:" if st.session_state.language == 'id' else "Select a categorical column for distribution analysis:", 
                                       st.session_state.categorical_columns)
        
        # Count plot
        fig, ax = plt.subplots(figsize=(12, 6))
        value_counts = data[selected_cat_col].value_counts()
        
        # If there are too many categories, show only top 20
        if len(value_counts) > 20:
            st.warning(f"Kolom ini memiliki {len(value_counts)} nilai unik. Hanya menampilkan 20 teratas." if st.session_state.language == 'id' else f"The column has {len(value_counts)} unique values. Showing only top 20.")
            value_counts = value_counts.head(20)
        
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        plt.title(f'Jumlah {selected_cat_col}' if st.session_state.language == 'id' else f'Count of {selected_cat_col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Bivariate analysis
    st.subheader("Analisis Bivariat" if st.session_state.language == 'id' else "Bivariate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("Pilih X-axis feature:" if st.session_state.language == 'id' else "Select X-axis feature:", data.columns)
    
    with col2:
        y_axis = st.selectbox("Pilih Y-axis feature:" if st.session_state.language == 'id' else "Select Y-axis feature:", [col for col in data.columns if col != x_axis])
    
    # Determine the plot type based on the data types
    x_is_numeric = data[x_axis].dtype in ['int64', 'float64']
    y_is_numeric = data[y_axis].dtype in ['int64', 'float64']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if x_is_numeric and y_is_numeric:
        # Scatter plot for numeric vs numeric
        sns.scatterplot(x=x_axis, y=y_axis, data=data, ax=ax)
        plt.title(f'Scatter plot of {x_axis} vs {y_axis}')
    elif x_is_numeric and not y_is_numeric:
        # Box plot for numeric vs categorical
        sns.boxplot(x=y_axis, y=x_axis, data=data, ax=ax)
        plt.title(f'Box plot of {x_axis} by {y_axis}')
    elif not x_is_numeric and y_is_numeric:
        # Box plot for categorical vs numeric
        sns.boxplot(x=x_axis, y=y_axis, data=data, ax=ax)
        plt.title(f'Box plot of {y_axis} by {x_axis}')
    else:
        # Count plot for categorical vs categorical
        pd.crosstab(data[x_axis], data[y_axis]).plot(kind='bar', stacked=True, ax=ax)
        plt.title(f'Stacked bar plot of {x_axis} and {y_axis}')
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Silakan unggah dataset di tab 'Data Upload' terlebih dahulu." if st.session_state.language == 'id' else "Please upload a dataset in the 'Data Upload' tab first.")


# --- Wizard Navigation ---
st.markdown("---")
st.markdown("### u23E9 Langkah Selanjutnya" if st.session_state.language == 'id' else "### u23E9 Next Step")
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("u2B05uFE0F Kembali ke Upload Data" if st.session_state.language == 'id' else "u2B05uFE0F Back to Data Upload", use_container_width=True):
        st.switch_page("pages/01_Data_Upload.py")
with col_next:
    if st.button("Lanjutkan ke Unsupervised Learning u27A1uFE0F" if st.session_state.language == 'id' else "Continue to Unsupervised Learning u27A1uFE0F", type="primary", use_container_width=True):
        st.switch_page("pages/02b_Unsupervised_Learning.py")
