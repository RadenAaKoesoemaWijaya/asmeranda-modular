import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
import warnings
warnings.filterwarnings('ignore')

# Import untuk LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow tidak tersedia. LSTM tidak dapat digunakan.")

# Import fungsi-fungsi dari utils.py
from utils import (
    is_timeseries, detect_timeseries_columns, prepare_timeseries_data,
    check_stationarity, plot_timeseries_analysis, create_features_from_date,
    create_lag_features, create_rolling_features
)

@st.cache_resource(show_spinner=False)
def train_arima_model(data, target_column, order=(1,1,1)):
    """
    Melatih model ARIMA untuk data timeseries
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    order : tuple, optional
        Order ARIMA (p,d,q)
        
    Returns:
    --------
    model : statsmodels.tsa.arima.model.ARIMAResults
        Model ARIMA yang telah dilatih
    """
    # Gunakan data yang sudah diimputasi jika tersedia, bukan drop sembarangan
    if data.isnull().any().any():
        # Coba gunakan imputasi yang lebih halus untuk data time series
        data_clean = data.copy()
        
        # Untuk kolom target, gunakan interpolasi time-based
        if data_clean[target_column].isnull().any():
            data_clean[target_column] = data_clean[target_column].interpolate(method='time')
            # Isi sisa dengan forward/backward fill
            data_clean[target_column] = data_clean[target_column].fillna(method='ffill').fillna(method='bfill')
        
        # Untuk kolom lain, gunakan metode yang sesuai
        for col in data_clean.columns:
            if data_clean[col].dtype in ['float64', 'int64'] and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].interpolate(method='linear')
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif data_clean[col].dtype == 'object' and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        data = data_clean
    
    # Validasi data tetap diperlukan
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model ARIMA")
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model ARIMA")
    
    if len(data) < 5:
        raise ValueError(f"Data terlalu sedikit untuk training model ARIMA. Minimal 5 data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Latih model ARIMA
    try:
        model = ARIMA(data[target_column], order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        raise ValueError(f"Gagal melatih model ARIMA: {str(e)}")

def calculate_forecast_metrics(actual, predicted):
    """
    Menghitung berbagai metrik evaluasi untuk forecasting
    
    Parameters:
    -----------
    actual : array-like
        Nilai aktual
    predicted : array-like
        Nilai prediksi
        
    Returns:
    --------
    dict
        Dictionary berisi MAE, MSE, RMSE, dan MAPE
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Hapus nilai NaN atau infinite
    mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'MAE': None,
            'MSE': None,
            'RMSE': None,
            'MAPE': None,
            'R2': None
        }
    
    # Hitung metrik
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    # Hindari pembagian dengan nol
    mask_nonzero = actual != 0
    if np.sum(mask_nonzero) > 0:
        mape = np.mean(np.abs((actual[mask_nonzero] - predicted[mask_nonzero]) / actual[mask_nonzero])) * 100
    else:
        mape = None
    
    # R-squared
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def evaluate_forecast_model(model, test_data, target_column, date_column=None):
    """
    Mengevaluasi model forecasting dengan berbagai metrik
    
    Parameters:
    -----------
    model : object
        Model yang telah dilatih
    test_data : pandas.DataFrame
        Data testing
    target_column : str
        Nama kolom target
    date_column : str, optional
        Nama kolom tanggal (untuk model ML)
        
    Returns:
    --------
    dict
        Hasil evaluasi model
    """
    try:
        # Validasi jumlah data testing yang cukup
        if len(test_data) < 2:
            return {
                'error': f'Data testing terlalu sedikit. Minimal 2 data dibutuhkan, tersedia: {len(test_data)}',
                'MAE': None,
                'MSE': None,
                'RMSE': None,
                'MAPE': None,
                'R2': None
            }
            
        # Validasi data target
        if test_data[target_column].isnull().all():
            return {
                'error': f'Kolom target {target_column} tidak memiliki nilai yang valid',
                'MAE': None,
                'MSE': None,
                'RMSE': None,
                'MAPE': None,
                'R2': None
            }
    
        y_pred = None
        y_actual = None
        
        if hasattr(model, 'predict'):
            # Untuk model ML (Random Forest, Gradient Boosting, dll)
            if date_column is not None:
                try:
                    from utils import create_features_from_date, create_lag_features, create_rolling_features
                    
                    df = test_data.copy()
                    df[date_column] = pd.to_datetime(df[date_column])
                    df = df.sort_values(by=date_column)
                    
                    # Buat fitur
                    df = create_features_from_date(df, date_column)
                    df = create_lag_features(df, target_column)
                    df = create_rolling_features(df, target_column)
                    
                    # Hapus NaN
                    df = df.dropna()
                    
                    # Validasi kembali setelah preprocessing
                    if len(df) < 2:
                        return {
                            'error': f'Data terlalu sedikit setelah preprocessing: {len(df)} data',
                            'MAE': None,
                            'MSE': None,
                            'RMSE': None,
                            'MAPE': None,
                            'R2': None
                        }
                    
                    # Dapatkan fitur yang digunakan
                    features = [col for col in df.columns if col != target_column and col != date_column]
                    
                    if len(features) > 0 and len(df) > 0:
                        X_test = df[features]
                        y_actual = df[target_column]
                        y_pred = model.predict(X_test)
                        
                        return calculate_forecast_metrics(y_actual, y_pred)
                    else:
                        return {
                            'error': 'Tidak cukup fitur atau data untuk evaluasi',
                            'MAE': None,
                            'MSE': None,
                            'RMSE': None,
                            'MAPE': None,
                            'R2': None
                        }
                        
                except Exception as e:
                    return {
                        'error': f'Error dalam evaluasi model ML: {str(e)}',
                        'MAE': None,
                        'MSE': None,
                        'RMSE': None,
                        'MAPE': None,
                        'R2': None
                    }
        
        elif hasattr(model, 'forecast'):
            # Untuk model ARIMA, SARIMA, Exponential Smoothing
            try:
                # Dapatkan prediksi untuk periode testing
                steps = len(test_data)
                if steps > 0:
                    forecast_result = model.forecast(steps=steps)
                    
                    if hasattr(forecast_result, 'values'):
                        y_pred = forecast_result.values
                    else:
                        y_pred = np.array(forecast_result)
                    
                    y_actual = test_data[target_column].values
                    
                    return calculate_forecast_metrics(y_actual, y_pred)
                else:
                    return {
                        'error': 'Tidak ada data untuk forecasting',
                        'MAE': None,
                        'MSE': None,
                        'RMSE': None,
                        'MAPE': None,
                        'R2': None
                    }
                    
            except Exception as e:
                return {
                    'error': f'Error dalam evaluasi model time series: {str(e)}',
                    'MAE': None,
                    'MSE': None,
                    'RMSE': None,
                    'MAPE': None,
                    'R2': None
                }
        
        else:
            return {
                'error': 'Tipe model tidak dikenali',
                'MAE': None,
                'MSE': None,
                'RMSE': None,
                'MAPE': None,
                'R2': None
            }
            
    except Exception as e:
        return {
            'error': f'Error umum dalam evaluasi: {str(e)}',
            'MAE': None,
            'MSE': None,
            'RMSE': None,
            'MAPE': None,
            'R2': None
        }

@st.cache_resource(show_spinner=False)
def train_sarima_model(data, target_column, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Melatih model SARIMA untuk data timeseries dengan komponen musiman
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    order : tuple, optional
        Order ARIMA (p,d,q)
    seasonal_order : tuple, optional
        Order SARIMA (P,D,Q,s)
        
    Returns:
    --------
    model : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Model SARIMA yang telah dilatih
    """
    # Gunakan data yang sudah diimputasi jika tersedia, bukan drop sembarangan
    if data.isnull().any().any():
        # Coba gunakan imputasi yang lebih halus untuk data time series
        data_clean = data.copy()
        
        # Untuk kolom target, gunakan interpolasi time-based
        if data_clean[target_column].isnull().any():
            data_clean[target_column] = data_clean[target_column].interpolate(method='time')
            # Isi sisa dengan forward/backward fill
            data_clean[target_column] = data_clean[target_column].fillna(method='ffill').fillna(method='bfill')
        
        # Untuk kolom lain, gunakan metode yang sesuai
        for col in data_clean.columns:
            if data_clean[col].dtype in ['float64', 'int64'] and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].interpolate(method='linear')
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif data_clean[col].dtype == 'object' and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        data = data_clean
    
    # Validasi data tetap diperlukan
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model SARIMA")
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model SARIMA")
    
    if len(data) < 10:
        raise ValueError(f"Data terlalu sedikit untuk training model SARIMA. Minimal 10 data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Latih model SARIMA
    try:
        model = SARIMAX(
            data[target_column],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        return model_fit
    except Exception as e:
        raise ValueError(f"Gagal melatih model SARIMA: {str(e)}")

@st.cache_resource(show_spinner=False)
def train_sarimax_model(data, target_column, exog_columns=None, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Melatih model SARIMAX untuk data timeseries dengan variabel eksogen
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    exog_columns : list, optional
        Daftar nama kolom eksogen
    order : tuple, optional
        Order ARIMA (p,d,q)
    seasonal_order : tuple, optional
        Order SARIMA (P,D,Q,s)
        
    Returns:
    --------
    model : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Model SARIMAX yang telah dilatih
    """
    # Gunakan data yang sudah diimputasi jika tersedia, bukan drop sembarangan
    if data.isnull().any().any():
        # Coba gunakan imputasi yang lebih halus untuk data time series
        data_clean = data.copy()
        
        # Untuk kolom target, gunakan interpolasi time-based
        if data_clean[target_column].isnull().any():
            data_clean[target_column] = data_clean[target_column].interpolate(method='time')
            # Isi sisa dengan forward/backward fill
            data_clean[target_column] = data_clean[target_column].fillna(method='ffill').fillna(method='bfill')
        
        # Untuk kolom lain, gunakan metode yang sesuai
        for col in data_clean.columns:
            if data_clean[col].dtype in ['float64', 'int64'] and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].interpolate(method='linear')
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif data_clean[col].dtype == 'object' and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        data = data_clean
    
    # Validasi data tetap diperlukan
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model SARIMAX")
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model SARIMAX")
    
    if len(data) < 10:
        raise ValueError(f"Data terlalu sedikit untuk training model SARIMAX. Minimal 10 data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Validasi kolom eksternal
    if exog_columns is None:
        exog_columns = []
    
    for col in exog_columns:
        if col not in data.columns:
            raise ValueError(f"Kolom eksternal '{col}' tidak ditemukan dalam data")
    
    # Latih model SARIMAX
    try:
        if exog_columns:
            exog_data = data[exog_columns]
        else:
            exog_data = None
            
        model = SARIMAX(
            data[target_column],
            exog=exog_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        return model_fit
    except Exception as e:
        raise ValueError(f"Gagal melatih model SARIMAX: {str(e)}")

@st.cache_resource(show_spinner=False)
def train_holt_winters(data, target_column, trend='add', seasonal='add', seasonal_periods=12):
    """
    Melatih model Holt-Winters Exponential Smoothing
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    trend : str, optional
        Tipe trend ('add', 'mul')
    seasonal : str, optional
        Tipe seasonal ('add', 'mul')
    seasonal_periods : int, optional
        Jumlah periode dalam satu siklus musiman
        
    Returns:
    --------
    model : statsmodels.tsa.holtwinters.ExponentialSmoothingResults
        Model Holt-Winters yang telah dilatih
    """
    # Pastikan data tidak memiliki nilai yang hilang
    data = data.dropna()
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model Holt-Winters")
    
    if len(data) < 2 * seasonal_periods:
        raise ValueError(f"Data terlalu sedikit untuk training model Holt-Winters. Minimal {2 * seasonal_periods} data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Latih model Holt-Winters
    try:
        model = HWES(
            data[target_column],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        raise ValueError(f"Gagal melatih model Holt-Winters: {str(e)}")

@st.cache_resource(show_spinner=False)
def train_lstm_model(data, target_column, look_back=60, epochs=100, batch_size=32, 
                     lstm_units=50, num_layers=2, dropout=0.2, recurrent_dropout=0.2, 
                     bidirectional=False, learning_rate=0.001, optimizer='adam'):
    """
    Melatih model LSTM untuk forecasting time series
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    look_back : int, optional
        Jumlah time steps untuk look back window
    epochs : int, optional
        Jumlah epochs untuk training
    batch_size : int, optional
        Batch size untuk training
    lstm_units : int, optional
        Jumlah unit LSTM
    num_layers : int, optional
        Jumlah layer LSTM
    dropout : float, optional
        Dropout rate untuk layer
    recurrent_dropout : float, optional
        Recurrent dropout rate
    bidirectional : bool, optional
        Gunakan bidirectional LSTM
    learning_rate : float, optional
        Learning rate untuk optimizer
    optimizer : str, optional
        Tipe optimizer ('adam', 'sgd', 'rmsprop')
        
    Returns:
    --------
    model : dict
        Dictionary berisi model LSTM, scaler, dan parameter
    """
    if not TENSORFLOW_AVAILABLE:
        raise ValueError("TensorFlow tidak tersedia. Silakan install TensorFlow terlebih dahulu.")
    
    # Gunakan data yang sudah diimputasi jika tersedia, bukan drop sembarangan
    if data.isnull().any().any():
        # Coba gunakan imputasi yang lebih halus untuk data time series
        data_clean = data.copy()
        
        # Untuk kolom target, gunakan interpolasi time-based
        if data_clean[target_column].isnull().any():
            data_clean[target_column] = data_clean[target_column].interpolate(method='time')
            # Isi sisa dengan forward/backward fill
            data_clean[target_column] = data_clean[target_column].fillna(method='ffill').fillna(method='bfill')
        
        # Untuk kolom lain, gunakan metode yang sesuai
        for col in data_clean.columns:
            if data_clean[col].dtype in ['float64', 'int64'] and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].interpolate(method='linear')
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif data_clean[col].dtype == 'object' and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        data = data_clean
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model LSTM")
    
    if len(data) < look_back + 10:
        raise ValueError(f"Data terlalu sedikit untuk training model LSTM. Minimal {look_back + 10} data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    try:
        # Normalisasi data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[[target_column]])
        
        # Buat sequences untuk LSTM
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data
        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Buat model LSTM dengan parameter arsitektur lanjutan
        model = Sequential()
        
        # Layer pertama
        if bidirectional:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=True if num_layers > 1 else False, 
                                     dropout=dropout, recurrent_dropout=recurrent_dropout, 
                                     input_shape=(look_back, 1))))
        else:
            model.add(LSTM(lstm_units, return_sequences=True if num_layers > 1 else False, 
                          dropout=dropout, recurrent_dropout=recurrent_dropout, 
                          input_shape=(look_back, 1)))
        
        # Layer tambahan jika num_layers > 1
        for i in range(1, num_layers):
            return_sequences = True if i < num_layers - 1 else False
            if bidirectional:
                model.add(Bidirectional(LSTM(lstm_units, return_sequences=return_sequences, 
                                           dropout=dropout, recurrent_dropout=recurrent_dropout)))
            else:
                model.add(LSTM(lstm_units, return_sequences=return_sequences, 
                             dropout=dropout, recurrent_dropout=recurrent_dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Setup optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            opt = Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=opt, loss='mean_squared_error')
        
        # Training model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                  validation_data=(X_test, y_test), verbose=0)
        
        return {
            'model': model,
            'scaler': scaler,
            'X_test': X_test,
            'y_test': y_test,
            'look_back': look_back,
            'lstm_units': lstm_units,
            'num_layers': num_layers,
            'dropout': dropout,
            'recurrent_dropout': recurrent_dropout,
            'bidirectional': bidirectional,
            'learning_rate': learning_rate,
            'optimizer': optimizer
        }
        
    except Exception as e:
        raise ValueError(f"Gagal melatih model LSTM: {str(e)}")

@st.cache_resource(show_spinner=False)
def train_exponential_smoothing(data, target_column, trend='add', seasonal='add', seasonal_periods=12):
    """
    Melatih model Exponential Smoothing untuk data timeseries
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    trend : str, optional
        Jenis trend ('add', 'mul', None)
    seasonal : str, optional
        Jenis seasonal ('add', 'mul', None)
    seasonal_periods : int, optional
        Periode musiman
        
    Returns:
    --------
    model : statsmodels.tsa.holtwinters.ExponentialSmoothingResults
        Model Exponential Smoothing yang telah dilatih
    """
    # Gunakan data yang sudah diimputasi jika tersedia, bukan drop sembarangan
    if data.isnull().any().any():
        # Coba gunakan imputasi yang lebih halus untuk data time series
        data_clean = data.copy()
        
        # Untuk kolom target, gunakan interpolasi time-based
        if data_clean[target_column].isnull().any():
            data_clean[target_column] = data_clean[target_column].interpolate(method='time')
            # Isi sisa dengan forward/backward fill
            data_clean[target_column] = data_clean[target_column].fillna(method='ffill').fillna(method='bfill')
        
        # Untuk kolom lain, gunakan metode yang sesuai
        for col in data_clean.columns:
            if data_clean[col].dtype in ['float64', 'int64'] and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].interpolate(method='linear')
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif data_clean[col].dtype == 'object' and data_clean[col].isnull().any():
                data_clean[col] = data_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        data = data_clean
    
    # Validasi data tetap diperlukan
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model Exponential Smoothing")
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model Exponential Smoothing")
    
    if len(data) < 5:
        raise ValueError(f"Data terlalu sedikit untuk training model Exponential Smoothing. Minimal 5 data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Latih model Exponential Smoothing
    try:
        if seasonal is not None and seasonal_periods is not None:
            model = ExponentialSmoothing(
                data[target_column],
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
        else:
            model = ExponentialSmoothing(
                data[target_column],
                trend=trend
            )
        
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        raise ValueError(f"Gagal melatih model Exponential Smoothing: {str(e)}")

@st.cache_resource(show_spinner=False)
def train_ml_forecaster(data, date_column, target_column, features=None, model_type='random_forest', **model_params):
    """
    Melatih model machine learning untuk forecasting dengan fitur lag dan rolling
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan kolom tanggal dan target
    date_column : str
        Nama kolom tanggal/waktu
    target_column : str
        Nama kolom target yang akan diprediksi
    features : list, optional
        Daftar kolom fitur tambahan (selain fitur lag dan rolling)
    model_type : str, optional
        Tipe model ('random_forest', 'gradient_boosting', 'linear')
    model_params : dict
        Parameter tambahan untuk model
        
    Returns:
    --------
    dict
        Dictionary berisi model yang telah dilatih dan informasi tambahan
    """
    # Buat salinan data
    df = data.copy()
    
    # Pastikan kolom tanggal adalah datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Urutkan data berdasarkan tanggal
    df = df.sort_values(by=date_column)
    
    # Buat fitur dari tanggal
    df = create_features_from_date(df, date_column)
    
    # Buat fitur lag
    df = create_lag_features(df, target_column)
    
    # Buat fitur rolling
    df = create_rolling_features(df, target_column)
    
    # Hapus baris dengan nilai NaN (karena fitur lag dan rolling)
    df = df.dropna()
    
    # Tentukan fitur yang akan digunakan
    date_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_month_start', 'is_month_end']
    lag_features = [col for col in df.columns if f'{target_column}_lag_' in col]
    rolling_features = [col for col in df.columns if f'{target_column}_rolling_' in col]
    
    # Gabungkan semua fitur
    all_features = date_features + lag_features + rolling_features
    
    # Tambahkan fitur tambahan jika ada
    if features is not None:
        all_features += [f for f in features if f in df.columns and f != target_column and f != date_column]
    
    # Pisahkan fitur dan target
    X = df[all_features]
    y = df[target_column]
    
    # Pilih model berdasarkan tipe
    if model_type == 'random_forest':
        model = RandomForestRegressor(**model_params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(**model_params)
    else:  # linear
        model = LinearRegression(**model_params)
    
    # Latih model
    model.fit(X, y)
    
    # Kembalikan model dan informasi tambahan
    return {
        'model': model,
        'features': all_features,
        'date_column': date_column,
        'target_column': target_column,
        'last_date': df[date_column].max(),
        'model_type': model_type
    }

def forecast_future(model_info, periods=10, freq='D'):
    """
    Membuat prediksi untuk periode di masa depan
    
    Parameters:
    -----------
    model_info : dict atau statsmodels model
        Model yang telah dilatih atau dictionary dengan informasi model ML
    periods : int, optional
        Jumlah periode yang akan diprediksi
    freq : str, optional
        Frekuensi data ('D' untuk harian, 'W' untuk mingguan, 'M' untuk bulanan, dll.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan tanggal dan hasil prediksi
    """
    # Cek tipe model
    if isinstance(model_info, dict) and 'model_type' in model_info:  # ML model or DLinear
        if model_info['model_type'] == 'dlinear':
            # DLinear model prediction
            model = model_info['model']
            history_length = model_info['history_length']
            last_date = model_info['last_date']
            
            res = model.predict(steps=periods, history_length=history_length)
            forecast = res['forecast']
            
            # Create dates for future periods
            if isinstance(last_date, pd.Timestamp):
                future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
            else:
                future_dates = range(periods)
                
            return pd.DataFrame({
                'date': future_dates,
                'forecast': forecast
            })
        else:
            # Ambil informasi dari model_info
            model = model_info['model']
            features = model_info['features']
            date_column = model_info['date_column']
            target_column = model_info['target_column']
            last_date = model_info['last_date']
            
            # Buat tanggal untuk periode masa depan
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
            
            # Buat DataFrame untuk prediksi
            future_df = pd.DataFrame({date_column: future_dates})
            
            # Buat fitur dari tanggal
            future_df = create_features_from_date(future_df, date_column)
            
            # Untuk model ML, kita perlu membuat fitur lag dan rolling secara manual
            # Ini memerlukan data historis, yang seharusnya disimpan dalam model_info
            # Untuk sederhananya, kita akan menggunakan nilai default untuk fitur lag dan rolling
            
            # Buat fitur lag dan rolling dengan nilai default (rata-rata dari data latih)
            lag_features = [f for f in features if f'{target_column}_lag_' in f]
            rolling_features = [f for f in features if f'{target_column}_rolling_' in f]
            
            # Gunakan nilai default untuk fitur lag dan rolling (misalnya 0)
            for feature in lag_features + rolling_features:
                future_df[feature] = 0
            
            # Prediksi menggunakan model ML
            future_df['forecast'] = model.predict(future_df[features])
            
            # Kembalikan hasil prediksi
            return future_df[[date_column, 'forecast']]
    
    else:  # Statsmodels model (ARIMA, SARIMA, Exponential Smoothing)
        # Prediksi menggunakan model statsmodels
        forecast = model_info.forecast(steps=periods)
        
        # Buat tanggal untuk periode masa depan
        try:
            if hasattr(model_info, 'model') and hasattr(model_info.model, 'data'):
                # Handle both regular DataFrame and PandasData object
                data_obj = model_info.model.data
                
                # Try different ways to get the index
                last_date = None
                
                # Method 1: Direct index access
                if hasattr(data_obj, 'index') and len(data_obj.index) > 0:
                    try:
                        last_date = data_obj.index[-1]
                    except (IndexError, TypeError):
                        pass
                
                # Method 2: Access through orig_endog for PandasData
                if last_date is None and hasattr(data_obj, 'orig_endog'):
                    orig_endog = data_obj.orig_endog
                    if hasattr(orig_endog, 'index') and len(orig_endog.index) > 0:
                        try:
                            last_date = orig_endog.index[-1]
                        except (IndexError, TypeError):
                            pass
                
                # Method 3: Access through _index attribute
                if last_date is None and hasattr(data_obj, '_index') and len(data_obj._index) > 0:
                    try:
                        last_date = data_obj._index[-1]
                    except (IndexError, TypeError):
                        pass
                
                # If still no date, use numerical index
                if last_date is None:
                    return pd.DataFrame({
                        'date': range(periods),
                        'forecast': forecast
                    })
                    
            else:
                # Fallback to numerical index
                return pd.DataFrame({
                    'date': range(periods),
                    'forecast': forecast
                })
        except Exception:
            # Fallback to numerical index if any error occurs
            return pd.DataFrame({
                'date': range(periods),
                'forecast': forecast
            })
        
        # Buat tanggal untuk periode masa depan
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
        else:
            future_dates = range(periods)
        
        # Kembalikan hasil prediksi
        return pd.DataFrame({
            'date': future_dates,
            'forecast': forecast
        })

def evaluate_forecast_model(model, test_data, target_column, date_column=None):
    """
    Mengevaluasi model forecasting menggunakan data test
    
    Parameters:
    -----------
    model : dict atau statsmodels model
        Model yang telah dilatih atau dictionary dengan informasi model ML
    test_data : pandas.DataFrame
        DataFrame dengan data test
    target_column : str
        Nama kolom target
    date_column : str, optional
        Nama kolom tanggal (untuk model ML)
        
    Returns:
    --------
    dict
        Dictionary dengan metrik evaluasi (MSE, RMSE, MAE, R2)
    """
    try:
        # Cek tipe model
        if isinstance(model, dict):
            if 'model_type' in model:  # ML model or DLinear
                if model['model_type'] == 'dlinear':
                    model_obj = model['model']
                    history_length = model['history_length']
                    res = model_obj.predict(steps=len(test_data), history_length=history_length)
                    predictions = res['forecast']
                    actual = test_data[target_column].values
                else:
                    # Untuk model ML
                    model_info = model
                    features = model_info['features']
                    date_column = model_info['date_column']
                    target_column = model_info['target_column']
                    
                    # Pastikan kolom tanggal adalah datetime
                    test_data[date_column] = pd.to_datetime(test_data[date_column])
                    
                    # Buat fitur dari tanggal
                    test_data = create_features_from_date(test_data, date_column)
                    
                    # Buat fitur lag dan rolling
                    test_data = create_lag_features(test_data, target_column)
                    test_data = create_rolling_features(test_data, target_column)
                    test_data = test_data.dropna()
                    
                    # Prediksi menggunakan model ML
                    predictions = model_info['model'].predict(test_data[features])
                    actual = test_data[target_column]
                
            elif 'model' in model and 'scaler' in model:  # LSTM model
                # Untuk model LSTM
                lstm_model = model['model']
                scaler = model['scaler']
                X_test = model['X_test']
                y_test = model['y_test']
                
                # Prediksi menggunakan model LSTM
                predictions_scaled = lstm_model.predict(X_test, verbose=0)
                predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
                actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
            else:  # Statsmodels model (ARIMA, SARIMA, Exponential Smoothing, etc.)
                # Prediksi menggunakan model statsmodels
                actual = test_data[target_column]
                
                # Validasi panjang data test
                if len(test_data) == 0:
                    raise ValueError("Data test kosong")
                
                # Prediksi untuk periode test
                forecast = model.forecast(steps=len(test_data))
                predictions = forecast
                
        else:  # Statsmodels model direct
            # Prediksi menggunakan model statsmodels
            actual = test_data[target_column]
            
            # Validasi panjang data test
            if len(test_data) == 0:
                raise ValueError("Data test kosong")
            
            # Prediksi untuk periode test
            forecast = model.forecast(steps=len(test_data))
            predictions = forecast
        
        # Konversi ke numpy array untuk perhitungan
        actual = np.array(actual).flatten()
        predictions = np.array(predictions).flatten()
        
        # Validasi panjang data
        if len(actual) != len(predictions):
            # Potong sesuai panjang terpendek
            min_len = min(len(actual), len(predictions))
            actual = actual[:min_len]
            predictions = predictions[:min_len]
        
        # Validasi data kosong
        if len(actual) == 0 or len(predictions) == 0:
            raise ValueError("Tidak ada data untuk evaluasi")
        
        # Hitung metrik evaluasi dengan penanganan error dan NaN
        try:
            # Validasi data untuk NaN atau infinite values
            mask = ~np.isnan(actual) & ~np.isnan(predictions) & ~np.isinf(actual) & ~np.isinf(predictions)
            actual_clean = actual[mask]
            predictions_clean = predictions[mask]
            
            if len(actual_clean) == 0 or len(predictions_clean) == 0:
                raise ValueError("Data evaluasi mengandung NaN/Inf values")
            
            mse = mean_squared_error(actual_clean, predictions_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_clean, predictions_clean)
            
            # Hitung R2 dengan penanganan error
            try:
                r2 = r2_score(actual_clean, predictions_clean)
            except Exception:
                r2 = 0.0
                
            # Hitung MAPE (Mean Absolute Percentage Error)
            try:
                # Hindari pembagian dengan nol
                mape_mask = actual_clean != 0
                if np.any(mape_mask):
                    mape = np.mean(np.abs((actual_clean[mape_mask] - predictions_clean[mape_mask]) / actual_clean[mape_mask])) * 100
                else:
                    mape = 0.0
            except Exception:
                mape = 0.0
                
            # Hitung SMAPE (Symmetric Mean Absolute Percentage Error)
            try:
                smape = np.mean(2 * np.abs(predictions_clean - actual_clean) / (np.abs(actual_clean) + np.abs(predictions_clean))) * 100
            except Exception:
                smape = 0.0
                
        except Exception as e:
            raise ValueError(f"Gagal menghitung metrik evaluasi: {str(e)}")
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'SMAPE': smape,
            'actual': actual,
            'predictions': predictions,
            'n_samples': len(actual)
        }
        
    except Exception as e:
        # Return error message in evaluation format
        return {
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'MAPE': np.nan,
            'SMAPE': np.nan,
            'actual': [],
            'predictions': [],
            'n_samples': 0,
            'error': str(e)
        }

def plot_forecast_results(train_data, test_data, forecast_data, target_column, date_column=None, figsize=(15, 8)):
    """
    Membuat plot hasil forecasting
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Data training
    test_data : pandas.DataFrame
        Data testing
    forecast_data : pandas.DataFrame
        Data hasil forecasting
    target_column : str
        Nama kolom target
    date_column : str, optional
        Nama kolom tanggal (jika tidak menggunakan index)
    figsize : tuple, optional
        Ukuran gambar
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi plot hasil forecasting
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data training
    if date_column is not None and date_column in train_data.columns:
        train_data = train_data.set_index(date_column)
    
    ax.plot(train_data.index, train_data[target_column], label='Data Training')
    
    # Plot data testing
    if date_column is not None and date_column in test_data.columns:
        test_data = test_data.set_index(date_column)
    
    ax.plot(test_data.index, test_data[target_column], label='Data Testing', color='green')
    
    # Plot hasil forecasting
    if 'ds' in forecast_data.columns and 'yhat' in forecast_data.columns:
        # Format dari forecast_future untuk model statsmodels
        ax.plot(forecast_data['ds'], forecast_data['yhat'], label='Forecast', color='red')
    elif date_column in forecast_data.columns and f'predicted_{target_column}' in forecast_data.columns:
        # Format dari forecast_future untuk model ML
        ax.plot(forecast_data[date_column], forecast_data[f'predicted_{target_column}'], label='Forecast', color='red')
    
    ax.set_title('Hasil Forecasting')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel(target_column)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig