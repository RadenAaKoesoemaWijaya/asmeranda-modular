import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def calculate_forecast_metrics(actual, predicted):
    """Menghitung metrik evaluasi untuk forecasting"""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()

        mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
        actual = actual[mask]
        predicted = predicted[mask]

        if len(actual) == 0 or len(predicted) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'count': 0}

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        mask_non_zero = actual != 0
        if np.any(mask_non_zero):
            mape = np.mean(np.abs((actual[mask_non_zero] - predicted[mask_non_zero]) / actual[mask_non_zero])) * 100
        else:
            mape = np.nan

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'count': len(actual)}
    except Exception:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'count': 0}


def plot_forecast_visualization(data, forecast_results, target_column, date_column=None):
    """Membuat visualisasi hasil forecasting"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Forecasting Analysis - {target_column}', fontsize=16, fontweight='bold')

        if date_column and date_column in data.columns:
            data = data.set_index(date_column)

        actual_data = data[target_column].dropna()

        ax1 = axes[0, 0]
        ax1.plot(actual_data.index, actual_data.values,
                 label='Actual', color='blue', linewidth=2, alpha=0.8)

        colors = ['red', 'green', 'orange', 'purple', 'brown']
        model_names = []

        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            forecast_values = forecast['forecast'] if isinstance(forecast, dict) and 'forecast' in forecast else forecast
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                start_idx = len(actual_data) - len(forecast_values)
                if start_idx >= 0:
                    forecast_index = actual_data.index[-len(forecast_values):]
                    ax1.plot(forecast_index, forecast_values,
                             label=f'{model_name} Forecast',
                             color=colors[i % len(colors)],
                             linewidth=2, linestyle='--')
                    model_names.append(model_name)

        ax1.set_title('Actual vs Forecast Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        ax2 = axes[0, 1]
        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            forecast_values = forecast['forecast'] if isinstance(forecast, dict) and 'forecast' in forecast else forecast
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                actual_slice = actual_data.values[-len(forecast_values):]
                residuals = actual_slice - forecast_values
                forecast_index = actual_data.index[-len(forecast_values):]
                ax2.scatter(forecast_index, residuals,
                            label=f'{model_name} Residuals',
                            color=colors[i % len(colors)], alpha=0.7)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Residual Analysis')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residual')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        ax3 = axes[1, 0]
        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            forecast_values = forecast['forecast'] if isinstance(forecast, dict) and 'forecast' in forecast else forecast
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                actual_slice = actual_data.values[-len(forecast_values):]
                residuals = actual_slice - forecast_values
                ax3.hist(residuals, bins=20, alpha=0.6,
                         label=f'{model_name} Errors', color=colors[i % len(colors)])

        ax3.set_title('Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.axis('off')

        metrics_data = []
        for model_name, forecast in forecast_results.items():
            forecast_values = forecast['forecast'] if isinstance(forecast, dict) and 'forecast' in forecast else forecast
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                actual_slice = actual_data.values[-len(forecast_values):]
                metrics = calculate_forecast_metrics(actual_slice, forecast_values)
                metrics_data.append({
                    'Model': model_name,
                    'RMSE': f"{metrics['rmse']:.4f}" if not np.isnan(metrics['rmse']) else "N/A",
                    'MAE': f"{metrics['mae']:.4f}" if not np.isnan(metrics['mae']) else "N/A",
                    'MAPE': f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "N/A",
                    'R²': f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A"
                })

        if metrics_data:
            table_data = [[d['Model'], d['RMSE'], d['MAE'], d['MAPE'], d['R²']] for d in metrics_data]
            table = ax4.table(cellText=table_data,
                              colLabels=['Model', 'RMSE', 'MAE', 'MAPE', 'R²'],
                              cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig
    except Exception as e:
        try:
            import streamlit as _st  # lazy import
            _st.error(f"Error dalam visualisasi forecasting: {e}")
        except Exception:
            pass
        return None


def build_forecast_summary(forecast_results, target_column, data):
    """
    Bangun ringkasan hasil forecasting dalam bentuk dict/None (pure).

    Parameters
    ----------
    forecast_results : dict
        {model_name: {'forecast': array-like}}.
    target_column : str
    data : pd.DataFrame
        Data aktual (kolom ``target_column``).

    Returns
    -------
    dict | None
        {
          'best_rmse': float,
          'best_mae': float,
          'best_mape': float | None,
          'best_r2': float,
          'metrics_summary': [ {Model, RMSE, MAE, MAPE, R², Data Points}, ... ]
        }
        Mengembalikan ``None`` bila tidak ada forecast valid.
    """
    metrics_summary: list[dict] = []
    actual_data = data[target_column].dropna()

    for model_name, forecast in forecast_results.items():
        forecast_values = (
            forecast["forecast"]
            if isinstance(forecast, dict) and "forecast" in forecast
            else forecast
        )
        if hasattr(forecast_values, "__len__") and len(forecast_values) > 0:
            actual_slice = actual_data.values[-len(forecast_values):]
            metrics = calculate_forecast_metrics(actual_slice, forecast_values)
            metrics_summary.append(
                {
                    "Model": model_name,
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
                    "MAPE": metrics["mape"],
                    "R²": metrics["r2"],
                    "Data Points": metrics["count"],
                }
            )

    if not metrics_summary:
        return None

    metrics_df = pd.DataFrame(metrics_summary)
    best_rmse = float(np.nanmin(metrics_df["RMSE"].values))
    best_mae = float(np.nanmin(metrics_df["MAE"].values))
    best_r2 = float(np.nanmax(metrics_df["R²"].values))
    if metrics_df["MAPE"].isna().all():
        best_mape: Optional[float] = None
    else:
        best_mape = float(np.nanmin(metrics_df["MAPE"].values))

    return {
        "best_rmse": best_rmse,
        "best_mae": best_mae,
        "best_mape": best_mape,
        "best_r2": best_r2,
        "metrics_summary": metrics_summary,
    }


def display_forecast_summary(forecast_results, target_column, data):
    """
    UI wrapper (Streamlit) - memanggil ``build_forecast_summary`` lalu
    menampilkan ke UI. Dipisah dari logika murni agar bisa dipakai
    di backend FastAPI tanpa import Streamlit.
    """
    try:
        import streamlit as _st  # lazy import untuk UI saja
    except Exception:
        return None

    try:
        _st.subheader(f"📊 Forecasting Summary - {target_column}")
        summary = build_forecast_summary(forecast_results, target_column, data)
        if summary is None:
            return None

        metrics_summary = summary["metrics_summary"]
        metrics_df = pd.DataFrame(metrics_summary)

        col1, col2, col3, col4 = _st.columns(4)
        with col1:
            _st.metric("Best RMSE", f"{summary['best_rmse']:.4f}")
        with col2:
            _st.metric("Best MAE", f"{summary['best_mae']:.4f}")
        with col3:
            if summary["best_mape"] is not None:
                _st.metric("Best MAPE", f"{summary['best_mape']:.2f}%")
            else:
                _st.metric("Best MAPE", "N/A")
        with col4:
            _st.metric("Best R²", f"{summary['best_r2']:.4f}")

        _st.dataframe(
            metrics_df.style.format(
                {
                    "RMSE": "{:.4f}",
                    "MAE": "{:.4f}",
                    "MAPE": "{:.2f}%",
                    "R²": "{:.4f}",
                }
            )
            .background_gradient(cmap="RdYlGn", subset=["R²"], axis=0)
            .background_gradient(cmap="RdYlGn_r", subset=["RMSE", "MAE", "MAPE"], axis=0)
        )
        return metrics_df
    except Exception as e:
        _st.error(f"Error dalam menampilkan ringkasan: {e}")
        return None
