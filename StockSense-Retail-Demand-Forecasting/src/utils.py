"""
Utility functions for Retail Store Demand Forecasting.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RetailForecast")


def get_project_root() -> str:
    """Returns absolute path to project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "") -> dict:
    """
    Computes standard regression & time series forecasting metrics:
    - MAE: Mean Absolute Error (in units)
    - RMSE: Root Mean Squared Error (in units)
    - R2: Coefficient of Determination
    - MAPE: Mean Absolute Percentage Error (%)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Safe MAPE calculation (avoiding division by zero)
    mask = (y_true != 0)
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    metrics = {
        "Model": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MAPE (%)": round(float(mape), 2)
    }
    return metrics


def save_model_artifact(model_obj, filepath: str) -> None:
    """Saves a fitted model/pipeline to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)
    logger.info(f"Model saved successfully to: {filepath}")


def load_model_artifact(filepath: str):
    """Loads a serialized model artifact from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    logger.info(f"Loading model artifact from: {filepath}")
    return joblib.load(filepath)
