"""
Model Evaluation and Explainability Module for Retail Demand Forecasting.
Provides feature importances, SHAP explainability, and residual diagnostics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.utils import logger, get_project_root, load_model_artifact, compute_metrics
from src.data_preprocessing import load_raw_data
from src.feature_engineering import generate_forecasting_features


def get_feature_names(preprocessor, cat_cols: list, num_cols: list) -> list:
    """Extracts transformed feature names from ColumnTransformer."""
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    return encoded_cat_names + num_cols


def plot_feature_importances(pipeline, cat_cols: list, num_cols: list, top_n: int = 20, save_path: str = None) -> pd.DataFrame:
    """Extracts and visualizes tree-based feature importances."""
    preprocessor = pipeline.named_steps['prep']
    estimator = pipeline.named_steps['model']

    feature_names = get_feature_names(preprocessor, cat_cols, num_cols)

    if not hasattr(estimator, 'feature_importances_'):
        logger.warning(f"Estimator {estimator.__class__.__name__} does not have feature_importances_ attribute.")
        return pd.DataFrame()

    importances = estimator.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_df.head(top_n), x='Importance', y='Feature', palette='viridis')
    plt.title(f"Top {top_n} Feature Importances ({estimator.__class__.__name__})", fontsize=14, pad=12)
    plt.xlabel("Relative Importance (Gain / Impurity Reduction)")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"Feature importance plot saved to: {save_path}")
    plt.close()

    return fi_df


def explain_with_shap(pipeline, X_sample: pd.DataFrame, cat_cols: list, num_cols: list,
                      save_path: str = None) -> None:
    """
    Computes TreeSHAP values on transformed features and creates summary plot.
    """
    preprocessor = pipeline.named_steps['prep']
    estimator = pipeline.named_steps['model']

    X_transformed = preprocessor.transform(X_sample)
    feature_names = get_feature_names(preprocessor, cat_cols, num_cols)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    logger.info(f"Computing TreeSHAP values for {len(X_sample)} sample observations...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed_df)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed_df, max_display=15, show=False)
    plt.title("SHAP Feature Attribution Summary (Impact on Predicted Demand)", fontsize=13, pad=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"SHAP summary plot saved to: {save_path}")
    plt.close()


def plot_residuals_and_forecast(y_test: pd.Series, y_pred: np.ndarray, dates_test: pd.Series,
                                save_path: str = None) -> None:
    """Generates residual diagnostics and actual vs forecast timeline plot."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[0], color="#2b5c8f", bins=40)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title("Forecast Residual Distribution (Actual - Predicted)", fontsize=13)
    axes[0].set_xlabel("Residual (Units)")
    axes[0].set_ylabel("Count")

    # 2. Aggregated daily demand: Actual vs Forecast
    timeline_df = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test,
        'Forecast': y_pred
    }).groupby('Date').sum().reset_index()

    axes[1].plot(timeline_df['Date'], timeline_df['Actual'], label='Actual Demand', color='#1f77b4', linewidth=2)
    axes[1].plot(timeline_df['Date'], timeline_df['Forecast'], label='XGBoost Forecast', color='#ff7f0e', linestyle='--', linewidth=2)
    axes[1].set_title("Aggregated Daily Demand: Actual vs Forecast (Holdout Test)", fontsize=13)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Total Demand Across All Stores/SKUs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"Residual and forecast timeline plot saved to: {save_path}")
    plt.close()


def evaluate_saved_model():
    """Loads saved model artifact and performs full explainability analysis."""
    root = get_project_root()
    model_path = os.path.join(root, "models", "demand_forecast_model.pkl")
    artifact = load_model_artifact(model_path)

    pipe = artifact["pipeline"]
    meta = artifact["feature_meta"]
    cat_cols = meta['cat_cols']
    num_cols = meta['num_cols']
    all_features = meta['all_features']
    target_col = meta['target_col']

    raw_df = load_raw_data()
    data, _ = generate_forecasting_features(raw_df)

    split_date = pd.to_datetime(artifact["split_date"])
    test_df = data[data['Date'] >= split_date].copy()

    X_test, y_test = test_df[all_features], test_df[target_col]
    y_pred = pipe.predict(X_test)

    test_metrics = compute_metrics(y_test, y_pred, model_name=artifact["model_name"])
    logger.info(f"Holdout Test Metrics: {test_metrics}")

    # Generate and save plots
    fi_plot_path = os.path.join(root, "reports", "feature_importance.png")
    shap_plot_path = os.path.join(root, "reports", "shap_summary.png")
    diag_plot_path = os.path.join(root, "reports", "residual_and_forecast_timeline.png")

    plot_feature_importances(pipe, cat_cols, num_cols, top_n=20, save_path=fi_plot_path)
    explain_with_shap(pipe, X_test.sample(1000, random_state=42), cat_cols, num_cols, save_path=shap_plot_path)
    plot_residuals_and_forecast(y_test, y_pred, test_df['Date'], save_path=diag_plot_path)

    logger.info("Model evaluation and explainability artifacts generated successfully.")


if __name__ == "__main__":
    evaluate_saved_model()
