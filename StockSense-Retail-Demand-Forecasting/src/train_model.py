"""
Model Training and Cross-Validation Pipeline for Retail Demand Forecasting.
Evaluates 5 models under strict time-based cross-validation and holdout evaluation.
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from src.utils import logger, compute_metrics, save_model_artifact, get_project_root
from src.data_preprocessing import load_raw_data
from src.feature_engineering import generate_forecasting_features


def create_preprocessors(cat_cols: list, num_cols: list) -> tuple[ColumnTransformer, ColumnTransformer]:
    """Creates preprocessing transformers for linear and tree-based estimators."""
    preprocessor_linear = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    preprocessor_tree = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    return preprocessor_linear, preprocessor_tree


def build_model_pipelines(cat_cols: list, num_cols: list) -> dict:
    """Builds scikit-learn pipelines for all 5 candidate models."""
    prep_linear, prep_tree = create_preprocessors(cat_cols, num_cols)

    models = {
        "Linear Regression": Pipeline([
            ('prep', prep_linear),
            ('model', LinearRegression())
        ]),
        "Decision Tree": Pipeline([
            ('prep', prep_tree),
            ('model', DecisionTreeRegressor(max_depth=12, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('prep', prep_tree),
            ('model', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
        ]),
        "Extra Trees": Pipeline([
            ('prep', prep_tree),
            ('model', ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
        ]),
        "XGBoost": Pipeline([
            ('prep', prep_tree),
            ('model', XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    return models


def run_time_series_cv(train_df: pd.DataFrame, feature_cols: list, target_col: str,
                       cat_cols: list, num_cols: list, n_splits: int = 4) -> dict:
    """
    Executes Date-Grouped Time-Series Cross Validation on the training data.
    Ensures no lookahead bias across temporal folds.
    """
    logger.info(f"Running {n_splits}-fold TimeSeriesSplit cross-validation on training data...")
    unique_dates = train_df['Date'].drop_duplicates().sort_values().values
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipelines = build_model_pipelines(cat_cols, num_cols)
    cv_summary = {}

    for name, pipe in pipelines.items():
        fold_metrics = []
        for fold, (train_dt_idx, val_dt_idx) in enumerate(tscv.split(unique_dates)):
            fold_train_dates = unique_dates[train_dt_idx]
            fold_val_dates = unique_dates[val_dt_idx]

            X_tr = train_df[train_df['Date'].isin(fold_train_dates)][feature_cols]
            y_tr = train_df[train_df['Date'].isin(fold_train_dates)][target_col]
            X_val = train_df[train_df['Date'].isin(fold_val_dates)][feature_cols]
            y_val = train_df[train_df['Date'].isin(fold_val_dates)][target_col]

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_val)
            metrics = compute_metrics(y_val, y_pred, model_name=f"{name}_Fold{fold+1}")
            fold_metrics.append(metrics)

        fold_df = pd.DataFrame(fold_metrics)
        cv_summary[name] = {
            "Mean_MAE": round(float(fold_df["MAE"].mean()), 4),
            "Mean_RMSE": round(float(fold_df["RMSE"].mean()), 4),
            "Mean_R2": round(float(fold_df["R2"].mean()), 4)
        }
        logger.info(f"CV [{name}] -> Mean MAE: {cv_summary[name]['Mean_MAE']:.2f}, "
                    f"Mean RMSE: {cv_summary[name]['Mean_RMSE']:.2f}, Mean R2: {cv_summary[name]['Mean_R2']:.4f}")

    return cv_summary


def train_and_evaluate_all(train_ratio: float = 0.8) -> tuple[pd.DataFrame, object, dict]:
    """
    Main training workflow:
    1. Loads data and engineers time-series features
    2. Performs time-aware 80/20 train/test split
    3. Runs TimeSeriesSplit cross validation
    4. Trains all 5 models and evaluates on out-of-time test set
    5. Saves best model artifact
    """
    raw_df = load_raw_data()
    data, meta = generate_forecasting_features(raw_df)

    cat_cols = meta['cat_cols']
    num_cols = meta['num_cols']
    all_features = meta['all_features']
    target_col = meta['target_col']

    # Chronological Train/Test Split
    unique_dates = data['Date'].drop_duplicates().sort_values().values
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]

    train_df = data[data['Date'] < split_date].copy()
    test_df = data[data['Date'] >= split_date].copy()

    logger.info(f"Train timeline: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    logger.info(f"Test timeline: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')} ({len(test_df)} rows)")

    # Run TimeSeriesSplit CV on Train portion
    cv_results = run_time_series_cv(train_df, all_features, target_col, cat_cols, num_cols)

    # Train all models on full Train dataset and evaluate on holdout Test set
    pipelines = build_model_pipelines(cat_cols, num_cols)
    X_train, y_train = train_df[all_features], train_df[target_col]
    X_test, y_test = test_df[all_features], test_df[target_col]

    evaluation_records = []
    best_model_name = None
    best_test_rmse = float("inf")
    best_pipeline = None

    for name, pipe in pipelines.items():
        logger.info(f"Fitting final pipeline for {name} on training set...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)

        # Predictions
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        train_m = compute_metrics(y_train, y_train_pred)
        test_m = compute_metrics(y_test, y_test_pred)

        record = {
            "Model": name,
            "CV MAE": cv_results[name]["Mean_MAE"],
            "CV RMSE": cv_results[name]["Mean_RMSE"],
            "CV R2": cv_results[name]["Mean_R2"],
            "Train MAE": train_m["MAE"],
            "Train RMSE": train_m["RMSE"],
            "Train R2": train_m["R2"],
            "Test MAE": test_m["MAE"],
            "Test RMSE": test_m["RMSE"],
            "Test R2": test_m["R2"],
            "Test MAPE (%)": test_m["MAPE (%)"],
            "Fit Time (s)": fit_time
        }
        evaluation_records.append(record)

        if test_m["RMSE"] < best_test_rmse:
            best_test_rmse = test_m["RMSE"]
            best_model_name = name
            best_pipeline = pipe

    results_table = pd.DataFrame(evaluation_records)
    logger.info("\n" + "="*80 + "\nFINAL MODEL BENCHMARK TABLE:\n" + results_table.to_string(index=False) + "\n" + "="*80)
    logger.info(f"Best Performing Model: {best_model_name} (Test RMSE: {best_test_rmse:.2f})")

    # Save artifact
    root = get_project_root()
    model_path = os.path.join(root, "models", "demand_forecast_model.pkl")
    save_model_artifact({
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "feature_meta": meta,
        "split_date": str(split_date),
        "results_table": results_table
    }, model_path)

    return results_table, best_pipeline, meta


if __name__ == "__main__":
    train_and_evaluate_all()
