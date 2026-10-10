"""
Feature Engineering Module for Retail Demand Forecasting.
Creates leakage-free calendar, pricing, promotion, lag, and rolling historical features.
"""

import pandas as pd
import numpy as np
from src.utils import logger


def generate_forecasting_features(df: pd.DataFrame, drop_warmup: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Constructs a comprehensive, leakage-free feature set for time-series demand forecasting.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe sorted by ['Store_ID', 'Product_ID', 'Date'].
        drop_warmup (bool): Whether to drop the initial 30 days of warmup period with NaN lags.
        
    Returns:
        tuple[pd.DataFrame, dict]:
            - Processed DataFrame containing engineered features
            - Dictionary with feature metadata ('cat_cols', 'num_cols', 'target_col')
    """
    logger.info("Starting feature engineering pipeline...")
    data = df.copy()

    # 1. Calendar & Temporal Features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['DayOfMonth'] = data['Date'].dt.day
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)

    # 2. Pricing, Competitor & Promotion Dynamics
    # Some columns (Promotion, Epidemic, Discount, Competitor_Pricing) are optional.
    # If a column is missing from the dataset, we simply skip the features built from it.
    if 'Promotion' in data.columns:
        data['Promotion_Num'] = data['Promotion'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    if 'Epidemic' in data.columns:
        data['Epidemic_Num'] = data['Epidemic'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    if 'Competitor_Pricing' in data.columns:
        data['Price_Difference'] = data['Price'] - data['Competitor_Pricing']
        data['Price_Ratio'] = data['Price'] / (data['Competitor_Pricing'] + 1e-6)
    if 'Promotion_Num' in data.columns and 'Discount' in data.columns:
        data['Promo_Discount_Interaction'] = data['Promotion_Num'] * data['Discount']

    # 3. Time Series Lags (grouped strictly by Store_ID and Product_ID)
    grouped = data.groupby(['Store_ID', 'Product_ID'])

    # Past Demand Lags
    for lag in [1, 2, 3, 7, 14, 30]:
        data[f'Demand_lag_{lag}'] = grouped['Demand'].shift(lag)

    # Past Realized Sales Lags (Historical units sold known before day t)
    for lag in [1, 7, 30]:
        data[f'Units_Sold_lag_{lag}'] = grouped['Units_Sold'].shift(lag)

    # Past Opening Inventory (Stock available at the start of period)
    data['Inventory_Level_lag_1'] = grouped['Inventory_Level'].shift(1)

    # 4. Rolling Window Statistics (Strictly shifted by 1 so target t is NEVER included)
    for w in [7, 14, 30]:
        data[f'Demand_rolling_mean_{w}'] = grouped['Demand'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        data[f'Demand_rolling_std_{w}'] = grouped['Demand'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        ).fillna(0)
        data[f'Units_Sold_rolling_mean_{w}'] = grouped['Units_Sold'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

    # 5. Handle warmup period
    if drop_warmup:
        initial_len = len(data)
        data = data.dropna(subset=['Demand_lag_30']).copy()
        logger.info(f"Dropped 30-day warmup period: {initial_len} -> {len(data)} rows remaining.")

    # Re-sort chronologically by Date across all entities for time-aware splitting
    data = data.sort_values(by=['Date', 'Store_ID', 'Product_ID']).reset_index(drop=True)

    # 6. Feature Metadata
    # Full wish-list of features. Anything whose source column is missing is filtered out below.
    cat_cols = ['Store_ID', 'Product_ID', 'Category', 'Region', 'Weather', 'Seasonality']
    num_cols = [
        'Price', 'Discount', 'Competitor_Pricing', 'Price_Difference', 'Price_Ratio',
        'Promotion_Num', 'Epidemic_Num', 'Promo_Discount_Interaction',
        'Year', 'Month', 'Quarter', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'IsWeekend',
        'Demand_lag_1', 'Demand_lag_2', 'Demand_lag_3', 'Demand_lag_7', 'Demand_lag_14', 'Demand_lag_30',
        'Units_Sold_lag_1', 'Units_Sold_lag_7', 'Units_Sold_lag_30',
        'Inventory_Level_lag_1',
        'Demand_rolling_mean_7', 'Demand_rolling_std_7',
        'Demand_rolling_mean_14', 'Demand_rolling_std_14',
        'Demand_rolling_mean_30', 'Demand_rolling_std_30',
        'Units_Sold_rolling_mean_7', 'Units_Sold_rolling_mean_30'
    ]
    missing = [c for c in cat_cols + num_cols if c not in data.columns]
    if missing:
        logger.warning(f"Skipping features not available in this dataset: {missing}")
    cat_cols = [c for c in cat_cols if c in data.columns]
    num_cols = [c for c in num_cols if c in data.columns]

    feature_meta = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'all_features': cat_cols + num_cols,
        'target_col': 'Demand'
    }

    logger.info(f"Feature engineering completed. Total features: {len(feature_meta['all_features'])} "
                f"({len(cat_cols)} categorical, {len(num_cols)} numerical).")
    return data, feature_meta
