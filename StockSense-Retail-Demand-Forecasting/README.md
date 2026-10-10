# 📦 StockSense — Retail Demand Forecasting & Inventory Optimization

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20SHAP-orange.svg)](https://scikit-learn.org/)

**StockSense** is an end-to-end Data Analytics and Machine Learning project that forecasts daily demand across multiple stores and products, and turns those forecasts into practical inventory decisions. It combines a leakage-free, time-aware forecasting pipeline with SQL-based business analysis.

> **In one line:** predict tomorrow's demand correctly, so shelves are neither empty nor overstocked.

---

## 📑 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Methodological Audit & Data Leakage Resolution](#-1-methodological-audit--data-leakage-resolution)
3. [Dataset & Panel Time-Series Architecture](#️-2-dataset--panel-time-series-architecture)
4. [Time-Aware Feature Engineering](#️-3-time-aware-feature-engineering)
5. [Multi-Model Benchmark & Results](#-4-multi-model-benchmark--results)
6. [Model Explainability](#-5-model-explainability)
7. [Business Intelligence & SQL Analysis](#️-6-business-intelligence--sql-analysis)
8. [Repository Structure](#-7-repository-structure)
9. [Getting Started](#-8-getting-started)
10. [Inventory Optimization Recommendations](#-9-inventory-optimization-recommendations)
11. [Technologies Used](#️-technologies-used)
12. [Project Highlights](#-project-highlights)

---

## 📌 Executive Summary

Accurate demand forecasting helps retail and supply chain teams reduce stockouts, control excess inventory, and improve replenishment planning. However, demand prediction pipelines can produce misleading results when they contain **data leakage** or **lookahead bias**, such as using same-day sales information or randomly splitting time-dependent data.

StockSense analyzes a multi-store retail dataset containing **73,100 daily observations across 5 stores and 20 products over 731 consecutive days** (1 Jan 2022 to 1 Jan 2024). The modeling pipeline was audited to identify and remove potential leakage, after which time-aware lag and rolling features were engineered and multiple regression models were evaluated using chronological validation.

### Key Results (current dataset)

| Metric | Value |
| :--- | :---: |
| **Model evaluated** | XGBoost Regressor |
| **Holdout Test MAE** | **89.16 units** |
| **Holdout Test RMSE** | **108.88 units** |
| **Holdout Test R²** | **-0.0088** |
| **Baseline (always predict the average) MAE** | 88.78 units |
| **Validation Strategy** | 20% chronological holdout test set (first 80% of dates train, last 20% test) |

> **Honest reading of these results:** on this dataset the model does **not** beat the simple average baseline. Yesterday's demand has almost no correlation with today's (r ≈ 0.0004), so lag and rolling features carry no signal. This is expected for synthetic data where each day is generated independently. An earlier version of this project, run on a different dataset with weather, promotion and competitor-price columns, reached R² = 0.70; those numbers do not apply to the data in this repository.

---

# 🔍 1. Methodological Audit & Data Leakage Resolution

In the earlier version of this project, baseline models produced artificially high scores, with R² values above 0.85. A methodological audit identified several potential sources of data leakage and lookahead bias, which StockSense resolves as follows.

| Feature / Method        | Audit Finding                                                                                             | Methodological Issue                                                                                      | Resolution                                                                                                                     |
| :---------------------- | :-------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| `Units_Sold`            | Represents realized customer sales during the current day and has a very high correlation with the target (r = 0.997 in the current dataset). | Same-day sales information may not be available when forecasting demand before the selling period.        | Removed from same-day predictors and converted into historical lag features such as `Units_Sold_lag_1` and `Units_Sold_lag_7`. |
| `Revenue`               | Revenue is calculated from realized sales and price: `Revenue = Units_Sold × Price`.                      | Revenue contains realized sales information and can introduce leakage when predicting same-period demand. | Removed from the predictive feature set.                                                                                       |
| `Units_Ordered`         | Represents restocking activity occurring during the current period.                                       | May reflect operational responses to current inventory or sales conditions.                               | Removed from same-day predictors.                                                                                              |
| Random train-test split | Randomly mixes observations from different dates.                                                         | Future observations can indirectly influence training when predicting earlier periods.                    | Replaced with chronological splitting and expanding-window `TimeSeriesSplit`.                                                  |

**Why this matters:** a model that scores well only because it "saw the future" will fail in production. The lower, honest score reflects what the model can actually deliver on unseen future data.

---

# 🏗️ 2. Dataset & Panel Time-Series Architecture

The dataset contains **73,100 daily observations** spanning 1 January 2022 to 1 January 2024. It represents 100 Store-Product combinations (5 stores × 20 products), each observed for 731 consecutive days, which makes it a **panel time series** (many parallel series observed over the same period).

### Columns in `data/sales_data.csv`

| Column | Meaning |
| :--- | :--- |
| `Date` | Day of the observation |
| `Store_ID`, `Product_ID` | Identify one of the 100 series |
| `Category`, `Region` | Product category and store region |
| `Inventory_Level` | Stock level |
| `Units_Sold` | Units sold that day |
| `Units_Ordered` | Units reordered that day |
| `Demand` | Demand value used as the forecasting target (called `Demand Forecast` in the original source file) |
| `Price` | Selling price |

`Revenue` is not stored in the file; the loader rebuilds it as `Units_Sold × Price`.

### Stores and Regions

* 5 stores: `S001` to `S005`
* 4 regions: North, South, East, West

### Products and Categories

* 20 unique products ranging from `P0001` to `P0020`
* 5 categories: Electronics, Clothing, Groceries, Toys, Furniture

### Optional columns

The pipeline can also use `Discount`, `Competitor_Pricing`, `Promotion`, `Epidemic`, `Weather` and `Seasonality` when they are present. They are **not** in the current dataset, so the features built from them are skipped automatically (the log prints a warning listing what was skipped). The SQL scripts in `scripts/` and the notebook were written for the fuller dataset and need those columns.

---

# ⚙️ 3. Time-Aware Feature Engineering

> **Note:** features that depend on columns missing from the current dataset (`Price_Difference`, `Price_Ratio`, `Promo_Discount_Interaction`, weather and season) are skipped automatically; the model currently uses 30 features.

Features are constructed using only information available **before** the forecast period. Historical demand and sales features are shifted so that the current target value is never included in the predictor set.

```text
Feature Pipeline
│
├── Calendar & Temporal Features
│   ├── Year
│   ├── Month
│   ├── Quarter
│   ├── DayOfWeek
│   ├── DayOfMonth
│   ├── WeekOfYear
│   └── IsWeekend
│
├── Market & Pricing Features
│   ├── Price_Difference
│   ├── Price_Ratio
│   └── Promo_Discount_Interaction
│
├── Historical Lag Features
│   ├── Demand_lag_1
│   ├── Demand_lag_2
│   ├── Demand_lag_3
│   ├── Demand_lag_7
│   ├── Demand_lag_14
│   ├── Demand_lag_30
│   ├── Units_Sold_lag_1
│   ├── Units_Sold_lag_7
│   └── Units_Sold_lag_30
│
├── Inventory Features
│   └── Inventory_Level_lag_1
│
└── Shifted Rolling Statistics
    ├── 7-day rolling mean
    ├── 14-day rolling mean
    ├── 30-day rolling mean
    └── Rolling standard deviation
```

### Warmup Handling

The first 30 observations for each time series are excluded where required to ensure that lag and rolling-window features are generated using genuine historical information.

---

# 📊 4. Model Benchmark

Models are evaluated using an **80% chronological training period** and a **20% unseen holdout test period**. The chronological split prevents the model from training on observations that occur after the period being predicted.

The training script compares five models (Linear Regression, Decision Tree, Random Forest, Extra Trees, XGBoost) with 4-fold expanding-window `TimeSeriesSplit`. To regenerate the full benchmark table on the current dataset, run:

```bash
python -m src.train_model
```

(Random Forest and Extra Trees are slow, so this can take several minutes.)

Result measured so far on the current dataset (XGBoost, 20% chronological holdout):

| Model | Test MAE | Test RMSE | Test R² |
| :--- | :---: | :---: | :---: |
| XGBoost Regressor | 89.16 | 108.88 | -0.0088 |
| Baseline: predict the average | 88.78 | – | – |

> **Note:** R² measures the proportion of variance explained by the model. It should not be interpreted as forecast accuracy. Forecasting error is evaluated primarily using MAE and RMSE.

---

# 🧠 5. Model Explainability

The best-performing XGBoost model was analyzed using **SHAP (SHapley Additive exPlanations)**.

The XGBoost model can be analyzed using **SHAP (SHapley Additive exPlanations)**. Run `python -m src.evaluate_model` to generate the plots, or open the **Model results** page of the Streamlit app for a feature-importance chart.

Because the current dataset contains almost no learnable time-series signal, importance values here mainly reflect noise. Features such as `Demand_rolling_mean_7`, `Demand_lag_1`, `Price_Difference` and `Promo_Discount_Interaction` were meaningful on the earlier, fuller dataset.

> The plots saved in the `reports/` directory (if present) come from the earlier dataset and are replaced when you run `evaluate_model`:
> `feature_importance.png`, `shap_summary.png` and `residual_and_forecast_timeline.png`.

---

# 🗄️ 6. Business Intelligence & SQL Analysis

> **Note:** these scripts were written for the fuller dataset. Parts that use `Weather`, `Epidemic`, `Discount`, `Competitor_Pricing` or `Promotion` need those columns, which are not in the current `data/sales_data.csv`.

Alongside the ML pipeline, StockSense includes modular SQL scripts for business analysis (a combined version is also available as `SQL_Store_Analysis.sql`).

### `Sales_revenue_metrics.sql`

Analyzes:

* Revenue by category
* Revenue by store
* Revenue concentration across products

### `inventory_demand_metrics.sql`

Analyzes:

* Inventory turnover
* Stock-to-demand coverage
* Potential demand-sales gaps

The demand-sales gap is calculated as:

`Demand - Units_Sold`

### `order_fulfillment_metrics.sql`

Analyzes:

* Replenishment patterns
* Regional fulfillment performance
* Supply buffer metrics

### `weather_impact.sql`

Analyzes:

* Revenue variation across weather conditions
* Demand patterns under different weather conditions
* Ranking and comparison using SQL window functions

### `epidemic_effect_demand_sales.sql`

Analyzes:

* Category resilience
* Demand changes during disruption periods
* Sales volatility under external disruptions

---

# 📁 7. Repository Structure

```text
StockSense-Retail-Demand-Forecasting/
│
├── app.py                                  # Streamlit dashboard
│
├── data/
│   └── sales_data.csv
│
├── notebooks/
│   └── Retail_Inventory_Demand_Forecasting.ipynb
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── scripts/
│   ├── Sales_revenue_metrics.sql
│   ├── inventory_demand_metrics.sql
│   ├── order_fulfillment_metrics.sql
│   ├── weather_impact.sql
│   └── epidemic_effect_demand_sales.sql
│
├── models/
│   └── demand_forecast_model.pkl
│
├── reports/
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── residual_and_forecast_timeline.png
│
├── Retail Inventory and Demand Forecasting.ipynb
├── SQL_Store_Analysis.sql
├── requirements.txt
└── README.md
```

---

# 🚀 8. Getting Started

## Prerequisites

* Python 3.10 or 3.11
* pip
* The dataset file `sales_data.csv`, placed inside the `data/` folder (or the project root); the loader checks both locations
* Always run commands from the **project root folder** (the one that contains `src/` and `app.py`)

## Installation

```bash
# Get the project (clone your copy of the repository, or unzip the archive)
cd StockSense-Retail-Demand-Forecasting

# Install dependencies
pip install -r requirements.txt
```

## Train and Evaluate Models

```bash
# Run the complete training and evaluation pipeline
python -m src.train_model

# Generate explainability and diagnostic plots
python -m src.evaluate_model
```

## Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` and has three pages: **Data overview**, **Explore data** (daily demand, category and region charts, per-store/product view) and **Model results** (XGBoost metrics, actual vs forecast, feature importance).

## Run the Jupyter Notebook

```bash
jupyter notebook "Retail Inventory and Demand Forecasting.ipynb"
```

---

# 💡 9. Inventory Optimization Recommendations

### 1. Dynamic Safety Stock

Use recent demand variability, such as `Demand_rolling_std_7`, to adjust safety stock levels dynamically rather than relying on fixed inventory buffers.

### 2. Promotion-Aware Replenishment

Plan inventory replenishment before major promotions to reduce the risk of stockouts during periods of increased demand.

### 3. Regular Model Retraining

Retrain the forecasting model periodically as new demand data becomes available to adapt to changing customer behavior and market conditions.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **XGBoost**
* **SHAP**
* **SQL**
* **Streamlit**
* **Jupyter Notebook**

---

# 📈 Project Highlights

* Audited the original modeling workflow for potential data leakage.
* Replaced random data splitting with chronological validation.
* Engineered time-aware lag and rolling features.
* Compared five regression models (benchmark script included).
* Built an interactive Streamlit dashboard for data exploration and model results.
* Evaluated models using MAE, RMSE, R², and MAPE.
* Used SHAP for model explainability.
* Performed SQL-based sales, inventory, fulfillment, weather, and disruption analysis.
* Generated actionable inventory optimization recommendations.
