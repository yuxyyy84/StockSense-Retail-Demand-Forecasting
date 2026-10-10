"""
StockSense Streamlit dashboard.
Run from the project folder with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data_preprocessing import load_raw_data
from src.feature_engineering import generate_forecasting_features
from src.train_model import build_model_pipelines
from src.utils import compute_metrics

# ---------- Page setup ----------
st.set_page_config(page_title="StockSense", page_icon="📦", layout="wide")
st.title("📦 StockSense — Retail Demand Forecasting")


# ---------- Helper functions ----------
# @st.cache_data remembers the result, so the file is not re-read on every click.
@st.cache_data
def get_raw_data():
    return load_raw_data()


# @st.cache_resource is the same idea, but for big objects like a trained model.
@st.cache_resource
def train_xgboost():
    raw = get_raw_data()
    data, meta = generate_forecasting_features(raw)

    # Chronological split: first 80% of dates = train, last 20% = test
    dates = data["Date"].drop_duplicates().sort_values().values
    split_date = dates[int(len(dates) * 0.8)]
    train = data[data["Date"] < split_date]
    test = data[data["Date"] >= split_date].copy()

    features = meta["all_features"]
    model = build_model_pipelines(meta["cat_cols"], meta["num_cols"])["XGBoost"]
    model.fit(train[features], train["Demand"])

    test["Predicted"] = model.predict(test[features])
    metrics = compute_metrics(test["Demand"], test["Predicted"].values)
    return model, test, meta, metrics


# ---------- Sidebar menu ----------
page = st.sidebar.radio("Go to", ["1. Data overview", "2. Explore data", "3. Model results"])
df = get_raw_data()

# ---------- Page 1 ----------
if page == "1. Data overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Stores", df["Store_ID"].nunique())
    c3.metric("Products", df["Product_ID"].nunique())
    c4.metric("Days", df["Date"].nunique())
    st.caption(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    st.subheader("First 100 rows")
    st.dataframe(df.head(100))
    st.subheader("Summary statistics")
    st.dataframe(df.describe().T)

# ---------- Page 2 ----------
elif page == "2. Explore data":
    st.subheader("Daily total demand vs units sold")
    daily = df.groupby("Date")[["Demand", "Units_Sold"]].sum()
    st.line_chart(daily.rolling(7).mean())  # 7-day moving average

    left, right = st.columns(2)
    with left:
        st.subheader("Demand by category")
        st.bar_chart(df.groupby("Category")["Demand"].sum())
    with right:
        st.subheader("Demand by region")
        st.bar_chart(df.groupby("Region")["Demand"].sum())

    st.subheader("Pick a store and product")
    store = st.selectbox("Store", sorted(df["Store_ID"].unique()))
    product = st.selectbox("Product", sorted(df["Product_ID"].unique()))
    one = df[(df["Store_ID"] == store) & (df["Product_ID"] == product)].set_index("Date")
    st.line_chart(one[["Demand", "Units_Sold", "Inventory_Level"]])

# ---------- Page 3 ----------
else:
    st.write("XGBoost is trained on the first 80% of dates and tested on the last 20%.")
    with st.spinner("Training XGBoost... (first time only, about a minute)"):
        model, test, meta, metrics = train_xgboost()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test MAE", metrics["MAE"])
    c2.metric("Test RMSE", metrics["RMSE"])
    c3.metric("Test R²", metrics["R2"])
    c4.metric("Test MAPE (%)", metrics["MAPE (%)"])

    baseline = float(np.abs(test["Demand"] - test["Demand"].mean()).mean())
    st.info(f"For comparison: always predicting the average demand gives MAE = {baseline:.2f}. "
            "If the model's MAE is not clearly lower, it is not adding value.")

    st.subheader("Actual vs forecast (total of all stores and products)")
    timeline = test.groupby("Date")[["Demand", "Predicted"]].sum()
    st.line_chart(timeline)

    st.subheader("Top 15 important features")
    names = list(model.named_steps["prep"].get_feature_names_out())
    imp = pd.Series(model.named_steps["model"].feature_importances_, index=names)
    top = imp.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)
