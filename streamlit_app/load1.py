import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------
# Utility Functions
# -------------------------------
def fix_missing_dates(df, date_col='Date', method='time'):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.groupby(df.index).mean(numeric_only=True)

    full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_range)

    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method=method)

    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill()

    df = df.loc[:df.dropna(how='all').index.max()]
    return df

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100,
        "SMAPE": np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100,
        "R²": r2_score(y_true, y_pred)
    }

# -------------------------------
# Extra Feature Engineering
# -------------------------------
def add_seasonal_features(df, target_col="Children discharged from HHS Care", period=7):
    result = seasonal_decompose(df[target_col], model="additive", period=period)
    df[f"{target_col}_trend"] = result.trend
    df[f"{target_col}_seasonal"] = result.seasonal
    df[f"{target_col}_resid"] = result.resid
    df = df.bfill().ffill()
    return df

def add_fourier_terms(df, target_col="Children discharged from HHS Care", K=3, period=7):
    t = np.arange(len(df))
    for k in range(1, K+1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df

# -------------------------------
# Forecasting Pipeline
# -------------------------------
def run_pipeline(df, target, label_icon, global_horizon, model_override, scenario, ci_width):
    st.header(f"{label_icon} Forecasting {target}")

    # Feature Engineering
    df[f'{target}_lag1'] = df[target].shift(1)
    df[f'{target}_lag7'] = df[target].shift(7)
    df[f'{target}_rolling_mean_7'] = df[target].rolling(7).mean()
    df[f'{target}_rolling_var_7'] = df[target].rolling(7).var()

    for col in ["Children in CBP custody","Children transferred out of CBP custody"]:
        df[f"{col}_lag7"] = df[col].shift(7)

    if target != "Children discharged from HHS Care":
        df['net_pressure'] = df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']
    else:
        df['net_pressure'] = df['Children transferred out of CBP custody']
        df = add_seasonal_features(df, target_col=target, period=7)
        df = add_fourier_terms(df, target_col=target, K=3, period=7)

    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['policy_change'] = (df.index >= "2025-01-01").astype(int)

    # Train/Test Split
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    features = [col for col in df.columns if col != target]
    exog_train = train[features].replace([np.inf, -np.inf], np.nan).bfill().ffill()
    exog_test  = test[features].replace([np.inf, -np.inf], np.nan).bfill().ffill()

    X_train, y_train = exog_train, train[target]
    X_test, y_test   = exog_test, test[target]

    results = {}

    # Baselines
    naive_pred = [y_train.iloc[-1]]*len(y_test)
    results["Naive"] = evaluate(y_test, naive_pred)

    ma_pred = [y_train.rolling(7).mean().iloc[-1]]*len(y_test)
    results["Moving Average"] = evaluate(y_test, ma_pred)

    # ARIMA grid search
    best_arima = None
    best_rmse = np.inf
    for p in [1,2]:
        for d in [1]:
            for q in [1,2]:
                try:
                    arima = ARIMA(y_train, order=(p,d,q)).fit()
                    pred = arima.forecast(len(y_test))
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_arima = arima
                except:
                    pass
    if best_arima:
        results["ARIMA"] = evaluate(y_test, best_arima.forecast(len(y_test)))

    # SARIMAX with exog
    try:
        sarimax = SARIMAX(y_train, exog=X_train, order=(1,0,1), seasonal_order=(1,1,1,7)).fit(disp=False, method='powell', maxiter=200)
        sarimax_pred = sarimax.forecast(steps=len(y_test), exog=X_test)
        results["Exog_SARIMAX"] = evaluate(y_test, sarimax_pred)
    except Exception as e:
        st.warning(f"SARIMAX failed: {e}")

    # ML Models
    lr = LinearRegression().fit(X_train, y_train)
    results["Linear Regression"] = evaluate(y_test, lr.predict(X_test))

    xgb = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=500, subsample=0.8)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = evaluate(y_test, xgb_pred)

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    gb.fit(X_train, y_train)
    results["GradientBoosting"] = evaluate(y_test, gb.predict(X_test))

    # Hybrid SARIMAX + XGBoost
    try:
        hybrid_pred = (sarimax_pred + xgb_pred) / 2
        results["Hybrid SARIMAX + XGBoost"] = evaluate(y_test, hybrid_pred)
    except Exception as e:
        st.warning(f"Hybrid model failed: {e}")

    # Results Table
    st.subheader("Model Performance")
    performance_df = pd.DataFrame(results).T
    st.write(performance_df)

    # Best model selection
    best_model_name = performance_df['RMSE'].idxmin()
    if model_override != "Auto":
        best_model_name = model_override
    st.success(f"✅ Best model selected for {target}: {best_model_name}")

    # Forecast
    horizon = global_horizon
    y_test_horizon = y_test.iloc[:horizon]
    future_forecast = None
    if best_model_name == "ARIMA":
        future_forecast = best_arima.forecast(steps=horizon)
    elif best_model_name == "Exog_SARIMAX":
        future_forecast = sarimax.forecast(steps=horizon, exog=X_test.iloc[:horizon])
    elif best_model_name == "Linear Regression":
        future_forecast = lr.predict(X_test.iloc[:horizon])
    elif best_model_name == "XGBoost":
        future_forecast = xgb.predict(X_test.iloc[:horizon])
    elif best_model_name == "GradientBoosting":
        future_forecast = gb.predict(X_test.iloc[:horizon])
    elif best_model_name == "Hybrid SARIMAX + XGBoost":
        sarimax_future = sarimax.forecast(steps=horizon, exog=X_test.iloc[:horizon])
        xgb_future = xgb.predict(X_test.iloc[:horizon])
        future_forecast = (sarimax_future + xgb_future) / 2
        
    # Scenario scaling
    if future_forecast is not None:
        if scenario == "Optimistic":
            future_forecast = future_forecast * 0.9
        elif scenario == "Pessimistic":
            future_forecast = future_forecast * 1.1

        # Plot Forecast with Confidence Interval
        fig, ax = plt.subplots(figsize=(12,5))
        y_test.plot(ax=ax, label="Actual")
        future_index = pd.date_range(y_test.index[-1], periods=horizon+1, freq='D')[1:]
        ax.plot(future_index, future_forecast, label=f"{best_model_name} Forecast ({scenario})")
        ax.fill_between(
            future_index,
            future_forecast * (1 - ci_width/100),
            future_forecast * (1 + ci_width/100),
            alpha=0.2,
            label=f"{ci_width}% Confidence Interval"
        )
        ax.legend()
        ax.set_title(f"Forecast for {target}")
        st.pyplot(fig)
    return y_test_horizon, future_forecast, best_model_name,results
 

def plot_actual_vs_predicted(y_test, y_pred, best_model_name, scenario, ci_width, target):
    """
    Plot Actual vs Predicted values with confidence interval shading.

    Parameters:
    - y_test: pandas Series of actual values (with DateTime index)
    - y_pred: array-like of predicted values (same length as y_test)
    - best_model_name: string, name of the model used
    - scenario: string, scenario label ("Baseline", "Optimistic", "Pessimistic")
    - ci_width: int, confidence interval percentage (e.g., 10 for ±10%)
    - target: string, target variable name
    """
    fig, ax = plt.subplots(figsize=(12,5))

    # Actual values
    y_test.plot(ax=ax, label="Actual", color="blue")

    # Predicted values
    ax.plot(y_test.index, y_pred, label=f"{best_model_name} Predicted ({scenario})", color="orange")

    # Confidence interval shading
    ax.fill_between(
        y_test.index,
        y_pred * (1 - ci_width/100),
        y_pred * (1 + ci_width/100),
        alpha=0.2,
        color="orange",
        label=f"{ci_width}% Confidence Interval"
    )

    ax.legend()
    ax.set_title(f"Actual vs Predicted for {target}")
    st.pyplot(fig)

def compare_with_baseline(results):
    """
    Compare each model's performance against baseline models (Naive, Moving Average).

    Parameters:
    - results: dict of model evaluation metrics (from run_pipeline)

    Returns:
    - comparison_df: DataFrame showing relative improvement over baselines
    """
    # Convert results dict to DataFrame
    df = pd.DataFrame(results).T

    # Baseline metrics
    baseline_mae = df.loc["Naive", "MAE"]
    baseline_rmse = df.loc["Naive", "RMSE"]

    comparisons = {}
    for model in df.index:
        if model not in ["Naive", "Moving Average"]:
            comparisons[model] = {
                "MAE vs Naive (%)": (baseline_mae - df.loc[model, "MAE"]) / baseline_mae * 100,
                "RMSE vs Naive (%)": (baseline_rmse - df.loc[model, "RMSE"]) / baseline_rmse * 100,
                "MAE vs Moving Avg (%)": (df.loc["Moving Average", "MAE"] - df.loc[model, "MAE"]) / df.loc["Moving Average", "MAE"] * 100,
                "RMSE vs Moving Avg (%)": (df.loc["Moving Average", "RMSE"] - df.loc[model, "RMSE"]) / df.loc["Moving Average", "RMSE"] * 100,
            }

    comparison_df = pd.DataFrame(comparisons).T
    return comparison_df

def calculate_kpis(y_test, y_pred, results, capacity_threshold=None):
    kpis = {}

    # Forecast Accuracy (%) → from MAPE
    best_model = min(results, key=lambda m: results[m]["RMSE"])
    best_mape = results[best_model]["MAPE"]
    kpis["Forecast Accuracy (%)"] = 100 - best_mape

    # Surge Lead Time → days before actual surge detected
    surge_lead_time = None
    if (y_pred is not None) and (y_test is not None):
        surge_threshold = y_test.mean() + 2 * y_test.std()
        surge_days = (y_pred > surge_threshold).nonzero()[0]
        if len(surge_days) > 0:
            surge_lead_time = surge_days[0]  # first day of surge
    kpis["Surge Lead Time (days)"] = surge_lead_time if surge_lead_time is not None else "N/A"

    # Capacity Breach Probability
    if capacity_threshold:
        breach_prob = (y_pred > capacity_threshold).mean()
        kpis["Capacity Breach Probability"] = breach_prob
    else:
        kpis["Capacity Breach Probability"] = "Threshold not set"

    # Forecast Stability Index → variance across models
    rmse_values = [results[m]["RMSE"] for m in results]
    stability_index = 1 / (np.std(rmse_values) + 1e-6)
    kpis["Forecast Stability Index"] = stability_index

    # Model Robustness → sensitivity to scenario scaling
    optimistic = y_pred * 0.9
    pessimistic = y_pred * 1.1
    robustness = 1 - (np.std([optimistic.mean(), pessimistic.mean(), y_pred.mean()]) / y_pred.mean())
    kpis["Model Robustness"] = robustness

    return kpis



# -------------------------------
# Main App
# -------------------------------
st.set_page_config(page_title="HHS_Unaccompanied_Alien_Children_Program Forecasting Dashboard", layout="wide")
st.title("📊 HHS_Unaccompanied_Alien_Children_Program Forecasting Dashboard")

# Sidebar Controls
st.sidebar.header("User Controls")
global_horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30)
model_override = st.sidebar.selectbox("Force Model Selection", 
                                      ["Auto", "ARIMA", "Exog_SARIMAX", "Linear Regression", 
                                       "XGBoost", "GradientBoosting", "Hybrid SARIMAX + XGBoost"])
scenario = st.sidebar.radio("Scenario", ["Baseline", "Optimistic", "Pessimistic"])
ci_width = st.sidebar.slider("Confidence Interval (%)", min_value=5, max_value=30, value=10)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    # Clean numeric columns
    df['Children in HHS Care'] = pd.to_numeric(
        df['Children in HHS Care'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df['Children apprehended and placed in CBP custody*'] = pd.to_numeric(
        df['Children apprehended and placed in CBP custody*'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df['Children in CBP custody'] = pd.to_numeric(
        df['Children in CBP custody'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df['Children transferred out of CBP custody'] = pd.to_numeric(
        df['Children transferred out of CBP custody'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df['Children discharged from HHS Care'] = pd.to_numeric(
        df['Children discharged from HHS Care'].astype(str).str.replace(',', ''), errors='coerce'
    )

    # Fix missing dates
    df = fix_missing_dates(df, date_col='Date')

    # Tabs for core modules
    tab1, tab2, tab3, tab4,tab5 = st.tabs([
        "Future Care Load Forecast",
        "Discharge Demand Forecast",
        "Model Comparison",
        "Confidence Interval",
        "KPI Dashboard"
    ])

    with tab1:
        run_pipeline(df, target="Children in HHS Care", label_icon="🏠", 
                     global_horizon=global_horizon, model_override=model_override, 
                     scenario=scenario, ci_width=ci_width)

    with tab2:
        run_pipeline(df, target="Children discharged from HHS Care", label_icon="🚪", 
                     global_horizon=global_horizon, model_override=model_override, 
                     scenario=scenario, ci_width=ci_width)

    with tab3:
        st.subheader("⚙️ Model Selection & Comparison")
        st.write("Compare ARIMA, SARIMAX, Regression, Boosting, Hybrid models side by side.")
        y_test_horizon, future_forecast, best_model_name, results = run_pipeline(
            df,
            target="Children in HHS Care",
            label_icon="🏠",
            global_horizon=global_horizon,
            model_override=model_override,
            scenario=scenario,
            ci_width=ci_width
        )
        comparison_df = compare_with_baseline(results)
        st.write(comparison_df)

    with tab4:
        st.subheader("📉 Confidence Interval Visualization")
        st.write("Visualize forecast uncertainty bands for chosen models.")

    # Example: run pipeline for one target, get predictions, then plot
    # You already compute y_test and y_pred inside run_pipeline.
    # To reuse them here, you can either:
    # (a) return y_test and y_pred from run_pipeline, or
    # (b) recompute them here for the chosen target.

    # Option (a): modify run_pipeline to return y_test, y_pred, best_model_name
        y_test_horizon, future_forecast, best_model_name, results = run_pipeline(
            df,
            target="Children in HHS Care",
            label_icon="🏠",
            global_horizon=global_horizon,
            model_override=model_override,
            scenario=scenario,
            ci_width=ci_width
        )

    # Then call plotting function
        plot_actual_vs_predicted(y_test_horizon, y_pred=future_forecast, best_model_name=best_model_name, scenario=scenario, ci_width=ci_width, target="Children in HHS Care")

    with tab5:
        st.subheader("📈 Key Performance Indicators (KPIs)")
        y_test_horizon, future_forecast, best_model_name, results = run_pipeline(
            df,
            target="Children in HHS Care",
            label_icon="🏠",
            global_horizon=global_horizon,
            model_override=model_override,
            scenario=scenario,
            ci_width=ci_width
        )

        kpis = calculate_kpis(y_test_horizon, future_forecast, results, capacity_threshold=5000)  # example threshold
        for kpi, value in kpis.items():
            st.metric(kpi, f"{value:.2f}" if isinstance(value, (int, float)) else value)