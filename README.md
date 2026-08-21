Forecasting Care Load and Discharge Demand in the HHS UAC Program

📌 Overview
This project develops a hybrid forecasting framework to predict care load and discharge demand in the U.S. Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) program. By combining statistical time series models (ARIMA, SARIMAX) with machine learning methods (XGBoost, Gradient Boosting), the system balances interpretability and predictive accuracy.

The forecasting dashboard provides actionable insights for capacity planning, surge preparedness, and policy evaluation.

🎯 Objectives
Forecast daily care load and discharge demand.

Improve robustness by integrating statistical and ML models.

Provide scenario-based forecasts (optimistic/pessimistic).

Enable early surge detection and capacity risk monitoring.

Support policymakers with interpretable and transparent outputs.

📂 Dataset & Preprocessing
Daily time series data: children in care, discharges, custody transfers.

Cleaning: date normalization, numeric conversion, interpolation.

Feature Engineering: lag features, rolling statistics, seasonal decomposition, Fourier terms, policy change indicators.

⚙️ Models Implemented
Baselines: Naive persistence, moving average.

ARIMA / SARIMAX: Captures trend, seasonality, and exogenous drivers.

Regression & Boosting Models: Linear Regression, Gradient Boosting, XGBoost.

Hybrid SARIMAX + XGBoost: Combines statistical rigor with ML flexibility.

📊 Evaluation Metrics
Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Mean Absolute Percentage Error (MAPE)

Symmetric MAPE (SMAPE)

R² (Coefficient of Determination)

📈 Key Insights
Hybrid SARIMAX + XGBoost consistently achieved the lowest RMSE and highest robustness.

Scenario planning (±10% scaling) improves preparedness under uncertainty.

Surge Lead Time metric enables proactive staffing and facility expansion.

Capacity Breach Probability quantifies risk of exceeding thresholds (e.g., 5,000 children in care).

Policy sensitivity analysis highlights structural shifts post-policy changes.

🔍 Recommendations
Adopt Hybrid Models for operational planning.

Integrate Forecasting Dashboard into daily decision-making.

Use Scenario-Based Planning for surge and lull conditions.

Monitor Capacity Thresholds with breach probability alerts.

Evaluate Policy Impacts using structural indicators.

Enhance Data Quality with granular and standardized reporting.

Expand KPI Usage for resilience assessment.

⚖️ Ethical & Policy Considerations
Forecasts must safeguard child welfare and avoid misuse.

Outputs should complement human judgment, not replace it.

Ensure transparency, accountability, and privacy protection.

Promote equitable resource allocation across facilities.

🚀 Future Work
Explore deep learning models (LSTMs, Transformers) for long-horizon forecasting.

Integrate external datasets (immigration trends, border activity, socio-political events).

Extend framework to multi-variate and real-time forecasting.

📚 References
Box et al. (2015). Time Series Analysis: Forecasting and Control.

Hyndman & Athanasopoulos (2018). Forecasting: Principles and Practice.

Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.

Bansak et al. (2018). Refugee Resettlement and Policy Design.

Shah et al. (2021). Hospital Admission Forecasting with ML.
