
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Electricity Bill Predictor",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# 2. DASHBOARD HEADING
# --------------------------------------------------

st.title("⚡ Electricity Bill Predictor")

st.caption("SMART ENERGY • DATA ANALYTICS • MACHINE LEARNING")

st.markdown(
    "Monitor electricity usage, explore future forecasts, "
    "evaluate prediction models, and estimate your electricity bill."
)

st.divider()
st.info(
    "Project Objective: This application analyzes historical "
    "electricity consumption, compares machine-learning models, "
    "forecasts future usage, and estimates electricity bills."
)

# --------------------------------------------------
# 3. FIND PROJECT FILES
# --------------------------------------------------

folder = Path(__file__).parent

daily_file = folder / "daily_electricity_consumption.csv"
forecast_file = folder / "future_electricity_forecast.csv"
results_file = folder / "model_evaluation_results.csv"

# --------------------------------------------------
# 4. SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("⚡ Bill Settings")

rate = st.sidebar.number_input(
    "Electricity rate (₹ per kWh)",
    min_value=0.0,
    value=8.0,
    step=0.5
)

st.sidebar.caption(
    "₹8/kWh is an example rate. Actual tariffs vary by location."
)

# --------------------------------------------------
# 5. LOAD DATA FILES
# --------------------------------------------------

daily = None
forecast = None
results = None

if daily_file.exists():
    daily = pd.read_csv(daily_file)

if forecast_file.exists():
    forecast = pd.read_csv(forecast_file)

if results_file.exists():
    results = pd.read_csv(results_file)

# --------------------------------------------------
# 6. HISTORICAL CONSUMPTION
# --------------------------------------------------

st.header("📊 Historical Consumption")

usage_column = None

if daily is not None:
    st.write("Preview of your daily electricity consumption:")

    st.dataframe(daily.head(10), use_container_width=True)

    numeric_columns = daily.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        usage_column = st.selectbox(
            "Choose the daily consumption column",
            numeric_columns
        )

        st.subheader("Daily Consumption Chart")
        st.line_chart(daily[usage_column])

        st.caption(
            "Check the selected column and its units before interpreting "
            "these values as kWh."
        )
    else:
        st.warning("No numeric consumption column was found.")

    st.write("Columns detected:", ", ".join(daily.columns))

else:
    st.warning(
        "daily_electricity_consumption.csv was not found. "
        "Place it in the same folder as app.py."
    )

# --------------------------------------------------
# 7. ELECTRICITY SUMMARY CARDS
# --------------------------------------------------

st.divider()
st.header("⚡ Electricity at a Glance")

if daily is not None and usage_column is not None:
    average_usage = daily[usage_column].mean()
    total_usage = daily[usage_column].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Daily Value",
        f"{average_usage:,.2f}"
    )

    col2.metric(
        "Total Recorded Value",
        f"{total_usage:,.2f}"
    )

    if forecast is not None:
        col3.metric(
            "Forecast Days",
            f"{len(forecast)}"
        )
    else:
        col3.metric("Forecast Days", "Unavailable")

else:
    st.info(
        "Load the daily consumption CSV to display summary cards."
    )

# --------------------------------------------------
# 8. FUTURE ELECTRICITY FORECAST
# --------------------------------------------------

st.divider()
st.header("🔮 Future Electricity Forecast")

if forecast is not None:
    st.write("Predicted electricity consumption for future days:")

    st.dataframe(forecast, use_container_width=True)

    forecast_numeric = forecast.select_dtypes(
        include="number"
    ).columns.tolist()

    if forecast_numeric:
        forecast_column = st.selectbox(
            "Choose the forecast column to display",
            forecast_numeric,
            key="forecast_column"
        )

        st.subheader("Future Consumption Chart")
        st.line_chart(forecast[forecast_column])

        st.caption(
            "Forecast values come from your saved model output. "
            "Check the selected column and units before interpreting them."
        )

    else:
        st.warning("No numeric forecast columns were found.")

else:
    st.warning(
        "future_electricity_forecast.csv was not found. "
        "Run main.py first to generate the forecast."
    )

# --------------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------------

st.divider()
st.header("🤖 Model Evaluation")

if results is not None:
    st.write("Comparison of the trained prediction models:")
    if {
        "actual_kwh",
        "linear_regression_kwh",
        "decision_tree_kwh"
    }.issubset(results.columns):

        linear_mae = (
            results["actual_kwh"]
            - results["linear_regression_kwh"]
        ).abs().mean()

        tree_mae = (
            results["actual_kwh"]
            - results["decision_tree_kwh"]
        ).abs().mean()

        col1, col2 = st.columns(2)

        col1.metric("Linear Regression MAE", f"{linear_mae:.2f}")
        col2.metric("Decision Tree MAE", f"{tree_mae:.2f}")

        st.success(
            "Better model on this test data: "
            + (
                "Linear Regression"
                if linear_mae < tree_mae
                else "Decision Tree"
                if tree_mae < linear_mae
                else "Both models have equal MAE"
            )
        )
    st.dataframe(results, use_container_width=True)

    st.caption(
        "For Mean Absolute Error (MAE), lower values generally indicate "
        "smaller prediction errors when models use the same target and units."
    )

    st.subheader("Model Evaluation Chart")

    numeric_results = results.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_results:
        metric_column = st.selectbox(
            "Choose a numeric evaluation metric",
            numeric_results,
            key="metric_column"
        )

        st.bar_chart(results[metric_column])

    else:
        st.info("No numeric evaluation metrics were found.")

else:
    st.warning(
        "model_evaluation_results.csv was not found. "
        "Run main.py first to generate model evaluation results."
    )

# --------------------------------------------------
# 10. ELECTRICITY BILL GENERATOR
# --------------------------------------------------

st.divider()
st.header("🧾 Electricity Bill Generator")

st.write(
    "Enter your electricity consumption to estimate your bill."
)

bill_usage = st.number_input(
    "Electricity consumed (kWh)",
    min_value=0.0,
    value=100.0,
    step=10.0
)

bill_amount = bill_usage * rate

col1, col2 = st.columns(2)

col1.metric(
    "Estimated Electricity Bill",
    f"₹{bill_amount:,.2f}"
)

col2.metric(
    "Selected Electricity Rate",
    f"₹{rate:,.2f}/kWh"
)

st.caption(
    "This is a basic estimate: it excludes fixed charges, taxes, "
    "slab rates, subsidies, and other adjustments."
)

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Electricity Bill Predictor | Student Data Analytics Project"
)
