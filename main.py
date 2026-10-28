
# ============================================================
# ELECTRICITY BILL PREDICTOR
# Required model: Linear Regression
# Optional model: Decision Tree Regressor
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

DATA_FILE = "household_power_consumption.txt"

# Example rate only; change it to the rate you want to demonstrate.
RATE_PER_KWH = 8.0

TEST_SIZE = 0.20
FORECAST_DAYS = 30

print("=" * 60)
print("             ELECTRICITY BILL PREDICTOR")
print("=" * 60)


# ============================================================
# 2. LOAD DATA AND CONVERT TXT TO CSV
# ============================================================

print("\nSTEP 1: Loading the dataset...")

if not os.path.exists(DATA_FILE):
    print("\nERROR: Dataset file not found.")
    print("Place household_power_consumption.txt beside main.py.")
    raise SystemExit(1)

data = pd.read_csv(
    DATA_FILE,
    sep=";",
    na_values=["?"],
    low_memory=False
)

# Save a CSV copy of the original data.
# Missing values are represented as empty CSV fields.
data.to_csv("household_power_consumption.csv", index=False)

print("Dataset loaded successfully!")
print("Converted TXT file to household_power_consumption.csv")
print("Original rows:", len(data))
print("Columns:", list(data.columns))


# ============================================================
# 3. CLEAN THE DATA
# ============================================================

print("\nSTEP 2: Cleaning the dataset...")

# Convert important columns to numeric values.
numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for column in numeric_columns:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

# Combine Date and Time into one datetime column.
data["datetime"] = pd.to_datetime(
    data["Date"].astype(str) + " " + data["Time"].astype(str),
    dayfirst=True,
    errors="coerce"
)

# Remove rows with missing power readings or invalid dates.
data = data.dropna(
    subset=["datetime", "Global_active_power"]
)

# Sort records by time.
data = data.sort_values("datetime")

# Remove duplicate timestamps, if any.
data = data.drop_duplicates(subset=["datetime"])

print("Rows after cleaning:", len(data))
print("Missing values in active power:",
      data["Global_active_power"].isna().sum())


# ============================================================
# 4. CALCULATE DAILY ELECTRICITY CONSUMPTION
# ============================================================

print("\nSTEP 3: Calculating daily electricity consumption...")

data = data.set_index("datetime")

# Global_active_power is measured in kilowatts (kW).
# Each original record represents one minute.
#
# Average power (kW) x 24 hours = estimated daily energy (kWh).
#
# Require at least 1,000 valid minute readings per day so
# severely incomplete days are not treated as full days.

daily_mean_power = (
    data["Global_active_power"]
    .resample("D")
    .mean()
)

daily_reading_count = (
    data["Global_active_power"]
    .resample("D")
    .count()
)

daily_usage = daily_mean_power * 24

daily_usage = daily_usage.where(
    daily_reading_count >= 1000
)

# Ensure the date index contains every calendar day.
# Interpolate missing daily estimates so lag features represent
# actual calendar days rather than merely the previous row.
daily_usage = daily_usage.asfreq("D")

daily_usage = daily_usage.interpolate(
    method="time",
    limit_direction="both"
)

daily_usage = daily_usage.dropna()

if len(daily_usage) < 30:
    print("Not enough daily data to train the models.")
    raise SystemExit(1)

daily_usage.name = "daily_kwh"

print("Daily data prepared!")
print("First recorded day:", daily_usage.index.min().date())
print("Last recorded day:", daily_usage.index.max().date())
print("Number of daily observations:", len(daily_usage))
print(f"Average daily consumption: {daily_usage.mean():.2f} kWh")
print(f"Maximum daily consumption: {daily_usage.max():.2f} kWh")
print(f"Minimum daily consumption: {daily_usage.min():.2f} kWh")

# Save daily consumption data.
daily_usage.to_csv(
    "daily_electricity_consumption.csv",
    header=True
)


# ============================================================
# 5. EXPLORE HISTORICAL CONSUMPTION PATTERNS
# ============================================================

print("\nSTEP 4: Exploring consumption patterns...")

# A. Daily consumption over time.
plt.figure(figsize=(12, 5))
plt.plot(
    daily_usage.index,
    daily_usage.values,
    linewidth=0.8
)
plt.title("Historical Daily Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Daily consumption (kWh)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("electricity_history.png", dpi=150)
plt.close()

# B. Average usage for each day of the week.
weekday_average = daily_usage.groupby(
    daily_usage.index.dayofweek
).mean()

weekday_names = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

plt.figure(figsize=(10, 5))
plt.bar(
    [weekday_names[int(day)] for day in weekday_average.index],
    weekday_average.values
)
plt.title("Average Electricity Consumption by Weekday")
plt.xlabel("Day of the week")
plt.ylabel("Average consumption (kWh)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("weekday_consumption.png", dpi=150)
plt.close()

# C. Average usage by month number.
monthly_pattern = daily_usage.groupby(
    daily_usage.index.month
).mean()

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(10, 5))
plt.bar(
    [month_names[int(month) - 1]
     for month in monthly_pattern.index],
    monthly_pattern.values
)
plt.title("Average Electricity Consumption by Month")
plt.xlabel("Month")
plt.ylabel("Average daily consumption (kWh)")
plt.tight_layout()
plt.savefig("monthly_consumption.png", dpi=150)
plt.close()

print("\nAverage consumption by weekday:")
for day, value in weekday_average.items():
    print(f"{weekday_names[int(day)]}: {value:.2f} kWh")

print("\nExploration graphs saved:")
print("- electricity_history.png")
print("- weekday_consumption.png")
print("- monthly_consumption.png")


# ============================================================
# 6. CREATE MACHINE LEARNING FEATURES
# ============================================================

print("\nSTEP 5: Creating prediction features...")

def create_features(series):
    """Build input features and the daily consumption target."""

    frame = pd.DataFrame(index=series.index)

    # Electricity consumption on the previous calendar day.
    frame["previous_day"] = series.shift(1)

    # Electricity consumption seven calendar days earlier.
    frame["previous_week"] = series.shift(7)

    # Calendar information.
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month

    # Value the model must learn to predict.
    frame["target"] = series

    return frame.dropna()


model_data = create_features(daily_usage)

feature_columns = [
    "previous_day",
    "previous_week",
    "day_of_week",
    "month"
]

X = model_data[feature_columns]
y = model_data["target"]

# Split chronologically: earlier data for training,
# later data for testing. Do not shuffle time-series data.
split_index = int(len(model_data) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

if len(X_train) == 0 or len(X_test) == 0:
    print("Not enough data for the training/testing split.")
    raise SystemExit(1)

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Chronological split completed.")


# ============================================================
# 7. TRAIN LINEAR REGRESSION
# ============================================================

print("\nSTEP 6: Training Linear Regression...")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = np.maximum(
    linear_model.predict(X_test),
    0
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

print("Linear Regression trained successfully!")


# ============================================================
# 8. OPTIONAL: TRAIN A DECISION TREE REGRESSOR
# ============================================================

print("\nSTEP 7: Training Decision Tree Regressor...")

tree_model = DecisionTreeRegressor(
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_predictions = np.maximum(
    tree_model.predict(X_test),
    0
)

tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)

print("Decision Tree trained successfully!")


# ============================================================
# 9. EVALUATE AND COMPARE MODELS USING MAE
# ============================================================

print("\nSTEP 8: Comparing model performance...")

print("\nMODEL EVALUATION")
print("-" * 45)
print(f"Linear Regression MAE: {linear_mae:.3f} kWh")
print(f"Decision Tree MAE:     {tree_mae:.3f} kWh")

if linear_mae < tree_mae:
    print("Lower test MAE: Linear Regression")
elif tree_mae < linear_mae:
    print("Lower test MAE: Decision Tree Regressor")
else:
    print("Both models have equal test MAE.")

print(
    "\nMAE is the average absolute difference between "
    "predicted and actual daily consumption."
)

# Save test predictions for inspection.
evaluation_results = pd.DataFrame({
    "actual_kwh": y_test,
    "linear_regression_kwh": linear_predictions,
    "decision_tree_kwh": tree_predictions
})

evaluation_results.to_csv(
    "model_evaluation_results.csv",
    index_label="date"
)

# Plot actual versus predicted results for both models.
plt.figure(figsize=(13, 6))

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual consumption",
    linewidth=1
)

plt.plot(
    y_test.index,
    linear_predictions,
    label="Linear Regression",
    linewidth=1
)

plt.plot(
    y_test.index,
    tree_predictions,
    label="Decision Tree",
    linewidth=1
)

plt.title("Actual vs Predicted Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Daily consumption (kWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

print("Evaluation graph saved: actual_vs_predicted.png")


# ============================================================
# 10. REFIT LINEAR REGRESSION ON ALL HISTORICAL DATA
# ============================================================

print("\nSTEP 9: Preparing the final forecasting model...")

# The test set was used for evaluation above.
# Now retrain on all available history for future forecasts.

final_model = LinearRegression()
final_model.fit(X, y)

print("Final Linear Regression model is ready.")


# ============================================================
# 11. FORECAST FUTURE ELECTRICITY CONSUMPTION
# ============================================================

print("\nSTEP 10: Forecasting the next 30 days...")

history = daily_usage.copy()
forecast_records = []

for _ in range(FORECAST_DAYS):

    # Predict the calendar day after the latest available day.
    forecast_date = history.index[-1] + pd.Timedelta(days=1)

    previous_day_date = forecast_date - pd.Timedelta(days=1)
    previous_week_date = forecast_date - pd.Timedelta(days=7)

    # Because history has a daily index, these dates should exist.
    previous_day_value = history.loc[previous_day_date]
    previous_week_value = history.loc[previous_week_date]

    new_features = pd.DataFrame(
        [[
            previous_day_value,
            previous_week_value,
            forecast_date.dayofweek,
            forecast_date.month
        ]],
        columns=feature_columns
    )

    predicted_kwh = float(
        final_model.predict(new_features)[0]
    )

    # Avoid physically impossible negative usage predictions.
    predicted_kwh = max(0.0, predicted_kwh)

    forecast_records.append({
        "date": forecast_date,
        "predicted_kwh": predicted_kwh,
        "estimated_cost_rupees": predicted_kwh * RATE_PER_KWH
    })

    # Add the predicted value to history so the next forecast
    # can use it as a previous-day or previous-week feature.
    history.loc[forecast_date] = predicted_kwh


forecast_df = pd.DataFrame(forecast_records)
forecast_df = forecast_df.set_index("date")

forecast_df.to_csv("future_electricity_forecast.csv")

print("\nNEXT 7 DAYS FORECAST")
print("-" * 60)
print(forecast_df.head(7).round(2).to_string())


# ============================================================
# 12. ESTIMATE FUTURE ELECTRICITY BILLS
# ============================================================

print("\nSTEP 11: Estimating electricity bills...")

next_7_days_kwh = forecast_df["predicted_kwh"].head(7).sum()
next_7_days_bill = next_7_days_kwh * RATE_PER_KWH

next_30_days_kwh = forecast_df["predicted_kwh"].sum()
next_30_days_bill = next_30_days_kwh * RATE_PER_KWH

print("\nBILL ESTIMATE")
print("-" * 45)
print(f"Assumed rate: Rs. {RATE_PER_KWH:.2f} per kWh")
print(f"Next 7 days consumption: {next_7_days_kwh:.2f} kWh")
print(f"Next 7 days estimated bill: Rs. {next_7_days_bill:.2f}")
print(f"Next 30 days consumption: {next_30_days_kwh:.2f} kWh")
print(f"Next 30 days estimated bill: Rs. {next_30_days_bill:.2f}")

# Plot the next seven days.
plt.figure(figsize=(10, 5))

plt.plot(
    forecast_df.index[:7],
    forecast_df["predicted_kwh"].iloc[:7],
    marker="o"
)

plt.title("Next 7 Days Electricity Consumption Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted consumption (kWh)")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("future_forecast.png", dpi=150)
plt.close()

print("\nForecast graph saved: future_forecast.png")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("                   PROJECT COMPLETE")
print("=" * 60)

print("\nFiles created:")
print("- household_power_consumption.csv")
print("- daily_electricity_consumption.csv")
print("- model_evaluation_results.csv")
print("- future_electricity_forecast.csv")
print("- electricity_history.png")
print("- weekday_consumption.png")
print("- monthly_consumption.png")
print("- actual_vs_predicted.png")
print("- future_forecast.png")

print("\nModels:")
print("- Linear Regression")
print("- Decision Tree Regressor")

print("\nImportant limitations:")
print("- The dataset represents a household in France.")
print("- The electricity rate is an assumed example rate.")
print("- The estimated bill excludes actual tariff slabs.")
print("  fixed charges, taxes and other utility fees.")
print("- Future forecasts are estimates, not guarantees.")

print("\nThank you for using the Electricity Bill Predictor!")

# Verify the saved experimental results
from pathlib import Path
import pandas as pd

results_file = Path(__file__).parent / "model_evaluation_results.csv"

results = pd.read_csv(results_file)

required_columns = [
    "actual_kwh",
    "linear_regression_kwh",
    "decision_tree_kwh"
]

if not set(required_columns).issubset(results.columns):
    raise ValueError("Required columns are missing from the results CSV.")

results = results[required_columns].dropna()

linear_mae = (
    results["actual_kwh"] - results["linear_regression_kwh"]
).abs().mean()

tree_mae = (
    results["actual_kwh"] - results["decision_tree_kwh"]
).abs().mean()

print("\n--- Experimental Results ---")
print(f"Test records evaluated: {len(results)}")
print(f"Linear Regression MAE: {linear_mae:.4f} kWh")
print(f"Decision Tree MAE: {tree_mae:.4f} kWh")

if linear_mae < tree_mae:
    print("Lower MAE: Linear Regression")
elif tree_mae < linear_mae:
    print("Lower MAE: Decision Tree")
else:
    print("Both models have equal MAE.")
