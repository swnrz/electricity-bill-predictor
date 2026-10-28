# Electricity Bill Predictor

## Live Demo

Try the Electricity Bill Predictor dashboard online:

**[Open the Live Website](https://electricity-bill-predictor-gao8w33rzksrei6rpg2hs3.streamlit.app/)**
## 1. Project Overview

The Electricity Bill Predictor is a Python-based data analytics and machine-learning project. It analyzes historical household electricity consumption, compares prediction models, forecasts future electricity usage, and estimates electricity costs.

## 2. Objectives

- Analyze historical electricity consumption patterns.
- Clean and prepare household electricity data.
- Train and compare Linear Regression and Decision Tree Regressor models.
- Forecast future electricity consumption.
- Estimate electricity costs using a user-adjustable example rate.
- Present results through an interactive Streamlit dashboard.

## 3. Dataset

The project uses the Individual Household Electric Power Consumption dataset. The data represents electricity measurements from a household in France, recorded from 16 December 2006 to 26 November 2010.

The original file contains 2,075,259 records. After the project's cleaning process, 2,049,280 rows remained. The analysis produced 1,442 daily observations.

## 4. Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## 5. Methodology

1. Load the household electricity dataset.
2. Handle missing values and clean the data.
3. Prepare daily electricity-consumption measurements.
4. Explore historical, weekday, and monthly consumption patterns.
5. Create time-based and previous-consumption prediction features.
6. Split the observations chronologically into training and testing sets.
7. Train Linear Regression and Decision Tree Regressor models.
8. Compare model predictions using Mean Absolute Error (MAE).
9. Generate a future consumption forecast.
10. Estimate electricity costs using an assumed rate.

## 6. Experimental Results

The executed experiment evaluated 287 test observations.

| Model | MAE |
|---|---:|
| Linear Regression | 4.3343 kWh |
| Decision Tree Regressor | 4.4487 kWh |

Linear Regression achieved the lower MAE on this test set. The difference was approximately 0.1144 kWh.

The observed average daily consumption was 26.16 kWh. The dataset's recorded daily values ranged from 4.17 kWh to 79.56 kWh.

These are results from the current experiment and should not be interpreted as a guarantee of future model performance.

## 7. Dashboard Features

- Historical consumption visualization
- Summary statistics
- Future electricity forecast
- Model evaluation and comparison
- Electricity bill estimation

## 8. Limitations

- The dataset represents a household in France and may not represent electricity usage in India.
- The example rate of ₹8/kWh is not an official electricity tariff.
- The bill estimate excludes tariff slabs, fixed charges, taxes, subsidies, and other fees.
- Forecasts are estimates and may differ from actual consumption.
- Model performance is evaluated on the available chronological test set and may change on other data.

## 9. Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the data-processing and model script:

```bash
python main.py
```

Run the interactive dashboard:

```bash
python -m streamlit run app.py
```

Keep the required dataset and generated CSV files in the locations expected by the code.

## 10. Reproducibility

The project saves processed data, model evaluation results, forecast results, and visualization files. Run the main script to regenerate the experimental outputs before reporting updated results.
