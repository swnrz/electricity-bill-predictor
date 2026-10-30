# Electricity Bill Predictor

## Machine-Learning-Based Household Energy Consumption Forecasting and Bill Estimation



**Project report — Singularity submission**  

**Date:** 30 October 2026



- **GitHub repository:** https://github.com/swnrz/electricity-bill-predictor

- **Live demo:** https://electricity-bill-predictor-gao8w33rzksrei6rpg2hs3.streamlit.app/


---



## Abstract



This project develops a Python application that analyzes historical household electricity consumption, forecasts future daily usage, compares regression models, and estimates electricity bills. It uses the Individual Household Electric Power Consumption dataset, daily aggregation, date-derived and historical-usage features, Linear Regression, and a Decision Tree Regressor comparison. A Streamlit dashboard presents historical consumption, forecast outputs, model evaluation, and an interactive bill estimator.



On the reported chronological test set of 287 daily observations, Linear Regression achieved a Mean Absolute Error (MAE) of **4.3343 kWh**, while the Decision Tree Regressor achieved **4.4487 kWh**. Linear Regression therefore had the lower MAE in this experiment. These are results recorded from the executed project; they have not been independently rerun for this report.



## 1. Project Objective and Scope



The objective is to forecast daily electricity consumption from historical usage data and use the forecast to estimate a future electricity bill. The project also explores consumption patterns, compares model performance using MAE, and presents results through a web dashboard.



The application is designed to:

1. Analyze historical electricity consumption.

2. Summarize consumption statistics and patterns.

3. Compare machine-learning regression models.

4. Forecast future electricity usage.

5. Estimate an electricity bill using an assumed per-unit rate.

6. Present results through a Streamlit web application.



## 2. Dataset and Preprocessing



The project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository. It contains household electric power measurements collected in France.



### Recorded dataset summary



| Dataset item | Recorded project detail |

|---|---:|

| Original records | 2,075,259 |

| Records after cleaning | 2,049,280 |

| Daily observations | 1,442 |

| Date range | 16 December 2006 – 26 November 2010 |

| Mean daily consumption | 26.16 kWh |

| Minimum daily consumption | 4.17 kWh |

| Maximum daily consumption | 79.56 kWh |



The preprocessing workflow converts the source data into a CSV-compatible format, handles missing or invalid readings, and aggregates the data into daily observations. The exact cleaning rules, column transformations, and excluded rows should be checked against \`main.py\` before a technical review.



## 3. Mathematical Formulas and Calculations



The formulas below explain the main mathematical ideas relevant to the project. Where the implementation details depend on the code, \`main.py\` is the source of truth.



### 3.1 Power and energy consumption



Energy is power multiplied by time:



\\[

E = P \times t

\\]



Where:

- \(E\) = energy in kilowatt-hours (kWh)

- \(P\) = power in kilowatts (kW)

- \(t\) = time in hours



For minute-level power readings, a daily energy calculation can be expressed as:



\\[

E\_{\text{day}} = \sum\_{i=1}^{N} P_i \times \frac{1}{60}

\\]



Here, \(P_i\) is the power reading in kW for minute \(i\), and \(N\) is the number of valid minute readings in that day. The factor \(1/60\) converts minutes to hours. Confirm that this conversion matches the exact source column and aggregation used in \`main.py\`.



### 3.2 Daily aggregation and average



Daily total consumption is the sum of the valid energy values assigned to a calendar day:



\\[

E\_{\text{day}} = \sum\_{i=1}^{N} E_i

\\]



Average daily consumption over \(D\) days is:



\\[

\bar{E} = \frac{\sum\_{d=1}^{D} E_d}{D}

\\]



The reported average daily consumption is 26.16 kWh.



### 3.3 Linear Regression prediction



\\[

\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_mx_m

\\]



Where:

- \(\hat{y}\) = predicted daily electricity consumption

- \(x_1,\ldots,x_m\) = input features

- \(\beta_0\) = intercept

- \(\beta_1,\ldots,\beta_m\) = learned model coefficients



The exact input features are determined by the feature-construction code.



### 3.4 Linear Regression least-squares objective



Linear Regression learns coefficients by minimizing the sum of squared errors:



\\[

\min\_{\beta_0,\ldots,\beta_m}

\sum\_{i=1}^{n}(y_i-\hat{y}\_i)^2

\\]



Here, \(y_i\) is the actual training value and \(\hat{y}\_i\) is the model prediction.



### 3.5 Decision Tree Regressor prediction



A Decision Tree Regressor divides the feature space into regions using learned decision rules. A common regression prediction at a terminal leaf is the mean of the training target values in that leaf:



\\[

\hat{y}(x) =

\frac{1}{N\_{\text{leaf}}}

\sum\_{i \in \text{leaf}(x)} y_i

\\]



The exact tree structure is learned from the training data.



### 3.6 Mean Absolute Error (MAE)



\\[

MAE = \frac{1}{n}\sum\_{i=1}^{n}|y_i-\hat{y}\_i|

\\]



Where:

- \(y_i\) = actual consumption

- \(\hat{y}\_i\) = predicted consumption

- \(n\) = number of test observations



The MAE unit is kWh because the target variable is daily electricity consumption. A lower MAE indicates a smaller average absolute prediction error on the evaluated test set.



### 3.7 Total consumption over a forecast horizon



\\[

E\_{\text{forecast}} = \sum\_{d=1}^{h}\hat{E}\_d

\\]



Where \(h\) is the number of forecast days and \(\hat{E}\_d\) is the predicted consumption for day \(d\).



### 3.8 Electricity bill estimation



\\[

\text{Estimated Bill (₹)}

\=

\text{Forecast Consumption (kWh)}

\times

\text{Rate (₹/kWh)}

\\]



For the illustrative rate used in the project:



\\[

\text{Estimated Bill} = \text{Forecast kWh} \times ₹8

\\]



This is a simplified estimate, not an official electricity tariff calculation.



### 3.9 Difference between model MAE values



Absolute difference:



\\[

\Delta MAE = MAE\_{\text{Decision Tree}} - MAE\_{\text{Linear Regression}}

\\]



Relative reduction compared with the Decision Tree:



\\[

\text{Relative reduction (\\%)}

\=

\frac{MAE\_{\text{Decision Tree}}-MAE\_{\text{Linear Regression}}}

{MAE\_{\text{Decision Tree}}}

\times 100

\\]



Using the recorded values:



\\[

4.4487 - 4.3343 = 0.1144\text{ kWh}

\\]



This comparison applies only to the reported test set.



## 4. Exploratory Analysis and Feature Engineering



Daily aggregation aligns the data with the objective of forecasting daily consumption. The project generates visual outputs for:

- Historical electricity consumption

- Weekday consumption patterns

- Monthly consumption patterns

- Actual-versus-predicted values

- Future electricity forecasts



Feature engineering uses date information and previous usage information to represent temporal patterns. Date-derived features can encode calendar structure, while lagged usage features represent recent consumption history. The exact feature columns and lag definitions should be checked in \`main.py\`.



## 5. Model Development and Experimental Evaluation



### Linear Regression



Linear Regression is a supervised regression algorithm that models the target as a weighted linear combination of input features. It is used to estimate daily electricity consumption.



### Decision Tree Regressor



A Decision Tree Regressor learns decision rules by splitting the feature space into regions and predicting a value for each region. It is included as a comparison model.



### Train-test split



The recorded run used:

- **Training observations:** 1,148

- **Testing observations:** 287



The project uses a chronological split: earlier observations are used for training and later observations for testing. This is suitable for time-dependent data because the model is evaluated on observations that occur after the training period.



### Experimental results



| Model | Test observations | MAE |

|---|---:|---:|

| Linear Regression | 287 | 4.3343 kWh |

| Decision Tree Regressor | 287 | 4.4487 kWh |



**Finding:** Linear Regression achieved the lower MAE by 0.1144 kWh on this particular test set. This does not prove that it will outperform the Decision Tree on every dataset or future period.



## 6. Forecasting and Bill Estimation Results



The project run reported the following forecast totals:



| Forecast horizon | Forecast consumption | Assumed rate | Recorded estimated bill |

|---|---:|---:|---:|

| 7 days | 191.05 kWh | ₹8/kWh | ₹1,528.38 |

| 30 days | 812.30 kWh | ₹8/kWh | ₹6,498.44 |



The displayed 7-day consumption of 191.05 kWh multiplied by ₹8 gives ₹1,528.40. The program's recorded bill is ₹1,528.38, which can occur because the calculation uses an unrounded forecast value while the displayed consumption is rounded. Use the exact values produced by the program when reproducing the result.



The ₹8/kWh rate is illustrative, not an official tariff. Actual bills may include slab rates, fixed charges, taxes, subsidies, meter charges, and other adjustments.



## 7. Limitations and Responsible Use



1. **Geographic representativeness:** The source dataset represents household usage in France and may not represent households in India or other regions.

2. **Tariff simplification:** The bill calculator uses an illustrative flat rate of ₹8/kWh and does not reproduce a real utility tariff structure.

3. **Forecast uncertainty:** Future usage depends on weather, occupancy, appliances, lifestyle, and other factors that may not be represented in the model features.

4. **Evaluation scope:** The reported MAE values apply to the specific chronological test split used in this run. Additional validation would strengthen the model comparison.

5. **Reproducibility:** Results depend on the dataset version, preprocessing, feature construction, split, and software environment.



## 8. Reproducibility and How to Run



### Project links



- **GitHub repository:** https://github.com/swnrz/electricity-bill-predictor




### Local execution



From the project directory:



```bash

# Install project dependencies

python -m pip install -r requirements.txt



# Run the analysis and model evaluation

python main.py



# Launch the Streamlit dashboard

python -m streamlit run app.py

```



The dataset must be available at the file path expected by the code. Check the input filename and path in \`main.py\` before reproducing the run. The \`.venv\` folder should not be uploaded to GitHub; dependencies should be installed from \`requirements.txt\`.



## 9. Conclusion



The Electricity Bill Predictor demonstrates a practical application of introductory machine learning and data analytics. It processes historical household electricity data, explores daily usage, compares Linear Regression and Decision Tree regression, forecasts future consumption, and estimates bills through an interactive dashboard.



In the recorded experiment, Linear Regression achieved a lower MAE (4.3343 kWh) than the Decision Tree Regressor (4.4487 kWh) on 287 chronological test observations. The application is an educational prototype; its forecasts and bill estimates should be interpreted in light of the dataset and tariff limitations described above.



Potential future improvements include testing on geographically representative datasets, evaluating additional time-series models, using rolling or walk-forward validation, incorporating relevant external variables, and implementing actual local tariff slabs and charges.



## 10. References



1. UCI Machine Learning Repository, [**Individual Household Electric Power Consumption, dataset 235**](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

2. Microsoft, [**ML for Beginners**](https://github.com/microsoft/ML-For-Beginners).

3. scikit-learn, [**Supervised Learning documentation**](https://scikit-learn.org/stable/supervised_learning.html).

4. Streamlit, [**Documentation**](https://docs.streamlit.io/).



---



## Submission note



The experimental metrics and summary statistics in this report are values previously recorded from the project execution. They have not been independently rerun while preparing this report. During a technical review, be prepared to show \`main.py\`, the prediction CSV, and the evaluation output, and to explain the exact feature columns, preprocessing steps, and chronological split used in the implementation.
