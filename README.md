# International Education Budget Predictor

A machine learning project that predicts the annual growth rate of key education-related costs for international students across four destination regions: **United States**, **United Kingdom**, **Australia**, and **Other countries worldwide**.

---

## Problem Statement

International students face significant financial uncertainty when planning their studies abroad. Tuition fees, rent, living costs, and insurance premiums all fluctuate year to year — and these fluctuations are closely tied to macroeconomic indicators like inflation and currency exchange rates.

This project builds a predictive model that estimates the **year-over-year growth rate** of four cost categories, enabling students and financial planners to make more informed budgeting decisions.

---

## Project Structure

```
Portfolio/
├── datasets/                        # Time-series and cross-sectional CSV data
│   ├── New International_Education_Costs.csv   # Main cross-sectional dataset (850+ records, 71 countries)
│   ├── US_Avg_Tuition.csv                      # Annual tuition growth rate — USA
│   ├── UK_Avg_Tuition.csv                      # Annual tuition growth rate — UK
│   ├── Aus_Avg_Tuition.csv                     # Annual tuition growth rate — Australia
│   ├── Other_Avg_Tuition.csv                   # Annual tuition growth rate — Other countries
│   ├── US_Inflation.csv                        # Annual inflation rate — USA
│   ├── UK_Inflation.csv                        # Annual inflation rate — UK
│   ├── Aus_Inflation.csv                       # Annual inflation rate — Australia
│   ├── Other_Inflation.csv                     # Annual inflation rate — Global average
│   ├── Growth_Rent_{US,UK,Aus,Other}.csv       # Annual rent growth rates
│   ├── Growth_LivCost_{US,UK,Aus,Other}.csv    # Annual living cost growth rates
│   └── Growth_Insurance_{US,UK,Aus,Other}.csv  # Annual insurance cost growth rates
│
├── data_prep.py             # Data loading, cleaning, and feature engineering
├── model_engine.py          # Model training pipeline (Random Forest)
├── debug_data_shape.py      # Utility script to inspect merged dataset shapes
├── explore.ipynb            # Exploratory Data Analysis notebook
├── budget_predictor_model.pkl  # Serialized trained model (joblib)
└── README.md
```

---

## Datasets

### Main Dataset — `New International_Education_Costs.csv`

Cross-sectional snapshot of universities across 71 countries (850+ records). Each row represents a specific program at a specific university.

| Column | Description |
|---|---|
| `Country` | Destination country |
| `City` | City where the university is located |
| `University` | Name of the institution |
| `Program` | Field of study (e.g., Computer Science, Data Science) |
| `Level` | Degree level: Bachelor, Master, PhD |
| `Duration_Years` | Program duration in years |
| `Tuition_USD` | Annual tuition fee in USD |
| `Living_Cost_Index` | Relative cost of living index (0–100 scale) |
| `Rent_USD` | Average monthly rent in USD |
| `Visa_Fee_USD` | Student visa application fee in USD |
| `Insurance_USD` | Annual health insurance cost in USD |
| `Exchange_Rate` | Local currency to USD exchange rate |

**Top represented countries:** UK (93), Australia (86), USA (78), Canada (76), Germany (33)

### Time-Series Growth Datasets

Sixteen CSV files (4 regions × 4 cost categories), each containing annual **percentage growth rates** with columns `Year` and `Growth`. These span approximately 2015–2025 and are the primary training signal for the model.

---

## Methodology

### 1. Data Preparation (`data_prep.py`)

- Loads all 21 CSV files into separate DataFrames
- Corrects data quality issues (e.g., city/university name typos, stale living cost index values)
- Applies updated `Living_Cost_Index` values for 30+ cities based on current benchmarks
- Computes derived features:
  - `Rent_Yearly`: monthly rent × 12
  - `Tuition_Yearly`: tuition × 2 (semester-based programs)
  - `Monthly_Living_Cost`: living cost index–adjusted monthly estimate including rent
- Fetches live **IDR/USD exchange rate growth** (10-year average) from Yahoo Finance via `yfinance`

### 2. Model Training (`model_engine.py`)

**Feature engineering:**
Each region's time-series datasets are merged on `Year` to produce one row per year per region, yielding a combined training set of ~40 rows.

| Feature | Description |
|---|---|
| `Country_Code` | Encoded region (US=0, UK=1, Aus=2, Other=3) |
| `x_inflation` | Annual inflation rate for that region |
| `Year` | Calendar year |
| `x_currency` | Average annual IDR/USD growth rate (same for all regions) |

**Multi-output targets:**

| Target | Description |
|---|---|
| `y_tuition` | Tuition fee growth rate (%) |
| `y_rent` | Rent growth rate (%) |
| `y_living` | Living cost growth rate (%) |
| `y_insur` | Insurance cost growth rate (%) |

**Model:** `sklearn.ensemble.RandomForestRegressor`
- `n_estimators = 200`
- `random_state = 42`
- 80/20 train-test split

**Evaluation metrics:** Mean Absolute Error (MAE) and R² Score

The trained model is serialized to `budget_predictor_model.pkl` using `joblib`.

### 3. Exploratory Data Analysis (`explore.ipynb`)

A Jupyter notebook covering:
- Preview and shape inspection of the main dataset
- Country-level distribution of records
- University-level tuition range analysis (e.g., USA institutions sorted by min/max tuition)
- Data quality fixes (typo corrections for university names)
- Notable entries: Harvard, Stanford, MIT, and 71 other countries

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | RandomForestRegressor, train-test split, metrics |
| `joblib` | Model serialization |
| `yfinance` | Live IDR/USD exchange rate data |
| `matplotlib` | Visualization (regression plots, correlation matrix) |
| `seaborn` | Heatmaps and regression plot styling |

Install dependencies:

```bash
pip install pandas scikit-learn joblib yfinance matplotlib seaborn
```

---

## Usage

### Train the model

```bash
python model_engine.py
```

This will:
1. Load and merge all time-series datasets
2. Fetch live currency exchange rate data
3. Train the Random Forest model
4. Print MAE and R² on the test set
5. Save the model to `budget_predictor_model.pkl`

### Inspect dataset shapes

```bash
python debug_data_shape.py
```

Prints the shape and column names of each regional dataset and their merged form — useful for verifying data alignment before training.

### Load the saved model for inference

```python
import joblib
import pandas as pd

model = joblib.load('budget_predictor_model.pkl')

# Example: predict cost growth for Australia in 2025
X_new = pd.DataFrame([{
    'Country_Code': 2,      # Aus
    'x_inflation': 3.5,     # %
    'Year': 2025,
    'x_currency': 2.1       # IDR/USD annual growth %
}])

predictions = model.predict(X_new)
print("Tuition, Rent, Living, Insurance growth (%):", predictions)
```

---

## Target Audience

This project is designed to help **Indonesian students** planning to study abroad understand how education costs evolve over time relative to inflation and currency depreciation — making it easier to prepare long-term financial plans.
