# Weather Forecasting System using Machine Learning

## Project Overview

This project implements an end-to-end weather forecasting pipeline using Machine Learning and Time Series Forecasting techniques. The system performs data preprocessing, feature engineering, model training, model comparison, and exposes the best-performing model through a FastAPI REST API.

The objective is to forecast future values of the target variable (`Total`) using historical observations and engineered time-series features.

---

## Features

* Data preprocessing and cleaning
* Feature engineering using lag and rolling window features
* Time-based train-test split
* Multiple forecasting models:

  * XGBoost Regressor
  * SARIMA
  * LSTM Neural Network
* Model comparison using Mean Absolute Error (MAE)
* Automatic best model selection
* FastAPI deployment for real-time predictions
* Interactive API documentation using Swagger UI

---

## Project Structure

```text
Weather_forecasting/
│
├── app.py
├── preprocessing.py
├── feature_engineering.py
├── train_test_split.py
├── train_xgboost.py
├── train_sarima.py
├── train_lstm.py
├── comparison.py
├── featured_weather_data.csv
├── xgboost_weather_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Processing Pipeline

### 1. Preprocessing

* Load raw dataset
* Handle missing values
* Convert data types
* Parse date columns

### 2. Feature Engineering

The following features are generated:

* lag_1
* lag_7
* lag_30
* rolling_mean_4
* rolling_std_4
* month
* quarter
* week
* year

### 3. Model Training

The following forecasting models are trained:

* XGBoost Regressor
* SARIMA
* LSTM

### 4. Model Evaluation

Evaluation Metric:

* Mean Absolute Error (MAE)

The model with the lowest MAE is selected as the final forecasting model.

---

## API Deployment

The best-performing model (XGBoost) is deployed using FastAPI.

### Run the API

```bash
uvicorn app:app --reload
```

### API Endpoint

#### Home

```http
GET /
```

Response:

```json
{
  "message": "Weather Forecasting API Running"
}
```

#### Prediction

```http
POST /predict
```

Request:

```json
{
  "lag_1": 100000000,
  "lag_7": 95000000,
  "lag_30": 90000000,
  "rolling_mean_4": 97000000,
  "rolling_std_4": 2000000,
  "month": 6,
  "quarter": 2,
  "week": 25,
  "year": 2024
}
```

Response:

```json
{
  "predicted_total": 15500434
}
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/weather-forecasting.git

cd weather-forecasting
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Comparison

| Model   | Description                                              |
| ------- | -------------------------------------------------------- |
| XGBoost | Gradient boosting model used as the final deployed model |
| SARIMA  | Classical statistical forecasting model                  |
| LSTM    | Deep learning time-series forecasting model              |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* TensorFlow / Keras
* Statsmodels
* FastAPI
* Uvicorn
* Joblib

---

## Future Improvements

* Hyperparameter tuning
* Docker deployment
* Cloud deployment (AWS/GCP/Azure)
* Automated retraining pipeline
* Model monitoring and drift detection


## Challenges Faced

### Prophet Model Compatibility Issue

The Prophet forecasting model was evaluated as part of the model comparison phase. However, it could not be successfully trained due to compatibility issues between the installed Prophet package and the NumPy version in the development environment.

The following error was encountered:

```text
AttributeError: 'Prophet' object has no attribute 'stan_backend'
```

and subsequently:

```text
AttributeError: np.float_ was removed in the NumPy 2.0 release
```

Despite attempts to resolve the issue by installing and configuring CmdStanPy and verifying Prophet installation, the incompatibility persisted. As a result, Prophet was excluded from the final model comparison.

The remaining models (XGBoost, SARIMA, and LSTM) were successfully implemented, trained, and evaluated. XGBoost achieved the lowest Mean Absolute Error (MAE) and was selected as the final deployed model.


---

## Author

Chirag Mishra
