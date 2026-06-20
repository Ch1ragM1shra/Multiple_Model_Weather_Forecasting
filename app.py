from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()


model = joblib.load(
    "xgboost_weather_model.pkl"
)


class WeatherInput(BaseModel):

    lag_1: float
    lag_7: float
    lag_30: float

    rolling_mean_4: float
    rolling_std_4: float

    month: int
    quarter: int
    week: int
    year: int


@app.get("/")
def home():

    return {
        "message":
        "Weather Forecasting API Running"
    }


@app.post("/predict")
def predict(data: WeatherInput):

    try:

        features = pd.DataFrame([{
            "lag_1": data.lag_1,
            "lag_7": data.lag_7,
            "lag_30": data.lag_30,
            "rolling_mean_4": data.rolling_mean_4,
            "rolling_std_4": data.rolling_std_4,
            "month": data.month,
            "quarter": data.quarter,
            "week": data.week,
            "year": data.year
        }])

        print(features)
        print(features.dtypes)

        prediction = model.predict(features)[0]

        return {
            "predicted_total": float(prediction)
        }

    except Exception as e:

        return {
            "error": str(e)
        }