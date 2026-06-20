from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd


def train_xgb():

    # Load engineered data
    df = pd.read_csv("featured_weather_data.csv")

    # Time-based train-test split
    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    features = [
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_mean_4",
        "rolling_std_4",
        "month",
        "quarter",
        "week",
        "year"
    ]

    X_train = train[features]
    y_train = train["Total"]

    X_test = test[features]
    y_test = test["Total"]

    print("Training XGBoost...")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    print(f"XGBoost MAE: {mae:,.2f}")

    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": preds
    })

    results.to_csv(
        "xgboost_predictions.csv",
        index=False
    )
    print("Predictions saved to xgboost_predictions.csv")


    import joblib
    joblib.dump(model,"xgboost_weather_model.pkl")
    return mae


if __name__ == "__main__":

    mae = train_xgb()

    print("\nFinal MAE:", f"{mae:,.2f}")




