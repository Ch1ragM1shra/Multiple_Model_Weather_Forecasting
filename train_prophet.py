import pandas as pd
import numpy as np

from prophet import Prophet
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("featured_weather_data.csv")


df["Date"] = pd.to_datetime(df["Date"])


df = df.sort_values(["State", "Date"])


train_list = []
test_list = []

for state in df["State"].unique():

    state_df = (
        df[df["State"] == state]
        .sort_values("Date")
    )

    split_idx = int(len(state_df) * 0.8)

    train_list.append(state_df.iloc[:split_idx])
    test_list.append(state_df.iloc[split_idx:])

train = pd.concat(train_list)
test = pd.concat(test_list)

scores = []

print("Training Prophet models...\n")

for state in df["State"].unique():

    train_state = train[
        train["State"] == state
    ]

    test_state = test[
        test["State"] == state
    ]

    try:

        prophet_df = train_state[
            ["Date", "Total"]
        ].copy()

        prophet_df.columns = [
            "ds",
            "y"
        ]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )

        model.fit(prophet_df)

        future = model.make_future_dataframe(
            periods=len(test_state),
            freq="MS"   # Monthly Start
        )

        forecast = model.predict(future)

        pred = (
            forecast["yhat"]
            .tail(len(test_state))
            .values
        )

        mae = mean_absolute_error(
            test_state["Total"],
            pred
        )

        scores.append(mae)

        print(
            f"{state}: MAE = {mae:,.2f}"
        )

    except Exception as e:

        print(
            f"Error for {state}: {e}"
        )


if len(scores) > 0:

    avg_mae = np.mean(scores)

    print("\n========================")
    print(f"States Trained : {len(scores)}")
    print(f"Average MAE    : {avg_mae:,.2f}")
    print("========================")

else:

    print("No states were successfully trained.")