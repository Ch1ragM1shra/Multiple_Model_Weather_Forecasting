import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error


def train_sarima():

    # Load engineered dataset
    df = pd.read_csv("featured_weather_data.csv")

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort data
    df = df.sort_values(["State", "Date"])

    # State-wise train/test split
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

    print("Training SARIMA models...\n")

    for state in df["State"].unique():

        train_state = (
            train[train["State"] == state]
            .sort_values("Date")
            .set_index("Date")
        )

        test_state = (
            test[test["State"] == state]
            .sort_values("Date")
            .set_index("Date")
        )

        if len(train_state) < 20 or len(test_state) == 0:
            continue

        try:

            model = SARIMAX(
                train_state["Total"],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted = model.fit(disp=False)

            pred = fitted.forecast(
                steps=len(test_state)
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

    if len(scores) == 0:
        raise ValueError(
            "No states were successfully trained."
        )

    avg_mae = np.mean(scores)

    print("\n========================")
    print(f"States Trained : {len(scores)}")
    print(f"Average MAE    : {avg_mae:,.2f}")
    print("========================")

    return avg_mae


if __name__ == "__main__":

    mae = train_sarima()

    print(
        "\nFinal SARIMA MAE:",
        f"{mae:,.2f}"
    )