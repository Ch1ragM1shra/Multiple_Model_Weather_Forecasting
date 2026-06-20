import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


def train_lstm():

    # Load engineered dataset
    df = pd.read_csv("featured_weather_data.csv")

    # Sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Use Total column
    series = df["Total"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series)

    # Create sequences
    window = 8

    X = []
    y = []

    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    # Train-test split
    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    print("Training LSTM...")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    model = Sequential([
        LSTM(
            64,
            return_sequences=True,
            input_shape=(window, 1)
        ),
        LSTM(32),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Predictions
    pred = model.predict(X_test, verbose=0)

    # Convert back to original scale
    pred = scaler.inverse_transform(pred)

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    # Metrics
    mae = mean_absolute_error(
        actual.flatten(),
        pred.flatten()
    )

    print(f"LSTM MAE: {mae:,.2f}")

    # Save predictions
    results = pd.DataFrame({
        "Actual": actual.flatten(),
        "Predicted": pred.flatten()
    })

    results.to_csv(
        "lstm_predictions.csv",
        index=False
    )

    print("Predictions saved to lstm_predictions.csv")

    return mae


if __name__ == "__main__":

    mae = train_lstm()

    print(
        "\nFinal LSTM MAE:",
        f"{mae:,.2f}"
    )