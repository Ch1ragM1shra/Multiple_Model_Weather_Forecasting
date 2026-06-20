import pandas as pd

# Load data
df = pd.read_csv(
    r"C:\Users\narendra mishra\OneDrive\Desktop\Forecasting Case- Study.xlsx - Sheet1.csv"
)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Clean Total column and convert to numeric
df["Total"] = (
    df["Total"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

# Sort data before creating time-series features
df = df.sort_values(["State", "Date"])

# Lag features
df["lag_1"] = df.groupby("State")["Total"].shift(1)
df["lag_7"] = df.groupby("State")["Total"].shift(7)
df["lag_30"] = df.groupby("State")["Total"].shift(30)

# Rolling statistics
df["rolling_mean_4"] = (
    df.groupby("State")["Total"]
      .transform(lambda x: x.rolling(window=4).mean())
)

df["rolling_std_4"] = (
    df.groupby("State")["Total"]
      .transform(lambda x: x.rolling(window=4).std())
)

# Date features
df = df.dropna(subset=["Date"])
df["month"] = df["Date"].dt.month
df["quarter"] = df["Date"].dt.quarter
df["week"] = df["Date"].dt.isocalendar().week
df["year"] = df["Date"].dt.year

# Remove rows with NaNs created by lag/rolling features
df = df.dropna()

# Save engineered dataset
df.to_csv("featured_weather_data.csv", index=False)

print("Feature engineering completed successfully!")
print(df.head())
print("\nData Shape:", df.shape)