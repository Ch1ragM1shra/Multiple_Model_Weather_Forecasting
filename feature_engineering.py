import pandas as pd


df = pd.read_csv(
    r"C:\Users\narendra mishra\OneDrive\Desktop\Forecasting Case- Study.xlsx - Sheet1.csv"
)


df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)


df["Total"] = (
    df["Total"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Total"] = pd.to_numeric(df["Total"], errors="coerce")


df = df.sort_values(["State", "Date"])


df["lag_1"] = df.groupby("State")["Total"].shift(1)
df["lag_7"] = df.groupby("State")["Total"].shift(7)
df["lag_30"] = df.groupby("State")["Total"].shift(30)


df["rolling_mean_4"] = (
    df.groupby("State")["Total"]
      .transform(lambda x: x.rolling(window=4).mean())
)

df["rolling_std_4"] = (
    df.groupby("State")["Total"]
      .transform(lambda x: x.rolling(window=4).std())
)


df = df.dropna(subset=["Date"])
df["month"] = df["Date"].dt.month
df["quarter"] = df["Date"].dt.quarter
df["week"] = df["Date"].dt.isocalendar().week
df["year"] = df["Date"].dt.year


df = df.dropna()


df.to_csv("featured_weather_data.csv", index=False)

print("Feature engineering completed successfully!")
print(df.head())
print("\nData Shape:", df.shape)