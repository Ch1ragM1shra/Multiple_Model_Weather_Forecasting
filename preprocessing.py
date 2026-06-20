import pandas as pd
import numpy as np
import holidays

df = pd.read_csv(
    r"C:\Users\narendra mishra\OneDrive\Desktop\Forecasting Case- Study.xlsx - Sheet1.csv"
)

df["Total"] = (
    df["Total"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed",
    dayfirst=True
)
df = df.sort_values(["State", "Date"])
india_holidays = holidays.India()
df["holiday_flag"] = df["Date"].apply(
    lambda x: 1 if x in india_holidays else 0
)