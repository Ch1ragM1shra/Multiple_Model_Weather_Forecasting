import pandas as pd
def split_data(df):

    split_date = (
        df["Date"].max()
        - pd.Timedelta(weeks=8)
    )

    train = df[
        df["Date"] <= split_date
    ]

    test = df[
        df["Date"] > split_date
    ]

    return train, test