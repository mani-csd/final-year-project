import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df["Conversion_Rate"] = df["Conversions"] / (df["Clicks"] + 1)

    df["Customer_Response"] = ((df["Conversion_Rate"] > 0.05) & (df["ROI"] > 1)).astype(int)


    id_cols = [c for c in df.columns if "ID" in c]
    df = df.drop(columns=id_cols)

    df = df.fillna(0)

    return df
