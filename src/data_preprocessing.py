import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df["Customer_Response"] = df["Conversions"].apply(lambda x: 1 if x > 0 else 0)

    id_cols = [c for c in df.columns if "ID" in c]
    df = df.drop(columns=id_cols)

    df = df.fillna(0)

    return df
