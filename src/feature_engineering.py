def create_features(df):
    df["Revenue_per_Click"] = df["Revenue_Generated"] / (df["Clicks"] + 1)
    df["Conversion_Rate"] = df["Conversions"] / (df["Clicks"] + 1)
    df["ROI_Score"] = df["ROI"] * df["Conversion_Rate"]
    return df
