from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train(df):
    df = pd.get_dummies(df, drop_first=True)
    
    features = ["Clicks", "Revenue_Generated", "ROI", "Discount_Level"]
    X = df[features]
    y = df["Customer_Response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 ,stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, X.columns
