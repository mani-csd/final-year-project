import joblib
import os
import pandas as pd
from src.data_preprocessing import load_data
from src.feature_engineering import create_features
from src.model_training import train
from src.model_evaluation import evaluate

print("Current working directory:", os.getcwd())

# 1️⃣ Load dataset
print("Loading data...")
df = load_data("data/raw/marketing_and_product_performance.csv")
print("Data loaded:", df.shape)

# 2️⃣ Feature engineering
print("Creating features...")
df = create_features(df)

# 3️⃣ Create Conversion Rate
df["Conversion_Rate"] = df["Conversions"] / (df["Clicks"] + 1)

# 4️⃣ Create target (Business Logic)
df["Customer_Response"] = ((df["Conversion_Rate"] > 0.05) & (df["ROI"] > 1)).astype(int)

# 5️⃣ Train model
print("Training model...")
model, X_test, y_test, features = train(df)

# 6️⃣ Evaluate model
print("Evaluating model...")
evaluate(model, X_test, y_test)

# 7️⃣ Ensure models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# 8️⃣ Save model and feature list
joblib.dump((model, features), "models/campaign_model.pkl")

print("Model and feature list saved to models/campaign_model.pkl")
