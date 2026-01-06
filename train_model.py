import joblib
import os
import pandas as pd
from src.data_preprocessing import load_data
from src.model_evaluation import evaluate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Current working directory:", os.getcwd())

# 1️⃣ Load dataset
print("Loading data...")
df = load_data("data/raw/marketing_and_product_performance.csv")
print("Data loaded:", df.shape)

# 2️⃣ Create Conversion Rate
df["Conversion_Rate"] = df["Conversions"] / (df["Clicks"] + 1)

# 3️⃣ Create target (Business Logic)
df["Customer_Response"] = ((df["Conversion_Rate"] > 0.05) & (df["ROI"] > 1)).astype(int)

# 4️⃣ Select ONLY UI input features
features = ["Clicks", "Revenue_Generated", "ROI", "Discount_Level"]
X = df[features]
y = df["Customer_Response"]

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6️⃣ Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Evaluate
print("Evaluating model...")
evaluate(model, X_test, y_test)

# 8️⃣ Save model and feature list
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump((model, features), "models/campaign_model.pkl")

print("Model and feature list saved to models/campaign_model.pkl")
