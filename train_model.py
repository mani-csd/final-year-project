import joblib
import os
from src.data_preprocessing import load_data
from src.feature_engineering import create_features
from src.model_training import train
from src.model_evaluation import evaluate

print("Current working directory:", os.getcwd())

# Load dataset
print("Loading data...")
df = load_data("data/raw/marketing_and_product_performance.csv")
print("Data loaded:", df.shape)

# Feature engineering
print("Creating features...")
df = create_features(df)

# Train model
print("Training model...")
model, X_test, y_test, features = train(df)

# Evaluate model
print("Evaluating model...")
evaluate(model, X_test, y_test)

# Ensure models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# Save model + features
joblib.dump((model, features), "models/campaign_model.pkl")

print("Model and feature list saved to models/campaign_model.pkl")
