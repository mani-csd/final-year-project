from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model, features = joblib.load(r"D:\marketing_campaign_project\models\campaign_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame(0, index=[0], columns=features)

    for key in data:
        if key in input_df.columns:
            input_df[key] = data[key]

    result = model.predict(input_df)[0]

    return jsonify({"Customer_Response": int(result)})

if __name__ == "__main__":
    app.run(debug=True)
