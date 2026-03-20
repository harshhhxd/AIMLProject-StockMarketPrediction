import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app      = Flask(__name__)
CORS(app)

# hardcoded base path
BASE_DIR = "/home/harsh/Desktop/my_projects/pr/Market stock prediction"

WINDOW_SIZE  = 90
FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Mid", "BB_Lower", "Sentiment"
]
TARGET_IDX = FEATURE_COLS.index("Close")
TICKERS    = ["AAPL", "TSLA", "RELIANCE_NS"]

print("Loading models ...")
MODELS  = {}
SCALERS = {}

for ticker in TICKERS:
    try:
        MODELS[ticker] = {
            "lstm"    : tf.keras.models.load_model(f"{BASE_DIR}/models/{ticker}_lstm.keras",     compile=False),
            "advanced": tf.keras.models.load_model(f"{BASE_DIR}/models/{ticker}_advanced.keras", compile=False)
        }
        with open(f"{BASE_DIR}/scalers/{ticker}_scaler.pkl", "rb") as f:
            SCALERS[ticker] = pickle.load(f)
        print(f"  ✓ {ticker} loaded")
    except Exception as e:
        print(f"  ✗ {ticker} failed: {e}")

print("All models loaded!")


def predict_prices(ticker, model_type):
    clean = ticker.upper().replace(".", "_")

    if clean not in MODELS:
        return None, f"{ticker} not supported. Use AAPL, TSLA or RELIANCE.NS"

    df     = pd.read_csv(f"{BASE_DIR}/data/{clean}_data.csv", index_col="Date", parse_dates=True)
    scaler = SCALERS[clean]
    model  = MODELS[clean][model_type]
    data   = df[FEATURE_COLS].dropna()
    scaled = scaler.transform(data)

    if len(scaled) < WINDOW_SIZE:
        return None, "Not enough data"

    last_seq    = scaled[-WINDOW_SIZE:]
    n_features  = last_seq.shape[1]
    predictions = []
    current_seq = last_seq.copy()

    for _ in range(7):
        pred = model.predict(
            current_seq.reshape(1, WINDOW_SIZE, n_features), verbose=0
        )[0][0]
        predictions.append(pred)
        new_row             = current_seq[-1].copy()
        new_row[TARGET_IDX] = pred
        current_seq         = np.vstack([current_seq[1:], new_row])

    dummy = np.zeros((7, n_features))
    dummy[:, TARGET_IDX] = predictions
    real_prices = scaler.inverse_transform(dummy)[:, TARGET_IDX]
    return real_prices.tolist(), None


@app.route("/health", methods=["GET", "POST"])
def health():
    return jsonify({
        "status" : "healthy",
        "models" : TICKERS,
        "message": "Stock prediction API is running"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        if not body:
            return jsonify({"error": "Request body required"}), 400

        ticker     = body.get("ticker", "")
        model_type = body.get("model_type", "lstm").lower()

        if not ticker:
            return jsonify({"error": "ticker is required"}), 400

        if model_type not in ["lstm", "advanced"]:
            return jsonify({"error": "model_type must be lstm or advanced"}), 400

        predictions, error = predict_prices(ticker, model_type)

        if error:
            return jsonify({"error": error}), 400

        return jsonify({
            "ticker"     : ticker.upper(),
            "model_type" : model_type,
            "currency"   : "INR" if "RELIANCE" in ticker.upper() else "USD",
            "predictions": [
                {"day": i+1, "label": f"Day {i+1}", "price": round(p, 2)}
                for i, p in enumerate(predictions)
            ]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    
    
