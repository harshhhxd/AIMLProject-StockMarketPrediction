"""
src/model.py
============
Handles LSTM model building, training, evaluation:
  - Standard Stacked LSTM (2 layers)
  - Bidirectional LSTM + Attention (advanced)
  - EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  - RMSE, MAE, MAPE metrics
  - Next 7 day recursive predictions
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout,
    Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Mid", "BB_Lower", "Sentiment"
]
TARGET_IDX = FEATURE_COLS.index("Close")


# ─────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────
def build_lstm_model(window_size, n_features):
    """
    Standard stacked LSTM with 2 layers.
    Architecture: Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(1)
    """
    inputs = Input(shape=(window_size, n_features))

    x = LSTM(128, return_sequences=True)(inputs)   # first LSTM layer
    x = Dropout(0.2)(x)                             # 20% dropout

    x = LSTM(64, return_sequences=False)(x)         # second LSTM layer
    x = Dropout(0.2)(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss      = "mse"
    )
    return model


def build_advanced_model(window_size, n_features):
    """
    Bidirectional LSTM + Multi-Head Attention.
    Reads sequence forward AND backward, focuses on important timesteps.
    """
    inputs = Input(shape=(window_size, n_features))

    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # attention mechanism
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x    = LayerNormalization()(attn + x)   # residual connection

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss      = "mse"
    )
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val, model_path):
    """
    Trains model with callbacks:
      - EarlyStopping   : stops if val_loss doesn't improve for 20 epochs
      - ModelCheckpoint : saves best model automatically
      - ReduceLROnPlateau: halves learning rate when stuck
    """
    callbacks = [
        EarlyStopping(
            monitor              = "val_loss",
            patience             = 20,
            restore_best_weights = True
        ),
        ModelCheckpoint(
            filepath       = model_path,
            monitor        = "val_loss",
            save_best_only = True,
            verbose        = 1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 5,
            min_lr   = 0.00001
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = 150,
        batch_size      = 32,
        callbacks       = callbacks,
        verbose         = 1
    )
    return history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def calculate_metrics(y_true, y_pred, label=""):
    """Calculates RMSE, MAE, MAPE on real price scale."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"  {label} Metrics:")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    MAPE : {mape:.4f}%")
    return rmse, mae, mape


def inverse_close(scaled_vals, scaler, n_features, target_idx):
    """Converts scaled predictions back to real prices."""
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, target_idx] = scaled_vals
    return scaler.inverse_transform(dummy)[:, target_idx]


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_next_7_days(model, last_sequence, scaler, n_features, target_idx):
    """
    Predicts next 7 days recursively.
    Each predicted day becomes input for the next prediction.
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(7):
        pred = model.predict(
            current_seq.reshape(1, current_seq.shape[0], n_features),
            verbose=0
        )[0][0]
        predictions.append(pred)

        new_row             = current_seq[-1].copy()
        new_row[target_idx] = pred
        current_seq         = np.vstack([current_seq[1:], new_row])

    dummy = np.zeros((7, n_features))
    dummy[:, target_idx] = predictions
    return scaler.inverse_transform(dummy)[:, target_idx]


# ─────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_all_models(processed_data, window_size=90):
    """Trains both models for all tickers and returns metrics."""
    all_metrics = {}

    for ticker, data in processed_data.items():
        print(f"\n{'='*50}\n  Training: {ticker}\n{'='*50}")

        X_train      = data["X_train"]
        y_train      = data["y_train"]
        X_val        = data["X_val"]
        y_val        = data["y_val"]
        X_test       = data["X_test"]
        y_test       = data["y_test"]
        scaler       = data["scaler"]
        n_features   = X_train.shape[2]
        clean_ticker = ticker.replace(".", "_")

        # train standard lstm
        print("\n  [1] Training Standard LSTM ...")
        lstm_model = build_lstm_model(window_size, n_features)
        lstm_path  = f"models/{clean_ticker}_lstm.keras"
        train_model(lstm_model, X_train, y_train, X_val, y_val, lstm_path)

        # train advanced model
        print("\n  [2] Training BiLSTM + Attention ...")
        adv_model = build_advanced_model(window_size, n_features)
        adv_path  = f"models/{clean_ticker}_advanced.keras"
        train_model(adv_model, X_train, y_train, X_val, y_val, adv_path)

        # evaluate
        print(f"\n  [3] Evaluating ...")
        lstm_preds = lstm_model.predict(X_test, verbose=0).flatten()
        adv_preds  = adv_model.predict(X_test,  verbose=0).flatten()

        lstm_real  = inverse_close(lstm_preds, scaler, n_features, TARGET_IDX)
        adv_real   = inverse_close(adv_preds,  scaler, n_features, TARGET_IDX)
        y_real     = inverse_close(y_test,      scaler, n_features, TARGET_IDX)

        lstm_metrics = calculate_metrics(y_real, lstm_real, "Standard LSTM")
        adv_metrics  = calculate_metrics(y_real, adv_real,  "BiLSTM+Attention")

        # next 7 days
        last_seq   = data["X_test"][-1]
        next7_lstm = predict_next_7_days(lstm_model, last_seq, scaler, n_features, TARGET_IDX)
        next7_adv  = predict_next_7_days(adv_model,  last_seq, scaler, n_features, TARGET_IDX)

        print(f"\n  Next 7 days Standard LSTM : {[round(p,2) for p in next7_lstm]}")
        print(f"  Next 7 days BiLSTM+Attn   : {[round(p,2) for p in next7_adv]}")

        all_metrics[ticker] = {
            "lstm"           : lstm_metrics,
            "advanced"       : adv_metrics,
            "lstm_preds_real": lstm_real,
            "adv_preds_real" : adv_real,
            "y_test_real"    : y_real,
            "next_7_lstm"    : next7_lstm,
            "next_7_adv"     : next7_adv
        }

    # print final summary
    print(f"\n{'='*65}")
    print("  FINAL METRICS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Stock':<15} {'Model':<25} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
    print(f"  {'-'*65}")

    for ticker, metrics in all_metrics.items():
        r, m, p = metrics["lstm"]
        print(f"  {ticker:<15} {'Standard LSTM':<25} {r:>8.4f} {m:>8.4f} {p:>8.4f}%")
        r, m, p = metrics["advanced"]
        print(f"  {'':<15} {'BiLSTM+Attention':<25} {r:>8.4f} {m:>8.4f} {p:>8.4f}%")
        print(f"  {'-'*65}")

    return all_metrics


if __name__ == "__main__":
    import pickle
    import pandas as pd
    from feature_engineering import run_feature_engineering

    # load data
    all_data = {}
    for ticker in ["AAPL", "TSLA", "RELIANCE.NS"]:
        clean = ticker.replace(".", "_")
        df    = pd.read_csv(f"data/{clean}_data.csv", index_col="Date", parse_dates=True)
        all_data[ticker] = df

    processed_data = run_feature_engineering(all_data)
    all_metrics    = train_all_models(processed_data)
