"""
src/feature_engineering.py
===========================
Handles preprocessing for LSTM:
  - MinMaxScaler normalization (fit on train only)
  - Sliding window sequence creation
  - Train/Val/Test splitting with no data leakage
  - Saving scalers as .pkl files
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WINDOW_SIZE  = 90
TARGET_COL   = "Close"
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.1
TEST_SPLIT   = 0.1
FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Mid", "BB_Lower", "Sentiment"
]
TARGET_IDX   = FEATURE_COLS.index(TARGET_COL)


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def normalize_data(df, ticker, scalers_dir="scalers"):
    """
    Normalizes features using MinMaxScaler.
    Scaler is fit ONLY on training data to prevent data leakage.
    Saves scaler to disk for later use in predictions.
    """
    print(f"\n[NORMALIZE] Processing {ticker} ...")

    data      = df[FEATURE_COLS].copy()
    n         = len(data)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    # split BEFORE scaling to prevent data leakage
    train_data = data.iloc[:train_end]
    val_data   = data.iloc[train_end:val_end]
    test_data  = data.iloc[val_end:]

    print(f"  Train: {len(train_data)} rows ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    print(f"  Val  : {len(val_data)} rows ({val_data.index[0].date()} to {val_data.index[-1].date()})")
    print(f"  Test : {len(test_data)} rows ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    # fit scaler ONLY on training data
    scaler       = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    val_scaled   = scaler.transform(val_data)
    test_scaled  = scaler.transform(test_data)

    # save scaler
    os.makedirs(scalers_dir, exist_ok=True)
    scaler_path = os.path.join(scalers_dir, f"{ticker.replace('.', '_')}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved → {scaler_path}")

    return train_scaled, val_scaled, test_scaled, scaler


# ─────────────────────────────────────────────
# SLIDING WINDOW SEQUENCES
# ─────────────────────────────────────────────
def create_sequences(data, window_size, target_col_idx):
    """
    Creates sliding window sequences for LSTM input.

    Example with window_size=3:
      Day1, Day2, Day3 → predict Day4
      Day2, Day3, Day4 → predict Day5

    Returns:
        X: shape (samples, window_size, features)
        y: shape (samples,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# DATA LEAKAGE CHECK
# ─────────────────────────────────────────────
def check_data_leakage(all_data):
    """Verifies no overlap between train/val/test splits."""
    print("\nDATA LEAKAGE CHECK")
    print("=" * 50)

    for ticker, df in all_data.items():
        n         = len(df)
        train_end = int(n * TRAIN_SPLIT)
        val_end   = int(n * (TRAIN_SPLIT + VAL_SPLIT))

        train_dates = df.index[:train_end]
        val_dates   = df.index[train_end:val_end]
        test_dates  = df.index[val_end:]

        print(f"\n{ticker}:")
        print(f"  Train/Val overlap  : {len(set(train_dates) & set(val_dates))} ← should be 0")
        print(f"  Val/Test overlap   : {len(set(val_dates) & set(test_dates))} ← should be 0")
        print(f"  Train/Test overlap : {len(set(train_dates) & set(test_dates))} ← should be 0")
        print(f"  Scaler fit on      : TRAIN ONLY ✅")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_feature_engineering(all_data, window_size=WINDOW_SIZE):
    """
    Runs full feature engineering pipeline for all tickers.
    Returns processed_data dict with train/val/test splits.
    """
    processed_data = {}

    for ticker, df in all_data.items():
        print(f"\n{'='*50}\n  Feature Engineering: {ticker}\n{'='*50}")

        train_scaled, val_scaled, test_scaled, scaler = normalize_data(df, ticker)

        X_train, y_train = create_sequences(train_scaled, window_size, TARGET_IDX)
        X_val,   y_val   = create_sequences(val_scaled,   window_size, TARGET_IDX)
        X_test,  y_test  = create_sequences(test_scaled,  window_size, TARGET_IDX)

        processed_data[ticker] = {
            "X_train": X_train, "y_train": y_train,
            "X_val"  : X_val,   "y_val"  : y_val,
            "X_test" : X_test,  "y_test" : y_test,
            "scaler" : scaler
        }

        print(f"\n  ✓ Shapes for {ticker}:")
        print(f"    X_train : {X_train.shape}")
        print(f"    X_val   : {X_val.shape}")
        print(f"    X_test  : {X_test.shape}")

    print("\n✅ Feature engineering done for all stocks!")
    check_data_leakage(all_data)
    return processed_data


if __name__ == "__main__":
    # load saved CSVs and run feature engineering
    all_data = {}
    for ticker in ["AAPL", "TSLA", "RELIANCE.NS"]:
        clean = ticker.replace(".", "_")
        df    = pd.read_csv(f"data/{clean}_data.csv", index_col="Date", parse_dates=True)
        all_data[ticker] = df

    processed_data = run_feature_engineering(all_data)
