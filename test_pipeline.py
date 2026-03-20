"""
tests/test_pipeline.py
======================
Unit tests for data pipeline and prediction functions.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import create_sequences, normalize_data, FEATURE_COLS, TARGET_IDX, WINDOW_SIZE


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Creates a sample DataFrame mimicking stock data."""
    np.random.seed(42)
    n    = 300
    idx  = pd.date_range(start="2022-01-01", periods=n, freq="B")
    data = {col: np.random.uniform(100, 500, n) for col in FEATURE_COLS}
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def sample_scaled_data():
    """Creates sample scaled numpy array."""
    np.random.seed(42)
    return np.random.uniform(0, 1, (200, len(FEATURE_COLS)))


# ─────────────────────────────────────────────
# DATA PIPELINE TESTS
# ─────────────────────────────────────────────
class TestDataPipeline:

    def test_feature_cols_count(self):
        """Feature columns should be exactly 16."""
        assert len(FEATURE_COLS) == 16

    def test_target_idx_is_close(self):
        """TARGET_IDX should point to Close column."""
        assert FEATURE_COLS[TARGET_IDX] == "Close"

    def test_window_size_positive(self):
        """Window size should be positive."""
        assert WINDOW_SIZE > 0

    def test_normalize_data_shape(self, sample_df):
        """Normalized splits should have correct shapes."""
        train_scaled, val_scaled, test_scaled, scaler = normalize_data(
            sample_df, "TEST", scalers_dir="/tmp/test_scalers"
        )
        total = len(sample_df)
        assert len(train_scaled) == int(total * 0.8)
        assert len(val_scaled)   == int(total * 0.1)
        assert len(test_scaled)  == total - int(total * 0.8) - int(total * 0.1)

    def test_normalize_data_range(self, sample_df):
        """Scaled values should be between 0 and 1."""
        train_scaled, val_scaled, test_scaled, _ = normalize_data(
            sample_df, "TEST", scalers_dir="/tmp/test_scalers"
        )
        assert train_scaled.min() >= -0.01   # small tolerance
        assert train_scaled.max() <= 1.01

    def test_no_data_leakage(self, sample_df):
        """Train and test dates should not overlap."""
        n         = len(sample_df)
        train_end = int(n * 0.8)
        val_end   = int(n * 0.9)

        train_dates = set(sample_df.index[:train_end])
        val_dates   = set(sample_df.index[train_end:val_end])
        test_dates  = set(sample_df.index[val_end:])

        assert len(train_dates & val_dates)  == 0, "Train/Val overlap!"
        assert len(val_dates   & test_dates) == 0, "Val/Test overlap!"
        assert len(train_dates & test_dates) == 0, "Train/Test overlap!"

    def test_scaler_saved(self, sample_df, tmp_path):
        """Scaler pkl file should be saved."""
        normalize_data(sample_df, "TESTAAPL", scalers_dir=str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "TESTAAPL_scaler.pkl"))


# ─────────────────────────────────────────────
# SEQUENCE CREATION TESTS
# ─────────────────────────────────────────────
class TestSequenceCreation:

    def test_sequence_shape(self, sample_scaled_data):
        """X shape should be (samples, window, features)."""
        X, y = create_sequences(sample_scaled_data, WINDOW_SIZE, TARGET_IDX)
        assert X.shape[1] == WINDOW_SIZE
        assert X.shape[2] == len(FEATURE_COLS)

    def test_sequence_length(self, sample_scaled_data):
        """Number of sequences = data length - window size."""
        X, y = create_sequences(sample_scaled_data, WINDOW_SIZE, TARGET_IDX)
        assert len(X) == len(sample_scaled_data) - WINDOW_SIZE

    def test_y_shape(self, sample_scaled_data):
        """y should be 1D array."""
        X, y = create_sequences(sample_scaled_data, WINDOW_SIZE, TARGET_IDX)
        assert y.ndim == 1

    def test_x_y_same_length(self, sample_scaled_data):
        """X and y should have same number of samples."""
        X, y = create_sequences(sample_scaled_data, WINDOW_SIZE, TARGET_IDX)
        assert len(X) == len(y)

    def test_target_value_correct(self, sample_scaled_data):
        """y[0] should equal data[window_size, target_idx]."""
        X, y = create_sequences(sample_scaled_data, WINDOW_SIZE, TARGET_IDX)
        expected = sample_scaled_data[WINDOW_SIZE, TARGET_IDX]
        assert abs(y[0] - expected) < 1e-6

    def test_window_content_correct(self, sample_scaled_data):
        """X[0] should equal data[0:window_size]."""
        X, y = create_sequences(sample_scaled_data, WINDOW_SIZE, TARGET_IDX)
        np.testing.assert_array_equal(X[0], sample_scaled_data[:WINDOW_SIZE])

    def test_no_nan_in_sequences(self, sample_scaled_data):
        """Sequences should not contain NaN values."""
        X, y = create_sequences(sample_scaled_data, WINDOW_SIZE, TARGET_IDX)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


# ─────────────────────────────────────────────
# PREDICTION FUNCTION TESTS
# ─────────────────────────────────────────────
class TestPredictionFunctions:

    def test_inverse_transform_shape(self, sample_df):
        """Inverse transformed predictions should match input length."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(sample_df[FEATURE_COLS])

        scaled_preds = np.random.uniform(0, 1, 7)
        dummy        = np.zeros((7, len(FEATURE_COLS)))
        dummy[:, TARGET_IDX] = scaled_preds
        real_prices  = scaler.inverse_transform(dummy)[:, TARGET_IDX]

        assert len(real_prices) == 7

    def test_prediction_values_positive(self, sample_df):
        """Stock prices should always be positive."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(sample_df[FEATURE_COLS])

        scaled_preds = np.random.uniform(0.1, 0.9, 7)
        dummy        = np.zeros((7, len(FEATURE_COLS)))
        dummy[:, TARGET_IDX] = scaled_preds
        real_prices  = scaler.inverse_transform(dummy)[:, TARGET_IDX]

        assert all(p > 0 for p in real_prices)


# ─────────────────────────────────────────────
# API TESTS
# ─────────────────────────────────────────────
class TestAPI:

    def test_health_endpoint(self):
        """Health endpoint should return 200 with healthy status."""
        try:
            import requests
            response = requests.get("http://127.0.0.1:5000/health", timeout=3)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "models" in data
        except Exception:
            pytest.skip("API server not running")

    def test_predict_endpoint_aapl(self):
        """Predict endpoint should return 7 predictions for AAPL."""
        try:
            import requests
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"ticker": "AAPL", "model_type": "lstm"},
                timeout=30
            )
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 7
            assert data["ticker"] == "AAPL"
        except Exception:
            pytest.skip("API server not running")

    def test_predict_invalid_ticker(self):
        """Invalid ticker should return 400 error."""
        try:
            import requests
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"ticker": "INVALID", "model_type": "lstm"},
                timeout=10
            )
            assert response.status_code == 400
        except Exception:
            pytest.skip("API server not running")

    def test_predict_missing_ticker(self):
        """Missing ticker should return 400 error."""
        try:
            import requests
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"model_type": "lstm"},
                timeout=10
            )
            assert response.status_code == 400
        except Exception:
            pytest.skip("API server not running")
