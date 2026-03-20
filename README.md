# Stock Market Prediction using LSTM Networks

A deep learning time-series forecasting system that predicts stock prices using historical OHLCV data, technical indicators, and financial news sentiment. Built with stacked LSTM and Bidirectional LSTM with Attention mechanisms.

---

## Project Structure

```
AIMLProjectStockMarketPrediction/
|
|-- src/
|   |-- data_collection.py        # Stock data download, technical indicators, sentiment analysis
|   |-- feature_engineering.py    # Normalization, sliding windows, train/val/test splits
|   |-- model.py                  # LSTM model architectures, training, evaluation
|
|-- api/
|   |-- api.py                    # Flask REST API with /predict and /health endpoints
|
|-- tests/
|   |-- test_pipeline.py          # Pytest unit tests for data pipeline and predictions
|
|-- notebooks/
|   |-- Data_Collection.ipynb     # Complete end-to-end walkthrough notebook
|
|-- data/                         # Generated CSV files (after running data collection)
|   |-- AAPL_data.csv
|   |-- TSLA_data.csv
|   |-- RELIANCE_NS_data.csv
|
|-- models/                       # Saved Keras models (after training)
|   |-- AAPL_lstm.keras
|   |-- AAPL_advanced.keras
|   |-- TSLA_lstm.keras
|   |-- TSLA_advanced.keras
|   |-- RELIANCE_NS_lstm.keras
|   |-- RELIANCE_NS_advanced.keras
|
|-- scalers/                      # Saved MinMaxScaler objects (after training)
|   |-- AAPL_scaler.pkl
|   |-- TSLA_scaler.pkl
|   |-- RELIANCE_NS_scaler.pkl
|
|-- plots/                        # Generated visualisation plots (after training)
|
|-- app.py                        # Streamlit web dashboard
|-- requirements.txt              # Pinned library versions
|-- .env.example                  # API keys template
|-- README.md
```

---

## Prerequisites

- Python 3.10 or 3.11 (recommended — tested on 3.11)
- CPU is sufficient — no GPU required
- Operating System: Ubuntu 22.04 / macOS / Windows 10+
- Minimum 8GB RAM recommended (FinBERT model requires ~2GB)
- Free API keys for Alpha Vantage (news sentiment)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AIMLProjectStockMarketPrediction.git
cd AIMLProjectStockMarketPrediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```
ALPHA_VANTAGE_KEY=your_key_here
```

Get a free Alpha Vantage key at: https://www.alphavantage.co/support/#api-key

---

## Data Fetching

Fetches 5 years of OHLCV data for AAPL, TSLA, and RELIANCE.NS. Computes technical indicators and sentiment scores. Saves to CSV files in the `data/` folder.

```bash
python src/data_collection.py
```

This will:
- Download historical price data from Yahoo Finance (2019 to present)
- Calculate SMA, EMA, RSI, MACD, and Bollinger Bands
- Fetch financial news headlines from Alpha Vantage
- Score headlines using FinBERT (finance-specific BERT model)
- Fill historical dates with price-derived sentiment as a baseline
- Save three CSV files in the `data/` folder

Expected runtime: 10 to 20 minutes (FinBERT model downloads on first run)

---

## Training the Model

Normalizes features, creates sliding window sequences, trains both model architectures, and saves the best checkpoints to `models/`.

```bash
python src/model.py
```

This will:
- Normalize all features using MinMaxScaler (fit on training data only)
- Create 90-day sliding window sequences for LSTM input
- Train Standard Stacked LSTM (2 layers: 128 and 64 units)
- Train Bidirectional LSTM with Multi-Head Attention
- Use EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks
- Report RMSE, MAE, and MAPE on the test set for each stock
- Save six model files to `models/` and three scaler files to `scalers/`

Model checkpoints are saved in `.keras` format. Training uses EarlyStopping with patience of 20 epochs.

---

## Running the Flask API

### Start the server

```bash
python api/api.py
```

Server starts at: http://localhost:5000

### Endpoints

**GET /health** — Check service status

```bash
curl http://localhost:5000/health
```

Response:
```json
{
    "status": "healthy",
    "models": ["AAPL", "TSLA", "RELIANCE_NS"],
    "message": "Stock prediction API is running"
}
```

**POST /predict** — Get next 7 day price predictions

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL", "model_type": "lstm"}'
```

Request body parameters:

| Parameter | Required | Values |
|-----------|----------|--------|
| ticker | Yes | AAPL, TSLA, RELIANCE.NS |
| model_type | No | lstm (default), advanced |

Response:
```json
{
    "ticker": "AAPL",
    "model_type": "lstm",
    "currency": "USD",
    "predictions": [
        {"day": 1, "label": "Day 1", "price": 241.94},
        {"day": 2, "label": "Day 2", "price": 241.45},
        {"day": 3, "label": "Day 3", "price": 240.98},
        {"day": 4, "label": "Day 4", "price": 240.53},
        {"day": 5, "label": "Day 5", "price": 240.14},
        {"day": 6, "label": "Day 6", "price": 239.79},
        {"day": 7, "label": "Day 7", "price": 239.48}
    ]
}
```

---

## Running the Web Dashboard

```bash
streamlit run app.py
```

Dashboard opens at: http://localhost:8501

Features:
- Stock selector dropdown (AAPL, TSLA, Reliance)
- Model selector (Standard LSTM or BiLSTM with Attention)
- Current price with daily change percentage
- RSI and sentiment score display
- Interactive 7-day price forecast chart with predicted values
- Full technical indicators chart (Price, RSI, MACD, Sentiment)
- Model performance metrics table (RMSE, MAE, MAPE)

---

## Running Tests

```bash
pytest tests/
```

Run with verbose output:

```bash
pytest tests/ -v
```

The test suite covers:
- Feature column count and target index validation
- Normalization output shape and value range (0 to 1)
- Data leakage verification across train, validation, and test splits
- Scaler file creation
- Sliding window sequence shape and content correctness
- NaN detection in sequences
- Inverse transform shape and positivity of predicted prices
- Flask API health and predict endpoints (requires running server)
- Error handling for invalid and missing ticker inputs

---

## Model Performance

Results on the held-out test set (last 10 percent of data):

| Stock | Model | RMSE | MAE | MAPE |
|-------|-------|------|-----|------|
| AAPL | Standard LSTM | 37.08 | 36.11 | 13.44% |
| AAPL | BiLSTM + Attention | 14.67 | 11.97 | 4.41% |
| TSLA | Standard LSTM | 19.54 | 16.17 | 3.71% |
| TSLA | BiLSTM + Attention | 42.88 | 39.79 | 9.17% |
| RELIANCE.NS | Standard LSTM | 31.91 | 26.37 | 1.80% |
| RELIANCE.NS | BiLSTM + Attention | 47.68 | 38.50 | 2.67% |

Best model per stock: BiLSTM with Attention for AAPL (4.41% MAPE), Standard LSTM for TSLA (3.71% MAPE), Standard LSTM for RELIANCE.NS (1.80% MAPE).

---

## Technical Architecture

### Data Pipeline

- Source: Yahoo Finance via yfinance library
- Date range: January 2019 to present (5 plus years)
- Stocks: AAPL (Apple), TSLA (Tesla), RELIANCE.NS (Reliance Industries)
- Features: 16 total (OHLCV, 10 technical indicators, 1 sentiment score)

### Technical Indicators

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| SMA | 20 and 50 day | Short and medium term trend |
| EMA | 20 day | Recent price momentum |
| RSI | 14 day | Overbought or oversold signal |
| MACD | 12, 26, 9 | Trend and momentum |
| Bollinger Bands | 20 day, 2 std dev | Volatility measurement |

### Sentiment Analysis

Two-layer approach:
- Recent dates (last 2 years): Real news headlines from Alpha Vantage scored by FinBERT (ProsusAI/finbert), a BERT model pretrained on financial text
- Historical dates (older than 2 years): Price-derived sentiment calculated as tanh(daily return multiplied by 10), normalised to range -1 to +1

### LSTM Architectures

Standard Stacked LSTM:
- Input shape: (90, 16)
- Layer 1: LSTM 128 units with return sequences, Dropout 0.2
- Layer 2: LSTM 64 units, Dropout 0.2
- Output: Dense 32 (ReLU), Dense 1
- Optimizer: Adam with learning rate 0.0005

Bidirectional LSTM with Attention:
- Input shape: (90, 16)
- Layer 1: Bidirectional LSTM 128 units, Dropout 0.3
- Layer 2: Bidirectional LSTM 64 units, Dropout 0.3
- Attention: Multi-Head Attention (4 heads, key dimension 32) with residual connection and Layer Normalization
- Output: GlobalAveragePooling1D, Dense 64 (ReLU), Dropout 0.2, Dense 32 (ReLU), Dense 1
- Optimizer: Adam with learning rate 0.0005

### Training Configuration

- Window size: 90 days
- Train split: 80 percent
- Validation split: 10 percent
- Test split: 10 percent
- Batch size: 32
- Max epochs: 150
- EarlyStopping patience: 20 epochs
- ReduceLROnPlateau factor: 0.5, patience: 5 epochs, minimum learning rate: 0.00001

---

## Demo Video Link

[Insert your demo video link here after uploading to ADR Lab Google Drive]

---

## Submission Details

- GitHub Repository: [Your forked repo link]
- Demo Video: [Google Drive link]
- Submission Email: adrlab2026@gmail.com
- Subject: [AI-ML Assignment] Harshad Jaiswal - Stock Market Prediction
