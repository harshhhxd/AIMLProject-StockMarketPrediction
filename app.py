
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

st.set_page_config(
    page_title = "Stock Market Prediction",
    page_icon  = "📈",
    layout     = "wide"
)

BASE_DIR     = "/home/harsh/Desktop/my_projects/pr/Market stock prediction"
WINDOW_SIZE  = 90
FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Mid", "BB_Lower", "Sentiment"
]
TARGET_IDX = FEATURE_COLS.index("Close")
TICKER_MAP  = {
    "Apple (AAPL)"          : "AAPL",
    "Tesla (TSLA)"          : "TSLA",
    "Reliance (RELIANCE.NS)": "RELIANCE_NS"
}

@st.cache_resource
def load_models():
    models  = {}
    scalers = {}
    for name, ticker in TICKER_MAP.items():
        try:
            models[ticker] = {
                "lstm"    : tf.keras.models.load_model(f"{BASE_DIR}/models/{ticker}_lstm.keras",     compile=False),
                "advanced": tf.keras.models.load_model(f"{BASE_DIR}/models/{ticker}_advanced.keras", compile=False)
            }
            with open(f"{BASE_DIR}/scalers/{ticker}_scaler.pkl", "rb") as f:
                scalers[ticker] = pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load {ticker}: {e}")
    return models, scalers


@st.cache_data
def load_data(ticker):
    return pd.read_csv(f"{BASE_DIR}/data/{ticker}_data.csv", index_col="Date", parse_dates=True)


def predict_next_7(model, scaler, df):
    data        = df[FEATURE_COLS].dropna()
    scaled      = scaler.transform(data)
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
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]


# load everything
models, scalers = load_models()

# ── sidebar ──────────────────────────────────
st.sidebar.title("📈 Stock Predictor")
st.sidebar.markdown("---")
selected_name  = st.sidebar.selectbox("Select Stock",  list(TICKER_MAP.keys()))
selected_model = st.sidebar.selectbox("Select Model",  ["Standard LSTM", "BiLSTM + Attention"])
days_history   = st.sidebar.slider("Days of History",  90, 500, 200)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
st.sidebar.markdown(f"- Window Size : {WINDOW_SIZE} days")
st.sidebar.markdown(f"- Features    : {len(FEATURE_COLS)}")
st.sidebar.markdown("- Sentiment   : FinBERT + Price-derived")
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Stocks:**")
st.sidebar.markdown("- AAPL (Apple)")
st.sidebar.markdown("- TSLA (Tesla)")
st.sidebar.markdown("- RELIANCE.NS (Reliance)")

ticker     = TICKER_MAP[selected_name]
model_type = "lstm" if selected_model == "Standard LSTM" else "advanced"

# ── main page ─────────────────────────────────
st.title(f"📈 Stock Market Prediction — {selected_name}")
st.markdown("---")

df     = load_data(ticker)
model  = models[ticker][model_type]
scaler = scalers[ticker]

# ── metrics row ───────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

current_price = float(df["Close"].iloc[-1])
prev_price    = float(df["Close"].iloc[-2])
price_change  = current_price - prev_price
pct_change    = (price_change / prev_price) * 100
sentiment     = float(df["Sentiment"].iloc[-1])
rsi           = float(df["RSI_14"].iloc[-1])
currency      = "₹" if "RELIANCE" in ticker else "$"

col1.metric("Current Price",  f"{currency}{current_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
col2.metric("RSI (14)",       f"{rsi:.2f}",                     "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Normal")
col3.metric("Sentiment",      f"{sentiment:.4f}",               "Positive 😊" if sentiment > 0 else "Negative 😟")
col4.metric("Total Records",  f"{len(df):,}")
col5.metric("Selected Model", selected_model)

st.markdown("---")

# ── forecast section ──────────────────────────
st.subheader(f"🔮 Next 7 Day Price Forecast — {selected_model}")

with st.spinner("Generating predictions ..."):
    next7 = predict_next_7(model, scaler, df)

last_date    = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq="B")

fig_forecast = go.Figure()

hist = df["Close"].tail(30)
fig_forecast.add_trace(go.Scatter(
    x=hist.index, y=hist.values,
    name="Historical", line=dict(color="black", width=2)
))
fig_forecast.add_trace(go.Scatter(
    x=[hist.index[-1], future_dates[0]],
    y=[float(hist.values[-1]), float(next7[0])],
    name="Bridge", line=dict(color="gray", width=1, dash="dot"),
    showlegend=False
))
fig_forecast.add_trace(go.Scatter(
    x=future_dates, y=next7,
    name="Predicted", line=dict(color="red", width=2, dash="dash"),
    mode="lines+markers+text",
    text=[f"{currency}{p:.1f}" for p in next7],
    textposition="top center"
))
fig_forecast.update_layout(
    title     = f"{selected_name} — Next 7 Days Forecast",
    xaxis_title = "Date",
    yaxis_title = f"Price ({currency})",
    height    = 400,
    hovermode = "x unified"
)
st.plotly_chart(fig_forecast, use_container_width=True)

# forecast table
forecast_df = pd.DataFrame({
    "Day"            : [f"Day {i+1}" for i in range(7)],
    "Date"           : future_dates.strftime("%Y-%m-%d"),
    "Predicted Price": [f"{currency}{p:.2f}" for p in next7]
})
st.dataframe(forecast_df, use_container_width=True)

st.markdown("---")

# ── technical indicators ──────────────────────
st.subheader("📊 Price History + Technical Indicators")

df_plot = df.tail(days_history)

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes   = True,
    vertical_spacing = 0.05,
    subplot_titles = (
        "Price + Moving Averages + Bollinger Bands",
        "RSI (14)",
        "MACD",
        "Sentiment Score"
    ),
    row_heights = [0.4, 0.2, 0.2, 0.2]
)

# price + MAs + BB
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"],    name="Close",    line=dict(color="black",  width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA_20"],   name="SMA 20",   line=dict(color="blue",   width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA_50"],   name="SMA 50",   line=dict(color="orange", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["EMA_20"],   name="EMA 20",   line=dict(color="green",  width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["BB_Upper"], name="BB Upper", line=dict(color="purple", width=1, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["BB_Lower"], name="BB Lower", line=dict(color="purple", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(128,0,128,0.1)"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI_14"], name="RSI", line=dict(color="purple", width=1)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD"],        name="MACD",   line=dict(color="blue", width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD_Signal"], name="Signal", line=dict(color="red",  width=1)), row=3, col=1)
colors = ["green" if v >= 0 else "red" for v in df_plot["MACD_Hist"]]
fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["MACD_Hist"], name="Histogram", marker_color=colors), row=3, col=1)

# sentiment
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["Sentiment"],
    name="Sentiment", line=dict(color="teal", width=1),
    fill="tozeroy", fillcolor="rgba(0,128,128,0.1)"
), row=4, col=1)

fig.update_layout(height=900, showlegend=True, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── model metrics ─────────────────────────────
st.subheader("📉 Model Performance Metrics")

metrics_data = {
    "Stock"   : ["AAPL", "AAPL", "TSLA", "TSLA", "RELIANCE.NS", "RELIANCE.NS"],
    "Model"   : ["Standard LSTM", "BiLSTM+Attention", "Standard LSTM", "BiLSTM+Attention", "Standard LSTM", "BiLSTM+Attention"],
    "RMSE"    : [37.08, 14.67, 19.54, 42.88, 31.91, 47.68],
    "MAE"     : [36.11, 11.97, 16.17, 39.79, 26.37, 38.50],
    "MAPE (%)" : [13.44, 4.41, 3.71, 9.17, 1.80, 2.67]
}
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")
st.markdown("Built with LSTM + FinBERT Sentiment Analysis | ADR Lab Assignment")
