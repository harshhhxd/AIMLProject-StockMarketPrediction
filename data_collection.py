"""
src/data_collection.py
======================
Handles all data collection:
  - Stock price download (yfinance)
  - Technical indicators (ta library)
  - News sentiment (FinBERT + Alpha Vantage + price-derived)
  - Saving clean CSVs
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import torch
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS           = ["AAPL", "TSLA", "RELIANCE.NS"]
START_DATE        = "2019-01-01"
END_DATE          = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR        = "data"
ALPHA_VANTAGE_KEY = "TBEBT3CFAEOZGUEU"   # replace with your key


# ─────────────────────────────────────────────
# STEP 1 — Download Stock Data
# ─────────────────────────────────────────────
def fetch_stock_data(ticker, start, end):
    """Downloads OHLCV data from Yahoo Finance."""
    print(f"Downloading {ticker} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    print(f"  ✓ {len(df)} rows downloaded")
    return df


# ─────────────────────────────────────────────
# STEP 2 — Technical Indicators
# ─────────────────────────────────────────────
def add_technical_indicators(df):
    """Adds SMA, EMA, RSI, MACD, Bollinger Bands."""
    close = df["Close"].squeeze()

    df["SMA_20"]      = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"]      = ta.trend.sma_indicator(close, window=50)
    df["EMA_20"]      = ta.trend.ema_indicator(close, window=20)
    df["RSI_14"]      = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close)
    df["BB_Upper"]    = bb.bollinger_hband()
    df["BB_Mid"]      = bb.bollinger_mavg()
    df["BB_Lower"]    = bb.bollinger_lband()

    df.dropna(inplace=True)
    print(f"  ✓ Indicators added. Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# STEP 3 — Sentiment Analysis
# ─────────────────────────────────────────────
def get_finbert_score(headlines, tokenizer, model):
    """Scores headlines with FinBERT, returns avg score -1 to +1."""
    if not headlines:
        return None
    scores = []
    for headline in headlines:
        inputs = tokenizer(
            headline, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
        scores.append(probs[0] - probs[1])  # positive minus negative
    return sum(scores) / len(scores)


def get_price_derived_sentiment(df):
    """Calculates sentiment from daily price movement as baseline."""
    daily_return = (df["Close"] - df["Open"]) / df["Open"]
    return np.tanh(daily_return * 10).squeeze()  # normalize to -1 to +1


def fetch_alpha_vantage_news(ticker, tokenizer, model):
    """Fetches 2 years of news from Alpha Vantage and scores with FinBERT."""
    print(f"  Fetching news from Alpha Vantage for {ticker} ...")
    clean_ticker = ticker.replace(".NS", "")
    end_date     = datetime.today()
    start_date   = end_date - timedelta(days=730)

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={clean_ticker}"
        f"&time_from={start_date.strftime('%Y%m%dT0000')}"
        f"&time_to={end_date.strftime('%Y%m%dT2359')}"
        f"&limit=1000"
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )

    try:
        response = requests.get(url)
        data     = response.json()

        if "feed" not in data:
            print(f"  ⚠ No news feed. Check API key.")
            return {}

        articles       = data["feed"]
        date_headlines = {}
        for article in articles:
            headline = article.get("title", "")
            pub_date = article.get("time_published", "")[:8]
            pub_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:]}"
            if headline and pub_date:
                date_headlines.setdefault(pub_date, []).append(headline)

        date_scores = {}
        for date, headlines in date_headlines.items():
            score = get_finbert_score(headlines, tokenizer, model)
            if score is not None:
                date_scores[date] = score

        print(f"  ✓ FinBERT scores for {len(date_scores)} dates")
        return date_scores

    except Exception as e:
        print(f"  ⚠ Alpha Vantage failed: {e}")
        return {}


def fetch_news_sentiment(ticker, df):
    """
    Combines price-derived sentiment (5 years baseline)
    with real FinBERT scored news (last 2 years).
    """
    print(f"\n  [SENTIMENT] Processing {ticker} ...")

    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model     = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    # fill all 5 years with price-derived sentiment as baseline
    df["Sentiment"] = get_price_derived_sentiment(df)

    # overwrite recent dates with real news scores
    date_scores = fetch_alpha_vantage_news(ticker, tokenizer, model)
    overwritten = 0
    for date, score in date_scores.items():
        mask = df.index.strftime("%Y-%m-%d") == date
        if mask.any():
            df.loc[mask, "Sentiment"] = score
            overwritten += 1

    print(f"  ✓ Real news: {overwritten} days | Price-derived: {len(df)-overwritten} days")
    return df


def fetch_reliance_news(df, get_finbert_score_fn, tokenizer, model):
    """Special handler for RELIANCE.NS using yfinance news."""
    print("  Fetching RELIANCE news from yfinance ...")
    df["Sentiment"] = get_price_derived_sentiment(df)

    try:
        news_list      = yf.Ticker("RELIANCE.NS").news
        date_headlines = {}
        for item in news_list:
            headline = item.get("content", {}).get("title") or item.get("title", "")
            pub_date = item.get("content", {}).get("pubDate", "")[:10]
            if not pub_date:
                ts = item.get("providerPublishTime", None)
                if ts:
                    pub_date = pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d")
            if headline and pub_date:
                date_headlines.setdefault(pub_date, []).append(headline)

        for date, headlines in date_headlines.items():
            score = get_finbert_score_fn(headlines, tokenizer, model)
            if score is not None:
                mask = df.index.strftime("%Y-%m-%d") == date
                if mask.any():
                    df.loc[mask, "Sentiment"] = score

        print(f"  ✓ yfinance headlines for {len(date_headlines)} dates")
    except Exception as e:
        print(f"  ⚠ yfinance news failed ({e})")

    return df


# ─────────────────────────────────────────────
# STEP 4 — Save CSV
# ─────────────────────────────────────────────
def save_data(df, ticker, output_dir):
    """Saves processed DataFrame to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{ticker.replace('.', '_')}_data.csv")
    df.to_csv(path)
    print(f"  ✓ Saved → {path}  ({len(df)} rows)")
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def collect_all(
    tickers    = TICKERS,
    start      = START_DATE,
    end        = END_DATE,
    output_dir = OUTPUT_DIR
):
    """Runs full data collection pipeline for all tickers."""
    all_data = {}

    for ticker in tickers:
        print(f"\n{'='*50}\n  Processing: {ticker}\n{'='*50}")
        try:
            df = fetch_stock_data(ticker, start, end)
            df = add_technical_indicators(df)
            df = fetch_news_sentiment(ticker, df)
            save_data(df, ticker, output_dir)
            all_data[ticker] = df
            time.sleep(1)
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\n  ALL DONE! Files saved in ./{output_dir}/\n")
    return all_data


if __name__ == "__main__":
    collect_all()
