#!/usr/bin/env python3
"""
alpaca_live_model_tp.py

Goal:
- Load best_params.json produced by alpaca_backtest_optimize_tp.py
- Fetch recent minute bars from Alpaca
- Build the same features
- Train a final model on a rolling window (or full backtest window)
- Live loop:
    If prob >= p_enter and within market hours, enter (market buy)
    Then manage exits:
      - If TP touched -> submit limit sell (or market) and close
      - If SL touched (optional) -> market sell
      - If horizon time exceeded -> market sell
  (One position at a time)

IMPORTANT:
- This is a template. Test on paper trading first.
- Minute-bar intrabar TP/SL detection in live trading: you only know OHLC after the bar closes.
  For "true" intraminute TP/SL, you'd need trades/quotes or shorter timeframe.
  This script manages based on completed minute bars.
- You can still place bracket orders (if your Alpaca account supports) to handle TP/SL server-side.

Requirements:
  pip install alpaca-py pandas numpy scikit-learn joblib

Env vars:
  ALPACA_API_KEY_1
  ALPACA_SECRET_KEY_1
"""

from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


NY = ZoneInfo("America/New_York")


# -------------------------
# Shared: data + features (same as backtest)
# -------------------------
def fetch_minute_bars(symbol: str, start: datetime, end: datetime, feed: str = "iex") -> pd.DataFrame:
    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=feed,
    )
    bars = client.get_stock_bars(req).df
    if bars is None or len(bars) == 0:
        return pd.DataFrame()

    bars = bars.reset_index()
    bars = bars.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "timestamp": "timestamp",
    })
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NY)
    bars = bars.sort_values("timestamp").set_index("timestamp")

    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol].copy()
        bars.drop(columns=["symbol"], inplace=True)

    return bars


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def buying_probability(rsi: float) -> float:
    if np.isnan(rsi):
        return np.nan
    if rsi < 30:
        return 1 - (rsi / 30)
    if rsi > 70:
        return 0.0
    return (40 - (rsi - 30)) / 40


def selling_probability(rsi: float) -> float:
    if np.isnan(rsi):
        return np.nan
    if rsi > 70:
        return (rsi - 70) / 30
    if rsi < 30:
        return 0.0
    return (rsi - 30) / 40


def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    day = df.index.date
    pv = (df["Close"] * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def ema(close: pd.Series, span: int = 20) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    m = exp1 - exp2
    s = m.ewm(span=signal, adjust=False).mean()
    return m, s


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    k = (df["Close"] - low_min) * 100 / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def obv(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].values
    vol = df["Volume"].values
    out = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        if close[i] > close[i-1]:
            out[i] = out[i-1] + vol[i]
        elif close[i] < close[i-1]:
            out[i] = out[i-1] - vol[i]
        else:
            out[i] = out[i-1]
    return pd.Series(out, index=df.index)


def price_oscillator(close: pd.Series, long_period: int = 26, short_period: int = 12) -> pd.Series:
    short_ema = close.ewm(span=short_period, adjust=False).mean()
    long_ema = close.ewm(span=long_period, adjust=False).mean()
    return short_ema - long_ema


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    rm = close.rolling(window=window).mean()
    rs = close.rolling(window=window).std()
    upper = rm + num_std * rs
    lower = rm - num_std * rs
    return rm, upper, lower


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["RSI"] = calculate_rsi(d["Close"])
    d["Buying Probability"] = d["RSI"].apply(buying_probability)
    d["Selling Probability"] = d["RSI"].apply(selling_probability)

    d["vwap"] = vwap_intraday(d)
    d["EMA"] = ema(d["Close"], span=20)
    d["IC"] = d["Close"] / d["vwap"]

    d["MACD"], d["Signal_Line"] = macd(d["Close"])
    d["%K"], d["%D"] = stochastic(d)
    d["OBV"] = obv(d)
    d["Price_Oscillator"] = price_oscillator(d["Close"])

    d["rolling_mean"], d["Bollinger_Upper"], d["Bollinger_Lower"] = bollinger(d["Close"])

    band_range = (d["Bollinger_Upper"] - d["Bollinger_Lower"]).replace(0, np.nan)
    for col in ["Open", "High", "Low", "Close"]:
        d[col + "_n"] = (d[col] - d["Bollinger_Lower"]) / band_range

    d["hour"] = d.index.hour
    d["minute"] = d.index.minute

    return d


# -------------------------
# Trading helpers
# -------------------------
def within_market_hours(now: datetime, market_open: dtime, market_close: dtime) -> bool:
    t = now.astimezone(NY).time()
    return (t >= market_open) and (t <= market_close)


@dataclass
class PositionState:
    in_pos: bool = False
    entry_time: datetime | None = None
    entry_px: float | None = None


def place_market_buy(trading: TradingClient, symbol: str, qty: int) -> None:
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    trading.submit_order(req)


def place_market_sell(trading: TradingClient, symbol: str, qty: int) -> None:
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    trading.submit_order(req)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", default="best_params.json")
    ap.add_argument("--symbol", default=None, help="Override symbol from params_json")
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--qty", type=int, default=10)
    ap.add_argument("--lookback_minutes", type=int, default=600, help="Bars to compute features + inference")
    ap.add_argument("--retrain_minutes", type=int, default=60, help="How often to retrain the model (minutes)")
    ap.add_argument("--model_out", default="final_model.joblib")
    ap.add_argument("--paper", action="store_true", help="Use paper trading endpoint (set in Alpaca account settings)")
    args = ap.parse_args()

    with open(args.params_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    symbol = args.symbol or cfg["symbol"]
    feature_cols = cfg["feature_cols"]
    model_params = cfg["model_params"]
    strat = cfg["best_strategy_params"]

    horizon = int(strat["horizon"])
    tp = float(strat["tp"])
    p_enter = float(strat["p_enter"])
    sl = strat.get("sl", None)
    sl = float(sl) if sl is not None else None

    market_open = dtime(*map(int, cfg["market_hours"]["open"].split(":")))
    market_close = dtime(*map(int, cfg["market_hours"]["close"].split(":")))

    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")

    trading = TradingClient(api_key=key, secret_key=secret, paper=args.paper)

    state = PositionState()
    model = None
    last_retrain = None

    print("LIVE CONFIG")
    print({"symbol": symbol, "horizon": horizon, "tp": tp, "p_enter": p_enter, "sl": sl, "qty": args.qty})

    while True:
        now = datetime.now(tz=NY)

        if not within_market_hours(now, market_open, market_close):
            time.sleep(10)
            continue

        # Fetch recent bars
        end = now
        start = end - timedelta(minutes=args.lookback_minutes + horizon + 5)
        df = fetch_minute_bars(symbol, start, end, feed=args.feed)
        if df.empty or len(df) < 60:
            time.sleep(5)
            continue

        df = add_features(df)
        df = df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close"]).copy()
        if len(df) < 200:
            time.sleep(5)
            continue

        # Retrain periodically on recent window (simple, robust)
        if (model is None) or (last_retrain is None) or ((now - last_retrain).total_seconds() >= args.retrain_minutes * 60):
            # Train label for the chosen horizon/tp on this rolling window
            y = make_label_hit_tp_local(df, horizon=horizon, tp=tp)
            train = df.dropna(subset=feature_cols).copy()
            y = y.reindex(train.index).dropna()

            X = train.loc[y.index, feature_cols]
            model = RandomForestClassifier(**model_params)
            model.fit(X, y.astype(int))

            joblib.dump(model, args.model_out)
            last_retrain = now
            print(f"[{now}] retrained model on {len(X)} rows -> saved {args.model_out}")

        # Inference at the latest fully-formed bar
        latest = df.iloc[-1]
        X_now = df.iloc[[-1]][feature_cols]
        prob = float(model.predict_proba(X_now)[:, 1][0])

        # Entry logic
        if not state.in_pos and prob >= p_enter:
            place_market_buy(trading, symbol, args.qty)
            state.in_pos = True
            state.entry_time = df.index[-1]  # using last bar close time as "decision time"
            state.entry_px = float(df["Close"].iloc[-1])
            print(f"[{now}] BUY signal prob={prob:.3f} entry_px~{state.entry_px:.4f}")

        # Manage open position based on completed bars
        if state.in_pos and state.entry_time is not None and state.entry_px is not None:
            minutes_in = int((df.index[-1] - state.entry_time).total_seconds() // 60)

            # Evaluate TP/SL on the bars since entry_time
            entry_px = state.entry_px
            tp_px = entry_px * (1 + tp)
            sl_px = entry_px * (1 - sl) if sl is not None else None

            df_after = df[df.index >= state.entry_time].copy()
            if len(df_after) > 0:
                hi = df_after["High"].max()
                lo = df_after["Low"].min()

                exit_reason = None
                if sl_px is not None and lo <= sl_px:
                    exit_reason = "SL"
                elif hi >= tp_px:
                    exit_reason = "TP"
                elif minutes_in >= horizon:
                    exit_reason = "TIME"

                if exit_reason is not None:
                    place_market_sell(trading, symbol, args.qty)
                    print(f"[{now}] SELL ({exit_reason}) minutes_in={minutes_in} prob_last={prob:.3f}")
                    state = PositionState()

        # Sleep until near next minute (simple pacing)
        time.sleep(5)


def make_label_hit_tp_local(df: pd.DataFrame, horizon: int, tp: float) -> pd.Series:
    """
    Local helper: label uses next-open entry in backtest,
    but for live retraining we approximate entry at next bar Open.
    We'll still construct y[t] based on Open[t+1] for consistency.
    """
    entry = df["Open"].shift(-1)
    start = 1
    highs = df["High"].to_numpy()
    fwd_max_high = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        j0 = i + start
        j1 = i + start + horizon
        if j1 <= len(df):
            fwd_max_high[i] = np.max(highs[j0:j1])
    target = entry.to_numpy() * (1 + tp)
    y = (fwd_max_high >= target).astype(float)
    return pd.Series(y, index=df.index)


if __name__ == "__main__":
    main()
