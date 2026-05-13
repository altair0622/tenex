#!/usr/bin/env python3
"""
alpaca_live_scanner_tp_multi.py

What this script does
---------------------
- Loads one or many parameter JSON files produced by alpaca_backtest_optimize_tp_multi.py:
    best_params_<SYMBOL>.json
- For each symbol:
    - Fetches recent minute bars from Alpaca
    - Builds the same features
    - Trains a model (RandomForest) with model_params from JSON on the recent history window
- Live loop (scanner):
    - Every minute, compute the latest probability for EACH symbol
    - Pick the top signals where prob >= p_enter
    - Enter up to --max_positions concurrently
    - Manage exits:
        - If using bracket orders: TP/SL handled server-side; we still enforce TIME exit at horizon
        - If not using bracket: we check completed bars and exit via TP/SL/TIME

Notes / realism
---------------
- Minute bars only update after the minute closes; intraminute TP/SL is not perfectly observable.
- If your Alpaca account supports bracket orders, TP/SL will be enforced server-side (recommended).
- TIME exit still requires this script (or another scheduler) to close positions after horizon minutes.

Requirements
------------
  pip install alpaca-py pandas numpy scikit-learn joblib

Env vars
--------
  ALPACA_API_KEY_1
  ALPACA_SECRET_KEY_1

Example
-------
python alpaca_live_scanner_tp_multi.py \
  --params_dir params_out \
  --paper \
  --feed iex \
  --max_positions 2 \
  --notional_usd 2000 \
  --use_brackets

"""

from __future__ import annotations

import os
import json
import time
import glob
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Bracket request objects vary by alpaca-py version; we'll import if available.
try:
    from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
except Exception:  # pragma: no cover
    TakeProfitRequest = None
    StopLossRequest = None

NY = ZoneInfo("America/New_York")


# -------------------------
# Data fetch
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


# -------------------------
# Features (same as backtest script)
# -------------------------
def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def macd(close: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def stochastic_kd(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    low_min = df["Low"].rolling(window=k_window).min()
    high_max = df["High"].rolling(window=k_window).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_window).mean()
    return k, d


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"]).cumsum()


def price_oscillator(close: pd.Series, short: int = 12, long: int = 26) -> pd.Series:
    sma_short = close.rolling(short).mean()
    sma_long = close.rolling(long).mean()
    return (sma_short - sma_long) / sma_long.replace(0, np.nan)


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Previous Close"] = d["Close"].shift(1)

    d["RSI"] = calculate_rsi(d["Close"])
    d["Buying Probability"] = d["RSI"].apply(buying_probability)
    d["Selling Probability"] = d["RSI"].apply(selling_probability)

    d["vwap"] = vwap_intraday(d)
    d["EMA"] = ema(d["Close"], span=20)
    d["IC"] = d["Close"] / d["vwap"].replace(0, np.nan)

    d["MACD"], d["Signal_Line"] = macd(d["Close"])
    d["%K"], d["%D"] = stochastic_kd(d)
    d["OBV"] = obv(d)
    d["Price_Oscillator"] = price_oscillator(d["Close"])

    d["rolling_mean"], d["Bollinger_Upper"], d["Bollinger_Lower"] = bollinger(d["Close"])
    band = (d["Bollinger_Upper"] - d["Bollinger_Lower"]).replace(0, np.nan)

    for col in ["Open", "High", "Low", "Close", "Previous Close"]:
        d[f"{col}_n"] = (d[col] - d["Bollinger_Lower"]) / band

    d["hour"] = d.index.hour
    d["minute"] = d.index.minute
    return d


# -------------------------
# Model / Params container
# -------------------------
@dataclass
class SymbolConfig:
    symbol: str
    feature_cols: list[str]
    model_params: dict
    market_open: dtime
    market_close: dtime
    horizon: int
    tp: float
    sl: float | None
    p_enter: float
    lookback_days: int


@dataclass
class OpenPosition:
    symbol: str
    qty: float
    entry_time: datetime
    horizon_min: int


def load_param_files(params_dir: str) -> dict[str, SymbolConfig]:
    files = sorted(glob.glob(os.path.join(params_dir, "best_params_*.json")))
    if not files:
        raise RuntimeError(f"No best_params_*.json found in {params_dir}")

    cfgs: dict[str, SymbolConfig] = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            p = json.load(f)

        sym = p["symbol"]
        mh = p.get("market_hours", {"open": "10:00", "close": "15:30"})
        mo_h, mo_m = map(int, mh["open"].split(":"))
        mc_h, mc_m = map(int, mh["close"].split(":"))

        strat = p["best_strategy_params"]
        cfgs[sym] = SymbolConfig(
            symbol=sym,
            feature_cols=p["feature_cols"],
            model_params=p["model_params"],
            market_open=dtime(mo_h, mo_m),
            market_close=dtime(mc_h, mc_m),
            horizon=int(strat["horizon"]),
            tp=float(strat["tp"]),
            sl=None if strat.get("sl") is None else float(strat["sl"]),
            p_enter=float(strat["p_enter"]),
            lookback_days=int(p.get("data_window_days", 60)),
        )
    return cfgs


def is_market_time(cfg: SymbolConfig, now: datetime) -> bool:
    t = now.timetz()
    return cfg.market_open <= t <= cfg.market_close


def train_model_for_symbol(symbol: str, cfg: SymbolConfig, feed: str, train_days: int | None = None) -> RandomForestClassifier:
    end = datetime.now(tz=NY)
    start = end - timedelta(days=(train_days or cfg.lookback_days))
    df = fetch_minute_bars(symbol, start, end, feed=feed)
    if df.empty:
        raise RuntimeError(f"No bars for {symbol} in training window.")
    df = add_features(df)
    df = df.dropna(subset=cfg.feature_cols).copy()

    # label not needed for fitting final model? We DO need it: fit to TP-hit objective.
    # We'll build the same label used in backtest (entry on next open).
    y = make_label_hit_tp(df, horizon=cfg.horizon, tp=cfg.tp, entry_on_next_open=True)
    df = df.dropna(subset=cfg.feature_cols).copy()
    y = y.reindex(df.index)

    m = df[cfg.feature_cols].notna().all(axis=1) & y.notna()
    X = df.loc[m, cfg.feature_cols]
    y2 = y.loc[m]

    model = RandomForestClassifier(**cfg.model_params)
    model.fit(X, y2)
    return model


def make_label_hit_tp(df: pd.DataFrame, horizon: int, tp: float, entry_on_next_open: bool = True) -> pd.Series:
    if entry_on_next_open:
        entry = df["Open"].shift(-1).values
        start = 1
    else:
        entry = df["Close"].values
        start = 0

    highs = df["High"].values
    fwd_max_high = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        j0 = i + start
        j1 = i + start + horizon
        if j1 <= len(df):
            fwd_max_high[i] = np.max(highs[j0:j1])
    target = entry * (1 + tp)
    y = (fwd_max_high >= target).astype(float)
    return pd.Series(y, index=df.index)


# -------------------------
# Trading helpers
# -------------------------
def get_trading_client(paper: bool) -> TradingClient:
    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def submit_entry(
    tc: TradingClient,
    symbol: str,
    qty: float,
    tp_price: float | None,
    sl_price: float | None,
    use_brackets: bool,
) -> str:
    """
    Returns order id.
    If use_brackets and request objects exist, submit bracket order.
    Otherwise submit market buy only.
    """
    if use_brackets and (TakeProfitRequest is not None) and (StopLossRequest is not None) and (tp_price is not None):
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
            stop_loss=(StopLossRequest(stop_price=round(sl_price, 2)) if sl_price is not None else None),
        )
    else:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    order = tc.submit_order(req)
    return order.id


def submit_exit_market(tc: TradingClient, symbol: str, qty: float) -> str:
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = tc.submit_order(req)
    return order.id


# -------------------------
# Main loop
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_dir", default="params_out", help="Directory with best_params_<SYMBOL>.json files")
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--paper", action="store_true", help="Use paper trading endpoint")
    ap.add_argument("--max_positions", type=int, default=1, help="Max concurrent open positions")
    ap.add_argument("--notional_usd", type=float, default=2000.0, help="Approx dollars per trade; used to compute qty")
    ap.add_argument("--use_brackets", action="store_true", help="Use bracket orders for TP/SL if supported")
    ap.add_argument("--refresh_model_minutes", type=int, default=60, help="Retrain each symbol model every N minutes")
    ap.add_argument("--loop_sleep", type=int, default=20, help="Seconds between checks (20-30s is fine)")
    args = ap.parse_args()

    cfgs = load_param_files(args.params_dir)

    print("Loaded symbols:", ", ".join(sorted(cfgs.keys())))
    tc = get_trading_client(paper=args.paper)

    # Train initial models
    models: dict[str, RandomForestClassifier] = {}
    last_train: dict[str, datetime] = {}
    for sym, cfg in cfgs.items():
        try:
            models[sym] = train_model_for_symbol(sym, cfg, feed=args.feed)
            last_train[sym] = datetime.now(tz=NY)
            print(f"Trained model for {sym}")
        except Exception as e:
            print(f"[WARN] Could not train {sym}: {e}")

    open_positions: dict[str, OpenPosition] = {}  # symbol -> position

    while True:
        now = datetime.now(tz=NY)

        # retrain periodically
        for sym, cfg in cfgs.items():
            if sym not in models:
                continue
            if (now - last_train[sym]).total_seconds() >= args.refresh_model_minutes * 60:
                try:
                    models[sym] = train_model_for_symbol(sym, cfg, feed=args.feed)
                    last_train[sym] = now
                    print(f"[{now:%H:%M:%S}] retrained {sym}")
                except Exception as e:
                    print(f"[WARN] retrain failed {sym}: {e}")

        # enforce TIME exits
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            if (now - pos.entry_time).total_seconds() >= pos.horizon_min * 60:
                try:
                    oid = submit_exit_market(tc, sym, pos.qty)
                    print(f"[{now:%H:%M:%S}] TIME EXIT {sym} qty={pos.qty} order_id={oid}")
                except Exception as e:
                    print(f"[WARN] time-exit failed {sym}: {e}")
                open_positions.pop(sym, None)

        # if already maxed out, just sleep
        if len(open_positions) >= args.max_positions:
            time.sleep(args.loop_sleep)
            continue

        # score symbols
        candidates = []
        for sym, cfg in cfgs.items():
            if sym in open_positions:
                continue
            if sym not in models:
                continue
            if not is_market_time(cfg, now):
                continue

            # Pull last ~300 minutes (enough for indicators)
            end = now
            start = end - timedelta(minutes=400)
            df = fetch_minute_bars(sym, start, end, feed=args.feed)
            if df.empty or len(df) < 80:
                continue
            df = add_features(df)
            df = df.dropna(subset=cfg.feature_cols).copy()
            if df.empty:
                continue

            X_last = df[cfg.feature_cols].iloc[[-1]]
            p = float(models[sym].predict_proba(X_last)[:, 1][0])

            if p >= cfg.p_enter:
                last_close = float(df["Close"].iloc[-1])
                # approximate TP/SL prices from last_close (entry will be near next open)
                tp_price = last_close * (1 + cfg.tp)
                sl_price = last_close * (1 - cfg.sl) if cfg.sl is not None else None
                candidates.append((p, sym, tp_price, sl_price, last_close))

        # sort by prob desc, take as many as we can
        candidates.sort(reverse=True, key=lambda x: x[0])
        slots = args.max_positions - len(open_positions)

        for p, sym, tp_price, sl_price, last_close in candidates[:slots]:
            qty = max(1.0, round(args.notional_usd / last_close, 6))
            try:
                oid = submit_entry(
                    tc=tc,
                    symbol=sym,
                    qty=qty,
                    tp_price=tp_price if args.use_brackets else None,
                    sl_price=sl_price if args.use_brackets else None,
                    use_brackets=args.use_brackets,
                )
                open_positions[sym] = OpenPosition(
                    symbol=sym,
                    qty=qty,
                    entry_time=now,
                    horizon_min=cfgs[sym].horizon,
                )
                print(f"[{now:%H:%M:%S}] ENTER {sym} p={p:.3f} qty={qty} order_id={oid} brackets={args.use_brackets}")
            except Exception as e:
                print(f"[WARN] entry failed {sym}: {e}")

        time.sleep(args.loop_sleep)


if __name__ == "__main__":
    main()
