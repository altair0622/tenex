#!/usr/bin/env python3
"""
alpaca_backtest_optimize_tp.py

Goal:
- Fetch 1-minute bars from Alpaca
- Build features (RSI, VWAP, EMA, MACD, Stoch, OBV, Price Oscillator, Bollinger + normalized OHLC as *_n)
- Create label: "Within H minutes after entry, does price touch +tp?"
- Walk-forward train -> predict probabilities
- Backtest strategy: enter if prob >= p_enter, exit via TP / (optional SL) / time-exit at H minutes
- Grid-search parameters (H, tp, p_enter, sl, model params) and export best set to JSON

Requirements:
  pip install alpaca-py pandas numpy scikit-learn joblib

Env vars:
  ALPACA_API_KEY_1
  ALPACA_SECRET_KEY_1
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


NY = ZoneInfo("America/New_York")


# -------------------------
# Alpaca data fetch
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
        raise RuntimeError("No bars returned. Check symbol, dates, and feed (iex/sip).")

    # alpaca-py returns MultiIndex (symbol, timestamp)
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

    # Keep only this symbol (if multi)
    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol].copy()
        bars.drop(columns=["symbol"], inplace=True)

    return bars


# -------------------------
# Indicators / features
# -------------------------
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
    # Reset per day (NY time)
    d = df.copy()
    day = d.index.date
    pv = (d["Close"] * d["Volume"]).groupby(day).cumsum()
    vv = d["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def ema(close: pd.Series, span: int = 20) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    m = exp1 - exp2
    s = m.ewm(span=signal, adjust=False).mean()
    return m, s


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
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


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
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

    # Preserve original OHLC; create normalized versions
    band_range = (d["Bollinger_Upper"] - d["Bollinger_Lower"]).replace(0, np.nan)
    for col in ["Open", "High", "Low", "Close"]:
        d[col + "_n"] = (d[col] - d["Bollinger_Lower"]) / band_range

    # Time features
    d["hour"] = d.index.hour
    d["minute"] = d.index.minute

    return d


# -------------------------
# Label: hit tp within horizon
# -------------------------
def make_label_hit_tp(df: pd.DataFrame, horizon: int, tp: float, entry_on_next_open: bool = True) -> pd.Series:
    """
    y[t] = 1 if within next H minutes, High reaches entry*(1+tp).
    entry = Open[t+1] (default) or Close[t]
    """
    if entry_on_next_open:
        entry = df["Open"].shift(-1)
        start = 1
    else:
        entry = df["Close"]
        start = 0

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


# -------------------------
# Walk-forward (rolling) predict
# -------------------------
def walkforward_train_predict(
    df_feat: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    train_size: int,
    test_size: int,
    model_params: dict,
) -> pd.Series:
    prob = pd.Series(index=df_feat.index, dtype=float)
    start = 0
    n = len(df_feat)

    while start + train_size + test_size <= n:
        tr = slice(start, start + train_size)
        te = slice(start + train_size, start + train_size + test_size)

        X_tr = df_feat.iloc[tr][feature_cols]
        y_tr = y.iloc[tr]
        X_te = df_feat.iloc[te][feature_cols]

        m = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr2, y_tr2 = X_tr[m], y_tr[m]

        model = RandomForestClassifier(**model_params)
        model.fit(X_tr2, y_tr2)

        prob.iloc[te] = model.predict_proba(X_te)[:, 1]
        start += test_size

    return prob


# -------------------------
# Backtest
# -------------------------
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    stats: dict


def backtest_prob_strategy(
    df: pd.DataFrame,
    prob: pd.Series,
    horizon: int,
    tp: float,
    p_enter: float,
    sl: float | None = None,
    entry_on_next_open: bool = True,
    fee_bps: float = 0.0,
) -> BacktestResult:
    """
    One-position-at-a-time.
    Enter when prob[t] >= p_enter.
    Entry price: Open[t+1] by default.
    Exit:
      - TP if High hits entry*(1+tp)
      - SL if Low hits entry*(1-sl) (optional) -- checked first (conservative)
      - else time-exit at Close of (entry_i + horizon - 1)
    """
    prob = prob.reindex(df.index)

    trades = []
    i = 0
    n = len(df)

    while i < n:
        if pd.isna(prob.iat[i]) or prob.iat[i] < p_enter:
            i += 1
            continue

        entry_i = i + (1 if entry_on_next_open else 0)
        if entry_i >= n:
            break

        entry_px = df["Open"].iat[entry_i] if entry_on_next_open else df["Close"].iat[i]
        tp_px = entry_px * (1 + tp)
        sl_px = entry_px * (1 - sl) if sl is not None else None

        last_i = min(entry_i + horizon - 1, n - 1)

        exit_reason = "TIME"
        exit_i = last_i

        for j in range(entry_i, last_i + 1):
            hi = df["High"].iat[j]
            lo = df["Low"].iat[j]
            if sl_px is not None and lo <= sl_px:
                exit_reason = "SL"
                exit_i = j
                break
            if hi >= tp_px:
                exit_reason = "TP"
                exit_i = j
                break

        if exit_reason == "TP":
            exit_px = tp_px
        elif exit_reason == "SL":
            exit_px = sl_px
        else:
            exit_px = df["Close"].iat[exit_i]

        fee = (fee_bps / 10000.0) * (entry_px + exit_px)
        ret = ((exit_px - entry_px) - fee) / entry_px

        trades.append({
            "entry_time": df.index[entry_i],
            "exit_time": df.index[exit_i],
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "exit_reason": exit_reason,
            "ret": float(ret),
            "signal_prob": float(prob.iat[i]),
        })

        i = exit_i + 1  # one position at a time

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return BacktestResult(trades_df, {"trades": 0})

    equity = (1 + trades_df["ret"]).cumprod()
    mdd = (equity / equity.cummax() - 1).min()

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["ret"] > 0).mean()),
        "avg_ret": float(trades_df["ret"].mean()),
        "median_ret": float(trades_df["ret"].median()),
        "tp_rate": float((trades_df["exit_reason"] == "TP").mean()),
        "sl_rate": float((trades_df["exit_reason"] == "SL").mean()) if sl is not None else None,
        "time_exit_rate": float((trades_df["exit_reason"] == "TIME").mean()),
        "equity_final": float(equity.iloc[-1]),
        "max_drawdown": float(mdd),
    }
    return BacktestResult(trades_df, stats)


# -------------------------
# Grid search
# -------------------------
def run_grid_search(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    tps: list[float],
    p_enters: list[float],
    sls: list[float | None],
    train_size: int,
    test_size: int,
    fee_bps: float,
    model_params: dict,
    min_trades: int,
    max_mdd: float,
) -> pd.DataFrame:
    rows = []

    for horizon in horizons:
        for tp in tps:
            y = make_label_hit_tp(df_feat, horizon=horizon, tp=tp, entry_on_next_open=True)

            prob = walkforward_train_predict(
                df_feat=df_feat,
                y=y,
                feature_cols=feature_cols,
                train_size=train_size,
                test_size=test_size,
                model_params=model_params,
            )

            ok = prob.notna()
            if ok.sum() < (train_size + test_size):
                continue

            # model-quality metric (optional reference)
            pr_auc = average_precision_score(y[ok], prob[ok])

            for p_enter in p_enters:
                for sl in sls:
                    bt = backtest_prob_strategy(
                        df=df_feat[ok],
                        prob=prob[ok],
                        horizon=horizon,
                        tp=tp,
                        p_enter=p_enter,
                        sl=sl,
                        entry_on_next_open=True,
                        fee_bps=fee_bps,
                    )
                    r = {
                        "horizon": horizon,
                        "tp": tp,
                        "p_enter": p_enter,
                        "sl": sl if sl is not None else None,
                        "pr_auc": float(pr_auc),
                        **bt.stats,
                    }
                    rows.append(r)

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    # Basic constraints (stability)
    res = res[res["trades"] >= min_trades].copy()
    res = res[res["max_drawdown"] >= -abs(max_mdd)].copy()

    # Score: primarily equity_final, then avg_ret, then lower drawdown
    res = res.sort_values(["equity_final", "avg_ret", "max_drawdown"], ascending=[False, False, False])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--days", type=int, default=60, help="Lookback days for backtest data")
    ap.add_argument("--feed", default="iex", help="iex or sip (depending on your subscription)")
    ap.add_argument("--market_open", default="10:00")
    ap.add_argument("--market_close", default="15:30")
    ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--test_size", type=int, default=500)
    ap.add_argument("--fee_bps", type=float, default=1.0)
    ap.add_argument("--out_json", default="best_params.json")
    args = ap.parse_args()

    end = datetime.now(tz=NY)
    start = end - timedelta(days=args.days)

    df = fetch_minute_bars(args.symbol, start, end, feed=args.feed)

    # market hours filter
    mo_h, mo_m = map(int, args.market_open.split(":"))
    mc_h, mc_m = map(int, args.market_close.split(":"))
    t = df.index.time
    df = df[(t >= datetime(2000,1,1,mo_h,mo_m).time()) & (t <= datetime(2000,1,1,mc_h,mc_m).time())].copy()

    df_feat = add_features(df)

    # Feature columns (close to your notebook + safe time features)
    feature_cols = [
        "RSI", "Buying Probability", "Selling Probability",
        "vwap", "EMA", "IC", "MACD", "Signal_Line", "%K", "%D", "OBV", "Price_Oscillator",
        "rolling_mean", "Bollinger_Upper", "Bollinger_Lower",
        "Open_n", "High_n", "Low_n", "Close_n",
        "hour", "minute",
    ]

    df_feat = df_feat.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()

    # Model params baseline (you can extend grid-search here too)
    model_params = dict(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    res = run_grid_search(
        df_feat=df_feat,
        feature_cols=feature_cols,
        horizons=[5, 10, 15],
        tps=[0.004, 0.005, 0.006],
        p_enters=[0.55, 0.60, 0.65, 0.70],
        sls=[None, 0.0025, 0.0035],
        train_size=args.train_size,
        test_size=args.test_size,
        fee_bps=args.fee_bps,
        model_params=model_params,
        min_trades=30,
        max_mdd=0.10,
    )

    if len(res) == 0:
        print("No valid parameter set found under constraints.")
        return

    best = res.iloc[0].to_dict()

    # Save best set (include model & features)
    payload = {
        "symbol": args.symbol,
        "asof": end.isoformat(),
        "data_window_days": args.days,
        "market_hours": {"open": args.market_open, "close": args.market_close},
        "feature_cols": feature_cols,
        "model_params": model_params,
        "best_strategy_params": {
            "horizon": int(best["horizon"]),
            "tp": float(best["tp"]),
            "p_enter": float(best["p_enter"]),
            "sl": None if pd.isna(best["sl"]) else float(best["sl"]),
            "fee_bps": float(args.fee_bps),
        },
        "best_backtest_stats": {k: best[k] for k in best.keys() if k not in ["horizon","tp","p_enter","sl"]},
    }

    out_path = args.out_json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Also save the whole grid result as CSV for inspection
    csv_path = out_path.replace(".json", "_grid.csv")
    res.to_csv(csv_path, index=False)

    print("Saved best params to:", out_path)
    print("Saved grid results to:", csv_path)
    print("Best:", payload["best_strategy_params"])
    print("Stats:", payload["best_backtest_stats"])


if __name__ == "__main__":
    main()
