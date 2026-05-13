#!/usr/bin/env python3
"""
4.1.6_backtest_select_best_etf_notional_highconviction.py

Goal:
- Compare a small ETF universe
- Use walk-forward evaluation focused on daily return on notional
- For each ETF, find the best params by avg_daily_return_on_notional
- Select the best ETF using the same metric
- Save transparent outputs for review

Default walk-forward:
- train_days = 10 trading days
- test_days = 1 trading day
- eval_days = 5 rolling test days
- step_days = 1 trading day

Primary selection metric:
- avg_daily_return_on_notional

Outputs:
- best_etf_selection_*.json
- best_etf_by_symbol_*.csv
- grid_results_*.csv
- daily_roll_results_*.csv
"""

from __future__ import annotations

import os
import csv
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

NY = ZoneInfo("America/New_York")

MARKET_OPEN = time(10, 0)
MARKET_CLOSE = time(15, 30)


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df.index.time >= MARKET_OPEN) & (df.index.time <= MARKET_CLOSE)].copy()


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
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - vol[i]
        else:
            out[i] = out[i - 1]
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


def make_label_hit_tp_local(df: pd.DataFrame, horizon: int, tp: float) -> pd.Series:
    entry = df["Open"].shift(-1)
    highs = df["High"].to_numpy()
    fwd_max_high = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        j0 = i + 1
        j1 = i + 1 + horizon
        if j1 <= len(df):
            fwd_max_high[i] = np.max(highs[j0:j1])
    target = entry.to_numpy() * (1 + tp)
    y = (fwd_max_high >= target).astype(float)
    return pd.Series(y, index=df.index)


@dataclass
class DayTrade:
    entry_ts: datetime
    exit_ts: datetime
    entry_px: float
    exit_px: float
    qty: int
    pnl: float
    exit_reason: str
    hold_minutes: int
    mfe: float
    mae: float
    pnl_pct_on_notional: float


FEATURE_COLS = [
    "RSI", "Buying Probability", "Selling Probability", "vwap", "EMA", "IC",
    "MACD", "Signal_Line", "%K", "%D", "OBV", "Price_Oscillator",
    "Open_n", "High_n", "Low_n", "Close_n", "hour", "minute"
]

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "random_state": 42,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}


def get_trading_days(index: pd.DatetimeIndex) -> list:
    return sorted(pd.Index(index.date).unique())


def simulate_day(
    test_df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: list[str],
    horizon: int,
    tp: float,
    sl: float,
    p_enter: float,
    qty: int,
) -> dict:
    df = test_df.copy().dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(df) < max(50, horizon + 5):
        return {
            "daily_pnl": 0.0,
            "daily_return_on_account": 0.0,
            "daily_return_on_notional": 0.0,
            "daily_notional_used": 0.0,
            "avg_entry_notional": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "pr_auc": np.nan,
            "mfe_sum": 0.0,
            "mae_sum": 0.0,
        }

    y = make_label_hit_tp_local(df, horizon=horizon, tp=tp)
    valid = df.dropna(subset=feature_cols).copy()
    y_valid = y.reindex(valid.index).dropna()
    X_valid = valid.loc[y_valid.index, feature_cols]
    probs_all = model.predict_proba(X_valid)[:, 1] if len(X_valid) else np.array([])
    pr_auc = average_precision_score(y_valid.astype(int), probs_all) if len(np.unique(y_valid.astype(int))) > 1 else np.nan

    total_pnl = 0.0
    total_notional = 0.0
    entry_notionals = []
    trades: list[DayTrade] = []

    in_pos = False
    entry_px = None
    entry_ts = None
    entry_i = None

    probs = pd.Series(index=df.index, dtype=float)
    probs.loc[X_valid.index] = probs_all

    idxs = list(df.index)
    for i, ts in enumerate(idxs[:-1]):
        prob = probs.get(ts, np.nan)
        if np.isnan(prob):
            continue

        if not in_pos and prob >= p_enter:
            entry_bar_i = i + 1
            if entry_bar_i >= len(df):
                continue
            entry_ts = idxs[entry_bar_i]
            entry_px = float(df.iloc[entry_bar_i]["Open"])
            entry_i = entry_bar_i
            in_pos = True
            entry_notional = entry_px * qty
            entry_notionals.append(entry_notional)
            total_notional += entry_notional
            continue

        if in_pos and entry_i is not None and entry_px is not None and entry_ts is not None:
            if i < entry_i:
                continue
            bars_since_entry = i - entry_i + 1
            hi = float(df.iloc[entry_i:i + 1]["High"].max())
            lo = float(df.iloc[entry_i:i + 1]["Low"].min())
            tp_px = entry_px * (1 + tp)
            sl_px = entry_px * (1 - sl)

            exit_reason = None
            exit_px = None
            exit_ts = ts
            if lo <= sl_px:
                exit_reason = "SL"
                exit_px = sl_px
            elif hi >= tp_px:
                exit_reason = "TP"
                exit_px = tp_px
            elif bars_since_entry >= horizon:
                exit_reason = "TIME"
                exit_px = float(df.loc[ts, "Close"])

            if exit_reason is not None:
                pnl = (exit_px - entry_px) * qty
                total_pnl += pnl
                hold_minutes = int((exit_ts - entry_ts).total_seconds() // 60)
                mfe = max(0.0, (hi - entry_px) * qty)
                mae = min(0.0, (lo - entry_px) * qty)
                pnl_pct_on_notional = pnl / (entry_px * qty) if entry_px * qty else 0.0
                trades.append(DayTrade(
                    entry_ts=entry_ts,
                    exit_ts=exit_ts,
                    entry_px=entry_px,
                    exit_px=exit_px,
                    qty=qty,
                    pnl=pnl,
                    exit_reason=exit_reason,
                    hold_minutes=hold_minutes,
                    mfe=mfe,
                    mae=mae,
                    pnl_pct_on_notional=pnl_pct_on_notional,
                ))
                in_pos = False
                entry_px = None
                entry_ts = None
                entry_i = None

    if in_pos and entry_i is not None and entry_px is not None and entry_ts is not None:
        last_ts = idxs[-1]
        exit_px = float(df.iloc[-1]["Close"])
        hi = float(df.iloc[entry_i:]["High"].max())
        lo = float(df.iloc[entry_i:]["Low"].min())
        pnl = (exit_px - entry_px) * qty
        total_pnl += pnl
        hold_minutes = int((last_ts - entry_ts).total_seconds() // 60)
        mfe = max(0.0, (hi - entry_px) * qty)
        mae = min(0.0, (lo - entry_px) * qty)
        pnl_pct_on_notional = pnl / (entry_px * qty) if entry_px * qty else 0.0
        trades.append(DayTrade(
            entry_ts=entry_ts,
            exit_ts=last_ts,
            entry_px=entry_px,
            exit_px=exit_px,
            qty=qty,
            pnl=pnl,
            exit_reason="EOD",
            hold_minutes=hold_minutes,
            mfe=mfe,
            mae=mae,
            pnl_pct_on_notional=pnl_pct_on_notional,
        ))

    wins = sum(1 for t in trades if t.pnl > 0)
    daily_notional_used = total_notional
    avg_entry_notional = float(np.mean(entry_notionals)) if entry_notionals else 0.0
    daily_return_on_notional = (total_pnl / daily_notional_used) if daily_notional_used > 0 else 0.0

    return {
        "daily_pnl": total_pnl,
        "daily_return_on_account": 0.0,
        "daily_return_on_notional": daily_return_on_notional,
        "daily_notional_used": daily_notional_used,
        "avg_entry_notional": avg_entry_notional,
        "total_trades": len(trades),
        "win_rate": (wins / len(trades)) if trades else 0.0,
        "pr_auc": pr_auc,
        "mfe_sum": float(sum(t.mfe for t in trades)),
        "mae_sum": float(sum(t.mae for t in trades)),
    }


def walkforward_evaluate(
    df: pd.DataFrame,
    symbol: str,
    feature_cols: list[str],
    model_params: dict,
    horizon: int,
    tp: float,
    sl: float,
    p_enter: float,
    qty: int,
    initial_cash: float,
    train_days: int,
    test_days: int,
    eval_days: int,
    step_days: int,
) -> tuple[dict, list[dict]]:
    df = df.copy().sort_index()
    trading_days = get_trading_days(df.index)
    needed_days = train_days + eval_days * step_days + test_days - 1
    if len(trading_days) < needed_days:
        raise ValueError(f"Not enough trading days for {symbol}. Needed at least {needed_days}, got {len(trading_days)}")

    daily_rows = []
    pr_aucs = []
    equity = initial_cash

    start_test_idx = len(trading_days) - (eval_days * step_days + test_days - 1)

    for roll_idx in range(eval_days):
        test_start_idx = start_test_idx + roll_idx * step_days
        test_end_idx = test_start_idx + test_days - 1
        train_start_idx = test_start_idx - train_days
        train_end_idx = test_start_idx - 1
        if train_start_idx < 0:
            continue

        train_days_set = set(trading_days[train_start_idx:train_end_idx + 1])
        test_days_set = set(trading_days[test_start_idx:test_end_idx + 1])

        index_dates = pd.Series(df.index.date, index=df.index)
        train_mask = index_dates.isin(train_days_set).to_numpy()
        test_mask = index_dates.isin(test_days_set).to_numpy()
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        train_df = train_df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
        test_df = test_df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(train_df) < 200 or len(test_df) < 50:
            continue

        y_train = make_label_hit_tp_local(train_df, horizon=horizon, tp=tp)
        train_valid = train_df.dropna(subset=feature_cols).copy()
        y_train = y_train.reindex(train_valid.index).dropna()
        X_train = train_valid.loc[y_train.index, feature_cols]
        if len(X_train) < 100 or len(np.unique(y_train.astype(int))) < 2:
            continue

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train.astype(int))

        day_result = simulate_day(
            test_df=test_df,
            model=model,
            feature_cols=feature_cols,
            horizon=horizon,
            tp=tp,
            sl=sl,
            p_enter=p_enter,
            qty=qty,
        )
        equity += day_result["daily_pnl"]
        daily_rows.append({
            "symbol": symbol,
            "test_date": min(test_days_set).isoformat(),
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "p_enter": p_enter,
            "qty": qty,
            "daily_pnl": day_result["daily_pnl"],
            "daily_return_on_account": day_result["daily_pnl"] / initial_cash,
            "daily_return_on_notional": day_result["daily_return_on_notional"],
            "daily_notional_used": day_result["daily_notional_used"],
            "avg_entry_notional": day_result["avg_entry_notional"],
            "total_trades": day_result["total_trades"],
            "win_rate": day_result["win_rate"],
            "pr_auc": day_result["pr_auc"],
            "mfe_sum": day_result["mfe_sum"],
            "mae_sum": day_result["mae_sum"],
            "equity_after_day": equity,
        })
        pr_aucs.append(day_result["pr_auc"])

    if not daily_rows:
        raise ValueError(f"No valid walk-forward rows for {symbol}")

    daily_df = pd.DataFrame(daily_rows)
    stats = {
        "pr_auc": float(np.nanmean(pr_aucs)) if len(pr_aucs) else np.nan,
        "avg_daily_pnl": float(daily_df["daily_pnl"].mean()),
        "median_daily_pnl": float(daily_df["daily_pnl"].median()),
        "worst_day_pnl": float(daily_df["daily_pnl"].min()),
        "best_day_pnl": float(daily_df["daily_pnl"].max()),
        "avg_daily_return_on_account": float(daily_df["daily_return_on_account"].mean()),
        "avg_daily_return_on_notional": float(daily_df["daily_return_on_notional"].mean()),
        "median_daily_return_on_notional": float(daily_df["daily_return_on_notional"].median()),
        "worst_day_return_on_notional": float(daily_df["daily_return_on_notional"].min()),
        "best_day_return_on_notional": float(daily_df["daily_return_on_notional"].max()),
        "positive_day_ratio": float((daily_df["daily_pnl"] > 0).mean()),
        "total_pnl": float(daily_df["daily_pnl"].sum()),
        "total_trades": int(daily_df["total_trades"].sum()),
        "avg_trades_per_day": float(daily_df["total_trades"].mean()),
        "avg_daily_notional_used": float(daily_df["daily_notional_used"].mean()),
        "median_daily_notional_used": float(daily_df["daily_notional_used"].median()),
        "avg_entry_notional": float(daily_df["avg_entry_notional"].replace(0, np.nan).mean()) if (daily_df["avg_entry_notional"] > 0).any() else 0.0,
        "eval_days_used": int(len(daily_df)),
        "train_days": train_days,
        "equity_final": float(initial_cash + daily_df["daily_pnl"].sum()),
    }
    return stats, daily_rows


def make_selection_reason(best_stats: dict, runner_up_symbol: str | None, runner_up_stats: dict | None) -> dict:
    if runner_up_symbol is None or runner_up_stats is None:
        return {
            "primary_metric": "avg_daily_return_on_notional",
            "summary": (
                f"{best_stats['symbol']} was selected because it was the only valid ETF with walk-forward results, "
                f"and its best params had the highest average daily return on notional."
            ),
        }
    return {
        "primary_metric": "avg_daily_return_on_notional",
        "summary": (
            f"{best_stats['symbol']} was selected because it had the highest average daily return on notional over the last "
            f"{best_stats['eval_days_used']} rolling test days after training on the prior {best_stats['train_days']} trading days for each test day."
        ),
        "runner_up_symbol": runner_up_symbol,
        "selected_avg_daily_return_on_notional": best_stats["avg_daily_return_on_notional"],
        "runner_up_avg_daily_return_on_notional": runner_up_stats["avg_daily_return_on_notional"],
        "selected_avg_daily_pnl": best_stats["avg_daily_pnl"],
        "runner_up_avg_daily_pnl": runner_up_stats["avg_daily_pnl"],
        "selected_positive_day_ratio": best_stats["positive_day_ratio"],
        "runner_up_positive_day_ratio": runner_up_stats["positive_day_ratio"],
        "selected_median_daily_return_on_notional": best_stats["median_daily_return_on_notional"],
        "runner_up_median_daily_return_on_notional": runner_up_stats["median_daily_return_on_notional"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,IWM,SOXL,TQQQ")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--out_dir", default="selector_out_416")
    ap.add_argument("--qty", type=int, default=10)
    ap.add_argument("--initial_cash", type=float, default=100000)
    ap.add_argument("--train_days", type=int, default=10)
    ap.add_argument("--test_days", type=int, default=1)
    ap.add_argument("--eval_days", type=int, default=15)
    ap.add_argument("--step_days", type=int, default=1)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "horizon": [20, 30, 45],
        "tp": [0.0075, 0.0100],
        "sl": [0.0035, 0.0050],
        "p_enter": [0.65, 0.70, 0.75, 0.80],
    }
    param_grid = list(product(grid["horizon"], grid["tp"], grid["sl"], grid["p_enter"]))

    end = datetime.now(tz=NY)
    start = end - timedelta(days=args.days)
    run_ts = datetime.now(tz=NY).strftime("%Y%m%d_%H%M%S")

    best_rows = []
    grid_rows = []
    daily_roll_rows = []

    for symbol in symbols:
        print(f"Fetching {symbol} from {start} to {end}")
        raw = fetch_minute_bars(symbol, start, end, feed=args.feed)
        if raw.empty:
            print(f"Skipping {symbol}: no data")
            continue
        raw = filter_market_hours(raw)
        if raw.empty:
            print(f"Skipping {symbol}: no data after market-hours filter")
            continue
        df = add_features(raw)

        symbol_best = None
        symbol_best_stats = None
        symbol_best_daily_rows = None

        for horizon, tp, sl, p_enter in param_grid:
            try:
                stats, daily_rows = walkforward_evaluate(
                    df=df,
                    symbol=symbol,
                    feature_cols=FEATURE_COLS,
                    model_params=MODEL_PARAMS,
                    horizon=horizon,
                    tp=tp,
                    sl=sl,
                    p_enter=p_enter,
                    qty=args.qty,
                    initial_cash=args.initial_cash,
                    train_days=args.train_days,
                    test_days=args.test_days,
                    eval_days=args.eval_days,
                    step_days=args.step_days,
                )
            except Exception as e:
                print(f"  {symbol} horizon={horizon} tp={tp} sl={sl} p_enter={p_enter} failed: {e}")
                continue

            row = {
                "symbol": symbol,
                "horizon": horizon,
                "tp": tp,
                "sl": sl,
                "p_enter": p_enter,
                **stats,
            }
            grid_rows.append(row)
            daily_roll_rows.extend(daily_rows)

            if symbol_best is None:
                symbol_best = row
                symbol_best_stats = stats
                symbol_best_daily_rows = daily_rows
            else:
                better = (
                    row["avg_daily_return_on_notional"] > symbol_best["avg_daily_return_on_notional"] or
                    (
                        np.isclose(row["avg_daily_return_on_notional"], symbol_best["avg_daily_return_on_notional"]) and
                        row["median_daily_return_on_notional"] > symbol_best["median_daily_return_on_notional"]
                    ) or
                    (
                        np.isclose(row["avg_daily_return_on_notional"], symbol_best["avg_daily_return_on_notional"]) and
                        np.isclose(row["median_daily_return_on_notional"], symbol_best["median_daily_return_on_notional"]) and
                        row["positive_day_ratio"] > symbol_best["positive_day_ratio"]
                    )
                )
                if better:
                    symbol_best = row
                    symbol_best_stats = stats
                    symbol_best_daily_rows = daily_rows

        if symbol_best is None:
            print(f"Skipping {symbol}: no valid grid result")
            continue

        print({
            "symbol": symbol_best["symbol"],
            "avg_daily_return_on_notional": round(symbol_best["avg_daily_return_on_notional"], 6),
            "avg_daily_pnl": round(symbol_best["avg_daily_pnl"], 4),
            "median_daily_return_on_notional": round(symbol_best["median_daily_return_on_notional"], 6),
            "positive_day_ratio": round(symbol_best["positive_day_ratio"], 4),
            "total_pnl": round(symbol_best["total_pnl"], 4),
            "total_trades": int(symbol_best["total_trades"]),
            "pr_auc": round(symbol_best["pr_auc"], 4) if pd.notna(symbol_best["pr_auc"]) else None,
            "horizon": symbol_best["horizon"],
            "tp": symbol_best["tp"],
            "sl": symbol_best["sl"],
            "p_enter": symbol_best["p_enter"],
        })
        best_rows.append(symbol_best)

    if not best_rows:
        raise RuntimeError("No valid ETF results were produced")

    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values(
        by=["avg_daily_return_on_notional", "median_daily_return_on_notional", "positive_day_ratio", "avg_daily_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    selected = best_df.iloc[0].to_dict()
    runner_up = best_df.iloc[1].to_dict() if len(best_df) > 1 else None

    selection_json = {
        "version": "4.1.6",
        "selection_mode": "single_best_etf_daily_return_on_notional_walkforward_highconviction",
        "selected_symbol": selected["symbol"],
        "universe": symbols,
        "feature_cols": FEATURE_COLS,
        "model_params": MODEL_PARAMS,
        "best_strategy_params": {
            "horizon": int(selected["horizon"]),
            "tp": float(selected["tp"]),
            "sl": float(selected["sl"]),
            "p_enter": float(selected["p_enter"]),
        },
        "best_backtest_stats": {
            "pr_auc": float(selected["pr_auc"]) if pd.notna(selected["pr_auc"]) else None,
            "avg_daily_pnl": float(selected["avg_daily_pnl"]),
            "median_daily_pnl": float(selected["median_daily_pnl"]),
            "worst_day_pnl": float(selected["worst_day_pnl"]),
            "best_day_pnl": float(selected["best_day_pnl"]),
            "avg_daily_return_on_account": float(selected["avg_daily_return_on_account"]),
            "avg_daily_return_on_notional": float(selected["avg_daily_return_on_notional"]),
            "median_daily_return_on_notional": float(selected["median_daily_return_on_notional"]),
            "worst_day_return_on_notional": float(selected["worst_day_return_on_notional"]),
            "best_day_return_on_notional": float(selected["best_day_return_on_notional"]),
            "positive_day_ratio": float(selected["positive_day_ratio"]),
            "total_pnl": float(selected["total_pnl"]),
            "total_trades": int(selected["total_trades"]),
            "avg_trades_per_day": float(selected["avg_trades_per_day"]),
            "avg_daily_notional_used": float(selected["avg_daily_notional_used"]),
            "median_daily_notional_used": float(selected["median_daily_notional_used"]),
            "avg_entry_notional": float(selected["avg_entry_notional"]),
            "eval_days_used": int(selected["eval_days_used"]),
            "train_days": int(selected["train_days"]),
            "equity_final": float(selected["equity_final"]),
        },
        "selection_reason": make_selection_reason(selected, runner_up["symbol"] if runner_up else None, runner_up),
        "walkforward": {
            "train_days": args.train_days,
            "test_days": args.test_days,
            "eval_days": args.eval_days,
            "step_days": args.step_days,
            "primary_metric": "avg_daily_return_on_notional",
        },
        "market_hours": {
            "open": "10:00",
            "close": "15:30",
        },
        "data_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "calendar_days": args.days,
            "feed": args.feed,
        },
        "qty": args.qty,
        "initial_cash": args.initial_cash,
    }

    print("\nSELECTED ETF")
    print(json.dumps(selection_json, indent=2))

    selection_path = out_dir / f"best_etf_selection_{run_ts}.json"
    best_by_symbol_path = out_dir / f"best_etf_by_symbol_{run_ts}.csv"
    grid_results_path = out_dir / f"grid_results_{run_ts}.csv"
    daily_roll_path = out_dir / f"daily_roll_results_{run_ts}.csv"

    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection_json, f, indent=2)

    best_df.to_csv(best_by_symbol_path, index=False)
    pd.DataFrame(grid_rows).to_csv(grid_results_path, index=False)
    pd.DataFrame(daily_roll_rows).to_csv(daily_roll_path, index=False)

    print(f"Saved selection JSON: {selection_path}")
    print(f"Saved best-by-symbol CSV: {best_by_symbol_path}")
    print(f"Saved grid results CSV: {grid_results_path}")
    print(f"Saved daily roll results CSV: {daily_roll_path}")


if __name__ == "__main__":
    main()
