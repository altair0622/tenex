#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


NY = ZoneInfo("America/New_York")
FEATURE_COLS = [
    "RSI", "Buying Probability", "Selling Probability", "vwap", "EMA", "IC",
    "MACD", "Signal_Line", "%K", "%D", "OBV", "Price_Oscillator",
    "Open_n", "High_n", "Low_n", "Close_n", "hour", "minute",
]


@dataclass
class ComboResult:
    symbol: str
    horizon: int
    tp: float
    sl: float | None
    p_enter: float
    rows: int
    train_rows: int
    test_rows: int
    pr_auc: float
    equity_final: float
    return_pct: float
    trades: int
    win_rate: float
    avg_pnl_per_trade: float


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
    mask_invalid = entry.isna().to_numpy() | np.isnan(fwd_max_high)
    y[mask_invalid] = np.nan
    return pd.Series(y, index=df.index)


def simulate_single_symbol(
    df_test: pd.DataFrame,
    probs: pd.Series,
    qty: int,
    initial_cash: float,
    tp: float,
    sl: float | None,
    horizon: int,
    p_enter: float,
) -> tuple[float, pd.DataFrame]:
    cash = float(initial_cash)
    in_pos = False
    entry_px = None
    entry_ts = None
    entry_bar_idx = None
    trades: list[dict] = []

    probs = probs.reindex(df_test.index)
    idx = list(range(len(df_test)))

    for i in idx[:-1]:
        ts = df_test.index[i]
        row = df_test.iloc[i]

        if in_pos and entry_bar_idx is not None:
            minutes_in = i - entry_bar_idx
            tp_px = entry_px * (1 + tp)
            sl_px = entry_px * (1 - sl) if sl is not None else None
            exit_reason = None
            exit_px = None

            if sl_px is not None and float(row["Low"]) <= sl_px:
                exit_reason = "SL"
                exit_px = sl_px
            elif float(row["High"]) >= tp_px:
                exit_reason = "TP"
                exit_px = tp_px
            elif minutes_in >= horizon:
                exit_reason = "TIME"
                exit_px = float(row["Close"])

            if exit_reason is not None:
                pnl = (exit_px - entry_px) * qty
                cash += exit_px * qty
                trades.append({
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "qty": qty,
                    "minutes_in": minutes_in,
                    "reason": exit_reason,
                    "pnl": pnl,
                })
                in_pos = False
                entry_px = None
                entry_ts = None
                entry_bar_idx = None
                continue

        if (not in_pos) and (float(probs.iloc[i]) >= p_enter):
            next_open = float(df_test.iloc[i + 1]["Open"])
            if cash >= next_open * qty:
                cash -= next_open * qty
                in_pos = True
                entry_px = next_open
                entry_ts = df_test.index[i + 1]
                entry_bar_idx = i + 1

    if in_pos and entry_px is not None:
        last_ts = df_test.index[-1]
        last_close = float(df_test.iloc[-1]["Close"])
        cash += last_close * qty
        trades.append({
            "entry_ts": entry_ts,
            "exit_ts": last_ts,
            "entry_px": entry_px,
            "exit_px": last_close,
            "qty": qty,
            "minutes_in": len(df_test) - 1 - (entry_bar_idx or 0),
            "reason": "EOD",
            "pnl": (last_close - entry_px) * qty,
        })

    trades_df = pd.DataFrame(trades)
    return cash, trades_df


def evaluate_symbol(
    symbol: str,
    df_raw: pd.DataFrame,
    horizons: list[int],
    tps: list[float],
    sls: list[float | None],
    p_enters: list[float],
    qty: int,
    initial_cash: float,
    train_ratio: float,
    model_params: dict,
) -> tuple[ComboResult, pd.DataFrame]:
    df = add_features(df_raw)
    df = df.dropna(subset=FEATURE_COLS + ["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(df) < 500:
        raise ValueError(f"Not enough usable rows for {symbol}: {len(df)}")

    results: list[ComboResult] = []
    best_trades = pd.DataFrame()
    best_key = None

    for horizon in horizons:
        for tp in tps:
            y = make_label_hit_tp_local(df, horizon=horizon, tp=tp)
            work = df.copy()
            work["y"] = y
            work = work.dropna(subset=FEATURE_COLS + ["y"]).copy()
            if len(work) < 500:
                continue

            split_idx = max(200, int(len(work) * train_ratio))
            if split_idx >= len(work) - 50:
                continue

            train = work.iloc[:split_idx].copy()
            test = work.iloc[split_idx:].copy()
            X_train = train[FEATURE_COLS]
            y_train = train["y"].astype(int)
            X_test = test[FEATURE_COLS]
            y_test = test["y"].astype(int)

            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)
            prob_test = pd.Series(model.predict_proba(X_test)[:, 1], index=test.index, name="prob")

            pr_auc = float(average_precision_score(y_test, prob_test)) if y_test.nunique() > 1 else math.nan

            for sl in sls:
                for p_enter in p_enters:
                    equity_final, trades_df = simulate_single_symbol(
                        df_test=test,
                        probs=prob_test,
                        qty=qty,
                        initial_cash=initial_cash,
                        tp=tp,
                        sl=sl,
                        horizon=horizon,
                        p_enter=p_enter,
                    )
                    trades = int(len(trades_df))
                    if trades > 0:
                        win_rate = float((trades_df["pnl"] > 0).mean())
                        avg_pnl = float(trades_df["pnl"].mean())
                    else:
                        win_rate = 0.0
                        avg_pnl = 0.0

                    result = ComboResult(
                        symbol=symbol,
                        horizon=horizon,
                        tp=tp,
                        sl=sl,
                        p_enter=p_enter,
                        rows=int(len(work)),
                        train_rows=int(len(train)),
                        test_rows=int(len(test)),
                        pr_auc=pr_auc,
                        equity_final=float(equity_final),
                        return_pct=float((equity_final / initial_cash) - 1.0),
                        trades=trades,
                        win_rate=win_rate,
                        avg_pnl_per_trade=avg_pnl,
                    )
                    results.append(result)

                    rank_key = (
                        result.equity_final,
                        0 if np.isnan(result.pr_auc) else result.pr_auc,
                        result.win_rate,
                        -abs(result.p_enter - 0.60),
                    )
                    if best_key is None or rank_key > best_key:
                        best_key = rank_key
                        best_trades = trades_df.copy()

    if not results:
        raise ValueError(f"No valid combo results for {symbol}")

    results_df = pd.DataFrame([asdict(r) for r in results]).sort_values(
        ["equity_final", "pr_auc", "win_rate"], ascending=[False, False, False]
    )
    top = results_df.iloc[0].to_dict()
    best = ComboResult(
        symbol=str(top["symbol"]),
        horizon=int(top["horizon"]),
        tp=float(top["tp"]),
        sl=None if pd.isna(top["sl"]) else float(top["sl"]),
        p_enter=float(top["p_enter"]),
        rows=int(top["rows"]),
        train_rows=int(top["train_rows"]),
        test_rows=int(top["test_rows"]),
        pr_auc=float(top["pr_auc"]) if not pd.isna(top["pr_auc"]) else math.nan,
        equity_final=float(top["equity_final"]),
        return_pct=float(top["return_pct"]),
        trades=int(top["trades"]),
        win_rate=float(top["win_rate"]),
        avg_pnl_per_trade=float(top["avg_pnl_per_trade"]),
    )
    return best, results_df


def resolve_output_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(tz=NY).strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,IWM,SOXL,TQQQ")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--qty", type=int, default=10)
    ap.add_argument("--initial_cash", type=float, default=100000)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--horizons", default="5,10,15")
    ap.add_argument("--tps", default="0.003,0.005,0.0075")
    ap.add_argument("--sls", default="0.002,0.0035")
    ap.add_argument("--p_enters", default="0.55,0.60,0.65")
    ap.add_argument("--market_open", default="09:30")
    ap.add_argument("--market_close", default="15:55")
    ap.add_argument("--out_dir", default="selector_out")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_samples_leaf", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    horizons = parse_int_list(args.horizons)
    tps = parse_float_list(args.tps)
    sls = parse_float_list(args.sls)
    p_enters = parse_float_list(args.p_enters)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    }

    end = datetime.now(tz=NY)
    start = end - timedelta(days=args.days)

    best_by_symbol: list[ComboResult] = []
    all_symbol_rows: list[pd.DataFrame] = []

    for symbol in symbols:
        print(f"Fetching {symbol} from {start} to {end}")
        df_raw = fetch_minute_bars(symbol, start, end, feed=args.feed)
        if df_raw.empty:
            print(f"Skipping {symbol}: no data")
            continue
        try:
            best, results_df = evaluate_symbol(
                symbol=symbol,
                df_raw=df_raw,
                horizons=horizons,
                tps=tps,
                sls=sls,
                p_enters=p_enters,
                qty=args.qty,
                initial_cash=args.initial_cash,
                train_ratio=args.train_ratio,
                model_params=model_params,
            )
            best_by_symbol.append(best)
            all_symbol_rows.append(results_df)
            print({
                "symbol": best.symbol,
                "equity_final": round(best.equity_final, 2),
                "return_pct": round(best.return_pct * 100, 2),
                "trades": best.trades,
                "pr_auc": None if np.isnan(best.pr_auc) else round(best.pr_auc, 4),
                "horizon": best.horizon,
                "tp": best.tp,
                "sl": best.sl,
                "p_enter": best.p_enter,
            })
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

    if not best_by_symbol:
        raise RuntimeError("No symbols produced valid backtest results")

    best_by_symbol_df = pd.DataFrame([asdict(x) for x in best_by_symbol]).sort_values(
        ["equity_final", "pr_auc", "win_rate"], ascending=[False, False, False]
    )
    selected = best_by_symbol_df.iloc[0].to_dict()
    selected_symbol = str(selected["symbol"])

    selection_payload = {
        "version": "4.1.3",
        "selection_mode": "single_best_etf",
        "selected_symbol": selected_symbol,
        "universe": symbols,
        "feature_cols": FEATURE_COLS,
        "model_params": model_params,
        "best_strategy_params": {
            "horizon": int(selected["horizon"]),
            "tp": float(selected["tp"]),
            "sl": None if pd.isna(selected["sl"]) else float(selected["sl"]),
            "p_enter": float(selected["p_enter"]),
        },
        "best_backtest_stats": {
            "rows": int(selected["rows"]),
            "train_rows": int(selected["train_rows"]),
            "test_rows": int(selected["test_rows"]),
            "pr_auc": None if pd.isna(selected["pr_auc"]) else float(selected["pr_auc"]),
            "equity_final": float(selected["equity_final"]),
            "return_pct": float(selected["return_pct"]),
            "trades": int(selected["trades"]),
            "win_rate": float(selected["win_rate"]),
            "avg_pnl_per_trade": float(selected["avg_pnl_per_trade"]),
        },
        "market_hours": {
            "open": args.market_open,
            "close": args.market_close,
        },
        "data_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": args.days,
            "feed": args.feed,
            "train_ratio": args.train_ratio,
        },
        "qty": args.qty,
        "initial_cash": args.initial_cash,
    }

    ts = resolve_output_name("best_etf_selection")
    json_path = out_dir / f"{ts}.json"
    per_symbol_path = out_dir / f"best_etf_by_symbol_{datetime.now(tz=NY).strftime('%Y%m%d_%H%M%S')}.csv"
    all_results_path = out_dir / f"grid_results_{datetime.now(tz=NY).strftime('%Y%m%d_%H%M%S')}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)

    best_by_symbol_df.to_csv(per_symbol_path, index=False)
    pd.concat(all_symbol_rows, ignore_index=True).to_csv(all_results_path, index=False)

    print("\nSELECTED ETF")
    print(json.dumps(selection_payload, indent=2))
    print(f"Saved selection JSON: {json_path}")
    print(f"Saved best-by-symbol CSV: {per_symbol_path}")
    print(f"Saved grid results CSV: {all_results_path}")


if __name__ == "__main__":
    main()
