#!/usr/bin/env python3
"""
alpaca_backtest_optimize_tp_multi.py

Goal
- Fetch 1-minute bars from Alpaca for multiple symbols
- Build the same features as the 4.1.2 single-symbol backtest
- Create label: "Within H minutes after entry, does price touch +tp?"
- Walk-forward train -> predict probabilities
- Backtest strategy: enter if prob >= p_enter, exit via TP / optional SL / time-exit
- Grid-search parameters and export best set to JSON per symbol

Costs model
- commission_per_order_usd : fixed dollar commission per order
- fee_bps                  : proportional fee in bps on entry+exit notionals
- slippage_bps             : worsens entry and exit fills conservatively
- notional_usd             : used to translate fixed commissions into return impact

Requirements
  pip install alpaca-py pandas numpy scikit-learn

Env vars
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

    band_range = (d["Bollinger_Upper"] - d["Bollinger_Lower"]).replace(0, np.nan)
    for col in ["Open", "High", "Low", "Close"]:
        d[col + "_n"] = (d[col] - d["Bollinger_Lower"]) / band_range

    d["hour"] = d.index.hour
    d["minute"] = d.index.minute

    return d


def make_label_hit_tp(df: pd.DataFrame, horizon: int, tp: float, entry_on_next_open: bool = True) -> pd.Series:
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


@dataclass
class WalkForwardResult:
    prob: pd.Series
    n_runs: int
    evaluated_bars: int


def walkforward_train_predict(
    df_feat: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    train_size: int,
    test_size: int,
    model_params: dict,
) -> WalkForwardResult:
    prob = pd.Series(index=df_feat.index, dtype=float)
    start = 0
    n = len(df_feat)
    n_runs = 0

    while start + train_size + test_size <= n:
        tr = slice(start, start + train_size)
        te = slice(start + train_size, start + train_size + test_size)

        X_tr = df_feat.iloc[tr][feature_cols]
        y_tr = y.iloc[tr]
        X_te = df_feat.iloc[te][feature_cols]

        m = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr2, y_tr2 = X_tr[m], y_tr[m]

        if len(X_tr2) == 0:
            start += test_size
            continue

        model = RandomForestClassifier(**model_params)
        model.fit(X_tr2, y_tr2)
        prob.iloc[te] = model.predict_proba(X_te)[:, 1]

        n_runs += 1
        start += test_size

    evaluated_bars = int(prob.notna().sum())
    return WalkForwardResult(prob=prob, n_runs=n_runs, evaluated_bars=evaluated_bars)


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
    commission_per_order_usd: float = 0.0,
    slippage_bps: float = 0.0,
    notional_usd: float = 10000.0,
) -> BacktestResult:
    prob = prob.reindex(df.index)

    evaluated_bars = int(prob.notna().sum())
    raw_signal_opportunities = int((prob >= p_enter).fillna(False).sum())

    trades = []
    i = 0
    n = len(df)
    blocked_signals_while_in_position = 0

    slip = slippage_bps / 10000.0
    fee = fee_bps / 10000.0

    while i < n:
        if pd.isna(prob.iat[i]) or prob.iat[i] < p_enter:
            i += 1
            continue

        entry_i = i + (1 if entry_on_next_open else 0)
        if entry_i >= n:
            break

        raw_entry_px = float(df["Open"].iat[entry_i] if entry_on_next_open else df["Close"].iat[i])
        entry_px = raw_entry_px * (1 + slip)

        tp_px = entry_px * (1 + tp)
        sl_px = entry_px * (1 - sl) if sl is not None else None

        last_i = min(entry_i + horizon - 1, n - 1)
        exit_reason = "TIME"
        exit_i = last_i

        for j in range(entry_i, last_i + 1):
            hi = float(df["High"].iat[j]) * (1 - slip)
            lo = float(df["Low"].iat[j]) * (1 - slip)
            if sl_px is not None and lo <= sl_px:
                exit_reason = "SL"
                exit_i = j
                exit_px = sl_px
                break
            if hi >= tp_px:
                exit_reason = "TP"
                exit_i = j
                exit_px = tp_px
                break
        else:
            exit_px = float(df["Close"].iat[exit_i]) * (1 - slip)

        shares = notional_usd / entry_px if notional_usd > 0 else 1.0
        entry_notional = shares * entry_px
        exit_notional = shares * exit_px
        proportional_cost = fee * (entry_notional + exit_notional)
        fixed_cost = 2.0 * commission_per_order_usd
        pnl = (exit_notional - entry_notional) - proportional_cost - fixed_cost
        ret = pnl / entry_notional

        trades.append({
            "entry_time": df.index[entry_i],
            "exit_time": df.index[exit_i],
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "exit_reason": exit_reason,
            "ret": float(ret),
            "signal_prob": float(prob.iat[i]),
        })

        if exit_i > i:
            blocked_signals_while_in_position += int(
                (prob.iloc[i + 1: exit_i + 1] >= p_enter).fillna(False).sum()
            )

        i = exit_i + 1

    trades_df = pd.DataFrame(trades)
    actual_entries = int(len(trades_df))

    stats = {
        "evaluated_bars": evaluated_bars,
        "raw_signal_opportunities": raw_signal_opportunities,
        "blocked_signals_while_in_position": blocked_signals_while_in_position,
        "actual_entries": actual_entries,
        "entry_rate_vs_evaluated_bars": float(actual_entries / evaluated_bars) if evaluated_bars > 0 else np.nan,
        "entry_rate_vs_raw_signals": float(actual_entries / raw_signal_opportunities) if raw_signal_opportunities > 0 else np.nan,
    }

    if len(trades_df) == 0:
        stats["trades"] = 0
        return BacktestResult(trades_df, stats)

    equity = (1 + trades_df["ret"]).cumprod()
    mdd = (equity / equity.cummax() - 1).min()

    stats.update({
        "trades": actual_entries,
        "win_rate": float((trades_df["ret"] > 0).mean()),
        "avg_ret": float(trades_df["ret"].mean()),
        "median_ret": float(trades_df["ret"].median()),
        "tp_rate": float((trades_df["exit_reason"] == "TP").mean()),
        "sl_rate": float((trades_df["exit_reason"] == "SL").mean()) if sl is not None else None,
        "time_exit_rate": float((trades_df["exit_reason"] == "TIME").mean()),
        "equity_final": float(equity.iloc[-1]),
        "max_drawdown": float(mdd),
    })
    return BacktestResult(trades_df, stats)


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
    commission_per_order_usd: float,
    slippage_bps: float,
    notional_usd: float,
) -> pd.DataFrame:
    rows = []

    for horizon in horizons:
        for tp in tps:
            y = make_label_hit_tp(df_feat, horizon=horizon, tp=tp, entry_on_next_open=True)

            wf = walkforward_train_predict(
                df_feat=df_feat,
                y=y,
                feature_cols=feature_cols,
                train_size=train_size,
                test_size=test_size,
                model_params=model_params,
            )

            prob = wf.prob
            ok = prob.notna()
            if ok.sum() < test_size:
                continue

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
                        commission_per_order_usd=commission_per_order_usd,
                        slippage_bps=slippage_bps,
                        notional_usd=notional_usd,
                    )

                    r = {
                        "horizon": horizon,
                        "tp": tp,
                        "p_enter": p_enter,
                        "sl": sl if sl is not None else None,
                        "walkforward_runs": wf.n_runs,
                        "pr_auc": float(pr_auc),
                        **bt.stats,
                    }
                    rows.append(r)

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    res = res[res["trades"] >= min_trades].copy()
    res = res[res["max_drawdown"] >= -abs(max_mdd)].copy()

    res = res.sort_values(
        ["equity_final", "avg_ret", "max_drawdown"],
        ascending=[False, False, False],
    )
    return res


def parse_symbols(s: str) -> list[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SOXL", help="Comma-separated symbols, e.g. SOXL,TQQQ,SPY")
    ap.add_argument("--days", type=int, default=90, help="Lookback days for backtest data")
    ap.add_argument("--feed", default="iex", help="iex or sip")
    ap.add_argument("--market_open", default="10:00")
    ap.add_argument("--market_close", default="15:30")
    ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--test_size", type=int, default=500)
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--commission_per_order_usd", type=float, default=0.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--notional_usd", type=float, default=10000.0)
    ap.add_argument("--out_dir", default="params_out")
    ap.add_argument("--save_full_grid", action="store_true")
    ap.add_argument("--min_trades", type=int, default=30)
    ap.add_argument("--max_mdd", type=float, default=0.10)
    args = ap.parse_args()

    end = datetime.now(tz=NY)
    start = end - timedelta(days=args.days)
    run_ts = end.strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.out_dir, exist_ok=True)

    mo_h, mo_m = map(int, args.market_open.split(":"))
    mc_h, mc_m = map(int, args.market_close.split(":"))

    symbols = parse_symbols(args.symbols)

    model_params = dict(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    summary_rows = []

    for sym in symbols:
        print(f"\n=== {sym} ===")
        df = fetch_minute_bars(sym, start, end, feed=args.feed)
        if df.empty:
            print("No data returned. Skipping.")
            continue

        t = df.index.time
        df = df[
            (t >= datetime(2000, 1, 1, mo_h, mo_m).time()) &
            (t <= datetime(2000, 1, 1, mc_h, mc_m).time())
        ].copy()

        df_feat = add_features(df)

        feature_cols = [
            "RSI", "Buying Probability", "Selling Probability",
            "vwap", "EMA", "IC", "MACD", "Signal_Line", "%K", "%D", "OBV", "Price_Oscillator",
            "rolling_mean", "Bollinger_Upper", "Bollinger_Lower",
            "Open_n", "High_n", "Low_n", "Close_n",
            "hour", "minute",
        ]

        df_feat = df_feat.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(df_feat) < (args.train_size + args.test_size):
            print(f"Not enough rows after feature dropna ({len(df_feat)}). Skipping.")
            continue

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
            min_trades=args.min_trades,
            max_mdd=args.max_mdd,
            commission_per_order_usd=args.commission_per_order_usd,
            slippage_bps=args.slippage_bps,
            notional_usd=args.notional_usd,
        )

        if len(res) == 0:
            print("No valid parameter set found under constraints.")
            continue

        best = res.iloc[0].to_dict()
        payload = {
            "symbol": sym,
            "asof": end.isoformat(),
            "data_window_days": args.days,
            "market_hours": {"open": args.market_open, "close": args.market_close},
            "feature_cols": feature_cols,
            "model_params": model_params,
            "cost_model": {
                "fee_bps": float(args.fee_bps),
                "commission_per_order_usd": float(args.commission_per_order_usd),
                "slippage_bps": float(args.slippage_bps),
                "notional_usd": float(args.notional_usd),
            },
            "best_strategy_params": {
                "horizon": int(best["horizon"]),
                "tp": float(best["tp"]),
                "p_enter": float(best["p_enter"]),
                "sl": None if pd.isna(best["sl"]) else float(best["sl"]),
                "fee_bps": float(args.fee_bps),
            },
            "best_backtest_stats": {
                k: best[k] for k in best.keys() if k not in ["horizon", "tp", "p_enter", "sl"]
            },
        }

        json_path = os.path.join(args.out_dir, f"best_params_{sym}_{run_ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if args.save_full_grid:
            grid_path = os.path.join(args.out_dir, f"grid_{sym}_{run_ts}.csv")
            res.to_csv(grid_path, index=False)
            print("Saved grid results to:", grid_path)

        print("Saved best params to:", json_path)
        print("Best:", payload["best_strategy_params"])

        stats = payload["best_backtest_stats"]
        print("\nDetailed backtest summary")
        print("walkforward_runs:", stats.get("walkforward_runs"))
        print("evaluated_bars:", stats.get("evaluated_bars"))
        print("raw_signal_opportunities:", stats.get("raw_signal_opportunities"))
        print("blocked_signals_while_in_position:", stats.get("blocked_signals_while_in_position"))
        print("actual_entries:", stats.get("actual_entries"))
        print("entry_rate_vs_evaluated_bars:", stats.get("entry_rate_vs_evaluated_bars"))
        print("entry_rate_vs_raw_signals:", stats.get("entry_rate_vs_raw_signals"))
        print("pr_auc:", stats.get("pr_auc"))
        print("win_rate:", stats.get("win_rate"))
        print("avg_ret:", stats.get("avg_ret"))
        print("equity_final:", stats.get("equity_final"))
        print("max_drawdown:", stats.get("max_drawdown"))

        summary_rows.append({
            "symbol": sym,
            **payload["best_strategy_params"],
            **payload["best_backtest_stats"],
            **payload["cost_model"],
        })

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values(
            ["equity_final", "avg_ret", "max_drawdown"],
            ascending=[False, False, False],
        )
        summary_path = os.path.join(args.out_dir, f"summary_best_{run_ts}.csv")
        summary.to_csv(summary_path, index=False)
        print("\nSaved summary:", summary_path)
        print(summary.head(20).to_string(index=False))
    else:
        print("\nNo symbols produced a valid result.")


if __name__ == "__main__":
    main()
