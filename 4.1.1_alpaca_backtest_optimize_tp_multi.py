#!/usr/bin/env python3
"""
alpaca_backtest_optimize_tp_multi.py

What this script does
---------------------
1) Pull 1-minute OHLCV bars from Alpaca for one OR many symbols.
2) Build features (RSI, VWAP, EMA, MACD, Stoch, OBV, Price Oscillator, Bollinger + *_n normalization).
3) Define label for the NEW goal:
      "If we enter, will price touch +tp within H minutes?"
4) Walk-forward (rolling) training -> probability predictions (no random split).
5) Strategy backtest:
      enter when prob >= p_enter
      exit via TP / optional SL / time-exit at H minutes
6) Grid-search parameters and save:
      - best_params_<SYMBOL>.json  (per-symbol best)
      - summary_best.csv           (one line per symbol)
      - grid_<SYMBOL>.csv          (optional full grid per symbol)

Costs model (important)
-----------------------
We support 3 cost components:
- commission_per_order_usd : fixed $ cost per order (buy or sell). Total per round trip = 2 * commission.
- fee_bps                 : proportional fee in basis points applied to (entry notional + exit notional)
- slippage_bps            : modeled by worsening fills:
                              entry price *= (1 + slippage_bps/10000)
                              exit  price *= (1 - slippage_bps/10000)

Because fixed commission depends on position size, we model position sizing via:
- notional_usd : dollars allocated per trade (used only for commission impact)
  shares = notional_usd / entry_price

Requirements
------------
  pip install alpaca-py pandas numpy scikit-learn

Env vars
--------
  ALPACA_API_KEY_1
  ALPACA_SECRET_KEY_1

Example
-------
python alpaca_backtest_optimize_tp_multi.py \
  --symbols SOXL,TQQQ,SPY \
  --days 90 \
  --feed iex \
  --commission_per_order_usd 0.35 \
  --slippage_bps 2 \
  --fee_bps 0 \
  --out_dir params_out
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
# Indicators / features (matches your notebook style)
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
    # Reset per day (NY)
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

    # IMPORTANT: keep raw OHLC, create *_n derived columns
    for col in ["Open", "High", "Low", "Close", "Previous Close"]:
        d[f"{col}_n"] = (d[col] - d["Bollinger_Lower"]) / band

    d["hour"] = d.index.hour
    d["minute"] = d.index.minute
    return d


# -------------------------
# Label: TP hit within H minutes after entry
# -------------------------
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
# Walk-forward predict
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

        mask = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr2, y_tr2 = X_tr[mask], y_tr[mask]

        if len(X_tr2) < 200:
            start += test_size
            continue

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
    sl: float | None,
    fee_bps: float,
    commission_per_order_usd: float,
    slippage_bps: float,
    notional_usd: float,
    entry_on_next_open: bool = True,
) -> BacktestResult:
    """
    One-position-at-a-time backtest on minute bars.

    Fill model:
      entry_px = raw_entry * (1 + slippage_bps/10000)
      exit_px  = raw_exit  * (1 - slippage_bps/10000)

    Costs:
      proportional: fee_bps applied to entry_notional + exit_notional
      fixed: 2 * commission_per_order_usd per round trip
      fixed impact converted to return via notional_usd
    """
    prob = prob.reindex(df.index)

    trades = []
    i = 0
    n = len(df)

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
        exit_reason, exit_i = "TIME", last_i

        for j in range(entry_i, last_i + 1):
            hi = float(df["High"].iat[j]) * (1 - slip)  # conservative: you sell lower
            lo = float(df["Low"].iat[j]) * (1 - slip)

            if sl_px is not None and lo <= sl_px:
                exit_reason, exit_i = "SL", j
                raw_exit_px = sl_px
                break
            if hi >= tp_px:
                exit_reason, exit_i = "TP", j
                raw_exit_px = tp_px
                break
        else:
            raw_exit_px = float(df["Close"].iat[exit_i]) * (1 - slip)

        exit_px = float(raw_exit_px)

        # position sizing only needed to model fixed commission impact
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
            "entry_px": entry_px,
            "exit_px": exit_px,
            "exit_reason": exit_reason,
            "ret": float(ret),
            "signal_prob": float(prob.iat[i]),
        })

        i = exit_i + 1

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
        "equity_final": float(equity.iloc[-1]),
        "max_drawdown": float(mdd),
        "tp_rate": float((trades_df["exit_reason"] == "TP").mean()),
        "time_rate": float((trades_df["exit_reason"] == "TIME").mean()),
        "sl_rate": float((trades_df["exit_reason"] == "SL").mean()) if sl is not None else None,
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
    model_params: dict,
    fee_bps: float,
    commission_per_order_usd: float,
    slippage_bps: float,
    notional_usd: float,
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

            for p_enter in p_enters:
                for sl in sls:
                    bt = backtest_prob_strategy(
                        df=df_feat[ok],
                        prob=prob[ok],
                        horizon=horizon,
                        tp=tp,
                        p_enter=p_enter,
                        sl=sl,
                        fee_bps=fee_bps,
                        commission_per_order_usd=commission_per_order_usd,
                        slippage_bps=slippage_bps,
                        notional_usd=notional_usd,
                        entry_on_next_open=True,
                    )
                    if bt.stats.get("trades", 0) < min_trades:
                        continue
                    if bt.stats.get("max_drawdown", 0) < -abs(max_mdd):
                        continue

                    rows.append({
                        "horizon": horizon,
                        "tp": tp,
                        "p_enter": p_enter,
                        "sl": np.nan if sl is None else sl,
                        **bt.stats,
                    })

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    res = res.sort_values(["equity_final", "avg_ret"], ascending=False).reset_index(drop=True)
    return res


def parse_symbols(s: str) -> list[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SOXL", help="Comma-separated symbols, e.g. SOXL,TQQQ,SPY")
    ap.add_argument("--days", type=int, default=90, help="Lookback days for backtest data")
    ap.add_argument("--feed", default="iex", help="iex or sip (depending on your subscription)")
    ap.add_argument("--market_open", default="10:00")
    ap.add_argument("--market_close", default="15:30")

    ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--test_size", type=int, default=500)

    # costs
    ap.add_argument("--fee_bps", type=float, default=0.0, help="Proportional fee in bps (round trip applied to entry+exit notionals)")
    ap.add_argument("--commission_per_order_usd", type=float, default=0.0, help="Fixed commission per order in USD (buy or sell)")
    ap.add_argument("--slippage_bps", type=float, default=0.0, help="Fill slippage in bps (worsen entry, worsen exit)")

    ap.add_argument("--notional_usd", type=float, default=10000.0, help="Position notional used for commission impact (does not change signal logic)")

    ap.add_argument("--out_dir", default="params_out", help="Directory to write outputs")
    ap.add_argument("--save_full_grid", action="store_true", help="If set, save full grid CSV per symbol")

    # constraints
    ap.add_argument("--min_trades", type=int, default=30)
    ap.add_argument("--max_mdd", type=float, default=0.10, help="Max allowed drawdown magnitude, e.g. 0.10 = -10%")

    args = ap.parse_args()

    end = datetime.now(tz=NY)
    start = end - timedelta(days=args.days)

    os.makedirs(args.out_dir, exist_ok=True)

    # market hours filter
    mo_h, mo_m = map(int, args.market_open.split(":"))
    mc_h, mc_m = map(int, args.market_close.split(":"))

    symbols = parse_symbols(args.symbols)

    # model baseline (extend grid if you want)
    model_params = dict(
        n_estimators=500,
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
            print("No data; skipping.")
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
            "Open_n", "High_n", "Low_n", "Close_n", "Previous Close_n",
            "hour", "minute",
        ]

        df_feat = df_feat.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(df_feat) < (args.train_size + args.test_size + 200):
            print(f"Not enough rows after feature dropna ({len(df_feat)}). Skipping.")
            continue

        res = run_grid_search(
            df_feat=df_feat,
            feature_cols=feature_cols,
            horizons=[5, 10, 15],
            tps=[0.004, 0.005, 0.006],
            p_enters=[0.55, 0.60, 0.65, 0.70, 0.75],
            sls=[None, 0.0025, 0.0035],
            train_size=args.train_size,
            test_size=args.test_size,
            model_params=model_params,
            fee_bps=args.fee_bps,
            commission_per_order_usd=args.commission_per_order_usd,
            slippage_bps=args.slippage_bps,
            notional_usd=args.notional_usd,
            min_trades=args.min_trades,
            max_mdd=args.max_mdd,
        )

        if res.empty:
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
            },
            "best_backtest_stats": {k: best[k] for k in best.keys() if k not in ["horizon", "tp", "p_enter", "sl"]},
        }

        out_json = os.path.join(args.out_dir, f"best_params_{sym}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if args.save_full_grid:
            out_grid = os.path.join(args.out_dir, f"grid_{sym}.csv")
            res.to_csv(out_grid, index=False)

        print("Saved:", out_json)
        print("Best params:", payload["best_strategy_params"])
        print("Stats:", payload["best_backtest_stats"])

        summary_rows.append({
            "symbol": sym,
            **payload["best_strategy_params"],
            **payload["best_backtest_stats"],
            **payload["cost_model"],
        })

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values(["equity_final", "avg_ret"], ascending=False)
        summary_path = os.path.join(args.out_dir, "summary_best.csv")
        summary.to_csv(summary_path, index=False)
        print("\nSaved summary:", summary_path)
        print(summary.head(10).to_string(index=False))
    else:
        print("\nNo symbols produced a valid result.")


if __name__ == "__main__":
    main()
