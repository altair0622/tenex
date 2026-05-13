#!/usr/bin/env python3
"""
4.1.5_live_trade_selected_etf_notional_review.py

Goal:
- Load the latest 4.1.5 selection JSON or a specified one
- Optionally override symbol from the terminal
- Trade a single selected ETF live or on paper
- Save detailed review artifacts for end-of-day analysis
- Emphasize return on notional in the review outputs

Outputs:
- live_events_*.csv
- live_trades_*.csv
- live_equity_*.csv
- live_summary_*.csv
- daily_review_summary.csv
"""

from __future__ import annotations

import os
import csv
import json
import time
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

NY = ZoneInfo("America/New_York")


def resolve_latest_match(pattern_or_file: str) -> Path:
    p = Path(pattern_or_file)
    if p.is_file():
        return p
    matches = sorted(Path().glob(pattern_or_file), key=lambda x: x.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern_or_file}")
    return matches[0]


def paired_best_by_symbol_csv(selection_path: Path) -> Path | None:
    name = selection_path.name
    if not name.startswith("best_etf_selection_"):
        return None
    ts = name.replace("best_etf_selection_", "").replace(".json", "")
    candidate = selection_path.parent / f"best_etf_by_symbol_{ts}.csv"
    return candidate if candidate.exists() else None


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
    return pd.Series((fwd_max_high >= target).astype(float), index=df.index)


def within_market_hours(now: datetime, market_open: dtime, market_close: dtime) -> bool:
    t = now.astimezone(NY).time()
    return market_open <= t <= market_close


def minute_sleep_seconds(now: datetime | None = None, lag_seconds: int = 2) -> int:
    now = now or datetime.now(tz=NY)
    return max(1, 60 - now.second + lag_seconds)


@dataclass
class PositionState:
    in_pos: bool = False
    entry_time: datetime | None = None
    entry_px: float | None = None
    qty: int = 0
    entry_order_id: str | None = None
    entry_signal_prob: float | None = None
    entry_notional: float = 0.0
    max_favorable_px: float | None = None
    max_adverse_px: float | None = None


@dataclass
class SessionMetrics:
    evaluated_bars: int = 0
    raw_signal_opportunities: int = 0
    blocked_signals_while_in_position: int = 0
    actual_entries: int = 0
    completed_exits: int = 0
    retrain_count: int = 0
    tp_exits: int = 0
    sl_exits: int = 0
    time_exits: int = 0
    eod_exits: int = 0
    gross_realized_pnl: float = 0.0


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)


def log_event(logger: CsvLogger, event: str, now: datetime, symbol: str, payload: dict) -> None:
    row = {"ts": now.isoformat(), "event": event, "symbol": symbol}
    row.update(payload)
    logger.write(row)


def get_actual_filled_avg_price(trading: TradingClient, order_id: str, timeout_seconds: int = 20) -> float | None:
    deadline = time.time() + timeout_seconds
    last_seen = None
    while time.time() < deadline:
        try:
            order = trading.get_order_by_id(order_id)
            last_seen = order
            filled_avg_price = getattr(order, "filled_avg_price", None)
            status = str(getattr(order, "status", ""))
            if filled_avg_price not in (None, ""):
                return float(filled_avg_price)
            if status.lower() in {"canceled", "rejected", "expired"}:
                return None
        except Exception:
            pass
        time.sleep(1)
    if last_seen is not None:
        filled_avg_price = getattr(last_seen, "filled_avg_price", None)
        if filled_avg_price not in (None, ""):
            return float(filled_avg_price)
    return None


def sync_position_state(trading: TradingClient, symbol: str, fallback_px: float | None = None) -> PositionState:
    try:
        positions = trading.get_all_positions()
    except Exception:
        return PositionState()

    for pos in positions:
        if getattr(pos, "symbol", None) == symbol:
            qty_raw = getattr(pos, "qty", 0)
            avg_raw = getattr(pos, "avg_entry_price", None)
            qty = int(float(qty_raw)) if qty_raw not in (None, "") else 0
            avg_entry_price = float(avg_raw) if avg_raw not in (None, "") else fallback_px
            entry_notional = (avg_entry_price or 0.0) * qty
            return PositionState(
                in_pos=qty > 0,
                entry_time=None,
                entry_px=avg_entry_price,
                qty=qty,
                entry_notional=entry_notional,
                max_favorable_px=avg_entry_price,
                max_adverse_px=avg_entry_price,
            )
    return PositionState()


def place_market_buy(trading: TradingClient, symbol: str, qty: int):
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    return trading.submit_order(req)


def place_market_sell(trading: TradingClient, symbol: str, qty: int):
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    return trading.submit_order(req)


def review_comment(summary: dict) -> str:
    parts = []
    if summary["net_pnl"] > 0:
        parts.append("The day finished positive.")
    elif summary["net_pnl"] < 0:
        parts.append("The day finished negative.")
    else:
        parts.append("The day finished flat.")

    if summary["blocked_signals"] > summary["actual_entries"]:
        parts.append("Blocked signals were high relative to entries.")
    if summary["time_exits"] > summary["tp_exits"] + summary["sl_exits"]:
        parts.append("Time exits dominated the day, so horizon tuning may matter.")
    if summary["avg_daily_return_on_notional"] >= 0.01:
        parts.append("The strategy met or exceeded the 1% daily return on notional target.")
    elif summary["avg_daily_return_on_notional"] > 0:
        parts.append("Return on notional was positive but below the 1% daily target.")
    else:
        parts.append("Return on notional was not positive today.")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection_json", default="selector_out_415/best_etf_selection_*.json")
    ap.add_argument("--symbols", default=None, help="Override selection symbol, for example TQQQ")
    ap.add_argument("--symbol", default=None, help="Alias for --symbols")
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--qty", type=int, default=None)
    ap.add_argument("--lookback_minutes", type=int, default=1200)
    ap.add_argument("--retrain_minutes", type=int, default=60)
    ap.add_argument("--model_out", default="final_model_415.joblib")
    ap.add_argument("--logs_dir", default="live_logs_415")
    ap.add_argument("--order_fill_timeout", type=int, default=20)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()

    selection_path = resolve_latest_match(args.selection_json)
    with selection_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    override_symbol = (args.symbols or args.symbol)
    best_by_symbol_path = paired_best_by_symbol_csv(selection_path)

    selected_symbol = cfg["selected_symbol"]
    feature_cols = cfg["feature_cols"]
    model_params = cfg["model_params"]
    strat = cfg["best_strategy_params"].copy()

    if override_symbol:
        override_symbol = override_symbol.strip().upper().split(",")[0]
        selected_symbol = override_symbol
        if best_by_symbol_path and best_by_symbol_path.exists():
            best_df = pd.read_csv(best_by_symbol_path)
            sub = best_df[best_df["symbol"].astype(str).str.upper() == override_symbol]
            if not sub.empty:
                row = sub.sort_values(
                    by=["avg_daily_return_on_notional", "median_daily_return_on_notional", "positive_day_ratio", "avg_daily_pnl"],
                    ascending=[False, False, False, False],
                ).iloc[0]
                strat = {
                    "horizon": int(row["horizon"]),
                    "tp": float(row["tp"]),
                    "sl": float(row["sl"]),
                    "p_enter": float(row["p_enter"]),
                }

    horizon = int(strat["horizon"])
    tp = float(strat["tp"])
    sl = float(strat["sl"])
    p_enter = float(strat["p_enter"])
    qty = args.qty if args.qty is not None else int(cfg.get("qty", 10))
    initial_cash = float(cfg.get("initial_cash", 100000))
    train_days = int(cfg.get("walkforward", {}).get("train_days", 10))

    market_open = dtime(*map(int, cfg["market_hours"]["open"].split(":")))
    market_close = dtime(*map(int, cfg["market_hours"]["close"].split(":")))

    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")

    trading = TradingClient(api_key=key, secret_key=secret, paper=args.paper)

    run_ts = datetime.now(tz=NY).strftime("%Y%m%d_%H%M%S")
    ymd = datetime.now(tz=NY).strftime("%Y%m%d")
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    event_log = CsvLogger(
        logs_dir / f"live_events_{selected_symbol}_{run_ts}.csv",
        [
            "ts", "event", "symbol", "prob", "close_px", "entry_px", "exit_px", "qty",
            "reason", "minutes_in", "tp_px", "sl_px", "raw_signal_count", "blocked_signal_count",
            "actual_entries", "broker_in_pos", "note"
        ],
    )
    trade_log = CsvLogger(
        logs_dir / f"live_trades_{selected_symbol}_{ymd}.csv",
        [
            "date", "symbol", "entry_ts", "exit_ts", "entry_px", "exit_px", "qty", "entry_notional",
            "pnl", "return_on_notional", "hold_minutes", "exit_reason", "entry_signal_prob",
            "mfe_dollars", "mae_dollars", "mfe_pct_on_notional", "mae_pct_on_notional"
        ],
    )
    equity_log = CsvLogger(
        logs_dir / f"live_equity_{selected_symbol}_{ymd}.csv",
        [
            "ts", "symbol", "close_px", "in_pos", "qty", "entry_px", "entry_notional",
            "realized_pnl", "unrealized_pnl", "total_pnl", "equity", "return_on_account", "return_on_notional_open_position"
        ],
    )
    summary_path = logs_dir / f"live_summary_{selected_symbol}_{run_ts}.csv"
    daily_review_path = logs_dir / "daily_review_summary.csv"

    state = sync_position_state(trading, selected_symbol)
    metrics = SessionMetrics()
    session_start = datetime.now(tz=NY)
    model = None
    last_retrain = None
    last_seen_bar_ts = None
    intraday_peak_equity = initial_cash
    max_intraday_drawdown = 0.0

    print("LIVE CONFIG")
    print({
        "symbol": selected_symbol,
        "horizon": horizon,
        "tp": tp,
        "sl": sl,
        "p_enter": p_enter,
        "qty": qty,
        "paper": args.paper,
        "primary_metric": cfg.get("walkforward", {}).get("primary_metric", "avg_daily_return_on_notional"),
        "selection_json": str(selection_path),
    })

    def save_summary(now: datetime, latest_close: float | None = None) -> None:
        nonlocal intraday_peak_equity, max_intraday_drawdown
        unrealized_pnl = 0.0
        open_notional = 0.0
        if state.in_pos and state.entry_px is not None and latest_close is not None:
            unrealized_pnl = (latest_close - state.entry_px) * state.qty
            open_notional = state.entry_px * state.qty
        total_pnl = metrics.gross_realized_pnl + unrealized_pnl
        equity = initial_cash + total_pnl
        intraday_peak_equity = max(intraday_peak_equity, equity)
        max_intraday_drawdown = min(max_intraday_drawdown, equity - intraday_peak_equity)

        row = {
            "session_start": session_start.isoformat(),
            "session_end": now.isoformat(),
            "symbol": selected_symbol,
            **asdict(metrics),
            "net_pnl": total_pnl,
            "realized_pnl": metrics.gross_realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "equity": equity,
            "max_intraday_drawdown": max_intraday_drawdown,
            "avg_daily_return_on_notional": (metrics.gross_realized_pnl / state.entry_notional) if (state.in_pos and state.entry_notional > 0) else 0.0,
        }
        file_exists = summary_path.exists() and summary_path.stat().st_size > 0
        with summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    try:
        while True:
            now = datetime.now(tz=NY)

            if not within_market_hours(now, market_open, market_close):
                save_summary(now, None)
                time.sleep(minute_sleep_seconds(now, lag_seconds=5))
                continue

            end = now
            start = end - timedelta(minutes=args.lookback_minutes + horizon + 10)
            df = fetch_minute_bars(selected_symbol, start, end, feed=args.feed)
            if df.empty or len(df) < 60:
                time.sleep(minute_sleep_seconds(now))
                continue

            df = add_features(df)
            df = df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
            if len(df) < 200:
                time.sleep(minute_sleep_seconds(now))
                continue

            latest_bar_ts = df.index[-1]
            if last_seen_bar_ts is not None and latest_bar_ts == last_seen_bar_ts:
                time.sleep(minute_sleep_seconds(now))
                continue
            last_seen_bar_ts = latest_bar_ts

            state = sync_position_state(trading, selected_symbol, fallback_px=state.entry_px)

            if model is None or last_retrain is None or (now - last_retrain).total_seconds() >= args.retrain_minutes * 60:
                y = make_label_hit_tp_local(df, horizon=horizon, tp=tp)
                train = df.dropna(subset=feature_cols).copy()
                y = y.reindex(train.index).dropna()
                X = train.loc[y.index, feature_cols]
                model = RandomForestClassifier(**model_params)
                model.fit(X, y.astype(int))
                joblib.dump(model, args.model_out)
                last_retrain = now
                metrics.retrain_count += 1
                log_event(event_log, "RETRAIN", now, selected_symbol, {
                    "prob": "", "close_px": float(df["Close"].iloc[-1]), "entry_px": "", "exit_px": "",
                    "qty": qty, "reason": "", "minutes_in": "", "tp_px": "", "sl_px": "",
                    "raw_signal_count": metrics.raw_signal_opportunities,
                    "blocked_signal_count": metrics.blocked_signals_while_in_position,
                    "actual_entries": metrics.actual_entries,
                    "broker_in_pos": state.in_pos,
                    "note": f"rows={len(X)}",
                })

            X_now = df.iloc[[-1]][feature_cols]
            prob = float(model.predict_proba(X_now)[:, 1][0])
            latest_close = float(df["Close"].iloc[-1])
            metrics.evaluated_bars += 1

            raw_signal = prob >= p_enter
            if raw_signal:
                metrics.raw_signal_opportunities += 1
                log_event(event_log, "RAW_SIGNAL", now, selected_symbol, {
                    "prob": prob, "close_px": latest_close, "entry_px": state.entry_px or "", "exit_px": "",
                    "qty": state.qty if state.in_pos else qty, "reason": "signal_seen", "minutes_in": "",
                    "tp_px": "", "sl_px": "", "raw_signal_count": metrics.raw_signal_opportunities,
                    "blocked_signal_count": metrics.blocked_signals_while_in_position,
                    "actual_entries": metrics.actual_entries, "broker_in_pos": state.in_pos, "note": ""
                })

            if raw_signal and state.in_pos:
                metrics.blocked_signals_while_in_position += 1
                log_event(event_log, "BLOCKED_SIGNAL", now, selected_symbol, {
                    "prob": prob, "close_px": latest_close, "entry_px": state.entry_px, "exit_px": "",
                    "qty": state.qty, "reason": "already_in_position", "minutes_in": "", "tp_px": "", "sl_px": "",
                    "raw_signal_count": metrics.raw_signal_opportunities,
                    "blocked_signal_count": metrics.blocked_signals_while_in_position,
                    "actual_entries": metrics.actual_entries, "broker_in_pos": state.in_pos, "note": ""
                })

            if (not state.in_pos) and raw_signal:
                buy_order = place_market_buy(trading, selected_symbol, qty)
                order_id = str(getattr(buy_order, "id", ""))
                fill_px = get_actual_filled_avg_price(trading, order_id, timeout_seconds=args.order_fill_timeout)
                entry_px = fill_px if fill_px is not None else latest_close
                entry_notional = entry_px * qty
                state = PositionState(
                    in_pos=True,
                    entry_time=latest_bar_ts,
                    entry_px=entry_px,
                    qty=qty,
                    entry_order_id=order_id,
                    entry_signal_prob=prob,
                    entry_notional=entry_notional,
                    max_favorable_px=entry_px,
                    max_adverse_px=entry_px,
                )
                metrics.actual_entries += 1
                log_event(event_log, "BUY", now, selected_symbol, {
                    "prob": prob, "close_px": latest_close, "entry_px": entry_px, "exit_px": "",
                    "qty": qty, "reason": "signal", "minutes_in": 0, "tp_px": entry_px * (1 + tp),
                    "sl_px": entry_px * (1 - sl), "raw_signal_count": metrics.raw_signal_opportunities,
                    "blocked_signal_count": metrics.blocked_signals_while_in_position,
                    "actual_entries": metrics.actual_entries, "broker_in_pos": True,
                    "note": f"entry_notional={entry_notional:.4f}",
                })

            state = sync_position_state(trading, selected_symbol, fallback_px=state.entry_px)

            if state.in_pos and state.entry_px is not None:
                if state.entry_time is None:
                    state.entry_time = latest_bar_ts
                state.max_favorable_px = max(state.max_favorable_px or state.entry_px, latest_close)
                state.max_adverse_px = min(state.max_adverse_px or state.entry_px, latest_close)

                minutes_in = int((latest_bar_ts - state.entry_time).total_seconds() // 60)
                entry_px = state.entry_px
                tp_px = entry_px * (1 + tp)
                sl_px = entry_px * (1 - sl)
                df_after = df[df.index >= state.entry_time].copy()

                if len(df_after) > 0:
                    hi = float(df_after["High"].max())
                    lo = float(df_after["Low"].min())
                    exit_reason = None
                    if lo <= sl_px:
                        exit_reason = "SL"
                    elif hi >= tp_px:
                        exit_reason = "TP"
                    elif minutes_in >= horizon:
                        exit_reason = "TIME"

                    if exit_reason is not None:
                        sell_order = place_market_sell(trading, selected_symbol, state.qty if state.qty > 0 else qty)
                        sell_order_id = str(getattr(sell_order, "id", ""))
                        exit_fill_px = get_actual_filled_avg_price(trading, sell_order_id, timeout_seconds=args.order_fill_timeout)
                        if exit_fill_px is None:
                            exit_fill_px = tp_px if exit_reason == "TP" else sl_px if exit_reason == "SL" else latest_close

                        trade_pnl = (exit_fill_px - entry_px) * (state.qty if state.qty > 0 else qty)
                        metrics.gross_realized_pnl += trade_pnl
                        metrics.completed_exits += 1
                        if exit_reason == "TP":
                            metrics.tp_exits += 1
                        elif exit_reason == "SL":
                            metrics.sl_exits += 1
                        elif exit_reason == "TIME":
                            metrics.time_exits += 1

                        entry_notional = entry_px * (state.qty if state.qty > 0 else qty)
                        return_on_notional = trade_pnl / entry_notional if entry_notional else 0.0
                        mfe_dollars = max(0.0, (hi - entry_px) * (state.qty if state.qty > 0 else qty))
                        mae_dollars = min(0.0, (lo - entry_px) * (state.qty if state.qty > 0 else qty))
                        trade_log.write({
                            "date": now.strftime("%Y-%m-%d"),
                            "symbol": selected_symbol,
                            "entry_ts": state.entry_time.isoformat() if state.entry_time else "",
                            "exit_ts": now.isoformat(),
                            "entry_px": entry_px,
                            "exit_px": exit_fill_px,
                            "qty": state.qty if state.qty > 0 else qty,
                            "entry_notional": entry_notional,
                            "pnl": trade_pnl,
                            "return_on_notional": return_on_notional,
                            "hold_minutes": minutes_in,
                            "exit_reason": exit_reason,
                            "entry_signal_prob": state.entry_signal_prob,
                            "mfe_dollars": mfe_dollars,
                            "mae_dollars": mae_dollars,
                            "mfe_pct_on_notional": (mfe_dollars / entry_notional) if entry_notional else 0.0,
                            "mae_pct_on_notional": (mae_dollars / entry_notional) if entry_notional else 0.0,
                        })
                        log_event(event_log, "SELL", now, selected_symbol, {
                            "prob": prob, "close_px": latest_close, "entry_px": entry_px, "exit_px": exit_fill_px,
                            "qty": state.qty if state.qty > 0 else qty, "reason": exit_reason, "minutes_in": minutes_in,
                            "tp_px": tp_px, "sl_px": sl_px,
                            "raw_signal_count": metrics.raw_signal_opportunities,
                            "blocked_signal_count": metrics.blocked_signals_while_in_position,
                            "actual_entries": metrics.actual_entries, "broker_in_pos": False,
                            "note": f"trade_pnl={trade_pnl:.4f}; return_on_notional={return_on_notional:.6f}",
                        })
                        state = PositionState()

            unrealized_pnl = 0.0
            open_return_on_notional = 0.0
            entry_notional = state.entry_px * state.qty if state.in_pos and state.entry_px is not None else 0.0
            if state.in_pos and state.entry_px is not None:
                unrealized_pnl = (latest_close - state.entry_px) * state.qty
                open_return_on_notional = unrealized_pnl / entry_notional if entry_notional else 0.0
            total_pnl = metrics.gross_realized_pnl + unrealized_pnl
            equity = initial_cash + total_pnl
            intraday_peak_equity = max(intraday_peak_equity, equity)
            max_intraday_drawdown = min(max_intraday_drawdown, equity - intraday_peak_equity)

            equity_log.write({
                "ts": now.isoformat(),
                "symbol": selected_symbol,
                "close_px": latest_close,
                "in_pos": state.in_pos,
                "qty": state.qty,
                "entry_px": state.entry_px if state.entry_px is not None else "",
                "entry_notional": entry_notional,
                "realized_pnl": metrics.gross_realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "equity": equity,
                "return_on_account": total_pnl / initial_cash if initial_cash else 0.0,
                "return_on_notional_open_position": open_return_on_notional,
            })

            save_summary(now, latest_close)
            time.sleep(minute_sleep_seconds(now))

    except KeyboardInterrupt:
        now = datetime.now(tz=NY)
        latest_close = None
        try:
            end = now
            start = end - timedelta(minutes=10)
            tail = fetch_minute_bars(selected_symbol, start, end, feed=args.feed)
            if not tail.empty:
                latest_close = float(tail["Close"].iloc[-1])
        except Exception:
            latest_close = None

        save_summary(now, latest_close)

        trades_df = pd.read_csv(trade_log.path) if trade_log.path.exists() and trade_log.path.stat().st_size > 0 else pd.DataFrame()
        total_trade_notional = float(trades_df["entry_notional"].sum()) if not trades_df.empty else 0.0
        avg_trade_return_on_notional = float(trades_df["return_on_notional"].mean()) if not trades_df.empty else 0.0
        net_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
        win_rate = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0
        avg_hold_minutes = float(trades_df["hold_minutes"].mean()) if not trades_df.empty else 0.0
        best_trade_pnl = float(trades_df["pnl"].max()) if not trades_df.empty else 0.0
        worst_trade_pnl = float(trades_df["pnl"].min()) if not trades_df.empty else 0.0
        avg_daily_return_on_notional = (net_pnl / total_trade_notional) if total_trade_notional > 0 else 0.0

        summary = {
            "date": now.strftime("%Y-%m-%d"),
            "symbol": selected_symbol,
            "selection_json": str(selection_path),
            "qty": qty,
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "p_enter": p_enter,
            "paper": args.paper,
            "net_pnl": net_pnl,
            "win_rate": win_rate,
            "avg_pnl_per_trade": float(trades_df["pnl"].mean()) if not trades_df.empty else 0.0,
            "avg_hold_minutes": avg_hold_minutes,
            "best_trade_pnl": best_trade_pnl,
            "worst_trade_pnl": worst_trade_pnl,
            "raw_signals": metrics.raw_signal_opportunities,
            "blocked_signals": metrics.blocked_signals_while_in_position,
            "actual_entries": metrics.actual_entries,
            "completed_exits": metrics.completed_exits,
            "tp_exits": metrics.tp_exits,
            "sl_exits": metrics.sl_exits,
            "time_exits": metrics.time_exits,
            "eod_exits": metrics.eod_exits,
            "max_intraday_drawdown": max_intraday_drawdown,
            "gross_realized_pnl": metrics.gross_realized_pnl,
            "avg_trade_return_on_notional": avg_trade_return_on_notional,
            "avg_daily_return_on_notional": avg_daily_return_on_notional,
            "total_trade_notional": total_trade_notional,
            "review_comment": "",
        }
        summary["review_comment"] = review_comment(summary)

        daily_logger = CsvLogger(daily_review_path, list(summary.keys()))
        daily_logger.write(summary)

        print("Stopped by user.")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
