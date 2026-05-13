#!/usr/bin/env python3
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


def resolve_latest_json(path_str: str) -> Path:
    p = Path(path_str)

    if p.is_file():
        return p

    if p.is_dir():
        candidates = sorted(
            p.glob("best_etf_selection_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No best_etf_selection JSON files found in directory: {p}")
        return candidates[0]

    if any(ch in path_str for ch in "*?[]"):
        candidates = sorted(
            Path().glob(path_str),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No JSON files matched pattern: {path_str}")
        return candidates[0]

    parent = p.parent if str(p.parent) != "" else Path(".")
    pattern = p.name
    if "." not in pattern:
        pattern = f"{pattern}*.json"

    candidates = sorted(
        parent.glob(pattern),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No JSON files matched: {parent / pattern}")
    return candidates[0]


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


@dataclass
class SessionMetrics:
    evaluated_bars: int = 0
    raw_signal_opportunities: int = 0
    blocked_signals_while_in_position: int = 0
    actual_entries: int = 0
    completed_exits: int = 0
    retrain_count: int = 0


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
    row = {
        "ts": now.isoformat(),
        "event": event,
        "symbol": symbol,
    }
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
            return PositionState(
                in_pos=qty > 0,
                entry_time=None,
                entry_px=avg_entry_price,
                qty=qty,
                entry_order_id=None,
                entry_signal_prob=None,
            )
    return PositionState()


def place_market_buy(trading: TradingClient, symbol: str, qty: int):
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    return trading.submit_order(req)


def place_market_sell(trading: TradingClient, symbol: str, qty: int):
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return trading.submit_order(req)


def save_session_summary(path: Path, session_start: datetime, now: datetime, symbol: str, metrics: SessionMetrics, cfg: dict) -> None:
    row = {
        "session_start": session_start.isoformat(),
        "session_end": now.isoformat(),
        "symbol": symbol,
        **asdict(metrics),
        "backtest_pr_auc": cfg.get("best_backtest_stats", {}).get("pr_auc"),
        "backtest_equity_final": cfg.get("best_backtest_stats", {}).get("equity_final"),
        "backtest_return_pct": cfg.get("best_backtest_stats", {}).get("return_pct"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selection_json",
        default="selector_out/best_etf_selection_*.json",
        help="JSON file, directory, or glob pattern. The most recent match will be used.",
    )
    ap.add_argument("--symbol", default=None, help="Override selected symbol from JSON")
    ap.add_argument("--feed", default=None, help="Override feed from JSON; default uses the selection JSON feed")
    ap.add_argument("--qty", type=int, default=None, help="Override qty from JSON")
    ap.add_argument("--lookback_minutes", type=int, default=800)
    ap.add_argument("--retrain_minutes", type=int, default=60)
    ap.add_argument("--model_out", default="final_model_selected_etf.joblib")
    ap.add_argument("--logs_dir", default="live_logs_4_1_3")
    ap.add_argument("--order_fill_timeout", type=int, default=20)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()

    selection_path = resolve_latest_json(args.selection_json)
    print(f"Using selection JSON: {selection_path}")

    with selection_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    symbol = args.symbol or cfg["selected_symbol"]
    feature_cols = cfg["feature_cols"]
    model_params = cfg["model_params"]
    strat = cfg["best_strategy_params"]
    backtest_stats = cfg.get("best_backtest_stats", {})

    horizon = int(strat["horizon"])
    tp = float(strat["tp"])
    p_enter = float(strat["p_enter"])
    sl = strat.get("sl", None)
    sl = float(sl) if sl is not None else None

    market_open = dtime(*map(int, cfg["market_hours"]["open"].split(":")))
    market_close = dtime(*map(int, cfg["market_hours"]["close"].split(":")))

    feed = args.feed or cfg.get("data_window", {}).get("feed", "iex")
    qty = args.qty if args.qty is not None else int(cfg.get("qty", 10))

    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")

    trading = TradingClient(api_key=key, secret_key=secret, paper=args.paper)

    run_ts = datetime.now(tz=NY).strftime("%Y%m%d_%H%M%S")
    logs_dir = Path(args.logs_dir)
    event_log = CsvLogger(
        logs_dir / f"live_selected_etf_events_{symbol}_{run_ts}.csv",
        [
            "ts", "event", "symbol", "prob", "close_px", "entry_px", "exit_px", "qty",
            "reason", "minutes_in", "tp_px", "sl_px", "raw_signal_count", "blocked_signal_count",
            "actual_entries", "broker_in_pos", "note"
        ],
    )
    summary_path = logs_dir / f"live_selected_etf_summary_{symbol}_{run_ts}.csv"

    state = sync_position_state(trading, symbol)
    metrics = SessionMetrics()
    session_start = datetime.now(tz=NY)
    model = None
    last_retrain = None
    last_seen_bar_ts = None

    print("LIVE CONFIG")
    print({
        "symbol": symbol,
        "horizon": horizon,
        "tp": tp,
        "p_enter": p_enter,
        "sl": sl,
        "qty": qty,
        "paper": args.paper,
        "feed": feed,
        "backtest_pr_auc": backtest_stats.get("pr_auc"),
        "backtest_equity_final": backtest_stats.get("equity_final"),
        "backtest_return_pct": backtest_stats.get("return_pct"),
    })

    if state.in_pos:
        print(f"Synced existing broker position for {symbol}: qty={state.qty}, entry_px={state.entry_px}")

    try:
        while True:
            now = datetime.now(tz=NY)

            if not within_market_hours(now, market_open, market_close):
                save_session_summary(summary_path, session_start, now, symbol, metrics, cfg)
                time.sleep(minute_sleep_seconds(now, lag_seconds=5))
                continue

            end = now
            start = end - timedelta(minutes=args.lookback_minutes + horizon + 5)
            df = fetch_minute_bars(symbol, start, end, feed=feed)
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

            state = sync_position_state(trading, symbol, fallback_px=state.entry_px)

            if (model is None) or (last_retrain is None) or ((now - last_retrain).total_seconds() >= args.retrain_minutes * 60):
                y = make_label_hit_tp_local(df, horizon=horizon, tp=tp)
                train = df.dropna(subset=feature_cols).copy()
                y = y.reindex(train.index).dropna()
                X = train.loc[y.index, feature_cols]
                model = RandomForestClassifier(**model_params)
                model.fit(X, y.astype(int))
                joblib.dump(model, args.model_out)
                last_retrain = now
                metrics.retrain_count += 1
                print(f"[{now}] retrained model on {len(X)} rows -> saved {args.model_out}")
                log_event(event_log, "RETRAIN", now, symbol, {
                    "prob": "",
                    "close_px": float(df["Close"].iloc[-1]),
                    "entry_px": "",
                    "exit_px": "",
                    "qty": qty,
                    "reason": "",
                    "minutes_in": "",
                    "tp_px": "",
                    "sl_px": "",
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

            if raw_signal and state.in_pos:
                metrics.blocked_signals_while_in_position += 1
                log_event(event_log, "BLOCKED_SIGNAL", now, symbol, {
                    "prob": prob,
                    "close_px": latest_close,
                    "entry_px": state.entry_px,
                    "exit_px": "",
                    "qty": state.qty,
                    "reason": "already_in_position",
                    "minutes_in": "",
                    "tp_px": "",
                    "sl_px": "",
                    "raw_signal_count": metrics.raw_signal_opportunities,
                    "blocked_signal_count": metrics.blocked_signals_while_in_position,
                    "actual_entries": metrics.actual_entries,
                    "broker_in_pos": state.in_pos,
                    "note": "",
                })

            if (not state.in_pos) and raw_signal:
                buy_order = place_market_buy(trading, symbol, qty)
                order_id = str(getattr(buy_order, "id", ""))
                fill_px = get_actual_filled_avg_price(trading, order_id, timeout_seconds=args.order_fill_timeout)
                entry_px = fill_px if fill_px is not None else latest_close

                state = PositionState(
                    in_pos=True,
                    entry_time=latest_bar_ts,
                    entry_px=entry_px,
                    qty=qty,
                    entry_order_id=order_id,
                    entry_signal_prob=prob,
                )
                metrics.actual_entries += 1
                print(f"[{now}] BUY signal prob={prob:.3f} entry_px={entry_px:.4f} order_id={order_id}")
                log_event(event_log, "BUY", now, symbol, {
                    "prob": prob,
                    "close_px": latest_close,
                    "entry_px": entry_px,
                    "exit_px": "",
                    "qty": qty,
                    "reason": "signal",
                    "minutes_in": 0,
                    "tp_px": entry_px * (1 + tp),
                    "sl_px": entry_px * (1 - sl) if sl is not None else "",
                    "raw_signal_count": metrics.raw_signal_opportunities,
                    "blocked_signal_count": metrics.blocked_signals_while_in_position,
                    "actual_entries": metrics.actual_entries,
                    "broker_in_pos": True,
                    "note": f"fill_source={'broker' if fill_px is not None else 'last_close_fallback'}",
                })

            state = sync_position_state(trading, symbol, fallback_px=state.entry_px)

            if state.in_pos and state.entry_px is not None:
                if state.entry_time is None:
                    state.entry_time = latest_bar_ts

                minutes_in = int((latest_bar_ts - state.entry_time).total_seconds() // 60)
                entry_px = state.entry_px
                tp_px = entry_px * (1 + tp)
                sl_px = entry_px * (1 - sl) if sl is not None else None

                df_after = df[df.index >= state.entry_time].copy()
                if len(df_after) > 0:
                    hi = float(df_after["High"].max())
                    lo = float(df_after["Low"].min())

                    exit_reason = None
                    if sl_px is not None and lo <= sl_px:
                        exit_reason = "SL"
                    elif hi >= tp_px:
                        exit_reason = "TP"
                    elif minutes_in >= horizon:
                        exit_reason = "TIME"

                    if exit_reason is not None:
                        sell_order = place_market_sell(trading, symbol, state.qty if state.qty > 0 else qty)
                        sell_order_id = str(getattr(sell_order, "id", ""))
                        exit_fill_px = get_actual_filled_avg_price(trading, sell_order_id, timeout_seconds=args.order_fill_timeout)
                        if exit_fill_px is None:
                            if exit_reason == "TP":
                                exit_fill_px = tp_px
                            elif exit_reason == "SL" and sl_px is not None:
                                exit_fill_px = sl_px
                            else:
                                exit_fill_px = latest_close

                        metrics.completed_exits += 1
                        print(f"[{now}] SELL ({exit_reason}) minutes_in={minutes_in} exit_px={exit_fill_px:.4f} prob_last={prob:.3f}")
                        log_event(event_log, "SELL", now, symbol, {
                            "prob": prob,
                            "close_px": latest_close,
                            "entry_px": entry_px,
                            "exit_px": exit_fill_px,
                            "qty": state.qty if state.qty > 0 else qty,
                            "reason": exit_reason,
                            "minutes_in": minutes_in,
                            "tp_px": tp_px,
                            "sl_px": sl_px if sl_px is not None else "",
                            "raw_signal_count": metrics.raw_signal_opportunities,
                            "blocked_signal_count": metrics.blocked_signals_while_in_position,
                            "actual_entries": metrics.actual_entries,
                            "broker_in_pos": False,
                            "note": f"order_id={sell_order_id}",
                        })
                        state = PositionState()

            save_session_summary(summary_path, session_start, now, symbol, metrics, cfg)
            time.sleep(minute_sleep_seconds(now))

    except KeyboardInterrupt:
        now = datetime.now(tz=NY)
        save_session_summary(summary_path, session_start, now, symbol, metrics, cfg)
        print("Stopped by user.")
        print("Final session metrics:")
        print(asdict(metrics))


if __name__ == "__main__":
    main()
