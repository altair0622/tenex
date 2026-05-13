#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import time
import argparse
import copy
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


def parse_symbol_override(symbols_arg: str | None) -> str | None:
    if not symbols_arg:
        return None
    parts = [x.strip().upper() for x in symbols_arg.split(",") if x.strip()]
    if not parts:
        return None
    return parts[0]


def resolve_symbol_row_csv(selection_path: Path) -> Path:
    name = selection_path.name
    if name.startswith("best_etf_selection_"):
        suffix = name[len("best_etf_selection_"):]
        candidate = selection_path.with_name(f"best_etf_by_symbol_{suffix}")
        if candidate.exists():
            return candidate
    matches = sorted(
        selection_path.parent.glob("best_etf_by_symbol_*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"Could not find best_etf_by_symbol CSV near: {selection_path}")
    return matches[0]


def apply_symbol_override(cfg: dict, selection_path: Path, symbol_override: str | None) -> dict:
    if not symbol_override:
        return cfg

    by_symbol_path = resolve_symbol_row_csv(selection_path)
    df = pd.read_csv(by_symbol_path)
    if "symbol" not in df.columns:
        raise ValueError(f"Missing 'symbol' column in {by_symbol_path}")

    rows = df[df["symbol"].astype(str).str.upper() == symbol_override.upper()].copy()
    if rows.empty:
        available = ", ".join(sorted(df["symbol"].astype(str).unique().tolist()))
        raise ValueError(f"Symbol {symbol_override} not found in {by_symbol_path}. Available: {available}")

    sort_cols = [c for c in ["avg_daily_pnl", "positive_day_ratio", "median_daily_pnl", "pr_auc", "total_pnl"] if c in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    row = rows.iloc[0]

    cfg2 = copy.deepcopy(cfg)
    cfg2["selected_symbol"] = symbol_override.upper()
    cfg2["best_strategy_params"] = {
        "horizon": int(row["horizon"]),
        "tp": float(row["tp"]),
        "sl": None if pd.isna(row["sl"]) else float(row["sl"]),
        "p_enter": float(row["p_enter"]),
    }
    cfg2["best_backtest_stats"] = {
        "pr_auc": None if pd.isna(row.get("pr_auc")) else float(row["pr_auc"]),
        "avg_daily_pnl": float(row["avg_daily_pnl"]),
        "median_daily_pnl": float(row["median_daily_pnl"]),
        "worst_day_pnl": float(row["worst_day_pnl"]),
        "best_day_pnl": float(row["best_day_pnl"]),
        "positive_day_ratio": float(row["positive_day_ratio"]),
        "avg_daily_return_pct": float(row["avg_daily_return_pct"]),
        "total_pnl": float(row["total_pnl"]),
        "total_trades": int(row["total_trades"]),
        "avg_trades_per_day": float(row["avg_trades_per_day"]),
        "eval_days_used": int(row["eval_days_used"]),
        "train_days": int(row["train_days"]),
    }
    cfg2["selection_reason"] = {
        "primary_metric": "manual_symbol_override",
        "summary": f"Live run manually overridden to use {symbol_override.upper()} with that symbol's best parameters from {by_symbol_path.name}.",
        "source_csv": by_symbol_path.name,
    }
    return cfg2


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
    pattern = p.name if "." in p.name else f"{p.name}*.json"
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
    close = df["Close"].to_numpy()
    vol = df["Volume"].to_numpy()
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


def compute_calendar_lookback_days(train_days: int, extra_buffer_days: int = 10) -> int:
    return max(30, int((train_days + extra_buffer_days) * 1.8))


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


@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    entry_ts: str
    entry_bar_ts: str
    entry_px: float
    entry_prob: float
    qty: int
    entry_order_id: str
    exit_ts: str | None = None
    exit_bar_ts: str | None = None
    exit_px: float | None = None
    exit_reason: str | None = None
    exit_order_id: str | None = None
    hold_minutes: int | None = None
    pnl_dollars: float | None = None
    pnl_pct: float | None = None
    mfe_dollars: float | None = None
    mae_dollars: float | None = None
    max_high_since_entry: float | None = None
    min_low_since_entry: float | None = None
    signal_close_px: float | None = None
    fill_source: str | None = None
    notes: str | None = None


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
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    return trading.submit_order(req)


def place_market_sell(trading: TradingClient, symbol: str, qty: int):
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    return trading.submit_order(req)


def build_live_train_and_infer_frames(
    df_raw: pd.DataFrame,
    feature_cols: list[str],
    train_days: int,
    horizon: int,
    tp: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = add_features(df_raw)
    df["y"] = make_label_hit_tp_local(df, horizon=horizon, tp=tp)
    df = df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
    if df.empty:
        raise ValueError("No usable rows after feature generation")

    all_dates = sorted(pd.Index(df.index.date).unique())
    if len(all_dates) < train_days + 1:
        raise ValueError(f"Need at least {train_days + 1} trading days, got {len(all_dates)}")

    infer_day = all_dates[-1]
    train_dates = all_dates[-(train_days + 1):-1]
    train = df[np.isin(df.index.date, train_dates)].copy()
    train = train.dropna(subset=["y"]).copy()
    infer = df[df.index.date == infer_day].copy()

    if train.empty or infer.empty:
        raise ValueError("Training or inference frame is empty")
    if train["y"].nunique() < 2:
        raise ValueError("Training labels need at least two classes")

    X_train = train[feature_cols]
    y_train = train["y"].astype(int)
    return X_train, y_train, infer


def compute_unrealized_pnl(state: PositionState, mark_px: float | None) -> float:
    if (not state.in_pos) or state.entry_px is None or mark_px is None:
        return 0.0
    return (float(mark_px) - float(state.entry_px)) * int(state.qty)


def append_trade_record(trade_logger: CsvLogger, rec: TradeRecord) -> None:
    trade_logger.write(asdict(rec))


def append_equity_snapshot(
    equity_logger: CsvLogger,
    now: datetime,
    symbol: str,
    latest_close: float,
    state: PositionState,
    realized_pnl: float,
    metrics: SessionMetrics,
    day_start_equity: float,
    current_trade_id: int | None,
) -> dict:
    unrealized_pnl = compute_unrealized_pnl(state, latest_close)
    total_pnl = realized_pnl + unrealized_pnl
    equity = day_start_equity + total_pnl
    row = {
        "ts": now.isoformat(),
        "symbol": symbol,
        "close_px": latest_close,
        "in_position": state.in_pos,
        "position_qty": state.qty,
        "entry_px": state.entry_px if state.entry_px is not None else "",
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "equity": equity,
        "current_trade_id": current_trade_id if current_trade_id is not None else "",
        "evaluated_bars": metrics.evaluated_bars,
        "raw_signal_count": metrics.raw_signal_opportunities,
        "blocked_signal_count": metrics.blocked_signals_while_in_position,
        "actual_entries": metrics.actual_entries,
        "completed_exits": metrics.completed_exits,
    }
    equity_logger.write(row)
    return row


def safe_stat(values: list[float], fn, default: float = 0.0) -> float:
    if not values:
        return default
    return float(fn(values))


def build_review_comment(summary: dict) -> str:
    comments = []

    if summary["total_trades"] == 0:
        comments.append("No completed trades today. Review whether threshold or market regime was too restrictive.")
    else:
        if summary["net_pnl"] > 0:
            comments.append("The day finished positive overall.")
        elif summary["net_pnl"] < 0:
            comments.append("The day finished negative overall, so entry quality and exit behavior should be reviewed.")
        else:
            comments.append("The day finished near flat.")

        if summary["blocked_signals"] > summary["actual_entries"]:
            comments.append("Blocked signals were high relative to actual entries, so max one-position logic may have limited opportunities.")

        if summary["time_exit_count"] > max(summary["tp_exit_count"], summary["sl_exit_count"]):
            comments.append("Time exits dominated today, which may mean horizon is too short or TP is too ambitious.")

        if summary["avg_hold_minutes"] > 0 and summary["avg_hold_minutes"] < 3:
            comments.append("Average hold time was very short, so the strategy may be reacting to noisy moves.")

        if summary["win_rate"] >= 0.6 and summary["avg_pnl_per_trade"] <= 0:
            comments.append("Win rate looked decent but average PnL per trade was weak, so winners may have been too small.")

        if summary["positive_signal_to_entry_ratio"] < 0.5 and summary["raw_signals"] > 0:
            comments.append("Many raw signals did not become entries, so thresholding or in-position blocking likely filtered opportunities.")

        if summary["max_intraday_drawdown"] < 0:
            comments.append("Intraday drawdown review is important because the strategy gave back gains during the session.")

    if not comments:
        comments.append("Session completed with limited signals and no strong anomaly in the review metrics.")
    return " ".join(comments)


def save_daily_review_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_daily_review_summary(
    *,
    session_start: datetime,
    now: datetime,
    symbol: str,
    metrics: SessionMetrics,
    cfg: dict,
    selection_path: Path,
    realized_pnl: float,
    trade_records: list[TradeRecord],
    equity_snapshots: list[dict],
    day_start_equity: float,
    current_state: PositionState,
    latest_close: float | None,
    latest_prob: float | None,
    current_trade_id: int | None,
) -> dict:
    closed = [t for t in trade_records if t.exit_ts is not None and t.pnl_dollars is not None]
    pnl_values = [float(t.pnl_dollars) for t in closed]
    hold_values = [int(t.hold_minutes) for t in closed if t.hold_minutes is not None]
    mfe_values = [float(t.mfe_dollars) for t in closed if t.mfe_dollars is not None]
    mae_values = [float(t.mae_dollars) for t in closed if t.mae_dollars is not None]

    win_count = sum(1 for x in pnl_values if x > 0)
    loss_count = sum(1 for x in pnl_values if x < 0)
    flat_count = sum(1 for x in pnl_values if x == 0)
    total_trades = len(closed)

    tp_exit_count = sum(1 for t in closed if t.exit_reason == "TP")
    sl_exit_count = sum(1 for t in closed if t.exit_reason == "SL")
    time_exit_count = sum(1 for t in closed if t.exit_reason == "TIME")
    manual_exit_count = sum(1 for t in closed if t.exit_reason not in {"TP", "SL", "TIME", None})

    unrealized_pnl = compute_unrealized_pnl(current_state, latest_close)
    net_pnl = realized_pnl + unrealized_pnl
    ending_equity = day_start_equity + net_pnl
    return_pct = 0.0 if day_start_equity == 0 else net_pnl / day_start_equity

    equity_values = [float(x["equity"]) for x in equity_snapshots]
    running_peak = None
    max_intraday_drawdown = 0.0
    best_intraday_equity = ending_equity
    worst_intraday_equity = ending_equity
    if equity_values:
        best_intraday_equity = max(equity_values)
        worst_intraday_equity = min(equity_values)
        for eq in equity_values:
            if running_peak is None or eq > running_peak:
                running_peak = eq
            dd = eq - running_peak
            if dd < max_intraday_drawdown:
                max_intraday_drawdown = dd

    summary = {
        "date": now.astimezone(NY).date().isoformat(),
        "session_start": session_start.isoformat(),
        "session_end": now.isoformat(),
        "symbol": symbol,
        "selection_json": selection_path.name,
        "selection_primary_metric": cfg.get("selection_reason", {}).get("primary_metric", ""),
        "selection_summary": cfg.get("selection_reason", {}).get("summary", ""),
        "paper": "",
        "start_equity": day_start_equity,
        "end_equity": ending_equity,
        "net_pnl": net_pnl,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "return_pct": return_pct,
        "total_trades": total_trades,
        "entries": metrics.actual_entries,
        "completed_exits": metrics.completed_exits,
        "win_count": win_count,
        "loss_count": loss_count,
        "flat_count": flat_count,
        "win_rate": 0.0 if total_trades == 0 else win_count / total_trades,
        "avg_pnl_per_trade": safe_stat(pnl_values, np.mean, 0.0),
        "median_pnl_per_trade": safe_stat(pnl_values, np.median, 0.0),
        "best_trade_pnl": safe_stat(pnl_values, max, 0.0),
        "worst_trade_pnl": safe_stat(pnl_values, min, 0.0),
        "avg_hold_minutes": safe_stat(hold_values, np.mean, 0.0),
        "median_hold_minutes": safe_stat(hold_values, np.median, 0.0),
        "avg_mfe_dollars": safe_stat(mfe_values, np.mean, 0.0),
        "avg_mae_dollars": safe_stat(mae_values, np.mean, 0.0),
        "raw_signals": metrics.raw_signal_opportunities,
        "blocked_signals": metrics.blocked_signals_while_in_position,
        "positive_signal_to_entry_ratio": 0.0 if metrics.raw_signal_opportunities == 0 else metrics.actual_entries / metrics.raw_signal_opportunities,
        "evaluated_bars": metrics.evaluated_bars,
        "retrain_count": metrics.retrain_count,
        "tp_exit_count": tp_exit_count,
        "sl_exit_count": sl_exit_count,
        "time_exit_count": time_exit_count,
        "manual_exit_count": manual_exit_count,
        "max_intraday_drawdown": max_intraday_drawdown,
        "best_intraday_equity": best_intraday_equity,
        "worst_intraday_equity": worst_intraday_equity,
        "open_position_at_end": current_state.in_pos,
        "open_position_qty": current_state.qty,
        "open_position_entry_px": current_state.entry_px if current_state.entry_px is not None else "",
        "open_position_unrealized_pnl": unrealized_pnl,
        "latest_close": latest_close if latest_close is not None else "",
        "latest_prob": latest_prob if latest_prob is not None else "",
        "current_trade_id": current_trade_id if current_trade_id is not None else "",
        "train_days": cfg.get("walkforward", {}).get("train_days", ""),
        "horizon": cfg.get("best_strategy_params", {}).get("horizon", ""),
        "tp": cfg.get("best_strategy_params", {}).get("tp", ""),
        "sl": cfg.get("best_strategy_params", {}).get("sl", ""),
        "p_enter": cfg.get("best_strategy_params", {}).get("p_enter", ""),
        "selection_avg_daily_pnl": cfg.get("best_backtest_stats", {}).get("avg_daily_pnl", ""),
        "selection_positive_day_ratio": cfg.get("best_backtest_stats", {}).get("positive_day_ratio", ""),
    }
    summary["review_comment"] = build_review_comment(summary)
    return summary


def save_session_summary(path: Path, session_start: datetime, now: datetime, symbol: str, metrics: SessionMetrics, cfg: dict) -> None:
    row = {
        "session_start": session_start.isoformat(),
        "session_end": now.isoformat(),
        "symbol": symbol,
        **asdict(metrics),
        "selection_avg_daily_pnl": cfg.get("best_backtest_stats", {}).get("avg_daily_pnl"),
        "selection_positive_day_ratio": cfg.get("best_backtest_stats", {}).get("positive_day_ratio"),
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
    ap.add_argument("--selection_json", default="selector_out_414/best_etf_selection_*.json")
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--symbols", default=None, help="Optional symbol override. Example: --symbols TQQQ")
    ap.add_argument("--symbol", default=None, help="Optional single-symbol override. Same effect as --symbols TQQQ")
    ap.add_argument("--qty", type=int, default=None)
    ap.add_argument("--lookback_calendar_days", type=int, default=None)
    ap.add_argument("--retrain_minutes", type=int, default=60)
    ap.add_argument("--model_out", default="final_model_414.joblib")
    ap.add_argument("--logs_dir", default="live_logs_414")
    ap.add_argument("--order_fill_timeout", type=int, default=20)
    ap.add_argument("--paper", action="store_true", help="Use paper trading endpoint")
    args = ap.parse_args()

    selection_path = resolve_latest_json(args.selection_json)
    print(f"Using selection JSON: {selection_path}")
    with selection_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    symbol_override = args.symbol or parse_symbol_override(args.symbols)
    cfg = apply_symbol_override(cfg, selection_path, symbol_override)

    symbol = cfg["selected_symbol"]
    feature_cols = cfg["feature_cols"]
    model_params = cfg["model_params"]
    strat = cfg["best_strategy_params"]
    walkforward = cfg.get("walkforward", {})

    horizon = int(strat["horizon"])
    tp = float(strat["tp"])
    p_enter = float(strat["p_enter"])
    sl = strat.get("sl", None)
    sl = float(sl) if sl is not None else None
    train_days = int(walkforward.get("train_days", 10))
    qty = args.qty if args.qty is not None else int(cfg.get("qty", 10))

    market_open = dtime(*map(int, cfg["market_hours"]["open"].split(":")))
    market_close = dtime(*map(int, cfg["market_hours"]["close"].split(":")))

    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")

    trading = TradingClient(api_key=key, secret_key=secret, paper=args.paper)

    run_ts = datetime.now(tz=NY).strftime("%Y%m%d_%H%M%S")
    run_date = datetime.now(tz=NY).strftime("%Y%m%d")
    logs_dir = Path(args.logs_dir)

    event_log = CsvLogger(
        logs_dir / f"live_events_{symbol}_{run_ts}.csv",
        [
            "ts", "event", "symbol", "prob", "close_px", "entry_px", "exit_px", "qty",
            "reason", "minutes_in", "tp_px", "sl_px", "raw_signal_count", "blocked_signal_count",
            "actual_entries", "broker_in_pos", "note"
        ],
    )
    trade_log = CsvLogger(
        logs_dir / f"live_trades_{symbol}_{run_date}.csv",
        [
            "trade_id", "symbol", "entry_ts", "entry_bar_ts", "entry_px", "entry_prob", "qty", "entry_order_id",
            "exit_ts", "exit_bar_ts", "exit_px", "exit_reason", "exit_order_id", "hold_minutes",
            "pnl_dollars", "pnl_pct", "mfe_dollars", "mae_dollars", "max_high_since_entry",
            "min_low_since_entry", "signal_close_px", "fill_source", "notes"
        ],
    )
    equity_log = CsvLogger(
        logs_dir / f"live_equity_{symbol}_{run_date}.csv",
        [
            "ts", "symbol", "close_px", "in_position", "position_qty", "entry_px",
            "realized_pnl", "unrealized_pnl", "total_pnl", "equity", "current_trade_id",
            "evaluated_bars", "raw_signal_count", "blocked_signal_count", "actual_entries", "completed_exits"
        ],
    )
    session_summary_path = logs_dir / f"live_summary_{symbol}_{run_ts}.csv"
    daily_review_path = logs_dir / "daily_review_summary.csv"

    state = sync_position_state(trading, symbol)
    metrics = SessionMetrics()
    session_start = datetime.now(tz=NY)
    model = None
    last_retrain = None
    last_seen_bar_ts = None
    current_trade: TradeRecord | None = None
    trade_records: list[TradeRecord] = []
    equity_snapshots: list[dict] = []
    latest_close_seen: float | None = None
    latest_prob_seen: float | None = None
    realized_pnl = 0.0
    trade_counter = 0
    day_start_equity = float(cfg.get("initial_cash", 100000))

    if state.in_pos:
        print(f"Synced existing broker position for {symbol}: qty={state.qty}, entry_px={state.entry_px}")

    print("LIVE CONFIG")
    print({
        "symbol": symbol,
        "horizon": horizon,
        "tp": tp,
        "p_enter": p_enter,
        "sl": sl,
        "qty": qty,
        "train_days": train_days,
        "paper": args.paper,
        "selection_avg_daily_pnl": cfg.get("best_backtest_stats", {}).get("avg_daily_pnl"),
        "selection_positive_day_ratio": cfg.get("best_backtest_stats", {}).get("positive_day_ratio"),
    })

    try:
        while True:
            now = datetime.now(tz=NY)

            if not within_market_hours(now, market_open, market_close):
                review_row = build_daily_review_summary(
                    session_start=session_start,
                    now=now,
                    symbol=symbol,
                    metrics=metrics,
                    cfg=cfg,
                    selection_path=selection_path,
                    realized_pnl=realized_pnl,
                    trade_records=trade_records,
                    equity_snapshots=equity_snapshots,
                    day_start_equity=day_start_equity,
                    current_state=state,
                    latest_close=latest_close_seen,
                    latest_prob=latest_prob_seen,
                    current_trade_id=current_trade.trade_id if current_trade else None,
                )
                review_row["paper"] = args.paper
                save_daily_review_summary(daily_review_path, review_row)
                save_session_summary(session_summary_path, session_start, now, symbol, metrics, cfg)
                time.sleep(minute_sleep_seconds(now, lag_seconds=5))
                continue

            lookback_days = args.lookback_calendar_days
            if lookback_days is None:
                lookback_days = compute_calendar_lookback_days(train_days)
            end = now
            start = end - timedelta(days=lookback_days)
            df_raw = fetch_minute_bars(symbol, start, end, feed=args.feed)
            if df_raw.empty or len(df_raw) < 200:
                time.sleep(minute_sleep_seconds(now))
                continue

            try:
                X_train, y_train, infer = build_live_train_and_infer_frames(
                    df_raw=df_raw,
                    feature_cols=feature_cols,
                    train_days=train_days,
                    horizon=horizon,
                    tp=tp,
                )
            except Exception as e:
                print(f"[{now}] skipped retrain/infer prep: {e}")
                time.sleep(minute_sleep_seconds(now))
                continue

            latest_bar_ts = infer.index[-1]
            if last_seen_bar_ts is not None and latest_bar_ts == last_seen_bar_ts:
                time.sleep(minute_sleep_seconds(now))
                continue
            last_seen_bar_ts = latest_bar_ts

            state = sync_position_state(trading, symbol, fallback_px=state.entry_px)

            if (model is None) or (last_retrain is None) or ((now - last_retrain).total_seconds() >= args.retrain_minutes * 60):
                model = RandomForestClassifier(**model_params)
                model.fit(X_train, y_train)
                joblib.dump(model, args.model_out)
                last_retrain = now
                metrics.retrain_count += 1
                print(f"[{now}] retrained model on {len(X_train)} rows -> saved {args.model_out}")
                log_event(event_log, "RETRAIN", now, symbol, {
                    "prob": "",
                    "close_px": float(infer["Close"].iloc[-1]),
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
                    "note": f"train_rows={len(X_train)} train_days={train_days}",
                })

            X_now = infer.iloc[[-1]][feature_cols]
            prob = float(model.predict_proba(X_now)[:, 1][0])
            latest_close = float(infer["Close"].iloc[-1])
            latest_close_seen = latest_close
            latest_prob_seen = prob
            metrics.evaluated_bars += 1

            raw_signal = prob >= p_enter
            if raw_signal:
                metrics.raw_signal_opportunities += 1
                log_event(event_log, "RAW_SIGNAL", now, symbol, {
                    "prob": prob,
                    "close_px": latest_close,
                    "entry_px": state.entry_px if state.entry_px is not None else "",
                    "exit_px": "",
                    "qty": state.qty if state.qty > 0 else qty,
                    "reason": "threshold_pass",
                    "minutes_in": "",
                    "tp_px": "",
                    "sl_px": "",
                    "raw_signal_count": metrics.raw_signal_opportunities,
                    "blocked_signal_count": metrics.blocked_signals_while_in_position,
                    "actual_entries": metrics.actual_entries,
                    "broker_in_pos": state.in_pos,
                    "note": "",
                })

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
                fill_source = "broker" if fill_px is not None else "last_close_fallback"

                state = PositionState(
                    in_pos=True,
                    entry_time=latest_bar_ts,
                    entry_px=entry_px,
                    qty=qty,
                    entry_order_id=order_id,
                    entry_signal_prob=prob,
                )
                metrics.actual_entries += 1
                trade_counter += 1
                current_trade = TradeRecord(
                    trade_id=trade_counter,
                    symbol=symbol,
                    entry_ts=now.isoformat(),
                    entry_bar_ts=latest_bar_ts.isoformat(),
                    entry_px=entry_px,
                    entry_prob=prob,
                    qty=qty,
                    entry_order_id=order_id,
                    signal_close_px=latest_close,
                    fill_source=fill_source,
                    notes="",
                )
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
                    "note": f"fill_source={fill_source} trade_id={trade_counter}",
                })

            state = sync_position_state(trading, symbol, fallback_px=state.entry_px)

            if state.in_pos and state.entry_px is not None:
                if state.entry_time is None:
                    state.entry_time = latest_bar_ts

                minutes_in = int((latest_bar_ts - state.entry_time).total_seconds() // 60)
                entry_px = state.entry_px
                tp_px = entry_px * (1 + tp)
                sl_px = entry_px * (1 - sl) if sl is not None else None

                infer_after = infer[infer.index >= state.entry_time].copy()
                if len(infer_after) > 0:
                    hi = float(infer_after["High"].max())
                    lo = float(infer_after["Low"].min())

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
                        trade_qty = state.qty if state.qty > 0 else qty
                        trade_pnl = (float(exit_fill_px) - float(entry_px)) * int(trade_qty)
                        realized_pnl += trade_pnl

                        print(f"[{now}] SELL ({exit_reason}) minutes_in={minutes_in} exit_px={exit_fill_px:.4f} prob_last={prob:.3f}")
                        log_event(event_log, "SELL", now, symbol, {
                            "prob": prob,
                            "close_px": latest_close,
                            "entry_px": entry_px,
                            "exit_px": exit_fill_px,
                            "qty": trade_qty,
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

                        if current_trade is None:
                            current_trade = TradeRecord(
                                trade_id=trade_counter + 1,
                                symbol=symbol,
                                entry_ts=state.entry_time.isoformat() if state.entry_time else now.isoformat(),
                                entry_bar_ts=state.entry_time.isoformat() if state.entry_time else latest_bar_ts.isoformat(),
                                entry_px=entry_px,
                                entry_prob=state.entry_signal_prob if state.entry_signal_prob is not None else prob,
                                qty=trade_qty,
                                entry_order_id=state.entry_order_id or "",
                                notes="Recovered trade record because in-memory entry was missing.",
                            )

                        current_trade.exit_ts = now.isoformat()
                        current_trade.exit_bar_ts = latest_bar_ts.isoformat()
                        current_trade.exit_px = float(exit_fill_px)
                        current_trade.exit_reason = exit_reason
                        current_trade.exit_order_id = sell_order_id
                        current_trade.hold_minutes = minutes_in
                        current_trade.pnl_dollars = trade_pnl
                        current_trade.pnl_pct = 0.0 if entry_px == 0 else (float(exit_fill_px) / float(entry_px) - 1.0)
                        current_trade.max_high_since_entry = hi
                        current_trade.min_low_since_entry = lo
                        current_trade.mfe_dollars = (hi - float(entry_px)) * int(trade_qty)
                        current_trade.mae_dollars = (lo - float(entry_px)) * int(trade_qty)
                        append_trade_record(trade_log, current_trade)
                        trade_records.append(current_trade)
                        current_trade = None
                        state = PositionState()

            snap = append_equity_snapshot(
                equity_logger=equity_log,
                now=now,
                symbol=symbol,
                latest_close=latest_close,
                state=state,
                realized_pnl=realized_pnl,
                metrics=metrics,
                day_start_equity=day_start_equity,
                current_trade_id=current_trade.trade_id if current_trade else None,
            )
            equity_snapshots.append(snap)

            review_row = build_daily_review_summary(
                session_start=session_start,
                now=now,
                symbol=symbol,
                metrics=metrics,
                cfg=cfg,
                selection_path=selection_path,
                realized_pnl=realized_pnl,
                trade_records=trade_records,
                equity_snapshots=equity_snapshots,
                day_start_equity=day_start_equity,
                current_state=state,
                latest_close=latest_close_seen,
                latest_prob=latest_prob_seen,
                current_trade_id=current_trade.trade_id if current_trade else None,
            )
            review_row["paper"] = args.paper
            save_daily_review_summary(daily_review_path, review_row)
            save_session_summary(session_summary_path, session_start, now, symbol, metrics, cfg)
            time.sleep(minute_sleep_seconds(now))

    except KeyboardInterrupt:
        now = datetime.now(tz=NY)
        if latest_close_seen is not None:
            snap = append_equity_snapshot(
                equity_logger=equity_log,
                now=now,
                symbol=symbol,
                latest_close=latest_close_seen,
                state=state,
                realized_pnl=realized_pnl,
                metrics=metrics,
                day_start_equity=day_start_equity,
                current_trade_id=current_trade.trade_id if current_trade else None,
            )
            equity_snapshots.append(snap)
        review_row = build_daily_review_summary(
            session_start=session_start,
            now=now,
            symbol=symbol,
            metrics=metrics,
            cfg=cfg,
            selection_path=selection_path,
            realized_pnl=realized_pnl,
            trade_records=trade_records,
            equity_snapshots=equity_snapshots,
            day_start_equity=day_start_equity,
            current_state=state,
            latest_close=latest_close_seen,
            latest_prob=latest_prob_seen,
            current_trade_id=current_trade.trade_id if current_trade else None,
        )
        review_row["paper"] = args.paper
        save_daily_review_summary(daily_review_path, review_row)
        save_session_summary(session_summary_path, session_start, now, symbol, metrics, cfg)
        print("Stopped by user.")
        print("Final session metrics:")
        print(asdict(metrics))
        print("Review summary:")
        print(review_row)


if __name__ == "__main__":
    main()
