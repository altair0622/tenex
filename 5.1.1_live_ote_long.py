#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

NY = ZoneInfo("America/New_York")


@dataclass
class LiveSetup:
    anchor_low_time: pd.Timestamp
    anchor_high_time: pd.Timestamp
    anchor_low: float
    anchor_high: float
    fib_50: float
    fib_62: float
    fib_79: float
    ote_low: float
    ote_high: float
    swing_range_pct: float
    valid_from: pd.Timestamp
    valid_until: pd.Timestamp


@dataclass
class PositionState:
    symbol: str
    entry_time: pd.Timestamp
    entry_bar_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    qty: int
    order_id: Optional[str] = None


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


def now_ny() -> datetime:
    return datetime.now(NY)


def parse_hhmm(value: str) -> dtime:
    hh, mm = value.split(":")
    return dtime(int(hh), int(mm))


def in_session(dt: datetime, start_hm: str, end_hm: str) -> bool:
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    start_dt = dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start_dt <= dt <= end_dt


def minute_sleep_seconds(now: Optional[datetime] = None, lag_seconds: int = 2) -> int:
    now = now or now_ny()
    return max(1, 60 - now.second + lag_seconds)


def resolve_latest_json(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(p.glob("best_etf_selection_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No best_etf_selection JSON files found in directory: {p}")
        return candidates[0]
    if any(ch in path_str for ch in "*?[]"):
        candidates = sorted(Path().glob(path_str), key=lambda x: x.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No JSON files matched pattern: {path_str}")
        return candidates[0]
    parent = p.parent if str(p.parent) != "" else Path(".")
    pattern = p.name if "." in p.name else f"{p.name}*.json"
    candidates = sorted(parent.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No JSON files matched: {parent / pattern}")
    return candidates[0]


def resolve_symbol_row_csv(selection_path: Path) -> Path:
    name = selection_path.name
    if name.startswith("best_etf_selection_"):
        suffix = name[len("best_etf_selection_"):]
        candidate = selection_path.with_name(f"best_etf_by_symbol_{suffix}")
        if candidate.exists():
            return candidate
    matches = sorted(selection_path.parent.glob("best_etf_by_symbol_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not find best_etf_by_symbol CSV near: {selection_path}")
    return matches[0]


def parse_symbol_override(symbols_arg: Optional[str]) -> Optional[str]:
    if not symbols_arg:
        return None
    parts = [x.strip().upper() for x in symbols_arg.split(",") if x.strip()]
    if not parts:
        return None
    return parts[0]


def apply_symbol_override(cfg: dict, selection_path: Path, symbol_override: Optional[str]) -> dict:
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
    sort_cols = [c for c in ["avg_daily_return_on_notional", "median_daily_return_on_notional", "positive_day_ratio", "total_pnl"] if c in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    row = rows.iloc[0]
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["selected_symbol"] = symbol_override.upper()
    cfg2["best_strategy_params"] = {
        "pivot_left": int(row["pivot_left"]),
        "pivot_right": int(row["pivot_right"]),
        "min_swing_range_pct": float(row["min_swing_range_pct"]),
        "setup_valid_minutes": int(row["setup_valid_minutes"]),
        "stop_buffer_pct": float(row["stop_buffer_pct"]),
        "max_wait_minutes_after_touch": int(row["max_wait_minutes_after_touch"]),
        "max_hold_minutes": int(row["max_hold_minutes"]),
        "rr_target": None if pd.isna(row["rr_target"]) else float(row["rr_target"]),
        "fee_bps_per_side": float(row["fee_bps_per_side"]),
    }
    cfg2["selection_reason"] = {
        "primary_metric": "manual_symbol_override",
        "summary": f"Live run manually overridden to use {symbol_override.upper()} with that symbol's best parameters from {by_symbol_path.name}.",
        "source_csv": by_symbol_path.name,
    }
    return cfg2


def build_clients(paper: bool) -> Tuple[TradingClient, StockHistoricalDataClient]:
    key = os.getenv("ALPACA_API_KEY_1") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY_1") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY_1 and ALPACA_SECRET_KEY_1, or APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    trading = TradingClient(key, secret, paper=paper)
    data_client = StockHistoricalDataClient(key, secret)
    return trading, data_client


def fetch_1m_bars(data_client: StockHistoricalDataClient, symbol: str, lookback_minutes: int, feed: str) -> pd.DataFrame:
    end = now_ny()
    start = end - timedelta(minutes=lookback_minutes)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=feed,
    )
    bars = data_client.get_stock_bars(req).df
    if bars.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
        bars = bars[bars["symbol"] == symbol].copy()
    else:
        bars = bars.reset_index()

    ts_col = "timestamp" if "timestamp" in bars.columns else "index"
    bars = bars.rename(columns={ts_col: "timestamp"})
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NY)
    out = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    x = df_1m.set_index("timestamp")
    df_5m = pd.DataFrame({
        "open": x["open"].resample("5min").first(),
        "high": x["high"].resample("5min").max(),
        "low": x["low"].resample("5min").min(),
        "close": x["close"].resample("5min").last(),
        "volume": x["volume"].resample("5min").sum(),
    }).dropna().reset_index()
    return df_5m


def detect_pivots_5m(df_5m: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    df = df_5m.copy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    highs = df["high"].values
    lows = df["low"].values
    for i in range(left, n - right):
        hi = highs[i]
        lo = lows[i]
        if np.all(hi > highs[i - left:i]) and np.all(hi > highs[i + 1:i + 1 + right]):
            swing_high[i] = True
        if np.all(lo < lows[i - left:i]) and np.all(lo < lows[i + 1:i + 1 + right]):
            swing_low[i] = True
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    df["pivot_confirm_time"] = df["timestamp"] + pd.Timedelta(minutes=5 * right)
    return df


def extract_confirmed_pivots(df_5m: pd.DataFrame, asof: pd.Timestamp) -> Tuple[List[Dict], List[Dict]]:
    highs: List[Dict] = []
    lows: List[Dict] = []
    part = df_5m[df_5m["pivot_confirm_time"] <= asof].copy()
    for _, row in part[part["swing_high"]].iterrows():
        highs.append({"pivot_time": row["timestamp"], "confirm_time": row["pivot_confirm_time"], "price": float(row["high"])})
    for _, row in part[part["swing_low"]].iterrows():
        lows.append({"pivot_time": row["timestamp"], "confirm_time": row["pivot_confirm_time"], "price": float(row["low"])})
    return highs, lows


def compute_setup_from_latest_structure(
    df_1m: pd.DataFrame,
    pivot_left: int,
    pivot_right: int,
    min_swing_range_pct: float,
    setup_valid_minutes: int,
) -> Optional[LiveSetup]:
    if len(df_1m) < 80:
        return None
    df_5m = resample_to_5m(df_1m)
    df_5m = detect_pivots_5m(df_5m, left=pivot_left, right=pivot_right)
    asof = df_1m.iloc[-1]["timestamp"]
    highs, lows = extract_confirmed_pivots(df_5m, asof=asof)
    if len(highs) < 2 or len(lows) < 2:
        return None

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    trend_up = (h2["price"] > h1["price"]) and (l2["price"] > l1["price"])
    if not trend_up:
        return None

    candidate_lows = [x for x in lows if x["pivot_time"] < h2["pivot_time"]]
    if not candidate_lows:
        return None

    anchor_low_info = candidate_lows[-1]
    anchor_high_info = h2
    anchor_low = float(anchor_low_info["price"])
    anchor_high = float(anchor_high_info["price"])
    if anchor_high <= anchor_low:
        return None

    swing_range_pct = (anchor_high - anchor_low) / anchor_low
    if swing_range_pct < min_swing_range_pct:
        return None

    diff = anchor_high - anchor_low
    fib_50 = anchor_low + 0.50 * diff
    fib_62 = anchor_low + 0.62 * diff
    fib_79 = anchor_low + 0.79 * diff
    valid_from = anchor_high_info["confirm_time"]
    valid_until = valid_from + pd.Timedelta(minutes=setup_valid_minutes)
    if asof > valid_until:
        return None

    return LiveSetup(
        anchor_low_time=anchor_low_info["pivot_time"],
        anchor_high_time=anchor_high_info["pivot_time"],
        anchor_low=anchor_low,
        anchor_high=anchor_high,
        fib_50=fib_50,
        fib_62=fib_62,
        fib_79=fib_79,
        ote_low=fib_62,
        ote_high=fib_79,
        swing_range_pct=swing_range_pct,
        valid_from=valid_from,
        valid_until=valid_until,
    )


def has_entry_signal(df_1m: pd.DataFrame, setup: LiveSetup, max_wait_minutes_after_touch: int) -> Tuple[bool, Optional[float]]:
    now_ts = df_1m.iloc[-1]["timestamp"]
    if now_ts < setup.valid_from or now_ts > setup.valid_until:
        return False, None

    window = df_1m[(df_1m["timestamp"] >= setup.valid_from) & (df_1m["timestamp"] <= now_ts)].copy()
    if window.empty:
        return False, None

    window["in_ote_zone"] = (window["low"] <= setup.ote_high) & (window["high"] >= setup.ote_low)
    ote_rows = window[window["in_ote_zone"]]
    if ote_rows.empty:
        return False, None

    first_touch = ote_rows.iloc[0]["timestamp"]
    if now_ts > first_touch + pd.Timedelta(minutes=max_wait_minutes_after_touch):
        return False, None

    if len(window) < 4:
        return False, None

    prev3_high = window["high"].shift(1).rolling(3).max().iloc[-1]
    last_close = float(window.iloc[-1]["close"])
    last_low = float(window.iloc[-1]["low"])
    if pd.isna(prev3_high):
        return False, None
    if last_low < setup.anchor_low:
        return False, None
    if last_close > float(prev3_high):
        return True, last_close
    return False, None


def get_open_position(trading: TradingClient, symbol: str) -> Optional[PositionState]:
    try:
        positions = trading.get_all_positions()
    except Exception:
        return None
    for pos in positions:
        if getattr(pos, "symbol", None) == symbol:
            qty = int(float(getattr(pos, "qty", 0)))
            entry_price = float(getattr(pos, "avg_entry_price", 0.0))
            return PositionState(
                symbol=symbol,
                entry_time=pd.Timestamp(now_ny()),
                entry_bar_time=pd.Timestamp(now_ny()),
                entry_price=entry_price,
                stop_price=0.0,
                target_price=0.0,
                qty=qty,
                order_id=None,
            )
    return None


def submit_market_buy(trading: TradingClient, symbol: str, qty: int):
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    return trading.submit_order(req)


def submit_market_sell(trading: TradingClient, symbol: str, qty: int):
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    return trading.submit_order(req)


def calc_qty(last_price: float, notional: float, max_qty: int) -> int:
    if last_price <= 0:
        return 0
    qty = int(notional // last_price)
    if max_qty > 0:
        qty = min(qty, max_qty)
    return max(qty, 0)


def compute_target_price(entry_price: float, stop_price: float, anchor_high: float, rr_target: Optional[float]) -> float:
    if rr_target is None:
        return anchor_high
    risk = entry_price - stop_price
    if risk <= 0:
        return anchor_high
    return entry_price + rr_target * risk


def live_loop(args: argparse.Namespace) -> None:
    selection_path = resolve_latest_json(args.selection_json)
    print(f"Using selection JSON: {selection_path}")
    with selection_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    symbol_override = args.symbol or parse_symbol_override(args.symbols)
    cfg = apply_symbol_override(cfg, selection_path, symbol_override)

    symbol = cfg["selected_symbol"]
    strat = cfg["best_strategy_params"]
    feed = args.feed or cfg.get("data_window", {}).get("feed", "iex")
    pivot_left = int(strat["pivot_left"])
    pivot_right = int(strat["pivot_right"])
    min_swing_range_pct = float(strat["min_swing_range_pct"])
    setup_valid_minutes = int(strat["setup_valid_minutes"])
    stop_buffer_pct = float(strat["stop_buffer_pct"])
    max_wait_minutes_after_touch = int(strat["max_wait_minutes_after_touch"])
    max_hold_minutes = int(strat["max_hold_minutes"])
    rr_target = strat.get("rr_target", None)
    rr_target = None if rr_target is None else float(rr_target)
    market_open = cfg.get("market_hours", {}).get("open", "10:00")
    market_close = cfg.get("market_hours", {}).get("close", "15:30")

    trading, data_client = build_clients(args.paper)
    state: Optional[PositionState] = None
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_ts = now_ny().strftime("%Y%m%d_%H%M%S")

    event_log = CsvLogger(logs_dir / f"live_events_{symbol}_{run_ts}.csv", [
        "ts", "symbol", "event", "close_px", "entry_px", "stop_px", "target_px", "qty", "note"
    ])

    print(json.dumps({
        "symbol": symbol,
        "paper": args.paper,
        "feed": feed,
        "pivot_left": pivot_left,
        "pivot_right": pivot_right,
        "min_swing_range_pct": min_swing_range_pct,
        "setup_valid_minutes": setup_valid_minutes,
        "stop_buffer_pct": stop_buffer_pct,
        "max_wait_minutes_after_touch": max_wait_minutes_after_touch,
        "max_hold_minutes": max_hold_minutes,
        "rr_target": rr_target,
        "notional_per_trade": args.notional_per_trade,
        "max_qty": args.max_qty,
        "selection_path": str(selection_path),
    }, indent=2))

    while True:
        dt = now_ny()
        if not in_session(dt, market_open, market_close):
            print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] outside session")
            time.sleep(minute_sleep_seconds(dt, lag_seconds=5))
            continue

        try:
            df_1m = fetch_1m_bars(data_client, symbol, args.lookback_minutes, feed)
            if df_1m.empty:
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} no bars")
                time.sleep(minute_sleep_seconds(dt))
                continue

            last_row = df_1m.iloc[-1]
            last_ts = pd.Timestamp(last_row["timestamp"])
            last_px = float(last_row["close"])

            open_pos = get_open_position(trading, symbol)
            if open_pos is not None and state is None:
                state = open_pos
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] synced broker position {symbol} qty={state.qty} entry={state.entry_price:.4f}")

            live_setup = compute_setup_from_latest_structure(
                df_1m=df_1m,
                pivot_left=pivot_left,
                pivot_right=pivot_right,
                min_swing_range_pct=min_swing_range_pct,
                setup_valid_minutes=setup_valid_minutes,
            )

            if state is None:
                if live_setup is None:
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} no valid setup | px={last_px:.4f}")
                    time.sleep(minute_sleep_seconds(dt))
                    continue

                signal, entry_px = has_entry_signal(
                    df_1m=df_1m,
                    setup=live_setup,
                    max_wait_minutes_after_touch=max_wait_minutes_after_touch,
                )

                print(
                    f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} px={last_px:.4f} "
                    f"ote=({live_setup.ote_low:.4f},{live_setup.ote_high:.4f}) signal={signal}"
                )

                if signal:
                    qty = calc_qty(last_price=entry_px, notional=args.notional_per_trade, max_qty=args.max_qty)
                    if qty > 0:
                        order = submit_market_buy(trading, symbol, qty)
                        stop_price = live_setup.anchor_low * (1.0 - stop_buffer_pct)
                        target_price = compute_target_price(float(entry_px), float(stop_price), float(live_setup.anchor_high), rr_target)
                        state = PositionState(
                            symbol=symbol,
                            entry_time=last_ts,
                            entry_bar_time=last_ts,
                            entry_price=float(entry_px),
                            stop_price=float(stop_price),
                            target_price=float(target_price),
                            qty=qty,
                            order_id=str(getattr(order, "id", "")),
                        )
                        print(
                            f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] BUY {symbol} qty={qty} entry~{entry_px:.4f} "
                            f"stop={stop_price:.4f} target={target_price:.4f}"
                        )
                        event_log.write({
                            "ts": dt.isoformat(),
                            "symbol": symbol,
                            "event": "BUY",
                            "close_px": last_px,
                            "entry_px": entry_px,
                            "stop_px": stop_price,
                            "target_px": target_price,
                            "qty": qty,
                            "note": f"order_id={getattr(order, 'id', '')}",
                        })
            else:
                hold_minutes = int((last_ts - state.entry_time).total_seconds() // 60)
                should_exit = None
                if last_px <= state.stop_price:
                    should_exit = "stop"
                elif last_px >= state.target_price:
                    should_exit = "target"
                elif hold_minutes >= max_hold_minutes:
                    should_exit = "time_exit"

                print(
                    f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] HOLD {symbol} px={last_px:.4f} entry={state.entry_price:.4f} "
                    f"stop={state.stop_price:.4f} target={state.target_price:.4f} held={hold_minutes}m"
                )

                if should_exit is not None:
                    submit_market_sell(trading, symbol, state.qty)
                    print(
                        f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] SELL {symbol} qty={state.qty} px~{last_px:.4f} reason={should_exit}"
                    )
                    event_log.write({
                        "ts": dt.isoformat(),
                        "symbol": symbol,
                        "event": "SELL",
                        "close_px": last_px,
                        "entry_px": state.entry_price,
                        "stop_px": state.stop_price,
                        "target_px": state.target_price,
                        "qty": state.qty,
                        "note": f"reason={should_exit}",
                    })
                    state = None

        except Exception as exc:
            print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} ERROR {exc}")
            event_log.write({
                "ts": dt.isoformat(),
                "symbol": symbol,
                "event": "ERROR",
                "close_px": "",
                "entry_px": "",
                "stop_px": "",
                "target_px": "",
                "qty": "",
                "note": str(exc),
            })

        time.sleep(minute_sleep_seconds(dt))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="5.1.1 OTE Long Live Runner")
    p.add_argument("--selection_json", default="selector_out_511/best_etf_selection_*.json")
    p.add_argument("--feed", default="")
    p.add_argument("--symbols", default=None, help="Optional symbol override. Example: --symbols TQQQ")
    p.add_argument("--symbol", default=None, help="Optional single-symbol override. Same effect as --symbols TQQQ")
    p.add_argument("--paper", action="store_true")
    p.add_argument("--lookback_minutes", type=int, default=390)
    p.add_argument("--notional_per_trade", type=float, default=1000.0)
    p.add_argument("--max_qty", type=int, default=0)
    p.add_argument("--logs_dir", default="live_logs_511")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    live_loop(args)
