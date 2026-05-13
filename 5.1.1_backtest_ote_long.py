#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time
from itertools import product
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

NY = ZoneInfo("America/New_York")
MARKET_OPEN = time(10, 0)
MARKET_CLOSE = time(15, 30)


@dataclass
class Setup:
    setup_id: int
    anchor_low_time: pd.Timestamp
    anchor_low_confirm_time: pd.Timestamp
    anchor_low: float
    anchor_high_time: pd.Timestamp
    anchor_high_confirm_time: pd.Timestamp
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
class Trade:
    symbol: str
    setup_id: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    hold_minutes: int
    gross_return: float
    net_return: float
    pnl_dollars: float
    notional: float
    trade_date: str
    anchor_low_time: pd.Timestamp
    anchor_high_time: pd.Timestamp
    ote_low: float
    ote_high: float


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
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "timestamp": "timestamp",
    })
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NY)
    bars = bars.sort_values("timestamp").set_index("timestamp")

    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol].copy()
        bars.drop(columns=["symbol"], inplace=True)

    bars["symbol"] = symbol
    return bars


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


def extract_confirmed_pivot_events(df_5m: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df_5m[df_5m["swing_high"]].iterrows():
        rows.append({
            "pivot_time": row["timestamp"],
            "confirm_time": row["pivot_confirm_time"],
            "price": float(row["high"]),
            "type": "H",
        })
    for _, row in df_5m[df_5m["swing_low"]].iterrows():
        rows.append({
            "pivot_time": row["timestamp"],
            "confirm_time": row["pivot_confirm_time"],
            "price": float(row["low"]),
            "type": "L",
        })
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values(["confirm_time", "pivot_time", "type"]).reset_index(drop=True)


def compute_fib_levels(anchor_low: float, anchor_high: float) -> Dict[str, float]:
    diff = anchor_high - anchor_low
    fib_50 = anchor_low + 0.50 * diff
    fib_62 = anchor_low + 0.62 * diff
    fib_79 = anchor_low + 0.79 * diff
    return {
        "fib_50": fib_50,
        "fib_62": fib_62,
        "fib_79": fib_79,
        "ote_low": fib_62,
        "ote_high": fib_79,
    }


def generate_all_setups(
    df_5m: pd.DataFrame,
    min_swing_range_pct: float = 0.004,
    setup_valid_minutes: int = 60,
) -> List[Setup]:
    events = extract_confirmed_pivot_events(df_5m)
    if events.empty:
        return []

    confirmed_highs: List[Dict] = []
    confirmed_lows: List[Dict] = []
    setups: List[Setup] = []
    seen_keys = set()
    setup_id = 1

    for _, event in events.iterrows():
        item = {
            "pivot_time": event["pivot_time"],
            "confirm_time": event["confirm_time"],
            "price": float(event["price"]),
            "type": event["type"],
        }

        if item["type"] == "H":
            confirmed_highs.append(item)
        else:
            confirmed_lows.append(item)

        if len(confirmed_highs) < 2 or len(confirmed_lows) < 2:
            continue

        h1, h2 = confirmed_highs[-2], confirmed_highs[-1]
        l1, l2 = confirmed_lows[-2], confirmed_lows[-1]
        trend_up = (h2["price"] > h1["price"]) and (l2["price"] > l1["price"])
        if not trend_up:
            continue

        candidate_lows = [x for x in confirmed_lows if x["pivot_time"] < h2["pivot_time"]]
        if not candidate_lows:
            continue

        anchor_low_info = candidate_lows[-1]
        anchor_high_info = h2
        anchor_low = float(anchor_low_info["price"])
        anchor_high = float(anchor_high_info["price"])
        if anchor_high <= anchor_low:
            continue

        swing_range_pct = (anchor_high - anchor_low) / anchor_low
        if swing_range_pct < min_swing_range_pct:
            continue

        key = (anchor_low_info["pivot_time"], anchor_high_info["pivot_time"])
        if key in seen_keys:
            continue

        fib = compute_fib_levels(anchor_low, anchor_high)
        valid_from = anchor_high_info["confirm_time"]
        valid_until = valid_from + pd.Timedelta(minutes=setup_valid_minutes)

        setups.append(Setup(
            setup_id=setup_id,
            anchor_low_time=anchor_low_info["pivot_time"],
            anchor_low_confirm_time=anchor_low_info["confirm_time"],
            anchor_low=anchor_low,
            anchor_high_time=anchor_high_info["pivot_time"],
            anchor_high_confirm_time=anchor_high_info["confirm_time"],
            anchor_high=anchor_high,
            fib_50=fib["fib_50"],
            fib_62=fib["fib_62"],
            fib_79=fib["fib_79"],
            ote_low=fib["ote_low"],
            ote_high=fib["ote_high"],
            swing_range_pct=swing_range_pct,
            valid_from=valid_from,
            valid_until=valid_until,
        ))
        seen_keys.add(key)
        setup_id += 1

    return setups


def add_prev3_break_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prev3_high"] = out["high"].shift(1).rolling(3).max()
    out["break_prev3_high"] = out["close"] > out["prev3_high"]
    return out


def compute_stop_price(anchor_low: float, stop_buffer_pct: float = 0.0005) -> float:
    return anchor_low * (1.0 - stop_buffer_pct)


def compute_target_price(entry_price: float, stop_price: float, setup: Setup, rr_target: Optional[float]) -> float:
    if rr_target is None:
        return setup.anchor_high
    risk = entry_price - stop_price
    if risk <= 0:
        return setup.anchor_high
    return entry_price + rr_target * risk


def find_entry_for_setup(
    df_1m: pd.DataFrame,
    setup: Setup,
    max_wait_minutes_after_touch: int = 10,
) -> Optional[Tuple[pd.Timestamp, float]]:
    window = df_1m[
        (df_1m["timestamp"] >= setup.valid_from) &
        (df_1m["timestamp"] <= setup.valid_until)
    ].copy()
    if window.empty:
        return None

    window["in_ote_zone"] = (window["low"] <= setup.ote_high) & (window["high"] >= setup.ote_low)
    window["below_anchor_low"] = window["low"] < setup.anchor_low
    window = add_prev3_break_signal(window)

    ote_rows = window[window["in_ote_zone"]]
    if ote_rows.empty:
        return None

    first_touch_time = ote_rows.iloc[0]["timestamp"]
    search_until = min(first_touch_time + pd.Timedelta(minutes=max_wait_minutes_after_touch), setup.valid_until)

    candidate = window[
        (window["timestamp"] >= first_touch_time) &
        (window["timestamp"] <= search_until)
    ].copy()
    candidate = candidate[~candidate["below_anchor_low"]]
    candidate = candidate[candidate["break_prev3_high"].fillna(False)]
    if candidate.empty:
        return None

    row = candidate.iloc[0]
    return row["timestamp"], float(row["close"])


def simulate_trade(
    df_1m: pd.DataFrame,
    symbol: str,
    setup: Setup,
    entry_time: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float,
    qty: int,
    max_hold_minutes: int = 20,
    fee_bps_per_side: float = 0.0,
) -> Trade:
    expiry_time = entry_time + pd.Timedelta(minutes=max_hold_minutes)
    future = df_1m[(df_1m["timestamp"] > entry_time) & (df_1m["timestamp"] <= expiry_time)].copy()

    exit_time = entry_time
    exit_price = entry_price
    exit_reason = "time_exit"

    for _, row in future.iterrows():
        bar_time = row["timestamp"]
        bar_low = float(row["low"])
        bar_high = float(row["high"])

        hit_stop = bar_low <= stop_price
        hit_target = bar_high >= target_price

        if hit_stop and hit_target:
            exit_time = bar_time
            exit_price = stop_price
            exit_reason = "stop_and_target_same_bar_assume_stop_first"
            break
        if hit_stop:
            exit_time = bar_time
            exit_price = stop_price
            exit_reason = "stop"
            break
        if hit_target:
            exit_time = bar_time
            exit_price = target_price
            exit_reason = "target"
            break
    else:
        if not future.empty:
            last_row = future.iloc[-1]
            exit_time = last_row["timestamp"]
            exit_price = float(last_row["close"])
            exit_reason = "time_exit"

    gross_return = (exit_price / entry_price) - 1.0
    fee_rate = fee_bps_per_side / 10000.0
    net_return = gross_return - (2.0 * fee_rate)
    notional = entry_price * qty
    pnl_dollars = net_return * notional
    hold_minutes = int((exit_time - entry_time).total_seconds() // 60)

    return Trade(
        symbol=symbol,
        setup_id=setup.setup_id,
        entry_time=entry_time,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        hold_minutes=hold_minutes,
        gross_return=gross_return,
        net_return=net_return,
        pnl_dollars=pnl_dollars,
        notional=notional,
        trade_date=entry_time.date().isoformat(),
        anchor_low_time=setup.anchor_low_time,
        anchor_high_time=setup.anchor_high_time,
        ote_low=setup.ote_low,
        ote_high=setup.ote_high,
    )


def backtest_symbol(
    df_1m_symbol: pd.DataFrame,
    symbol: str,
    pivot_left: int,
    pivot_right: int,
    min_swing_range_pct: float,
    setup_valid_minutes: int,
    stop_buffer_pct: float,
    max_wait_minutes_after_touch: int,
    max_hold_minutes: int,
    fee_bps_per_side: float,
    allow_overlapping_positions: bool,
    rr_target: Optional[float],
    qty: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_1m = df_1m_symbol.copy().sort_values("timestamp").reset_index(drop=True)
    df_5m = resample_to_5m(df_1m)
    df_5m = detect_pivots_5m(df_5m, left=pivot_left, right=pivot_right)
    setups = generate_all_setups(df_5m, min_swing_range_pct=min_swing_range_pct, setup_valid_minutes=setup_valid_minutes)

    trades: List[Trade] = []
    next_free_time = pd.Timestamp.min.tz_localize(df_1m["timestamp"].dt.tz) if df_1m["timestamp"].dt.tz is not None else pd.Timestamp.min

    for setup in setups:
        if not allow_overlapping_positions and setup.valid_from < next_free_time:
            continue

        entry = find_entry_for_setup(df_1m=df_1m, setup=setup, max_wait_minutes_after_touch=max_wait_minutes_after_touch)
        if entry is None:
            continue

        entry_time, entry_price = entry
        if not allow_overlapping_positions and entry_time < next_free_time:
            continue

        stop_price = compute_stop_price(setup.anchor_low, stop_buffer_pct=stop_buffer_pct)
        target_price = compute_target_price(entry_price, stop_price, setup, rr_target)
        if stop_price >= entry_price or target_price <= entry_price:
            continue

        trade = simulate_trade(
            df_1m=df_1m,
            symbol=symbol,
            setup=setup,
            entry_time=entry_time,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            qty=qty,
            max_hold_minutes=max_hold_minutes,
            fee_bps_per_side=fee_bps_per_side,
        )
        trades.append(trade)
        if not allow_overlapping_positions:
            next_free_time = trade.exit_time

    setups_df = pd.DataFrame([asdict(s) for s in setups]) if setups else pd.DataFrame()
    trades_df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()

    if trades_df.empty:
        daily_df = pd.DataFrame(columns=["trade_date", "symbol", "daily_pnl", "daily_notional", "daily_return_on_notional", "trades"])
    else:
        daily_df = (
            trades_df.groupby(["trade_date", "symbol"], as_index=False)
            .agg(
                daily_pnl=("pnl_dollars", "sum"),
                daily_notional=("notional", "sum"),
                trades=("setup_id", "count"),
            )
        )
        daily_df["daily_return_on_notional"] = np.where(
            daily_df["daily_notional"] > 0,
            daily_df["daily_pnl"] / daily_df["daily_notional"],
            0.0,
        )
    return setups_df, trades_df, daily_df


def compute_equity_curve(trades_df: pd.DataFrame, start_equity: float = 1.0) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["entry_time", "equity", "cummax", "drawdown"])
    out = trades_df.sort_values("entry_time").copy()
    out["equity"] = start_equity * (1.0 + out["net_return"]).cumprod()
    out["cummax"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["cummax"] - 1.0
    return out


def summarize_symbol_result(symbol: str, params: dict, trades_df: pd.DataFrame, daily_df: pd.DataFrame, initial_cash: float) -> dict:
    if trades_df.empty:
        return {
            "symbol": symbol,
            **params,
            "avg_daily_return_on_notional": 0.0,
            "median_daily_return_on_notional": 0.0,
            "worst_day_return_on_notional": 0.0,
            "best_day_return_on_notional": 0.0,
            "positive_day_ratio": 0.0,
            "avg_daily_pnl": 0.0,
            "median_daily_pnl": 0.0,
            "total_pnl": 0.0,
            "total_trades": 0,
            "avg_trades_per_day": 0.0,
            "avg_daily_notional_used": 0.0,
            "median_daily_notional_used": 0.0,
            "avg_entry_notional": 0.0,
            "equity_final": initial_cash,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "eval_days_used": 0,
        }

    equity_df = compute_equity_curve(trades_df)
    return {
        "symbol": symbol,
        **params,
        "avg_daily_return_on_notional": float(daily_df["daily_return_on_notional"].mean()) if not daily_df.empty else 0.0,
        "median_daily_return_on_notional": float(daily_df["daily_return_on_notional"].median()) if not daily_df.empty else 0.0,
        "worst_day_return_on_notional": float(daily_df["daily_return_on_notional"].min()) if not daily_df.empty else 0.0,
        "best_day_return_on_notional": float(daily_df["daily_return_on_notional"].max()) if not daily_df.empty else 0.0,
        "positive_day_ratio": float((daily_df["daily_pnl"] > 0).mean()) if not daily_df.empty else 0.0,
        "avg_daily_pnl": float(daily_df["daily_pnl"].mean()) if not daily_df.empty else 0.0,
        "median_daily_pnl": float(daily_df["daily_pnl"].median()) if not daily_df.empty else 0.0,
        "total_pnl": float(trades_df["pnl_dollars"].sum()),
        "total_trades": int(len(trades_df)),
        "avg_trades_per_day": float(daily_df["trades"].mean()) if not daily_df.empty else 0.0,
        "avg_daily_notional_used": float(daily_df["daily_notional"].mean()) if not daily_df.empty else 0.0,
        "median_daily_notional_used": float(daily_df["daily_notional"].median()) if not daily_df.empty else 0.0,
        "avg_entry_notional": float(trades_df["notional"].mean()),
        "equity_final": float(initial_cash * (1.0 + trades_df["net_return"]).prod()),
        "max_drawdown": float(equity_df["drawdown"].min()) if not equity_df.empty else 0.0,
        "win_rate": float((trades_df["net_return"] > 0).mean()),
        "eval_days_used": int(daily_df["trade_date"].nunique()) if not daily_df.empty else 0,
    }


def make_selection_reason(best_stats: dict, runner_up_symbol: Optional[str], runner_up_stats: Optional[dict]) -> dict:
    if runner_up_symbol is None or runner_up_stats is None:
        return {
            "primary_metric": "avg_daily_return_on_notional",
            "summary": (
                f"{best_stats['symbol']} was selected because it was the only valid ETF with OTE backtest results, "
                f"and its best params had the highest average daily return on notional."
            ),
        }
    return {
        "primary_metric": "avg_daily_return_on_notional",
        "summary": (
            f"{best_stats['symbol']} was selected because it had the highest average daily return on notional "
            f"among the OTE long parameter grid over the evaluated period."
        ),
        "runner_up_symbol": runner_up_symbol,
        "selected_avg_daily_return_on_notional": best_stats["avg_daily_return_on_notional"],
        "runner_up_avg_daily_return_on_notional": runner_up_stats["avg_daily_return_on_notional"],
        "selected_total_pnl": best_stats["total_pnl"],
        "runner_up_total_pnl": runner_up_stats["total_pnl"],
        "selected_positive_day_ratio": best_stats["positive_day_ratio"],
        "runner_up_positive_day_ratio": runner_up_stats["positive_day_ratio"],
    }


def run_selector(args: argparse.Namespace) -> None:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rr_values = []
    for x in args.rr_targets.split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() == "none":
            rr_values.append(None)
        else:
            rr_values.append(float(x))

    if not rr_values:
        rr_values = [None]
	    
    grid = {
        "pivot_left": [int(x.strip()) for x in args.pivot_left_grid.split(",") if x.strip()],
        "pivot_right": [int(x.strip()) for x in args.pivot_right_grid.split(",") if x.strip()],
        "min_swing_range_pct": [float(x.strip()) for x in args.min_swing_range_grid.split(",") if x.strip()],
        "stop_buffer_pct": [float(x.strip()) for x in args.stop_buffer_grid.split(",") if x.strip()],
        "max_wait_minutes_after_touch": [int(x.strip()) for x in args.max_wait_grid.split(",") if x.strip()],
        "max_hold_minutes": [int(x.strip()) for x in args.max_hold_grid.split(",") if x.strip()],
        "rr_target": rr_values,
    }
    param_grid = list(product(
        grid["pivot_left"],
        grid["pivot_right"],
        grid["min_swing_range_pct"],
        grid["stop_buffer_pct"],
        grid["max_wait_minutes_after_touch"],
        grid["max_hold_minutes"],
        grid["rr_target"],
    ))

    end = datetime.now(tz=NY)
    start = end - timedelta(days=args.days)
    run_ts = datetime.now(tz=NY).strftime("%Y%m%d_%H%M%S")

    best_rows = []
    grid_rows = []
    daily_rows_all = []
    trade_rows_all = []

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

        df = raw.reset_index().rename(columns={"index": "timestamp"})
        symbol_best = None

        for pivot_left, pivot_right, min_swing_range_pct, stop_buffer_pct, max_wait, max_hold, rr_target in param_grid:
            params = {
                "pivot_left": pivot_left,
                "pivot_right": pivot_right,
                "min_swing_range_pct": min_swing_range_pct,
                "setup_valid_minutes": args.setup_valid_minutes,
                "stop_buffer_pct": stop_buffer_pct,
                "max_wait_minutes_after_touch": max_wait,
                "max_hold_minutes": max_hold,
                "rr_target": rr_target,
                "fee_bps_per_side": args.fee_bps_per_side,
                "qty": args.qty,
            }
            try:
                setups_df, trades_df, daily_df = backtest_symbol(
                    df_1m_symbol=df,
                    symbol=symbol,
                    pivot_left=pivot_left,
                    pivot_right=pivot_right,
                    min_swing_range_pct=min_swing_range_pct,
                    setup_valid_minutes=args.setup_valid_minutes,
                    stop_buffer_pct=stop_buffer_pct,
                    max_wait_minutes_after_touch=max_wait,
                    max_hold_minutes=max_hold,
                    fee_bps_per_side=args.fee_bps_per_side,
                    allow_overlapping_positions=args.allow_overlapping_positions,
                    rr_target=rr_target,
                    qty=args.qty,
                )
            except Exception as e:
                print(f"  {symbol} params={params} failed: {e}")
                continue

            stats = summarize_symbol_result(symbol, params, trades_df, daily_df, args.initial_cash)
            grid_rows.append(stats)

            if not daily_df.empty:
                daily_df = daily_df.copy()
                for k, v in params.items():
                    daily_df[k] = v
                daily_rows_all.extend(daily_df.to_dict("records"))
            if not trades_df.empty:
                trades_df = trades_df.copy()
                for k, v in params.items():
                    trades_df[k] = v
                trade_rows_all.extend(trades_df.to_dict("records"))

            if symbol_best is None:
                symbol_best = stats
            else:
                better = (
                    stats["avg_daily_return_on_notional"] > symbol_best["avg_daily_return_on_notional"] or
                    (
                        np.isclose(stats["avg_daily_return_on_notional"], symbol_best["avg_daily_return_on_notional"]) and
                        stats["median_daily_return_on_notional"] > symbol_best["median_daily_return_on_notional"]
                    ) or
                    (
                        np.isclose(stats["avg_daily_return_on_notional"], symbol_best["avg_daily_return_on_notional"]) and
                        np.isclose(stats["median_daily_return_on_notional"], symbol_best["median_daily_return_on_notional"]) and
                        stats["positive_day_ratio"] > symbol_best["positive_day_ratio"]
                    )
                )
                if better:
                    symbol_best = stats

        if symbol_best is None:
            print(f"Skipping {symbol}: no valid grid result")
            continue

        print({
            "symbol": symbol_best["symbol"],
            "avg_daily_return_on_notional": round(symbol_best["avg_daily_return_on_notional"], 6),
            "total_pnl": round(symbol_best["total_pnl"], 2),
            "positive_day_ratio": round(symbol_best["positive_day_ratio"], 4),
            "total_trades": int(symbol_best["total_trades"]),
            "pivot_left": symbol_best["pivot_left"],
            "pivot_right": symbol_best["pivot_right"],
            "min_swing_range_pct": symbol_best["min_swing_range_pct"],
            "stop_buffer_pct": symbol_best["stop_buffer_pct"],
            "max_wait_minutes_after_touch": symbol_best["max_wait_minutes_after_touch"],
            "max_hold_minutes": symbol_best["max_hold_minutes"],
            "rr_target": symbol_best["rr_target"],
        })
        best_rows.append(symbol_best)

    if not best_rows:
        raise RuntimeError("No valid ETF results were produced")

    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values(
        by=["avg_daily_return_on_notional", "median_daily_return_on_notional", "positive_day_ratio", "total_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    selected = best_df.iloc[0].to_dict()
    runner_up = best_df.iloc[1].to_dict() if len(best_df) > 1 else None

    selection_json = {
        "version": "5.1.1",
        "selection_mode": "single_best_etf_ote_long_daily_return_on_notional",
        "selected_symbol": selected["symbol"],
        "universe": symbols,
        "strategy_type": "OTE_LONG_RULE_BASED",
        "best_strategy_params": {
            "pivot_left": int(selected["pivot_left"]),
            "pivot_right": int(selected["pivot_right"]),
            "min_swing_range_pct": float(selected["min_swing_range_pct"]),
            "setup_valid_minutes": int(selected["setup_valid_minutes"]),
            "stop_buffer_pct": float(selected["stop_buffer_pct"]),
            "max_wait_minutes_after_touch": int(selected["max_wait_minutes_after_touch"]),
            "max_hold_minutes": int(selected["max_hold_minutes"]),
            "rr_target": None if pd.isna(selected["rr_target"]) else float(selected["rr_target"]),
            "fee_bps_per_side": float(selected["fee_bps_per_side"]),
        },
        "best_backtest_stats": {
            "avg_daily_return_on_notional": float(selected["avg_daily_return_on_notional"]),
            "median_daily_return_on_notional": float(selected["median_daily_return_on_notional"]),
            "worst_day_return_on_notional": float(selected["worst_day_return_on_notional"]),
            "best_day_return_on_notional": float(selected["best_day_return_on_notional"]),
            "positive_day_ratio": float(selected["positive_day_ratio"]),
            "avg_daily_pnl": float(selected["avg_daily_pnl"]),
            "median_daily_pnl": float(selected["median_daily_pnl"]),
            "total_pnl": float(selected["total_pnl"]),
            "total_trades": int(selected["total_trades"]),
            "avg_trades_per_day": float(selected["avg_trades_per_day"]),
            "avg_daily_notional_used": float(selected["avg_daily_notional_used"]),
            "median_daily_notional_used": float(selected["median_daily_notional_used"]),
            "avg_entry_notional": float(selected["avg_entry_notional"]),
            "equity_final": float(selected["equity_final"]),
            "max_drawdown": float(selected["max_drawdown"]),
            "win_rate": float(selected["win_rate"]),
            "eval_days_used": int(selected["eval_days_used"]),
        },
        "selection_reason": make_selection_reason(selected, runner_up["symbol"] if runner_up else None, runner_up),
        "market_hours": {
            "open": MARKET_OPEN.strftime("%H:%M"),
            "close": MARKET_CLOSE.strftime("%H:%M"),
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
    trade_log_path = out_dir / f"trade_log_{run_ts}.csv"

    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection_json, f, indent=2)

    best_df.to_csv(best_by_symbol_path, index=False)
    pd.DataFrame(grid_rows).to_csv(grid_results_path, index=False)
    pd.DataFrame(daily_rows_all).to_csv(daily_roll_path, index=False)
    pd.DataFrame(trade_rows_all).to_csv(trade_log_path, index=False)

    print(f"Saved selection JSON: {selection_path}")
    print(f"Saved best-by-symbol CSV: {best_by_symbol_path}")
    print(f"Saved grid results CSV: {grid_results_path}")
    print(f"Saved daily roll results CSV: {daily_roll_path}")
    print(f"Saved trade log CSV: {trade_log_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="5.1.1 OTE Long Backtest Selector")
    p.add_argument("--symbols", default="SPY,QQQ,IWM,SOXL,TQQQ")
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--feed", default="iex")
    p.add_argument("--out_dir", default="selector_out_511")
    p.add_argument("--qty", type=int, default=10)
    p.add_argument("--initial_cash", type=float, default=100000)
    p.add_argument("--setup_valid_minutes", type=int, default=60)
    p.add_argument("--fee_bps_per_side", type=float, default=1.0)
    p.add_argument("--allow_overlapping_positions", action="store_true")
    p.add_argument("--pivot_left_grid", default="2")
    p.add_argument("--pivot_right_grid", default="2")
    p.add_argument("--min_swing_range_grid", default="0.004,0.005")
    p.add_argument("--stop_buffer_grid", default="0.0005,0.0010")
    p.add_argument("--max_wait_grid", default="10")
    p.add_argument("--max_hold_grid", default="20,30")
    p.add_argument("--rr_targets", default="none,1.5")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_selector(args)
