from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.base import BaseStrategy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core strategy logic
# ---------------------------------------------------------------------------

def _resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    x = df_1m.set_index("timestamp")
    df_5m = pd.DataFrame({
        "open":   x["open"].resample("5min").first(),
        "high":   x["high"].resample("5min").max(),
        "low":    x["low"].resample("5min").min(),
        "close":  x["close"].resample("5min").last(),
        "volume": x["volume"].resample("5min").sum(),
    }).dropna().reset_index()
    return df_5m


def _detect_pivots_5m(df_5m: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    df = df_5m.copy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    highs = df["high"].values
    lows = df["low"].values
    for i in range(left, n - right):
        hi, lo = highs[i], lows[i]
        if np.all(hi > highs[i - left:i]) and np.all(hi > highs[i + 1:i + 1 + right]):
            swing_high[i] = True
        if np.all(lo < lows[i - left:i]) and np.all(lo < lows[i + 1:i + 1 + right]):
            swing_low[i] = True
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    df["pivot_confirm_time"] = df["timestamp"] + pd.Timedelta(minutes=5 * right)
    return df


def _extract_pivot_events(df_5m: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df_5m[df_5m["swing_high"]].iterrows():
        rows.append({"pivot_time": row["timestamp"], "confirm_time": row["pivot_confirm_time"],
                     "price": float(row["high"]), "type": "H"})
    for _, row in df_5m[df_5m["swing_low"]].iterrows():
        rows.append({"pivot_time": row["timestamp"], "confirm_time": row["pivot_confirm_time"],
                     "price": float(row["low"]), "type": "L"})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["confirm_time", "pivot_time", "type"]).reset_index(drop=True)


def _compute_fib_levels(anchor_low: float, anchor_high: float) -> Dict[str, float]:
    diff = anchor_high - anchor_low
    return {
        "fib_50": anchor_low + 0.50 * diff,
        "fib_62": anchor_low + 0.62 * diff,
        "fib_79": anchor_low + 0.79 * diff,
        "ote_low": anchor_low + 0.62 * diff,
        "ote_high": anchor_low + 0.79 * diff,
    }


def _generate_setups(
    df_5m: pd.DataFrame,
    min_swing_range_pct: float,
    setup_valid_minutes: int,
) -> List[Setup]:
    events = _extract_pivot_events(df_5m)
    if events.empty:
        return []

    confirmed_highs: List[Dict] = []
    confirmed_lows: List[Dict] = []
    setups: List[Setup] = []
    seen_keys: set = set()
    setup_id = 1

    for _, event in events.iterrows():
        item = {"pivot_time": event["pivot_time"], "confirm_time": event["confirm_time"],
                "price": float(event["price"]), "type": event["type"]}
        if item["type"] == "H":
            confirmed_highs.append(item)
        else:
            confirmed_lows.append(item)

        if len(confirmed_highs) < 2 or len(confirmed_lows) < 2:
            continue

        h1, h2 = confirmed_highs[-2], confirmed_highs[-1]
        l1, l2 = confirmed_lows[-2], confirmed_lows[-1]
        if not (h2["price"] > h1["price"] and l2["price"] > l1["price"]):
            continue

        candidate_lows = [x for x in confirmed_lows if x["pivot_time"] < h2["pivot_time"]]
        if not candidate_lows:
            continue

        al = candidate_lows[-1]
        ah = h2
        anchor_low, anchor_high = float(al["price"]), float(ah["price"])
        if anchor_high <= anchor_low:
            continue

        swing_range_pct = (anchor_high - anchor_low) / anchor_low
        if swing_range_pct < min_swing_range_pct:
            continue

        key = (al["pivot_time"], ah["pivot_time"])
        if key in seen_keys:
            continue
        seen_keys.add(key)

        fib = _compute_fib_levels(anchor_low, anchor_high)
        valid_from = ah["confirm_time"]
        valid_until = valid_from + pd.Timedelta(minutes=setup_valid_minutes)

        setups.append(Setup(
            setup_id=setup_id,
            anchor_low_time=al["pivot_time"],
            anchor_low_confirm_time=al["confirm_time"],
            anchor_low=anchor_low,
            anchor_high_time=ah["pivot_time"],
            anchor_high_confirm_time=ah["confirm_time"],
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
        setup_id += 1

    return setups


def _find_entry(
    df_1m: pd.DataFrame,
    setup: Setup,
    max_wait_minutes_after_touch: int,
) -> Optional[Tuple[pd.Timestamp, float]]:
    window = df_1m[
        (df_1m["timestamp"] >= setup.valid_from) &
        (df_1m["timestamp"] <= setup.valid_until)
    ].copy()
    if window.empty:
        return None

    window["in_ote_zone"] = (window["low"] <= setup.ote_high) & (window["high"] >= setup.ote_low)
    window["below_anchor_low"] = window["low"] < setup.anchor_low
    window["prev3_high"] = window["high"].shift(1).rolling(3).max()
    window["break_prev3_high"] = window["close"] > window["prev3_high"]

    ote_rows = window[window["in_ote_zone"]]
    if ote_rows.empty:
        return None

    first_touch = ote_rows.iloc[0]["timestamp"]
    search_until = min(first_touch + pd.Timedelta(minutes=max_wait_minutes_after_touch), setup.valid_until)

    candidate = window[
        (window["timestamp"] >= first_touch) &
        (window["timestamp"] <= search_until) &
        (~window["below_anchor_low"]) &
        (window["break_prev3_high"].fillna(False))
    ]
    if candidate.empty:
        return None

    row = candidate.iloc[0]
    return row["timestamp"], float(row["close"])


def _simulate_trade(
    df_1m: pd.DataFrame,
    symbol: str,
    setup: Setup,
    entry_time: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float,
    qty: int,
    max_hold_minutes: int,
    fee_bps_per_side: float,
) -> Trade:
    expiry = entry_time + pd.Timedelta(minutes=max_hold_minutes)
    future = df_1m[(df_1m["timestamp"] > entry_time) & (df_1m["timestamp"] <= expiry)].copy()

    exit_time, exit_price, exit_reason = entry_time, entry_price, "time_exit"

    for _, row in future.iterrows():
        bar_low, bar_high = float(row["low"]), float(row["high"])
        if bar_low <= stop_price and bar_high >= target_price:
            exit_time, exit_price, exit_reason = row["timestamp"], stop_price, "stop_and_target_same_bar"
            break
        if bar_low <= stop_price:
            exit_time, exit_price, exit_reason = row["timestamp"], stop_price, "stop"
            break
        if bar_high >= target_price:
            exit_time, exit_price, exit_reason = row["timestamp"], target_price, "target"
            break
    else:
        if not future.empty:
            last = future.iloc[-1]
            exit_time, exit_price = last["timestamp"], float(last["close"])

    gross_return = (exit_price / entry_price) - 1.0
    net_return = gross_return - 2.0 * (fee_bps_per_side / 10_000.0)
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


def _run_backtest(
    df_1m: pd.DataFrame,
    symbol: str,
    pivot_left: int,
    pivot_right: int,
    min_swing_range_pct: float,
    setup_valid_minutes: int,
    stop_buffer_pct: float,
    max_wait_minutes_after_touch: int,
    max_hold_minutes: int,
    fee_bps_per_side: float,
    rr_target: Optional[float],
    qty: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_1m.copy().sort_values("timestamp").reset_index(drop=True)
    df_5m = _resample_to_5m(df)
    df_5m = _detect_pivots_5m(df_5m, left=pivot_left, right=pivot_right)
    setups = _generate_setups(df_5m, min_swing_range_pct, setup_valid_minutes)

    trades: List[Trade] = []
    next_free_time = (
        pd.Timestamp.min.tz_localize(df["timestamp"].dt.tz)
        if df["timestamp"].dt.tz is not None
        else pd.Timestamp.min
    )

    for setup in setups:
        if setup.valid_from < next_free_time:
            continue

        entry = _find_entry(df, setup, max_wait_minutes_after_touch)
        if entry is None:
            continue

        entry_time, entry_price = entry
        if entry_time < next_free_time:
            continue

        stop_price = setup.anchor_low * (1.0 - stop_buffer_pct)
        if rr_target is None:
            target_price = setup.anchor_high
        else:
            risk = entry_price - stop_price
            target_price = setup.anchor_high if risk <= 0 else entry_price + rr_target * risk

        if stop_price >= entry_price or target_price <= entry_price:
            continue

        trade = _simulate_trade(
            df_1m=df,
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
        next_free_time = trade.exit_time

    trades_df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()

    if trades_df.empty:
        daily_df = pd.DataFrame(columns=["trade_date", "daily_pnl", "daily_notional"])
    else:
        daily_df = (
            trades_df.groupby("trade_date", as_index=False)
            .agg(daily_pnl=("pnl_dollars", "sum"), daily_notional=("notional", "sum"))
        )

    return trades_df, daily_df


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class OTELong511(BaseStrategy):
    """
    OTE (Optimal Trade Entry) Long strategy — version 5.1.1.1
    Identifies uptrending structure on 5-min pivots, enters on a retracement
    into the 62–79% Fibonacci OTE zone with a momentum confirmation.
    """

    model_id = "ote_long"
    version = "5.1.1.1"
    strategy_type = "OTE_LONG_RULE_BASED"

    def __init__(
        self,
        pivot_left_grid: List[int] = None,
        pivot_right_grid: List[int] = None,
        min_swing_range_grid: List[float] = None,
        stop_buffer_grid: List[float] = None,
        max_wait_grid: List[int] = None,
        max_hold_grid: List[int] = None,
        rr_target_grid: List[Optional[float]] = None,
        setup_valid_minutes: int = 60,
    ):
        self._pivot_left_grid = pivot_left_grid or [2]
        self._pivot_right_grid = pivot_right_grid or [2]
        self._min_swing_range_grid = min_swing_range_grid or [0.004, 0.005]
        self._stop_buffer_grid = stop_buffer_grid or [0.0005, 0.0010]
        self._max_wait_grid = max_wait_grid or [10]
        self._max_hold_grid = max_hold_grid or [20, 30]
        self._rr_target_grid = rr_target_grid if rr_target_grid is not None else [None, 1.5]
        self._setup_valid_minutes = setup_valid_minutes

    def param_grid(self) -> List[dict]:
        combos = product(
            self._pivot_left_grid,
            self._pivot_right_grid,
            self._min_swing_range_grid,
            self._stop_buffer_grid,
            self._max_wait_grid,
            self._max_hold_grid,
            self._rr_target_grid,
        )
        return [
            {
                "pivot_left": pl,
                "pivot_right": pr,
                "min_swing_range_pct": ms,
                "stop_buffer_pct": sb,
                "max_wait_minutes_after_touch": mw,
                "max_hold_minutes": mh,
                "rr_target": rt,
                "setup_valid_minutes": self._setup_valid_minutes,
            }
            for pl, pr, ms, sb, mw, mh, rt in combos
        ]

    def backtest(
        self,
        df_1m: pd.DataFrame,
        symbol: str,
        params: dict,
        fee_bps_per_side: float,
        qty: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return _run_backtest(
            df_1m=df_1m,
            symbol=symbol,
            pivot_left=params["pivot_left"],
            pivot_right=params["pivot_right"],
            min_swing_range_pct=params["min_swing_range_pct"],
            setup_valid_minutes=params["setup_valid_minutes"],
            stop_buffer_pct=params["stop_buffer_pct"],
            max_wait_minutes_after_touch=params["max_wait_minutes_after_touch"],
            max_hold_minutes=params["max_hold_minutes"],
            fee_bps_per_side=fee_bps_per_side,
            rr_target=params["rr_target"],
            qty=qty,
        )
