#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time as pytime
import warnings
from datetime import datetime, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

BASE_SCRIPT = Path(__file__).with_name("4.1.7_backtest_select_best_etf_notional_featureboost.py")

spec = importlib.util.spec_from_file_location("bt417", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load base script: {BASE_SCRIPT}")

base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

VERSION = "4.1.7.3-live"
DEFAULT_SELECTION_DIR = Path("selector_out_4173")

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)

MODEL_PARAMS_4173 = dict(base.MODEL_PARAMS)
MODEL_PARAMS_4173["n_jobs"] = 1

MARKET_OPEN = time(10, 0)
MARKET_CLOSE = time(15, 30)


def now_ny() -> datetime:
    return datetime.now(tz=base.NY)


def load_selection(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_selection_json(selection_dir: Path, symbol: str = "") -> Path:
    if not selection_dir.exists():
        raise FileNotFoundError(f"Selection directory does not exist: {selection_dir}")

    pattern = "best_*_selection_*.json"
    candidates = sorted(selection_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if symbol:
        symbol = symbol.strip().lower()
        symbol_candidates = [p for p in candidates if f"best_{symbol}_selection_" in p.name.lower()]
        if symbol_candidates:
            return symbol_candidates[0]
        raise FileNotFoundError(f"No selection JSON found for symbol={symbol} in {selection_dir}")

    if not candidates:
        raise FileNotFoundError(f"No selection JSON files found in {selection_dir}")

    return candidates[0]


def get_alpaca_clients(paper: bool) -> tuple[TradingClient, StockHistoricalDataClient]:
    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")
    trading = TradingClient(api_key=key, secret_key=secret, paper=paper)
    data_client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    return trading, data_client


def fetch_recent_bars(
    data_client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str,
) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=feed,
    )
    bars = data_client.get_stock_bars(req).df
    if bars is None or len(bars) == 0:
        return pd.DataFrame()

    bars = bars.reset_index()
    bars = bars.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "timestamp": "timestamp",
        }
    )
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(base.NY)
    bars = bars.sort_values("timestamp").set_index("timestamp")

    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol].copy()
        bars.drop(columns=["symbol"], inplace=True)

    bars = bars[(bars.index.time >= MARKET_OPEN) & (bars.index.time <= MARKET_CLOSE)].copy()
    return bars


def get_trading_days(index: pd.DatetimeIndex) -> list:
    return sorted(pd.Index(index.date).unique())


def train_model_from_recent_window(
    df: pd.DataFrame,
    horizon: int,
    tp: float,
    train_days: int,
):
    trading_days = get_trading_days(df.index)
    if len(trading_days) < train_days + 1:
        raise RuntimeError(f"Not enough trading days to train. Need at least {train_days + 1}, got {len(trading_days)}")

    train_days_set = set(trading_days[-train_days:])
    idx_dates = pd.Series(df.index.date, index=df.index)
    train_df = df.loc[idx_dates.isin(train_days_set).to_numpy()].copy()
    train_df = train_df.dropna(subset=base.FEATURE_COLS + ["Open", "High", "Low", "Close", "Volume"]).copy()

    if len(train_df) < 150:
        raise RuntimeError(f"Training dataframe too small after cleanup: {len(train_df)} rows")

    y_train = base.make_label_hit_tp_local(train_df, horizon=horizon, tp=tp)
    train_valid = train_df.dropna(subset=base.FEATURE_COLS).copy()
    y_train = y_train.reindex(train_valid.index).dropna()
    X_train = train_valid.loc[y_train.index, base.FEATURE_COLS]

    if len(X_train) < 100 or len(np.unique(y_train.astype(int))) < 2:
        raise RuntimeError("Training set does not have enough usable rows or lacks both classes")

    model = base.RandomForestClassifier(**MODEL_PARAMS_4173)
    model.fit(X_train, y_train.astype(int))
    return model


def latest_signal_probability(model, df: pd.DataFrame) -> tuple[datetime, float]:
    valid = df.dropna(subset=base.FEATURE_COLS).copy()
    if valid.empty:
        raise RuntimeError("No valid feature rows available for inference")
    latest_ts = valid.index[-1]
    latest_x = valid.loc[[latest_ts], base.FEATURE_COLS]
    prob = float(model.predict_proba(latest_x)[:, 1][0])
    return latest_ts, prob


def get_latest_trade_price(data_client: StockHistoricalDataClient, symbol: str) -> float:
    req = StockLatestTradeRequest(symbol_or_symbols=symbol)
    latest = data_client.get_stock_latest_trade(req)
    trade_obj = latest.get(symbol)
    if trade_obj is None:
        raise RuntimeError(f"Could not fetch latest trade for {symbol}")
    return float(trade_obj.price)


def get_open_position_qty(trading: TradingClient, symbol: str) -> int:
    try:
        pos = trading.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0


def submit_market_buy(trading: TradingClient, symbol: str, qty: int):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    return trading.submit_order(order_data=order)


def submit_market_sell(trading: TradingClient, symbol: str, qty: int):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return trading.submit_order(order_data=order)


def in_market_window(ts: datetime) -> bool:
    t = ts.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def load_recent_data_for_model(
    data_client: StockHistoricalDataClient,
    symbol: str,
    calendar_days: int,
    feed: str,
) -> pd.DataFrame:
    end = now_ny()
    start = end - timedelta(days=calendar_days)
    raw = fetch_recent_bars(data_client, symbol=symbol, start=start, end=end, feed=feed)
    if raw.empty:
        raise RuntimeError("No recent bars fetched")
    return base.add_features(raw)


def sleep_until_next_minute(second_offset: int = 2):
    now = now_ny()
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    target = next_minute + timedelta(seconds=second_offset)
    sleep_seconds = max(1.0, (target - now).total_seconds())
    pytime.sleep(sleep_seconds)


def main():
    ap = argparse.ArgumentParser(description="4.1.7.3 live runner from latest coarse-to-fine selection JSON")
    ap.add_argument("--selection_json", default="")
    ap.add_argument("--selection_dir", default=str(DEFAULT_SELECTION_DIR))
    ap.add_argument("--symbol", default="")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--retrain_every_minutes", type=int, default=15)
    ap.add_argument("--calendar_days_override", type=int, default=0)
    ap.add_argument("--feed_override", default="")
    ap.add_argument("--minute_offset_seconds", type=int, default=2)
    args = ap.parse_args()

    if args.selection_json:
        selection_path = Path(args.selection_json)
    else:
        selection_path = find_latest_selection_json(
            selection_dir=Path(args.selection_dir),
            symbol=args.symbol,
        )

    selection = load_selection(selection_path)
    symbol = str(selection["selected_symbol"]).upper()
    params = selection["best_strategy_params"]
    walk = selection.get("walkforward", {})
    data_window = selection.get("data_window", {})

    horizon = int(params["horizon"])
    tp = float(params["tp"])
    sl = float(params["sl"])
    p_enter = float(params["p_enter"])
    qty = int(selection.get("qty", 10))
    train_days = int(walk.get("train_days", 7))
    feed = args.feed_override or data_window.get("feed", "iex")
    calendar_days = args.calendar_days_override or int(data_window.get("calendar_days", 30))

    trading, data_client = get_alpaca_clients(paper=args.paper)

    state = {
        "in_position": False,
        "entry_ts": None,
        "entry_px": None,
        "entry_qty": 0,
        "last_retrain_ts": None,
        "model": None,
        "last_processed_bar_minute": None,
    }

    print(f"Starting {VERSION}")
    print(f"selection_json={selection_path}")
    print(
        f"symbol={symbol} horizon={horizon} tp={tp} sl={sl} p_enter={p_enter} "
        f"qty={qty} train_days={train_days} feed={feed} rf_n_jobs={MODEL_PARAMS_4173['n_jobs']}"
    )
    print(f"paper={args.paper}")

    while True:
        try:
            now = now_ny()
            if not in_market_window(now):
                print(f"[{now.isoformat()}] Outside market window {MARKET_OPEN} - {MARKET_CLOSE}")
                sleep_until_next_minute(args.minute_offset_seconds)
                continue

            need_retrain = (
                state["model"] is None
                or state["last_retrain_ts"] is None
                or (now - state["last_retrain_ts"]).total_seconds() >= args.retrain_every_minutes * 60
            )

            if need_retrain:
                recent_df = load_recent_data_for_model(
                    data_client=data_client,
                    symbol=symbol,
                    calendar_days=calendar_days,
                    feed=feed,
                )
                model = train_model_from_recent_window(
                    df=recent_df,
                    horizon=horizon,
                    tp=tp,
                    train_days=train_days,
                )
                state["model"] = model
                state["last_retrain_ts"] = now
                print(f"[{now.isoformat()}] Model retrained")

            recent_df = load_recent_data_for_model(
                data_client=data_client,
                symbol=symbol,
                calendar_days=calendar_days,
                feed=feed,
            )
            latest_ts, prob = latest_signal_probability(state["model"], recent_df)

            bar_minute_key = latest_ts.strftime("%Y-%m-%d %H:%M")
            if state["last_processed_bar_minute"] == bar_minute_key:
                sleep_until_next_minute(args.minute_offset_seconds)
                continue

            state["last_processed_bar_minute"] = bar_minute_key
            last_price = get_latest_trade_price(data_client, symbol)

            broker_qty = get_open_position_qty(trading, symbol)
            state["in_position"] = broker_qty > 0
            if not state["in_position"]:
                state["entry_ts"] = None
                state["entry_px"] = None
                state["entry_qty"] = 0

            print(
                f"[{now.isoformat()}] latest_bar={latest_ts.isoformat()} "
                f"prob={prob:.4f} price={last_price:.4f} in_position={state['in_position']}"
            )

            if not state["in_position"] and prob >= p_enter:
                submit_market_buy(trading, symbol, qty)
                state["in_position"] = True
                state["entry_ts"] = now
                state["entry_px"] = last_price
                state["entry_qty"] = qty
                print(f"[{now.isoformat()}] BUY submitted | prob={prob:.4f} entry_px_ref={last_price:.4f} qty={qty}")

            elif state["in_position"] and state["entry_px"] is not None and state["entry_ts"] is not None:
                tp_px = state["entry_px"] * (1 + tp)
                sl_px = state["entry_px"] * (1 - sl)
                held_minutes = int((now - state["entry_ts"]).total_seconds() // 60)

                should_exit = False
                exit_reason = None

                if last_price <= sl_px:
                    should_exit = True
                    exit_reason = "SL"
                elif last_price >= tp_px:
                    should_exit = True
                    exit_reason = "TP"
                elif held_minutes >= horizon:
                    should_exit = True
                    exit_reason = "TIME"

                if should_exit:
                    exit_qty = get_open_position_qty(trading, symbol)
                    if exit_qty > 0:
                        submit_market_sell(trading, symbol, exit_qty)
                        print(
                            f"[{now.isoformat()}] SELL submitted | reason={exit_reason} "
                            f"held_minutes={held_minutes} entry_px_ref={state['entry_px']:.4f} "
                            f"last_price={last_price:.4f} qty={exit_qty}"
                        )
                    state["in_position"] = False
                    state["entry_ts"] = None
                    state["entry_px"] = None
                    state["entry_qty"] = 0

            sleep_until_next_minute(args.minute_offset_seconds)

        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print(f"[{now_ny().isoformat()}] loop error: {e}")
            sleep_until_next_minute(args.minute_offset_seconds)


if __name__ == "__main__":
    main()
