#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import importlib.util
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
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

VERSION = "4.1.7.1-live"


def load_selection(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_trading_client(paper: bool) -> TradingClient:
    key = os.getenv("ALPACA_API_KEY_1")
    secret = os.getenv("ALPACA_SECRET_KEY_1")
    if not key or not secret:
        raise RuntimeError("Missing env vars ALPACA_API_KEY_1 / ALPACA_SECRET_KEY_1")
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def latest_prob_for_symbol(symbol: str, feature_cols: list[str], model_params: dict, horizon: int, tp: float, train_days: int, feed: str) -> tuple[float, pd.DataFrame]:
    end = datetime.now(tz=base.NY)
    start = end - timedelta(days=max(30, train_days * 4))
    raw = base.fetch_minute_bars(symbol, start, end, feed=feed)
    raw = base.filter_market_hours(raw)
    df = base.add_features(raw).sort_index()
    if df.empty:
        raise RuntimeError(f"No intraday data available for {symbol}")

    trading_days = base.get_trading_days(df.index)
    if len(trading_days) < train_days + 1:
        raise RuntimeError(f"Not enough trading days for training {symbol}")

    train_days_set = set(trading_days[-(train_days + 1):-1])
    today_set = {trading_days[-1]}

    index_dates = pd.Series(df.index.date, index=df.index)
    train_df = df.loc[index_dates.isin(train_days_set).to_numpy()].copy()
    test_df = df.loc[index_dates.isin(today_set).to_numpy()].copy()

    train_df = train_df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
    test_df = test_df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(train_df) < 100 or len(test_df) < 20:
        raise RuntimeError(f"Insufficient rows after feature prep for {symbol}")

    y_train = base.make_label_hit_tp_local(train_df, horizon=horizon, tp=tp)
    train_valid = train_df.dropna(subset=feature_cols).copy()
    y_train = y_train.reindex(train_valid.index).dropna()
    X_train = train_valid.loc[y_train.index, feature_cols]
    if len(X_train) < 100 or len(np.unique(y_train.astype(int))) < 2:
        raise RuntimeError(f"Training label quality too weak for {symbol}")

    model = base.RandomForestClassifier(**model_params)
    model.fit(X_train, y_train.astype(int))

    test_valid = test_df.dropna(subset=feature_cols).copy()
    X_test = test_valid[feature_cols]
    probs = model.predict_proba(X_test)[:, 1]
    latest_prob = float(probs[-1])
    return latest_prob, test_valid


def get_position_qty(client: TradingClient, symbol: str) -> int:
    try:
        pos = client.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0


def submit_market_order(client: TradingClient, symbol: str, qty: int, side: OrderSide):
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
    return client.submit_order(order_data=req)


def within_session(now_dt: datetime, open_str: str, close_str: str) -> bool:
    oh, om = map(int, open_str.split(":"))
    ch, cm = map(int, close_str.split(":"))
    hhmm = now_dt.hour * 100 + now_dt.minute
    return (oh * 100 + om) <= hhmm <= (ch * 100 + cm)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic live runner from selection JSON")
    ap.add_argument("--selection_json", required=True)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--poll_seconds", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    selection = load_selection(Path(args.selection_json))
    symbol = selection["selected_symbol"].strip().upper()
    feature_cols = selection["feature_cols"]
    model_params = selection["model_params"]
    params = selection["best_strategy_params"]
    walk = selection["walkforward"]
    market_hours = selection.get("market_hours", {"open": "10:00", "close": "15:30"})
    qty = int(selection.get("qty", 10))

    horizon = int(params["horizon"])
    tp = float(params["tp"])
    sl = float(params["sl"])
    p_enter = float(params["p_enter"])
    train_days = int(walk["train_days"])

    client = make_trading_client(args.paper)

    entry_state = {
        "has_position": False,
        "entry_price": None,
        "entry_time": None,
    }

    print(f"Starting {VERSION} for {symbol} | paper={args.paper}")
    print(f"Params: horizon={horizon} tp={tp} sl={sl} p_enter={p_enter} qty={qty}")

    while True:
        now_dt = datetime.now(tz=base.NY)
        if not within_session(now_dt, market_hours["open"], market_hours["close"]):
            print(f"[{now_dt.isoformat()}] Outside session window for {symbol}")
            if args.once:
                break
            time.sleep(args.poll_seconds)
            continue

        try:
            latest_prob, intraday_df = latest_prob_for_symbol(
                symbol=symbol,
                feature_cols=feature_cols,
                model_params=model_params,
                horizon=horizon,
                tp=tp,
                train_days=train_days,
                feed=args.feed,
            )
            last_close = float(intraday_df.iloc[-1]["Close"])
        except Exception as e:
            print(f"[{now_dt.isoformat()}] Model refresh failed for {symbol}: {e}")
            if args.once:
                break
            time.sleep(args.poll_seconds)
            continue

        open_qty = get_position_qty(client, symbol)
        if open_qty > 0:
            entry_state["has_position"] = True

        print(f"[{now_dt.isoformat()}] {symbol} prob={latest_prob:.4f} last_close={last_close:.4f} open_qty={open_qty}")

        if not entry_state["has_position"] and open_qty == 0 and latest_prob >= p_enter:
            order = submit_market_order(client, symbol, qty, OrderSide.BUY)
            entry_state["has_position"] = True
            entry_state["entry_price"] = last_close
            entry_state["entry_time"] = now_dt
            print(f"BUY submitted for {symbol} qty={qty} at approx {last_close:.4f} | order={order.id}")

        elif entry_state["has_position"] and open_qty > 0:
            entry_price = entry_state["entry_price"] or last_close
            entry_time = entry_state["entry_time"] or now_dt
            tp_px = entry_price * (1 + tp)
            sl_px = entry_price * (1 - sl)
            held_minutes = int((now_dt - entry_time).total_seconds() // 60)

            should_exit = False
            exit_reason = None
            if last_close >= tp_px:
                should_exit = True
                exit_reason = "TP"
            elif last_close <= sl_px:
                should_exit = True
                exit_reason = "SL"
            elif held_minutes >= horizon:
                should_exit = True
                exit_reason = "TIME"

            if should_exit:
                order = submit_market_order(client, symbol, open_qty, OrderSide.SELL)
                print(f"SELL submitted for {symbol} qty={open_qty} reason={exit_reason} at approx {last_close:.4f} | order={order.id}")
                entry_state = {
                    "has_position": False,
                    "entry_price": None,
                    "entry_time": None,
                }

        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
