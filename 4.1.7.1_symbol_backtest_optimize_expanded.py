#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import importlib.util
import json
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

BASE_SCRIPT = Path(__file__).with_name("4.1.7_backtest_select_best_etf_notional_featureboost.py")
spec = importlib.util.spec_from_file_location("bt417", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

VERSION = "4.1.7.1"


def choose_best_row(df: pd.DataFrame, min_total_trades: int, min_positive_day_ratio: float) -> pd.Series:
    if df.empty:
        raise ValueError("No grid rows to choose from")

    eligible = df[
        (df["total_trades"] >= min_total_trades)
        & (df["positive_day_ratio"] >= min_positive_day_ratio)
    ].copy()

    pool = eligible if not eligible.empty else df.copy()
    pool = pool.sort_values(
        by=[
            "avg_daily_return_on_notional",
            "median_daily_return_on_notional",
            "positive_day_ratio",
            "total_pnl",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return pool.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Expanded single-symbol optimizer based on 4.1.7 featureboost")
    ap.add_argument("--symbol", default="TQQQ")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--out_dir", default="selector_out_4171_symbol")
    ap.add_argument("--qty", type=int, default=10)
    ap.add_argument("--initial_cash", type=float, default=100000)
    ap.add_argument("--train_days", type=int, default=10)
    ap.add_argument("--test_days", type=int, default=1)
    ap.add_argument("--eval_days", type=int, default=20)
    ap.add_argument("--step_days", type=int, default=1)
    ap.add_argument("--min_total_trades", type=int, default=20)
    ap.add_argument("--min_positive_day_ratio", type=float, default=0.25)
    args = ap.parse_args()

    symbol = args.symbol.strip().upper()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "horizon": [15, 20, 25, 30],
        "tp": [0.0050, 0.0075, 0.0100],
        "sl": [0.0030, 0.0035, 0.0050],
        "p_enter": [0.55, 0.60, 0.65, 0.70],
    }
    param_grid = list(product(grid["horizon"], grid["tp"], grid["sl"], grid["p_enter"]))

    end = datetime.now(tz=base.NY)
    start = end - timedelta(days=args.days)
    run_ts = datetime.now(tz=base.NY).strftime("%Y%m%d_%H%M%S")

    print(f"Fetching {symbol} from {start} to {end}")
    raw = base.fetch_minute_bars(symbol, start, end, feed=args.feed)
    if raw.empty:
        raise RuntimeError(f"No data fetched for {symbol}")
    raw = base.filter_market_hours(raw)
    if raw.empty:
        raise RuntimeError(f"No data for {symbol} after market-hours filter")
    df = base.add_features(raw)

    grid_rows = []
    daily_roll_rows = []

    for horizon, tp, sl, p_enter in param_grid:
        try:
            stats, daily_rows = base.walkforward_evaluate(
                df=df,
                symbol=symbol,
                feature_cols=base.FEATURE_COLS,
                model_params=base.MODEL_PARAMS,
                horizon=horizon,
                tp=tp,
                sl=sl,
                p_enter=p_enter,
                qty=args.qty,
                initial_cash=args.initial_cash,
                train_days=args.train_days,
                test_days=args.test_days,
                eval_days=args.eval_days,
                step_days=args.step_days,
            )
        except Exception as e:
            print(f"  {symbol} horizon={horizon} tp={tp} sl={sl} p_enter={p_enter} failed: {e}")
            continue

        row = {
            "symbol": symbol,
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "p_enter": p_enter,
            **stats,
        }
        grid_rows.append(row)
        daily_roll_rows.extend(daily_rows)

    if not grid_rows:
        raise RuntimeError("No valid grid results were produced")

    grid_df = pd.DataFrame(grid_rows)
    selected = choose_best_row(
        grid_df,
        min_total_trades=args.min_total_trades,
        min_positive_day_ratio=args.min_positive_day_ratio,
    ).to_dict()

    runner_pool = grid_df.sort_values(
        by=[
            "avg_daily_return_on_notional",
            "median_daily_return_on_notional",
            "positive_day_ratio",
            "total_pnl",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    runner_up = runner_pool.iloc[1].to_dict() if len(runner_pool) > 1 else None

    eligible_count = int(((grid_df["total_trades"] >= args.min_total_trades) & (grid_df["positive_day_ratio"] >= args.min_positive_day_ratio)).sum())

    selection_json = {
        "version": VERSION,
        "base_version": "4.1.7",
        "selection_mode": "single_symbol_expanded_grid_featureboost",
        "selected_symbol": symbol,
        "feature_cols": base.FEATURE_COLS,
        "model_params": base.MODEL_PARAMS,
        "grid": grid,
        "practical_filter": {
            "min_total_trades": args.min_total_trades,
            "min_positive_day_ratio": args.min_positive_day_ratio,
            "eligible_grid_count": eligible_count,
        },
        "best_strategy_params": {
            "horizon": int(selected["horizon"]),
            "tp": float(selected["tp"]),
            "sl": float(selected["sl"]),
            "p_enter": float(selected["p_enter"]),
        },
        "best_backtest_stats": {
            "pr_auc": float(selected["pr_auc"]) if pd.notna(selected["pr_auc"]) else None,
            "avg_daily_pnl": float(selected["avg_daily_pnl"]),
            "median_daily_pnl": float(selected["median_daily_pnl"]),
            "worst_day_pnl": float(selected["worst_day_pnl"]),
            "best_day_pnl": float(selected["best_day_pnl"]),
            "avg_daily_return_on_account": float(selected["avg_daily_return_on_account"]),
            "avg_daily_return_on_notional": float(selected["avg_daily_return_on_notional"]),
            "median_daily_return_on_notional": float(selected["median_daily_return_on_notional"]),
            "worst_day_return_on_notional": float(selected["worst_day_return_on_notional"]),
            "best_day_return_on_notional": float(selected["best_day_return_on_notional"]),
            "positive_day_ratio": float(selected["positive_day_ratio"]),
            "total_pnl": float(selected["total_pnl"]),
            "total_trades": int(selected["total_trades"]),
            "avg_trades_per_day": float(selected["avg_trades_per_day"]),
            "avg_daily_notional_used": float(selected["avg_daily_notional_used"]),
            "median_daily_notional_used": float(selected["median_daily_notional_used"]),
            "avg_entry_notional": float(selected["avg_entry_notional"]),
            "eval_days_used": int(selected["eval_days_used"]),
            "train_days": int(selected["train_days"]),
            "equity_final": float(selected["equity_final"]),
        },
        "selection_reason": {
            "primary_metric": "avg_daily_return_on_notional",
            "summary": (
                f"{symbol} best params were selected from an expanded grid using 4.1.7 featureboost. "
                f"Rows meeting the practical trade filter were prioritized before ranking by avg_daily_return_on_notional."
            ),
            "runner_up": runner_up,
        },
        "walkforward": {
            "train_days": args.train_days,
            "test_days": args.test_days,
            "eval_days": args.eval_days,
            "step_days": args.step_days,
            "primary_metric": "avg_daily_return_on_notional",
        },
        "market_hours": {
            "open": "10:00",
            "close": "15:30",
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

    print("\nSELECTED PARAMS")
    print(json.dumps(selection_json, indent=2))

    selection_path = out_dir / f"best_{symbol.lower()}_selection_{run_ts}.json"
    grid_results_path = out_dir / f"grid_results_{symbol.lower()}_{run_ts}.csv"
    daily_roll_path = out_dir / f"daily_roll_results_{symbol.lower()}_{run_ts}.csv"

    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection_json, f, indent=2)

    grid_df.to_csv(grid_results_path, index=False)
    pd.DataFrame(daily_roll_rows).to_csv(daily_roll_path, index=False)

    print(f"Saved selection JSON: {selection_path}")
    print(f"Saved grid results CSV: {grid_results_path}")
    print(f"Saved daily roll results CSV: {daily_roll_path}")


if __name__ == "__main__":
    main()
