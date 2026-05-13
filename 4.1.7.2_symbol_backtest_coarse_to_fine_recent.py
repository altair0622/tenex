#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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

VERSION = "4.1.7.2"


def now_ny() -> datetime:
    return datetime.now(tz=base.NY)


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def score_row(
    row: dict,
    min_total_trades: int,
    min_positive_day_ratio: float,
    penalty_trade_shortfall: float = 0.0002,
    penalty_dayratio_shortfall: float = 0.0002,
) -> float:
    score = float(row.get("avg_daily_return_on_notional", -np.inf))
    trades = int(row.get("total_trades", 0))
    pos_ratio = float(row.get("positive_day_ratio", 0.0))

    if trades < min_total_trades:
        score -= penalty_trade_shortfall * (min_total_trades - trades)

    if pos_ratio < min_positive_day_ratio:
        score -= penalty_dayratio_shortfall * (min_positive_day_ratio - pos_ratio)

    return score


def choose_best_row(df: pd.DataFrame, min_total_trades: int, min_positive_day_ratio: float) -> pd.Series:
    if df.empty:
        raise ValueError("No candidate rows to choose from")

    ranked = df.copy()
    ranked["practical_score"] = ranked.apply(
        lambda r: score_row(
            r.to_dict(),
            min_total_trades=min_total_trades,
            min_positive_day_ratio=min_positive_day_ratio,
        ),
        axis=1,
    )
    ranked = ranked.sort_values(
        by=[
            "practical_score",
            "avg_daily_return_on_notional",
            "median_daily_return_on_notional",
            "positive_day_ratio",
            "total_pnl",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def coarse_grid() -> dict:
    return {
        "horizon": [15, 20, 30],
        "tp": [0.0050, 0.0075, 0.0100],
        "sl": [0.0035, 0.0050],
        "p_enter": [0.55, 0.60, 0.65],
    }


def fine_grid_around(top_rows: pd.DataFrame) -> dict:
    horizons = set()
    tps = set()
    sls = set()
    p_enters = set()

    for _, row in top_rows.iterrows():
        h = int(row["horizon"])
        tp = float(row["tp"])
        sl = float(row["sl"])
        pe = float(row["p_enter"])

        horizons.update([max(10, h - 5), h, h + 5])
        tps.update([max(0.0030, round(tp - 0.0010, 4)), round(tp, 4), round(tp + 0.0010, 4)])
        sls.update([max(0.0020, round(sl - 0.0005, 4)), round(sl, 4), round(sl + 0.0005, 4)])
        p_enters.update([
            max(0.45, round(pe - 0.05, 2)),
            max(0.45, round(pe - 0.02, 2)),
            round(pe, 2),
            min(0.85, round(pe + 0.02, 2)),
            min(0.85, round(pe + 0.05, 2)),
        ])

    return {
        "horizon": sorted(horizons),
        "tp": sorted(tps),
        "sl": sorted(sls),
        "p_enter": sorted(p_enters),
    }


def build_param_grid(grid: dict) -> list[tuple[int, float, float, float]]:
    return list(product(grid["horizon"], grid["tp"], grid["sl"], grid["p_enter"]))


def fetch_and_prepare(symbol: str, days: int, feed: str) -> tuple[pd.DataFrame, datetime, datetime]:
    end = now_ny()
    start = end - timedelta(days=days)

    print(f"Fetching {symbol} from {start} to {end}")
    raw = base.fetch_minute_bars(symbol, start, end, feed=feed)
    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    raw = base.filter_market_hours(raw)
    if raw.empty:
        raise RuntimeError(f"No data for {symbol} after market-hours filter")

    df = base.add_features(raw)
    return df, start, end


def evaluate_grid(
    df: pd.DataFrame,
    symbol: str,
    param_grid: list[tuple[int, float, float, float]],
    qty: int,
    initial_cash: float,
    train_days: int,
    test_days: int,
    eval_days: int,
    step_days: int,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    daily_roll_rows = []
    total = len(param_grid)

    for idx, (horizon, tp, sl, p_enter) in enumerate(param_grid, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}] symbol={symbol} horizon={horizon} tp={tp} sl={sl} p_enter={p_enter}")

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
                qty=qty,
                initial_cash=initial_cash,
                train_days=train_days,
                test_days=test_days,
                eval_days=eval_days,
                step_days=step_days,
            )
        except Exception as e:
            print(f"  failed: {e}")
            continue

        row = {
            "symbol": symbol,
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "p_enter": p_enter,
            **stats,
        }
        rows.append(row)
        daily_roll_rows.extend(daily_rows)

    return pd.DataFrame(rows), daily_roll_rows


def should_run_fine_search(
    coarse_best: pd.Series,
    min_avg_daily_return_on_notional: float,
    min_total_pnl: float,
    force_fine: bool,
) -> bool:
    if force_fine:
        return True

    coarse_metric = float(coarse_best.get("avg_daily_return_on_notional", -np.inf))
    coarse_pnl = float(coarse_best.get("total_pnl", -np.inf))
    return (coarse_metric < min_avg_daily_return_on_notional) or (coarse_pnl < min_total_pnl)


def make_selection_json(
    selected: dict,
    symbol: str,
    stage_used: str,
    coarse_best: dict,
    coarse_grid_def: dict,
    fine_grid_def: dict | None,
    fine_triggered: bool,
    practical_filters: dict,
    data_window: dict,
    qty: int,
    initial_cash: float,
    train_days: int,
    test_days: int,
    eval_days: int,
    step_days: int,
) -> dict:
    return {
        "version": VERSION,
        "selection_mode": "single_symbol_recent_recency_weighted_coarse_to_fine_featureboost",
        "selected_symbol": symbol,
        "feature_cols": base.FEATURE_COLS,
        "model_params": base.MODEL_PARAMS,
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
            "primary_metric": "practical_score_then_avg_daily_return_on_notional",
            "summary": (
                f"{symbol} was optimized with a recent-window coarse-to-fine search. "
                f"The selected params came from the {stage_used} stage and were ranked by a practical score "
                f"that starts from avg_daily_return_on_notional and penalizes too-few-trade or too-low-positive-day setups."
            ),
            "stage_used": stage_used,
            "fine_triggered": fine_triggered,
            "coarse_best_snapshot": {
                "horizon": int(coarse_best["horizon"]),
                "tp": float(coarse_best["tp"]),
                "sl": float(coarse_best["sl"]),
                "p_enter": float(coarse_best["p_enter"]),
                "avg_daily_return_on_notional": float(coarse_best["avg_daily_return_on_notional"]),
                "total_pnl": float(coarse_best["total_pnl"]),
                "total_trades": int(coarse_best["total_trades"]),
                "positive_day_ratio": float(coarse_best["positive_day_ratio"]),
            },
        },
        "search_design": {
            "coarse_grid": coarse_grid_def,
            "fine_grid": fine_grid_def,
            "stage_used": stage_used,
            "fine_triggered": fine_triggered,
            "practical_filters": practical_filters,
        },
        "walkforward": {
            "train_days": train_days,
            "test_days": test_days,
            "eval_days": eval_days,
            "step_days": step_days,
            "primary_metric": "avg_daily_return_on_notional",
        },
        "market_hours": {
            "open": "10:00",
            "close": "15:30",
        },
        "data_window": data_window,
        "qty": qty,
        "initial_cash": initial_cash,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="4.1.7.2 recent-window coarse-to-fine optimizer for a single symbol")
    ap.add_argument("--symbol", default="TQQQ")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--feed", default="iex")
    ap.add_argument("--out_dir", default="selector_out_4172")
    ap.add_argument("--qty", type=int, default=10)
    ap.add_argument("--initial_cash", type=float, default=100000)
    ap.add_argument("--train_days", type=int, default=7)
    ap.add_argument("--test_days", type=int, default=1)
    ap.add_argument("--eval_days", type=int, default=5)
    ap.add_argument("--step_days", type=int, default=1)
    ap.add_argument("--top_n_for_fine", type=int, default=3)
    ap.add_argument("--min_total_trades", type=int, default=8)
    ap.add_argument("--min_positive_day_ratio", type=float, default=0.20)
    ap.add_argument("--min_avg_daily_return_on_notional", type=float, default=0.0008)
    ap.add_argument("--min_total_pnl", type=float, default=3.0)
    ap.add_argument("--force_fine", action="store_true")
    args = ap.parse_args()

    symbol = normalize_symbol(args.symbol)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, start, end = fetch_and_prepare(symbol=symbol, days=args.days, feed=args.feed)

    coarse_grid_def = coarse_grid()
    coarse_param_grid = build_param_grid(coarse_grid_def)

    print("\nRunning coarse search...")
    coarse_results_df, coarse_daily_rows = evaluate_grid(
        df=df,
        symbol=symbol,
        param_grid=coarse_param_grid,
        qty=args.qty,
        initial_cash=args.initial_cash,
        train_days=args.train_days,
        test_days=args.test_days,
        eval_days=args.eval_days,
        step_days=args.step_days,
    )
    if coarse_results_df.empty:
        raise RuntimeError("No valid coarse-search results were produced")

    coarse_best = choose_best_row(
        coarse_results_df,
        min_total_trades=args.min_total_trades,
        min_positive_day_ratio=args.min_positive_day_ratio,
    )

    fine_triggered = should_run_fine_search(
        coarse_best=coarse_best,
        min_avg_daily_return_on_notional=args.min_avg_daily_return_on_notional,
        min_total_pnl=args.min_total_pnl,
        force_fine=args.force_fine,
    )

    final_results_df = coarse_results_df.copy()
    final_daily_rows = list(coarse_daily_rows)
    final_best = coarse_best
    fine_grid_def = None
    stage_used = "coarse"

    if fine_triggered:
        print("\nCoarse best did not clear the thresholds. Running fine search...")
        coarse_ranked = coarse_results_df.copy()
        coarse_ranked["practical_score"] = coarse_ranked.apply(
            lambda r: score_row(
                r.to_dict(),
                min_total_trades=args.min_total_trades,
                min_positive_day_ratio=args.min_positive_day_ratio,
            ),
            axis=1,
        )
        coarse_ranked = coarse_ranked.sort_values(
            by=["practical_score", "avg_daily_return_on_notional", "total_pnl"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        top_rows = coarse_ranked.head(args.top_n_for_fine)
        fine_grid_def = fine_grid_around(top_rows)
        fine_param_grid = build_param_grid(fine_grid_def)

        fine_results_df, fine_daily_rows = evaluate_grid(
            df=df,
            symbol=symbol,
            param_grid=fine_param_grid,
            qty=args.qty,
            initial_cash=args.initial_cash,
            train_days=args.train_days,
            test_days=args.test_days,
            eval_days=args.eval_days,
            step_days=args.step_days,
        )
        if not fine_results_df.empty:
            fine_best = choose_best_row(
                fine_results_df,
                min_total_trades=args.min_total_trades,
                min_positive_day_ratio=args.min_positive_day_ratio,
            )
            stage_used = "fine"
            final_best = fine_best
            final_results_df = pd.concat(
                [
                    coarse_results_df.assign(search_stage="coarse"),
                    fine_results_df.assign(search_stage="fine"),
                ],
                ignore_index=True,
            )
            final_daily_rows.extend(fine_daily_rows)
        else:
            final_results_df = coarse_results_df.assign(search_stage="coarse")
    else:
        print("\nCoarse best cleared the thresholds. Fine search skipped.")
        final_results_df = coarse_results_df.assign(search_stage="coarse")

    run_ts = now_ny().strftime("%Y%m%d_%H%M%S")
    practical_filters = {
        "min_total_trades": args.min_total_trades,
        "min_positive_day_ratio": args.min_positive_day_ratio,
        "min_avg_daily_return_on_notional_to_skip_fine": args.min_avg_daily_return_on_notional,
        "min_total_pnl_to_skip_fine": args.min_total_pnl,
    }
    data_window = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "calendar_days": args.days,
        "feed": args.feed,
    }

    selection_json = make_selection_json(
        selected=final_best.to_dict(),
        symbol=symbol,
        stage_used=stage_used,
        coarse_best=coarse_best.to_dict(),
        coarse_grid_def=coarse_grid_def,
        fine_grid_def=fine_grid_def,
        fine_triggered=fine_triggered,
        practical_filters=practical_filters,
        data_window=data_window,
        qty=args.qty,
        initial_cash=args.initial_cash,
        train_days=args.train_days,
        test_days=args.test_days,
        eval_days=args.eval_days,
        step_days=args.step_days,
    )

    selection_path = out_dir / f"best_{symbol.lower()}_selection_{run_ts}.json"
    grid_results_path = out_dir / f"grid_results_{symbol.lower()}_{run_ts}.csv"
    daily_roll_path = out_dir / f"daily_roll_results_{symbol.lower()}_{run_ts}.csv"
    summary_path = out_dir / f"search_summary_{symbol.lower()}_{run_ts}.json"

    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection_json, f, indent=2)

    final_results_df.to_csv(grid_results_path, index=False)
    pd.DataFrame(final_daily_rows).to_csv(daily_roll_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": VERSION,
                "symbol": symbol,
                "coarse_grid": coarse_grid_def,
                "fine_grid": fine_grid_def,
                "fine_triggered": fine_triggered,
                "stage_used": stage_used,
                "coarse_best": {
                    "horizon": int(coarse_best["horizon"]),
                    "tp": float(coarse_best["tp"]),
                    "sl": float(coarse_best["sl"]),
                    "p_enter": float(coarse_best["p_enter"]),
                    "avg_daily_return_on_notional": float(coarse_best["avg_daily_return_on_notional"]),
                    "total_pnl": float(coarse_best["total_pnl"]),
                    "total_trades": int(coarse_best["total_trades"]),
                    "positive_day_ratio": float(coarse_best["positive_day_ratio"]),
                },
                "final_best": {
                    "horizon": int(final_best["horizon"]),
                    "tp": float(final_best["tp"]),
                    "sl": float(final_best["sl"]),
                    "p_enter": float(final_best["p_enter"]),
                    "avg_daily_return_on_notional": float(final_best["avg_daily_return_on_notional"]),
                    "total_pnl": float(final_best["total_pnl"]),
                    "total_trades": int(final_best["total_trades"]),
                    "positive_day_ratio": float(final_best["positive_day_ratio"]),
                    "search_stage": stage_used,
                },
                "practical_filters": practical_filters,
            },
            f,
            indent=2,
        )

    print("\nSELECTED PARAMS")
    print(json.dumps(selection_json["best_strategy_params"], indent=2))
    print("\nSELECTED STATS")
    print(json.dumps(selection_json["best_backtest_stats"], indent=2))
    print(f"\nSaved selection JSON: {selection_path}")
    print(f"Saved grid results CSV: {grid_results_path}")
    print(f"Saved daily roll results CSV: {daily_roll_path}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
