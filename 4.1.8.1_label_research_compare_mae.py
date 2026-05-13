from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_SCRIPT = Path(__file__).with_name("4.1.7_backtest_select_best_etf_notional_featureboost.py")

spec = importlib.util.spec_from_file_location("bt417", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

VERSION = "4.1.8.1"


@dataclass
class LabelSpec:
    name: str
    mode: str
    mfe_target: float | None = None
    mae_limit: float | None = None
    rr_min: float | None = None
    quality_threshold: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "mfe_target": self.mfe_target,
            "mae_limit": self.mae_limit,
            "rr_min": self.rr_min,
            "quality_threshold": self.quality_threshold,
        }


def _forward_arrays(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry = df["Open"].shift(-1).to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)

    fwd_max_high = np.full(len(df), np.nan, dtype=float)
    fwd_min_low = np.full(len(df), np.nan, dtype=float)
    first_tp_idx = np.full(len(df), np.nan, dtype=float)
    first_sl_idx = np.full(len(df), np.nan, dtype=float)

    return entry, highs, lows, fwd_max_high, fwd_min_low, first_tp_idx, first_sl_idx


def make_label(df: pd.DataFrame, horizon: int, tp: float, sl: float, spec: LabelSpec) -> pd.Series:
    entry, highs, lows, fwd_max_high, fwd_min_low, first_tp_idx, first_sl_idx = _forward_arrays(df, horizon)
    target = entry * (1 + tp)
    stop = entry * (1 - sl)

    for i in range(len(df)):
        j0 = i + 1
        j1 = i + 1 + horizon
        if j1 > len(df):
            continue
        h_slice = highs[j0:j1]
        l_slice = lows[j0:j1]
        fwd_max_high[i] = np.max(h_slice)
        fwd_min_low[i] = np.min(l_slice)

        tp_hits = np.where(h_slice >= target[i])[0]
        sl_hits = np.where(l_slice <= stop[i])[0]
        if len(tp_hits):
            first_tp_idx[i] = int(tp_hits[0]) + 1
        if len(sl_hits):
            first_sl_idx[i] = int(sl_hits[0]) + 1

    mfe = (fwd_max_high / entry) - 1.0
    mae = 1.0 - (fwd_min_low / entry)
    hit_tp = fwd_max_high >= target
    hit_sl = fwd_min_low <= stop

    if spec.mode == "hit_tp_within_h":
        y = hit_tp.astype(float)

    elif spec.mode == "tp_before_sl":
        tp_first = np.where(np.isnan(first_tp_idx), np.inf, first_tp_idx)
        sl_first = np.where(np.isnan(first_sl_idx), np.inf, first_sl_idx)
        y = (tp_first < sl_first).astype(float)

    elif spec.mode == "mfe_at_least":
        thresh = spec.mfe_target if spec.mfe_target is not None else tp
        y = (mfe >= thresh).astype(float)

    elif spec.mode == "tp_and_low_mae":
        mae_limit = spec.mae_limit if spec.mae_limit is not None else sl * 0.5
        y = (hit_tp & (mae <= mae_limit)).astype(float)

    elif spec.mode == "quality_ratio":
        rr_min = spec.rr_min if spec.rr_min is not None else 1.5
        quality_threshold = spec.quality_threshold if spec.quality_threshold is not None else tp * 0.25
        ratio = np.where(mae > 0, mfe / mae, np.nan)
        quality_score = mfe - mae
        y = ((ratio >= rr_min) & (quality_score >= quality_threshold)).astype(float)

    else:
        raise ValueError(f"Unknown label mode: {spec.mode}")

    y[np.isnan(entry)] = np.nan
    y[np.isnan(fwd_max_high)] = np.nan
    return pd.Series(y, index=df.index, name="label")


def walkforward_evaluate_label(
    df: pd.DataFrame,
    symbol: str,
    feature_cols: list[str],
    model_params: dict,
    horizon: int,
    tp: float,
    sl: float,
    p_enter: float,
    qty: int,
    initial_cash: float,
    train_days: int,
    test_days: int,
    eval_days: int,
    step_days: int,
    label_spec: LabelSpec,
) -> tuple[dict, pd.DataFrame]:
    trading_days = base.get_trading_days(df.index)
    if len(trading_days) < train_days + test_days + 1:
        return {}, pd.DataFrame()

    daily_rows = []
    equity = initial_cash

    latest_test_start = len(trading_days) - eval_days
    valid_pr_aucs: list[float] = []

    for test_start_idx in range(max(train_days, latest_test_start), len(trading_days), step_days):
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= len(trading_days):
            break

        train_start_idx = test_start_idx - train_days
        train_end_idx = test_start_idx - 1
        train_days_set = set(trading_days[train_start_idx:train_end_idx + 1])
        test_days_set = set(trading_days[test_start_idx:test_end_idx + 1])

        index_dates = pd.Series(df.index.date, index=df.index)
        train_mask = index_dates.isin(train_days_set).to_numpy()
        test_mask = index_dates.isin(test_days_set).to_numpy()
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        train_df = train_df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
        test_df = test_df.dropna(subset=feature_cols + ["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(train_df) < 200 or len(test_df) < 50:
            continue

        y_train = make_label(train_df, horizon=horizon, tp=tp, sl=sl, spec=label_spec)
        train_valid = train_df.dropna(subset=feature_cols).copy()
        y_train = y_train.reindex(train_valid.index).dropna()
        X_train = train_valid.loc[y_train.index, feature_cols]
        if len(X_train) < 100 or len(np.unique(y_train.astype(int))) < 2:
            continue

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train.astype(int))

        day_result = base.simulate_day(
            test_df=test_df,
            model=model,
            feature_cols=feature_cols,
            horizon=horizon,
            tp=tp,
            sl=sl,
            p_enter=p_enter,
            qty=qty,
        )
        equity += day_result["daily_pnl"]
        daily_rows.append({
            "symbol": symbol,
            "label_name": label_spec.name,
            "label_mode": label_spec.mode,
            "test_date": min(test_days_set).isoformat(),
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "p_enter": p_enter,
            "daily_pnl": day_result["daily_pnl"],
            "daily_return_on_account": day_result["daily_pnl"] / initial_cash,
            "daily_return_on_notional": day_result["daily_return_on_notional"],
            "daily_notional_used": day_result["daily_notional_used"],
            "avg_entry_notional": day_result["avg_entry_notional"],
            "total_trades": day_result["total_trades"],
            "win_rate": day_result["win_rate"],
            "pr_auc": day_result["pr_auc"],
            "mfe_sum": day_result["mfe_sum"],
            "mae_sum": day_result["mae_sum"],
            "equity": equity,
        })
        if pd.notna(day_result.get("pr_auc")):
            valid_pr_aucs.append(float(day_result["pr_auc"]))

    if not daily_rows:
        return {}, pd.DataFrame()

    daily_df = pd.DataFrame(daily_rows)
    stats = {
        "symbol": symbol,
        "label_name": label_spec.name,
        "label_mode": label_spec.mode,
        "mae_limit": label_spec.mae_limit,
        "horizon": horizon,
        "tp": tp,
        "sl": sl,
        "p_enter": p_enter,
        "pr_auc": float(np.mean(valid_pr_aucs)) if valid_pr_aucs else np.nan,
        "avg_daily_pnl": float(daily_df["daily_pnl"].mean()),
        "median_daily_pnl": float(daily_df["daily_pnl"].median()),
        "worst_day_pnl": float(daily_df["daily_pnl"].min()),
        "best_day_pnl": float(daily_df["daily_pnl"].max()),
        "avg_daily_return_on_account": float(daily_df["daily_return_on_account"].mean()),
        "avg_daily_return_on_notional": float(daily_df["daily_return_on_notional"].mean()),
        "median_daily_return_on_notional": float(daily_df["daily_return_on_notional"].median()),
        "worst_day_return_on_notional": float(daily_df["daily_return_on_notional"].min()),
        "best_day_return_on_notional": float(daily_df["daily_return_on_notional"].max()),
        "positive_day_ratio": float((daily_df["daily_pnl"] > 0).mean()),
        "total_pnl": float(daily_df["daily_pnl"].sum()),
        "total_trades": int(daily_df["total_trades"].sum()),
        "avg_trades_per_day": float(daily_df["total_trades"].mean()),
        "avg_daily_notional_used": float(daily_df["daily_notional_used"].mean()),
        "median_daily_notional_used": float(daily_df["daily_notional_used"].median()),
        "avg_entry_notional": float(daily_df["avg_entry_notional"].replace(0, np.nan).mean()) if (daily_df["avg_entry_notional"] > 0).any() else 0.0,
        "eval_days_used": int(len(daily_df)),
        "train_days": int(train_days),
        "equity_final": float(equity),
    }
    return stats, daily_df


DEFAULT_LABEL_SPECS = [
    LabelSpec(name="baseline_hit_tp", mode="hit_tp_within_h"),
    LabelSpec(name="tp_before_sl", mode="tp_before_sl"),
    LabelSpec(name="tp_and_low_mae_015", mode="tp_and_low_mae", mae_limit=0.0015),
    LabelSpec(name="tp_and_low_mae_020", mode="tp_and_low_mae", mae_limit=0.0020),
    LabelSpec(name="tp_and_low_mae_025", mode="tp_and_low_mae", mae_limit=0.0025),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare research label definitions, including MAE-threshold variants, under the same walk-forward backtest")
    parser.add_argument("--symbol", type=str, default="TQQQ")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--feed", type=str, default="iex")
    parser.add_argument("--out_dir", type=str, default="label_research_out_4181")
    parser.add_argument("--train_days", type=int, default=10)
    parser.add_argument("--test_days", type=int, default=1)
    parser.add_argument("--eval_days", type=int, default=15)
    parser.add_argument("--step_days", type=int, default=1)
    parser.add_argument("--qty", type=int, default=10)
    parser.add_argument("--initial_cash", type=float, default=100000.0)
    args = parser.parse_args()

    model_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "random_state": 42,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    }

    grid = {
        "horizon": [15, 20, 30],
        "tp": [0.0050, 0.0075, 0.0100],
        "sl": [0.0035, 0.0050],
        "p_enter": [0.55, 0.60, 0.65, 0.70],
    }
    param_grid = list(product(grid["horizon"], grid["tp"], grid["sl"], grid["p_enter"]))

    end = pd.Timestamp.now(tz="America/New_York")
    start = end - pd.Timedelta(days=args.days)

    print(f"Fetching {args.symbol} from {start} to {end}")
    df = base.fetch_minute_bars(args.symbol, start.to_pydatetime(), end.to_pydatetime(), feed=args.feed)
    df = base.filter_market_hours(df)
    df = base.add_features(df)
    df = df.dropna().copy()

    results = []
    daily_all = []

    for label_spec in DEFAULT_LABEL_SPECS:
        print(f"\nLABEL: {label_spec.name} ({label_spec.mode})")
        best_stats = None
        best_daily = None
        best_metric = -np.inf

        for horizon, tp, sl, p_enter in param_grid:
            stats, daily_df = walkforward_evaluate_label(
                df=df,
                symbol=args.symbol,
                feature_cols=base.FEATURE_COLS,
                model_params=model_params,
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
                label_spec=label_spec,
            )
            if not stats:
                continue
            metric = stats["avg_daily_return_on_notional"]
            if pd.notna(metric) and metric > best_metric:
                best_metric = metric
                best_stats = stats
                best_daily = daily_df.copy()

        if best_stats is None:
            print("  no valid result")
            continue

        print(json.dumps(best_stats, indent=2))
        results.append(best_stats)
        daily_all.append(best_daily)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(results).sort_values(
        ["avg_daily_return_on_notional", "total_pnl", "positive_day_ratio"],
        ascending=[False, False, False],
    )
    daily_df = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()

    summary_path = out_dir / f"label_compare_summary_{stamp}.csv"
    daily_path = out_dir / f"label_compare_daily_{stamp}.csv"
    json_path = out_dir / f"label_compare_best_{stamp}.json"

    results_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": VERSION,
            "symbol": args.symbol,
            "grid": grid,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "eval_days": args.eval_days,
            "label_specs": [x.to_dict() for x in DEFAULT_LABEL_SPECS],
            "best_by_label": results,
        }, f, ensure_ascii=False, indent=2)

    print("\nBEST LABELS")
    if not results_df.empty:
        print(results_df[[
            "label_name", "label_mode", "horizon", "tp", "sl", "p_enter",
            "avg_daily_return_on_notional", "total_pnl", "total_trades",
            "positive_day_ratio", "pr_auc"
        ]].to_string(index=False))
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved daily CSV: {daily_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
