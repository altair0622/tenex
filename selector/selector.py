from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
REGISTRY_PATH = Path("registry/all_results.csv")
CHAMPION_PATH = Path("registry/champion.json")

# Minimum thresholds — results below these are not considered for champion.
MIN_EVAL_DAYS = 5        # must have been evaluated on at least 5 trading days
MIN_TOTAL_TRADES = 15    # must have at least 15 trades total


def load_registry(registry_path: Path = REGISTRY_PATH) -> pd.DataFrame:
    if not registry_path.exists() or registry_path.stat().st_size == 0:
        raise FileNotFoundError(f"Registry not found: {registry_path}. Run backtest first.")
    return pd.read_csv(registry_path)


def load_champion(champion_path: Path = CHAMPION_PATH) -> Optional[dict]:
    if not champion_path.exists():
        return None
    with champion_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_champion(champion: dict, champion_path: Path) -> None:
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    with champion_path.open("w", encoding="utf-8") as f:
        json.dump(champion, f, indent=2, default=str)


def select_champion(
    registry_path: Path = REGISTRY_PATH,
    champion_path: Path = CHAMPION_PATH,
    min_eval_days: int = MIN_EVAL_DAYS,
    min_total_trades: int = MIN_TOTAL_TRADES,
) -> dict:
    """
    Read all backtest results from the registry, find the best (model, symbol, params)
    combination, compare against the current champion, and update champion.json if
    the new candidate is better.

    Ranking order (all descending):
      1. avg_daily_roi_pct  (primary — fee-adjusted return on capital deployed)
      2. positive_day_ratio (secondary)
      3. win_rate           (tertiary)
    """
    df = load_registry(registry_path)

    # Apply quality filters
    df = df[df["eval_days"] >= min_eval_days]
    df = df[df["total_trades"] >= min_total_trades]

    if df.empty:
        raise RuntimeError(
            f"No results meet minimum criteria "
            f"(eval_days >= {min_eval_days}, total_trades >= {min_total_trades}). "
            f"Run more backtests or relax thresholds."
        )

    # Among runs for the same (model, symbol, params), keep the most recent
    if "run_ts" in df.columns:
        df = df.sort_values("run_ts", ascending=False)
        key_cols = [c for c in ["model_id", "version", "symbol",
                                 "pivot_left", "pivot_right", "min_swing_range_pct",
                                 "stop_buffer_pct", "max_wait_minutes_after_touch",
                                 "max_hold_minutes", "rr_target"]
                    if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="first")

    # Rank by primary → secondary → tertiary
    df = df.sort_values(
        ["avg_daily_roi_pct", "positive_day_ratio", "win_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = df.iloc[0].to_dict()
    best["crowned_at"] = datetime.now(tz=NY).isoformat()

    current_champion = load_champion(champion_path)

    if current_champion is None:
        _save_champion(best, champion_path)
        _print_champion_update(None, best, "NEW")
    elif best["avg_daily_roi_pct"] > current_champion.get("avg_daily_roi_pct", float("-inf")):
        best["dethroned"] = {
            "model_id": current_champion.get("model_id"),
            "version": current_champion.get("version"),
            "symbol": current_champion.get("symbol"),
            "avg_daily_roi_pct": current_champion.get("avg_daily_roi_pct"),
            "crowned_at": current_champion.get("crowned_at"),
        }
        _save_champion(best, champion_path)
        _print_champion_update(current_champion, best, "DETHRONED")
    else:
        _print_champion_update(current_champion, best, "DEFENDED")
        best = current_champion  # return the actual champion, not the challenger

    _print_leaderboard(df.head(10))
    return best


def _print_champion_update(current: Optional[dict], challenger: dict, outcome: str) -> None:
    print(f"\n{'='*60}")
    if outcome == "NEW":
        print("CHAMPION SET (first run)")
    elif outcome == "DETHRONED":
        print("CHAMPION UPDATED!")
        print(f"  Previous: {current['model_id']} v{current['version']} | "
              f"{current['symbol']} | {current['avg_daily_roi_pct']:.4f}% daily ROI")
    else:
        print("CHAMPION DEFENDED")
        print(f"  Current:    {current['model_id']} v{current['version']} | "
              f"{current['symbol']} | {current['avg_daily_roi_pct']:.4f}% daily ROI")
        print(f"  Challenger: {challenger['model_id']} v{challenger['version']} | "
              f"{challenger['symbol']} | {challenger['avg_daily_roi_pct']:.4f}% daily ROI")
        return

    c = challenger
    print(f"  Model:            {c['model_id']} v{c['version']}")
    print(f"  Symbol:           {c['symbol']}")
    print(f"  avg_daily_roi:    {c['avg_daily_roi_pct']:.4f}%")
    print(f"  positive_days:    {c.get('positive_day_ratio', 0):.1%}")
    print(f"  win_rate:         {c.get('win_rate', 0):.1%}")
    print(f"  total_trades:     {c.get('total_trades', 0)}")
    print(f"  max_drawdown:     {c.get('max_drawdown_pct', 0):.2f}%")
    print(f"  eval_days:        {c.get('eval_days', 0)}")
    print(f"{'='*60}\n")


def _print_leaderboard(df: pd.DataFrame) -> None:
    print("\nTop 10 Leaderboard (all-time registry):")
    cols = [c for c in ["model_id", "version", "symbol", "avg_daily_roi_pct",
                         "positive_day_ratio", "win_rate", "total_trades",
                         "max_drawdown_pct", "eval_days"]
            if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
