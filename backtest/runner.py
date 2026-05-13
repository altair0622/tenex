from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd

from backtest.data import build_data_client, fetch_minute_bars
from backtest.metrics import compute_metrics
from models.base import BaseStrategy

NY = ZoneInfo("America/New_York")
REGISTRY_PATH = Path("registry/all_results.csv")
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_all_models(
    strategies: List[BaseStrategy],
    symbols: List[str],
    days: int = 45,
    feed: str = "iex",
    fee_bps_per_side: float = 1.0,
    qty: int = 10,
    initial_cash: float = 100_000.0,
    registry_path: Path = REGISTRY_PATH,
) -> pd.DataFrame:
    """
    Run every (strategy × symbol × params) combination, compute standardized
    metrics, append results to the registry CSV, and return a DataFrame of
    this run's results.
    """
    client = build_data_client()
    end = datetime.now(tz=NY)
    start = end - timedelta(days=days)
    run_ts = end.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"\n{'='*60}")
    print(f"Backtest run: {run_ts}")
    print(f"Symbols: {symbols} | Days: {days} | Feed: {feed}")
    print(f"Fee: {fee_bps_per_side} bps/side | Qty: {qty} | Cash: ${initial_cash:,.0f}")
    print(f"{'='*60}\n")

    all_results: List[dict] = []

    for strategy in strategies:
        grid = strategy.param_grid()
        print(f"[{strategy.model_id} v{strategy.version}] {len(grid)} param combos × {len(symbols)} symbols "
              f"= {len(grid) * len(symbols)} runs")

        for symbol in symbols:
            print(f"  Fetching {symbol}...", end=" ", flush=True)
            df_1m = fetch_minute_bars(client, symbol, start, end, feed)
            if df_1m.empty:
                print("no data, skipping.")
                continue
            print(f"{len(df_1m)} bars")

            for params in grid:
                try:
                    trades_df, daily_df = strategy.backtest(df_1m, symbol, params, fee_bps_per_side, qty)
                    metrics = compute_metrics(
                        symbol=symbol,
                        model_id=strategy.model_id,
                        version=strategy.version,
                        strategy_type=strategy.strategy_type,
                        params=params,
                        trades_df=trades_df,
                        daily_df=daily_df,
                        initial_cash=initial_cash,
                    )
                    metrics["run_ts"] = run_ts
                    metrics["days_lookback"] = days
                    metrics["fee_bps_per_side"] = fee_bps_per_side
                    metrics["qty"] = qty
                    all_results.append(metrics)
                except Exception as exc:
                    print(f"    ERROR {symbol} {params}: {exc}")

    if not all_results:
        print("No results produced.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    _append_to_registry(results_df, registry_path)

    # Save this run separately so past runs are never lost
    run_path = registry_path.parent / f"run_{run_ts.replace(':', '').replace('-', '').replace('T', '_')}.csv"
    results_df.to_csv(run_path, index=False)

    print(f"\nRun complete. {len(results_df)} results saved to {registry_path}")
    print(f"This run snapshot:    {run_path}")
    return results_df


def _append_to_registry(df: pd.DataFrame, path: Path) -> None:
    """
    Merge new results into the registry CSV.
    Reads existing file first so column sets can differ across runs
    (e.g. OTE Long vs RF-TP have different param columns).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path)
            combined = pd.concat([existing, df], ignore_index=True)
        except Exception:
            # If the existing file is corrupt, overwrite with fresh data
            combined = df
    else:
        combined = df
    combined.to_csv(path, index=False)
