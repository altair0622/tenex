#!/usr/bin/env python3
"""
run_all.py — Main entry point for the strategy research framework.

Usage:
    python run_all.py
    python run_all.py --symbols SOXL,TQQQ,SPY --days 30
    python run_all.py --symbols TQQQ --days 60 --feed sip

What it does:
    1. Runs all registered strategies × all symbols × all param combos
    2. Appends results to registry/all_results.csv
    3. Compares against current champion in registry/champion.json
    4. Prints leaderboard and updates champion if a better model is found
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backtest.runner import run_all_models
from models.ote_long_511 import OTELong511
from models.rf_tp import RFTPStrategy
from selector.selector import select_champion

# ---------------------------------------------------------------------------
# Register all strategies here. Add new models to this list.
# ---------------------------------------------------------------------------
STRATEGIES = [
    OTELong511(),
    # RF-TP series — rolling walk-forward
    RFTPStrategy("4.1.1"),
    RFTPStrategy("4.1.2"),
    # RF-TP series — day walk-forward, small-tp grid
    RFTPStrategy("4.1.3"),
    RFTPStrategy("4.1.4"),
    RFTPStrategy("4.1.5"),
    RFTPStrategy("4.1.5tf"),
    # RF-TP series — day walk-forward, high-conviction grid
    RFTPStrategy("4.1.6"),
    RFTPStrategy("4.1.6.1"),
    RFTPStrategy("4.1.7"),
    # RF-TP series — day walk-forward, expanded / coarse grids
    RFTPStrategy("4.1.7.1"),
    RFTPStrategy("4.1.7.2"),
    RFTPStrategy("4.1.7.3"),
    # RF-TP series — day walk-forward, feature-bundle
    RFTPStrategy("4.1.9"),
]

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "SOXL", "TQQQ"]
REGISTRY_PATH = Path("registry/all_results.csv")
CHAMPION_PATH = Path("registry/champion.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strategy research: backtest all models, update champion")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                   help="Comma-separated list of symbols (default: SPY,QQQ,IWM,SOXL,TQQQ)")
    p.add_argument("--days", type=int, default=45,
                   help="Lookback window in calendar days (default: 45)")
    p.add_argument("--feed", default="iex",
                   help="Alpaca data feed: 'iex' (free) or 'sip' (paid, default: iex)")
    p.add_argument("--fee_bps", type=float, default=1.0,
                   help="One-way fee in basis points (default: 1.0)")
    p.add_argument("--qty", type=int, default=10,
                   help="Shares per trade (default: 10)")
    p.add_argument("--initial_cash", type=float, default=100_000.0,
                   help="Starting equity for equity curve (default: 100000)")
    p.add_argument("--min_eval_days", type=int, default=5,
                   help="Minimum trading days a result must have to be eligible for champion (default: 5)")
    p.add_argument("--min_trades", type=int, default=15,
                   help="Minimum total trades required for champion eligibility (default: 15)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Step 1: Run all strategies
    results_df = run_all_models(
        strategies=STRATEGIES,
        symbols=symbols,
        days=args.days,
        feed=args.feed,
        fee_bps_per_side=args.fee_bps,
        qty=args.qty,
        initial_cash=args.initial_cash,
        registry_path=REGISTRY_PATH,
    )

    if results_df.empty:
        print("No results — check your Alpaca credentials and symbol list.")
        return

    # Step 2: Determine champion
    champion = select_champion(
        registry_path=REGISTRY_PATH,
        champion_path=CHAMPION_PATH,
        min_eval_days=args.min_eval_days,
        min_total_trades=args.min_trades,
    )

    print(f"Champion saved to: {CHAMPION_PATH}")
    print(f"Full registry at:  {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
