from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_metrics(
    symbol: str,
    model_id: str,
    version: str,
    strategy_type: str,
    params: dict,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    initial_cash: float = 100_000.0,
) -> dict:
    """
    Compute standardized performance metrics for one (model, symbol, params) run.

    Primary metric: avg_daily_roi_pct
        = average over trading days of (daily_pnl / daily_notional * 100)
        Intuitive reading: "on average this model returned X% of the capital
        deployed on any given trading day."
    """
    base = {
        "model_id": model_id,
        "version": version,
        "strategy_type": strategy_type,
        "symbol": symbol,
        **params,
    }

    empty = {
        **base,
        "avg_daily_roi_pct": 0.0,
        "median_daily_roi_pct": 0.0,
        "best_day_roi_pct": 0.0,
        "worst_day_roi_pct": 0.0,
        "positive_day_ratio": 0.0,
        "avg_daily_pnl": 0.0,
        "total_pnl": 0.0,
        "total_trades": 0,
        "avg_trades_per_day": 0.0,
        "win_rate": 0.0,
        "max_drawdown_pct": 0.0,
        "equity_final": initial_cash,
        "eval_days": 0,
    }

    if trades_df is None or trades_df.empty or daily_df is None or daily_df.empty:
        return empty

    daily = daily_df.copy()
    daily["daily_roi_pct"] = np.where(
        daily["daily_notional"] > 0,
        daily["daily_pnl"] / daily["daily_notional"] * 100.0,
        0.0,
    )

    n_days = int(daily["trade_date"].nunique()) if "trade_date" in daily.columns else len(daily)
    equity_df = _equity_curve(trades_df, initial_cash)

    return {
        **base,
        # --- primary metric ---
        "avg_daily_roi_pct": float(daily["daily_roi_pct"].mean()),
        # --- secondary metrics ---
        "median_daily_roi_pct": float(daily["daily_roi_pct"].median()),
        "best_day_roi_pct": float(daily["daily_roi_pct"].max()),
        "worst_day_roi_pct": float(daily["daily_roi_pct"].min()),
        "positive_day_ratio": float((daily["daily_pnl"] > 0).mean()),
        # --- trade-level metrics ---
        "avg_daily_pnl": float(daily["daily_pnl"].mean()),
        "total_pnl": float(trades_df["pnl_dollars"].sum()),
        "total_trades": int(len(trades_df)),
        "avg_trades_per_day": float(len(trades_df) / n_days) if n_days > 0 else 0.0,
        "win_rate": float((trades_df["net_return"] > 0).mean()),
        # --- risk metrics ---
        "max_drawdown_pct": float(equity_df["drawdown"].min() * 100.0) if not equity_df.empty else 0.0,
        "equity_final": float(initial_cash + trades_df["pnl_dollars"].sum()),
        "eval_days": n_days,
    }


def _equity_curve(trades_df: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    out = trades_df.sort_values("entry_time").copy()
    out["cum_pnl"] = out["pnl_dollars"].cumsum()
    out["equity"] = initial_cash + out["cum_pnl"]
    out["cummax"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["cummax"] - 1.0
    return out
