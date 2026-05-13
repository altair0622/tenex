from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class BaseStrategy(ABC):
    """
    Every trading model must subclass this and implement the two methods below.
    The backtest runner calls param_grid() to discover what to sweep, then calls
    backtest() once per (symbol, params) combination.
    """

    model_id: str = ""      # e.g. "ote_long"
    version: str = ""       # e.g. "5.1.1.1"
    strategy_type: str = "" # e.g. "OTE_LONG_RULE_BASED"

    @abstractmethod
    def param_grid(self) -> List[dict]:
        """
        Return a list of parameter dicts to sweep.
        Each dict will be passed as-is to backtest().
        """
        ...

    @abstractmethod
    def backtest(
        self,
        df_1m: pd.DataFrame,
        symbol: str,
        params: dict,
        fee_bps_per_side: float,
        qty: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run a single backtest for the given symbol and params.

        Parameters
        ----------
        df_1m : 1-minute bars, market-hours only.
                Required columns: timestamp, open, high, low, close, volume
        symbol : ticker string
        params : one element from param_grid()
        fee_bps_per_side : one-way fee in basis points (e.g. 1.0 = 0.01%)
        qty : number of shares per trade

        Returns
        -------
        trades_df : one row per trade.
            Required columns: entry_time, exit_time, entry_price, exit_price,
                              net_return, pnl_dollars, notional, exit_reason,
                              trade_date (YYYY-MM-DD string)
        daily_df : one row per trading day that had at least one trade.
            Required columns: trade_date, daily_pnl, daily_notional
        """
        ...
