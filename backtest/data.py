from __future__ import annotations

import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

NY = ZoneInfo("America/New_York")
MARKET_OPEN = time(10, 0)
MARKET_CLOSE = time(15, 30)


def build_data_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY_1") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY_1") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials not found. "
            "Set ALPACA_API_KEY_1 and ALPACA_SECRET_KEY_1 environment variables."
        )
    return StockHistoricalDataClient(key, secret)


def fetch_minute_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Fetch 1-minute bars from Alpaca and return a clean DataFrame
    filtered to market hours (10:00–15:30 ET).

    Columns: timestamp, open, high, low, close, volume
    """
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=feed,
    )
    raw = client.get_stock_bars(req).df
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()
        raw = raw[raw["symbol"] == symbol].copy()
    else:
        raw = raw.reset_index()

    ts_col = "timestamp" if "timestamp" in raw.columns else raw.columns[0]
    raw = raw.rename(columns={ts_col: "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(NY)

    t = raw["timestamp"].dt.time
    raw = raw[(t >= MARKET_OPEN) & (t <= MARKET_CLOSE)].copy()

    out = raw[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out
