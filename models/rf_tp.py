"""
models/rf_tp.py — RF-TP (Random Forest, Take-Profit) strategy series.

Covers versions 4.1.1 through 4.1.9 (excluding 4.1.8/4.1.8.1 research files).
Two walk-forward modes:
  - rolling: train on N bars, test on next M, slide (v4.1.1, v4.1.2)
  - day:     train on N trading days, test on 1 day, repeat (v4.1.3+)
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from models.base import BaseStrategy

# ---------------------------------------------------------------------------
# Feature column sets
# ---------------------------------------------------------------------------

FEAT_V411 = [
    "RSI", "Buying Probability", "Selling Probability",
    "vwap", "EMA", "IC", "MACD", "Signal_Line", "%K", "%D",
    "OBV", "Price_Oscillator",
    "rolling_mean", "Bollinger_Upper", "Bollinger_Lower",
    "Open_n", "High_n", "Low_n", "Close_n",
    "hour", "minute",
]

FEAT_STANDARD = [
    "RSI", "Buying Probability", "Selling Probability",
    "vwap", "EMA", "IC", "MACD", "Signal_Line", "%K", "%D",
    "OBV", "Price_Oscillator",
    "Open_n", "High_n", "Low_n", "Close_n",
    "hour", "minute",
]

FEAT_BUNDLE = FEAT_STANDARD + [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "hl_range_pct", "oc_body_pct", "range_pos_15",
]

# ---------------------------------------------------------------------------
# RF model parameter sets
# ---------------------------------------------------------------------------

RF_V411 = {
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "random_state": 42,
    "class_weight": "balanced",
    "n_jobs": -1,
}

RF_V45 = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "random_state": 42,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}

RF_V473 = {**RF_V45, "n_jobs": 1}

# ---------------------------------------------------------------------------
# Param grids
# ---------------------------------------------------------------------------

GRID_A = {
    "horizon": [5, 10, 15],
    "tp":      [0.004, 0.005, 0.006],
    "sl":      [None, 0.0025, 0.0035],
    "p_enter": [0.55, 0.60, 0.65, 0.70],
}

GRID_B = {
    "horizon": [10, 15],
    "tp":      [0.003, 0.005],
    "sl":      [0.002, 0.0035],
    "p_enter": [0.55, 0.60, 0.65],
}

GRID_C = {
    "horizon": [20, 30, 45],
    "tp":      [0.0075, 0.0100],
    "sl":      [0.0035, 0.0050],
    "p_enter": [0.65, 0.70, 0.75, 0.80],
}

GRID_D = {
    "horizon": [15, 20, 25, 30],
    "tp":      [0.0050, 0.0075, 0.0100],
    "sl":      [0.0030, 0.0035, 0.0050],
    "p_enter": [0.55, 0.60, 0.65, 0.70],
}

GRID_E = {
    "horizon": [15, 20, 30],
    "tp":      [0.0050, 0.0075, 0.0100],
    "sl":      [0.0035, 0.0050],
    "p_enter": [0.55, 0.60, 0.65],
}

GRID_F = {
    "horizon": [20],
    "tp":      [0.0075, 0.0100],
    "sl":      [0.0050],
    "p_enter": [0.65, 0.70],
}

# ---------------------------------------------------------------------------
# Version configurations
# ---------------------------------------------------------------------------

_DAY_BASE = {"wf_mode": "day", "train_days": 10, "eval_days": 5}

VERSION_CONFIGS: Dict[str, dict] = {
    "4.1.1": {
        "wf_mode": "rolling", "train_size": 2000, "test_size": 500,
        "features": FEAT_V411, "rf_params": RF_V411, "grid": GRID_A,
    },
    "4.1.2": {
        "wf_mode": "rolling", "train_size": 2000, "test_size": 500,
        "features": FEAT_V411, "rf_params": RF_V411, "grid": GRID_A,
    },
    "4.1.3":   {**_DAY_BASE, "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_B},
    "4.1.4":   {**_DAY_BASE, "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_B},
    "4.1.5":   {**_DAY_BASE, "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_B},
    "4.1.5tf": {**_DAY_BASE, "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_B},
    "4.1.6":   {**_DAY_BASE, "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_C},
    "4.1.6.1": {**_DAY_BASE, "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_C},
    "4.1.7":   {**_DAY_BASE, "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_C},
    "4.1.7.1": {
        "wf_mode": "day", "train_days": 10, "eval_days": 20,
        "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_D,
    },
    "4.1.7.2": {
        "wf_mode": "day", "train_days": 7, "eval_days": 5,
        "features": FEAT_STANDARD, "rf_params": RF_V45, "grid": GRID_E,
    },
    "4.1.7.3": {
        "wf_mode": "day", "train_days": 7, "eval_days": 5,
        "features": FEAT_STANDARD, "rf_params": RF_V473, "grid": GRID_E,
    },
    "4.1.9": {
        "wf_mode": "day", "train_days": 10, "eval_days": 15,
        "features": FEAT_BUNDLE, "rf_params": RF_V45, "grid": GRID_F,
    },
}

# ---------------------------------------------------------------------------
# Indicator / feature helpers
# ---------------------------------------------------------------------------

def _prepare_df(df_1m: pd.DataFrame) -> pd.DataFrame:
    d = df_1m.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    return d.set_index("timestamp")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _buying_prob(rsi: float) -> float:
    if np.isnan(rsi): return np.nan
    if rsi < 30: return 1 - rsi / 30
    if rsi > 70: return 0.0
    return (40 - (rsi - 30)) / 40


def _selling_prob(rsi: float) -> float:
    if np.isnan(rsi): return np.nan
    if rsi > 70: return (rsi - 70) / 30
    if rsi < 30: return 0.0
    return (rsi - 30) / 40


def _vwap(df: pd.DataFrame) -> pd.Series:
    day = df.index.date
    pv = (df["Close"] * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def _obv(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].values
    vol = df["Volume"].values
    out = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - vol[i]
        else:
            out[i] = out[i - 1]
    return pd.Series(out, index=df.index)


def _add_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    d = df.copy()

    d["RSI"] = _rsi(d["Close"])
    d["Buying Probability"] = d["RSI"].apply(_buying_prob)
    d["Selling Probability"] = d["RSI"].apply(_selling_prob)

    d["vwap"] = _vwap(d)
    d["EMA"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["IC"] = d["Close"] / d["vwap"]

    exp12 = d["Close"].ewm(span=12, adjust=False).mean()
    exp26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = exp12 - exp26
    d["Signal_Line"] = d["MACD"].ewm(span=9, adjust=False).mean()

    low_min = d["Low"].rolling(14).min()
    high_max = d["High"].rolling(14).max()
    d["%K"] = (d["Close"] - low_min) * 100 / (high_max - low_min).replace(0, np.nan)
    d["%D"] = d["%K"].rolling(3).mean()

    d["OBV"] = _obv(d)
    d["Price_Oscillator"] = exp12 - exp26  # same as MACD numerically, kept for naming

    rm = d["Close"].rolling(20).mean()
    rs = d["Close"].rolling(20).std()
    d["rolling_mean"] = rm
    d["Bollinger_Upper"] = rm + 2 * rs
    d["Bollinger_Lower"] = rm - 2 * rs
    band_range = (d["Bollinger_Upper"] - d["Bollinger_Lower"]).replace(0, np.nan)
    for col in ("Open", "High", "Low", "Close"):
        d[col + "_n"] = (d[col] - d["Bollinger_Lower"]) / band_range

    d["hour"] = d.index.hour
    d["minute"] = d.index.minute

    bundle_cols = {"ret_1", "ret_3", "ret_5", "ret_10", "hl_range_pct", "oc_body_pct", "range_pos_15"}
    if bundle_cols & set(features):
        d["ret_1"] = d["Close"].pct_change(1)
        d["ret_3"] = d["Close"].pct_change(3)
        d["ret_5"] = d["Close"].pct_change(5)
        d["ret_10"] = d["Close"].pct_change(10)
        d["hl_range_pct"] = (d["High"] - d["Low"]) / d["Close"].replace(0, np.nan)
        d["oc_body_pct"] = (d["Close"] - d["Open"]).abs() / d["Open"].replace(0, np.nan)
        roll_lo = d["Low"].rolling(15).min()
        roll_hi = d["High"].rolling(15).max()
        d["range_pos_15"] = (d["Close"] - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)

    return d


def _make_label(df: pd.DataFrame, horizon: int, tp: float) -> pd.Series:
    entry = df["Open"].shift(-1)
    highs = df["High"].to_numpy()
    fwd = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        j0, j1 = i + 1, i + 1 + horizon
        if j1 <= len(df):
            fwd[i] = highs[j0:j1].max()
    target = entry.to_numpy() * (1 + tp)
    return pd.Series((fwd >= target).astype(float), index=df.index)


def _get_trading_days(index: pd.DatetimeIndex) -> list:
    return sorted(pd.Index(index.date).unique())


# ---------------------------------------------------------------------------
# Rolling walk-forward (4.1.1 / 4.1.2)
# ---------------------------------------------------------------------------

def _walkforward_rolling(
    df_feat: pd.DataFrame,
    y: pd.Series,
    features: list,
    train_size: int,
    test_size: int,
    rf_params: dict,
) -> pd.Series:
    prob = pd.Series(np.nan, index=df_feat.index)
    n = len(df_feat)
    start = 0

    while start + train_size + test_size <= n:
        tr = slice(start, start + train_size)
        te = slice(start + train_size, start + train_size + test_size)

        X_tr = df_feat.iloc[tr][features]
        y_tr = y.iloc[tr]
        X_te = df_feat.iloc[te][features]

        mask = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr2, y_tr2 = X_tr[mask], y_tr[mask]

        if len(X_tr2) >= 50 and len(np.unique(y_tr2.astype(int))) == 2:
            clf = RandomForestClassifier(**rf_params)
            clf.fit(X_tr2, y_tr2.astype(int))
            prob.iloc[te] = clf.predict_proba(X_te)[:, 1]

        start += test_size

    return prob


def _simulate_rolling_trades(
    df: pd.DataFrame,
    prob: pd.Series,
    horizon: int,
    tp: float,
    p_enter: float,
    sl: Optional[float],
    fee_bps_per_side: float,
    qty: int,
) -> List[dict]:
    trades: List[dict] = []
    n = len(df)
    i = 0

    while i < n:
        p = prob.iat[i] if not pd.isna(prob.iat[i]) else -1.0
        if p < p_enter:
            i += 1
            continue

        entry_i = i + 1
        if entry_i >= n:
            break

        entry_px = float(df["Open"].iat[entry_i])
        tp_px = entry_px * (1 + tp)
        sl_px = entry_px * (1 - sl) if sl is not None else None
        last_i = min(entry_i + horizon - 1, n - 1)

        exit_reason = "TIME"
        exit_i = last_i

        for j in range(entry_i, last_i + 1):
            hi, lo = float(df["High"].iat[j]), float(df["Low"].iat[j])
            if sl_px is not None and lo <= sl_px:
                exit_reason, exit_i = "SL", j
                break
            if hi >= tp_px:
                exit_reason, exit_i = "TP", j
                break

        exit_px = (tp_px if exit_reason == "TP"
                   else sl_px if exit_reason == "SL"
                   else float(df["Close"].iat[exit_i]))

        notional = entry_px * qty
        gross_pnl = (exit_px - entry_px) * qty
        fee = 2 * fee_bps_per_side / 10_000 * notional
        pnl_dollars = gross_pnl - fee
        net_return = pnl_dollars / notional

        entry_ts = df.index[entry_i]
        exit_ts = df.index[exit_i]

        trades.append({
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "exit_reason": exit_reason,
            "net_return": net_return,
            "pnl_dollars": pnl_dollars,
            "notional": notional,
            "trade_date": entry_ts.date().isoformat(),
        })
        i = exit_i + 1

    return trades


# ---------------------------------------------------------------------------
# Day-based walk-forward (4.1.3+)
# ---------------------------------------------------------------------------

def _simulate_day_trades(
    test_df: pd.DataFrame,
    model: RandomForestClassifier,
    features: list,
    horizon: int,
    tp: float,
    sl: Optional[float],
    p_enter: float,
    qty: int,
    fee_bps_per_side: float,
) -> List[dict]:
    df = test_df.copy().dropna(subset=features + ["Open", "High", "Low", "Close", "Volume"])
    if len(df) < max(50, horizon + 5):
        return []

    valid = df.dropna(subset=features)
    if valid.empty:
        return []

    probs = pd.Series(np.nan, index=df.index)
    probs.loc[valid.index] = model.predict_proba(valid[features])[:, 1]

    idxs = list(df.index)
    n = len(idxs)
    trades: List[dict] = []

    in_pos = False
    entry_px: Optional[float] = None
    entry_ts = None
    entry_i: Optional[int] = None

    for i, ts in enumerate(idxs[:-1]):
        p = probs.get(ts, np.nan)
        if np.isnan(p):
            continue

        if not in_pos and p >= p_enter:
            ei = i + 1
            if ei >= n:
                continue
            entry_ts = idxs[ei]
            entry_px = float(df.iloc[ei]["Open"])
            entry_i = ei
            in_pos = True
            continue

        if in_pos and entry_i is not None and entry_px is not None:
            if i < entry_i:
                continue
            bars_since_entry = i - entry_i + 1
            window_hi = float(df.iloc[entry_i:i + 1]["High"].max())
            window_lo = float(df.iloc[entry_i:i + 1]["Low"].min())
            tp_px = entry_px * (1 + tp)
            sl_px = entry_px * (1 - sl) if sl is not None else None

            exit_reason = exit_px = None
            if sl_px is not None and window_lo <= sl_px:
                exit_reason, exit_px = "SL", sl_px
            elif window_hi >= tp_px:
                exit_reason, exit_px = "TP", tp_px
            elif bars_since_entry >= horizon:
                exit_reason = "TIME"
                exit_px = float(df.loc[ts, "Close"])

            if exit_reason is not None:
                notional = entry_px * qty
                fee = 2 * fee_bps_per_side / 10_000 * notional
                pnl_dollars = (exit_px - entry_px) * qty - fee
                net_return = pnl_dollars / notional
                trades.append({
                    "entry_time": entry_ts,
                    "exit_time": ts,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "exit_reason": exit_reason,
                    "net_return": net_return,
                    "pnl_dollars": pnl_dollars,
                    "notional": notional,
                    "trade_date": entry_ts.date().isoformat(),
                })
                in_pos = False
                entry_px = entry_ts = entry_i = None

    if in_pos and entry_i is not None and entry_px is not None:
        exit_px = float(df.iloc[-1]["Close"])
        notional = entry_px * qty
        fee = 2 * fee_bps_per_side / 10_000 * notional
        pnl_dollars = (exit_px - entry_px) * qty - fee
        net_return = pnl_dollars / notional
        trades.append({
            "entry_time": entry_ts,
            "exit_time": idxs[-1],
            "entry_price": entry_px,
            "exit_price": exit_px,
            "exit_reason": "EOD",
            "net_return": net_return,
            "pnl_dollars": pnl_dollars,
            "notional": notional,
            "trade_date": entry_ts.date().isoformat(),
        })

    return trades


def _trades_to_dfs(trades: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not trades:
        return pd.DataFrame(), pd.DataFrame()
    trades_df = pd.DataFrame(trades)
    daily_df = (
        trades_df.groupby("trade_date", as_index=False)
        .agg(daily_pnl=("pnl_dollars", "sum"), daily_notional=("notional", "sum"))
    )
    return trades_df, daily_df


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class RFTPStrategy(BaseStrategy):
    model_id = "rf_tp"
    strategy_type = "RF_TP_ML"

    def __init__(self, version: str):
        if version not in VERSION_CONFIGS:
            raise ValueError(f"Unknown RF-TP version: {version}. Options: {list(VERSION_CONFIGS)}")
        self.version = version
        cfg = VERSION_CONFIGS[version]
        self._wf_mode: str = cfg["wf_mode"]
        self._features: list = cfg["features"]
        self._rf_params: dict = cfg["rf_params"]
        self._grid_def: dict = cfg["grid"]
        # rolling-specific
        self._train_size: Optional[int] = cfg.get("train_size")
        self._test_size: Optional[int] = cfg.get("test_size")
        # day-specific
        self._train_days: Optional[int] = cfg.get("train_days")
        self._eval_days: Optional[int] = cfg.get("eval_days")

        # Caches — valid for the lifetime of this instance (one run_all call)
        # Keyed by (id(df_1m), symbol, ...) — id is stable within one symbol's grid loop
        self._feat_cache: dict = {}         # (df_id, symbol) -> df_feat
        self._prob_cache: dict = {}         # (df_id, symbol, horizon, tp) -> prob series  [rolling]
        self._model_cache: dict = {}        # (df_id, symbol, roll_idx, horizon, tp) -> (model, test_df)

    def param_grid(self) -> List[dict]:
        g = self._grid_def
        return [
            {"horizon": h, "tp": tp, "sl": sl, "p_enter": pe}
            for h, tp, sl, pe in product(g["horizon"], g["tp"], g["sl"], g["p_enter"])
        ]

    def backtest(
        self,
        df_1m: pd.DataFrame,
        symbol: str,
        params: dict,
        fee_bps_per_side: float,
        qty: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_id = id(df_1m)
        feat_key = (df_id, symbol)

        if feat_key not in self._feat_cache:
            df = _prepare_df(df_1m)
            df_feat = _add_features(df, self._features)
            df_feat = df_feat.dropna(
                subset=self._features + ["Open", "High", "Low", "Close", "Volume"]
            ).copy()
            self._feat_cache[feat_key] = df_feat
        df_feat = self._feat_cache[feat_key]

        if self._wf_mode == "rolling":
            return self._backtest_rolling(df_id, symbol, df_feat, params, fee_bps_per_side, qty)
        else:
            return self._backtest_day(df_id, symbol, df_feat, params, fee_bps_per_side, qty)

    # ------------------------------------------------------------------
    # Rolling walk-forward (4.1.1 / 4.1.2)
    # ------------------------------------------------------------------

    def _backtest_rolling(
        self,
        df_id: int,
        symbol: str,
        df_feat: pd.DataFrame,
        params: dict,
        fee_bps_per_side: float,
        qty: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        horizon = params["horizon"]
        tp = params["tp"]
        sl = params["sl"]
        p_enter = params["p_enter"]

        prob_key = (df_id, symbol, horizon, tp)
        if prob_key not in self._prob_cache:
            y = _make_label(df_feat, horizon, tp)
            prob = _walkforward_rolling(
                df_feat, y, self._features,
                self._train_size, self._test_size, self._rf_params,
            )
            self._prob_cache[prob_key] = prob
        prob = self._prob_cache[prob_key]

        trades = _simulate_rolling_trades(
            df_feat, prob, horizon, tp, p_enter, sl, fee_bps_per_side, qty
        )
        return _trades_to_dfs(trades)

    # ------------------------------------------------------------------
    # Day-based walk-forward (4.1.3+)
    # ------------------------------------------------------------------

    def _backtest_day(
        self,
        df_id: int,
        symbol: str,
        df_feat: pd.DataFrame,
        params: dict,
        fee_bps_per_side: float,
        qty: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        horizon = params["horizon"]
        tp = params["tp"]
        sl = params["sl"]
        p_enter = params["p_enter"]

        trading_days = _get_trading_days(df_feat.index)
        n = len(trading_days)
        needed = self._train_days + 1  # at least 1 eval day
        if n < needed:
            return pd.DataFrame(), pd.DataFrame()

        start_test_idx = max(self._train_days, n - self._eval_days)
        all_trades: List[dict] = []

        for roll_idx in range(start_test_idx, n):
            model_key = (df_id, symbol, roll_idx, horizon, tp)

            if model_key not in self._model_cache:
                train_start = roll_idx - self._train_days
                train_set = set(trading_days[train_start:roll_idx])
                test_set = {trading_days[roll_idx]}

                idx_dates = pd.Series(df_feat.index.date, index=df_feat.index)
                train_df = df_feat.loc[idx_dates.isin(train_set).to_numpy()].copy()
                test_df = df_feat.loc[idx_dates.isin(test_set).to_numpy()].copy()

                train_df = train_df.dropna(subset=self._features)
                test_df = test_df.dropna(subset=self._features)

                model = None
                if len(train_df) >= 100 and len(test_df) >= 10:
                    y_train = _make_label(train_df, horizon, tp)
                    y_train = y_train.reindex(train_df.index).dropna()
                    X_train = train_df.loc[y_train.index, self._features]
                    if len(X_train) >= 50 and len(np.unique(y_train.astype(int))) == 2:
                        model = RandomForestClassifier(**self._rf_params)
                        model.fit(X_train, y_train.astype(int))

                self._model_cache[model_key] = (model, test_df)

            model, test_df = self._model_cache[model_key]
            if model is None or test_df.empty:
                continue

            day_trades = _simulate_day_trades(
                test_df, model, self._features,
                horizon, tp, sl, p_enter, qty, fee_bps_per_side,
            )
            all_trades.extend(day_trades)

        return _trades_to_dfs(all_trades)
