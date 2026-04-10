"""
features.py — Shared feature engineering.
Used identically by pipeline.py (batch) and trader.py (live rolling buffer).
No normalisation — XGBoost/tree models don't need it.
No tick_index_norm — causes train/live mismatch.
"""

import numpy as np
import pandas as pd
from config import SMA_WINDOWS, MOMENTUM_PERIOD, ZSCORE_WINDOW, VOL_WINDOW, PREDICTION_HORIZON

# These are the exact column names produced — order matters for model loading
FEATURE_NAMES: list[str] = []   # populated by get_feature_names()


def get_feature_names() -> list[str]:
    """Return the canonical ordered list of feature column names."""
    cols = []
    # Price derivatives
    cols += ["returns", "log_returns", "price_diff"]
    # SMA features
    for w in SMA_WINDOWS:
        cols += [f"sma_{w}", f"dist_from_sma_{w}", f"pct_from_sma_{w}"]
    # Momentum
    cols += [f"momentum_{MOMENTUM_PERIOD}", f"roc_{MOMENTUM_PERIOD}"]
    # Z-score (mean reversion signal)
    cols += [f"zscore_{ZSCORE_WINDOW}"]
    # Volatility regime
    cols += [f"rolling_vol_{VOL_WINDOW}", "vol_regime_low", "vol_regime_high"]
    # Tick direction / microstructure
    cols += ["tick_direction", "streak"]
    # Order flow
    cols += ["order_flow_10", "order_flow_20"]
    # Time features (cyclical — no raw integers that shift between train/live)
    cols += ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    return cols


FEATURE_NAMES = get_feature_names()


def compute_features_from_series(prices: list | np.ndarray,
                                  timestamps=None) -> dict:
    """
    Compute features from a raw price list (used by live trader).
    Returns a single dict row of feature values.
    Requires len(prices) >= 55 to produce valid results.
    """
    p = np.asarray(prices, dtype=np.float64)
    n = len(p)

    if n < 55:
        return {}

    row = {}

    # ── Price derivatives ──
    row["price_diff"]  = float(p[-1] - p[-2])
    row["returns"]     = float(row["price_diff"] / p[-2]) if p[-2] != 0 else 0.0
    row["log_returns"] = float(np.log(p[-1] / p[-2])) if p[-2] > 0 else 0.0

    # ── SMA features ──
    for w in SMA_WINDOWS:
        window = p[-w:] if n >= w else p
        sma = float(np.mean(window))
        row[f"sma_{w}"]          = sma
        row[f"dist_from_sma_{w}"]= float(p[-1] - sma)
        row[f"pct_from_sma_{w}"] = float((p[-1] - sma) / sma * 100) if sma != 0 else 0.0

    # ── Momentum ──
    n_back = MOMENTUM_PERIOD
    if n > n_back:
        row[f"momentum_{n_back}"] = float(p[-1] - p[-1 - n_back])
        row[f"roc_{n_back}"]      = float((p[-1] - p[-1 - n_back]) / p[-1 - n_back] * 100) \
                                    if p[-1 - n_back] != 0 else 0.0
    else:
        row[f"momentum_{n_back}"] = 0.0
        row[f"roc_{n_back}"]      = 0.0

    # ── Z-score (mean reversion) ──
    w = ZSCORE_WINDOW
    window_z = p[-w:] if n >= w else p
    mu  = float(np.mean(window_z))
    std = float(np.std(window_z))
    row[f"zscore_{w}"] = float((p[-1] - mu) / std) if std > 0 else 0.0

    # ── Rolling volatility ──
    wv = VOL_WINDOW
    window_v = p[-wv:] if n >= wv else p
    log_rets_v = np.log(window_v[1:] / window_v[:-1]) if len(window_v) > 1 else np.array([0.0])
    rv = float(np.std(log_rets_v))
    row[f"rolling_vol_{wv}"] = rv

    # Vol regime (relative to longer-window vol)
    w_long = min(100, n)
    long_lr = np.log(p[-w_long+1:] / p[-w_long:-1]) if w_long > 1 else np.array([rv])
    long_vol = float(np.std(long_lr)) if len(long_lr) > 1 else rv
    row["vol_regime_low"]  = int(rv < long_vol * 0.8)
    row["vol_regime_high"] = int(rv > long_vol * 1.2)

    # ── Tick direction & streak ──
    diffs = np.diff(p[-20:])
    tick_dir = int(np.sign(diffs[-1])) if len(diffs) > 0 else 0
    row["tick_direction"] = tick_dir

    streak = 0
    for d in reversed(diffs):
        s = int(np.sign(d))
        if s == tick_dir:
            streak += 1
        else:
            break
    row["streak"] = streak * tick_dir

    # ── Order flow ──
    all_diffs = np.diff(p)
    all_dirs  = np.sign(all_diffs)
    for w in [10, 20]:
        window_d = all_dirs[-w:] if len(all_dirs) >= w else all_dirs
        row[f"order_flow_{w}"] = float(np.sum(window_d) / max(len(window_d), 1))

    # ── Time features (if timestamps provided) ──
    if timestamps is not None:
        try:
            ts = pd.Timestamp(timestamps[-1])
            row["hour_sin"]   = float(np.sin(2 * np.pi * ts.hour   / 24))
            row["hour_cos"]   = float(np.cos(2 * np.pi * ts.hour   / 24))
            row["minute_sin"] = float(np.sin(2 * np.pi * ts.minute / 60))
            row["minute_cos"] = float(np.cos(2 * np.pi * ts.minute / 60))
        except Exception:
            row["hour_sin"] = row["hour_cos"] = 0.0
            row["minute_sin"] = row["minute_cos"] = 0.0
    else:
        row["hour_sin"] = row["hour_cos"] = 0.0
        row["minute_sin"] = row["minute_cos"] = 0.0

    return row


def compute_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for a full DataFrame (used by pipeline.py).
    df must have columns: timestamp, price, symbol
    Returns df with feature columns added (NO normalisation).
    NO tick_index_norm.
    """
    p = df["price"].values
    n = len(p)

    # ── Price derivatives ──
    df["price_diff"]  = df["price"].diff()
    df["returns"]     = df["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))

    # ── SMA features ──
    for w in SMA_WINDOWS:
        sma = df["price"].rolling(w, min_periods=1).mean()
        df[f"sma_{w}"]           = sma
        df[f"dist_from_sma_{w}"] = df["price"] - sma
        df[f"pct_from_sma_{w}"]  = ((df["price"] - sma) / sma) * 100

    # ── Momentum ──
    nn = MOMENTUM_PERIOD
    df[f"momentum_{nn}"] = df["price"].diff(nn)
    df[f"roc_{nn}"]      = df["price"].pct_change(nn) * 100

    # ── Z-score ──
    w = ZSCORE_WINDOW
    roll_mu  = df["price"].rolling(w, min_periods=1).mean()
    roll_std = df["price"].rolling(w, min_periods=1).std().replace(0, np.nan)
    df[f"zscore_{w}"] = (df["price"] - roll_mu) / roll_std

    # ── Rolling volatility ──
    wv = VOL_WINDOW
    df[f"rolling_vol_{wv}"] = df["log_returns"].rolling(wv, min_periods=1).std()

    # Vol regime
    long_vol = df["log_returns"].rolling(100, min_periods=1).std()
    df["vol_regime_low"]  = (df[f"rolling_vol_{wv}"] < long_vol * 0.8).astype(int)
    df["vol_regime_high"] = (df[f"rolling_vol_{wv}"] > long_vol * 1.2).astype(int)

    # ── Tick direction & streak ──
    tick_dir = np.sign(df["price"].diff()).fillna(0).astype(int)
    df["tick_direction"] = tick_dir

    # Vectorised streak
    group = (tick_dir != tick_dir.shift(1)).cumsum()
    df["streak"] = (tick_dir * (tick_dir.groupby(group).cumcount() + 1)).fillna(0).astype(int)

    # ── Order flow ──
    for w in [10, 20]:
        df[f"order_flow_{w}"] = tick_dir.rolling(w, min_periods=1).sum() / w

    # ── Time features ──
    ts = df["timestamp"].dt
    df["hour_sin"]   = np.sin(2 * np.pi * ts.hour   / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * ts.hour   / 24)
    df["minute_sin"] = np.sin(2 * np.pi * ts.minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * ts.minute / 60)

    # ── Labels ──
    future_price        = df["price"].shift(-PREDICTION_HORIZON)
    df["target"]        = (future_price > df["price"]).astype(int)
    df["future_return"] = (future_price - df["price"]) / df["price"]

    return df
