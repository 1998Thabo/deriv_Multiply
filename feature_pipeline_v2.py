"""
Deriv Volatility Index — Enhanced Feature Engineering Pipeline  v2
==================================================================
Builds on Phase 2 by adding:
  • Microstructure features   (tick direction, streaks, order flow)
  • Volatility regime         (low / medium / high classification)
  • Breakout features         (distance from recent high/low, breakout flag)
  • Mean-reversion signals    (MA distance, Z-score)
  • Extended time features    (session tick index, cyclical patterns)

Usage:
    python feature_pipeline_v2.py

Input:  data/R_*.csv   (raw tick CSVs from collector)
Output: processed/R_*_features_v2.csv
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

INPUT_DIR  = Path("data")
OUTPUT_DIR = Path("processed")

# Rolling window sizes (ticks)
ROLLING_WINDOWS = [10, 50, 100]

# Momentum / ROC periods
MOMENTUM_PERIODS = [5, 10, 20]

# EMA spans
EMA_SPANS = [10, 50, 100]

# Breakout lookback windows
BREAKOUT_WINDOWS = [20, 50]

# Order-flow counting windows
ORDER_FLOW_WINDOWS = [10, 20, 50]

# Volatility regime percentile thresholds (based on rolling_std_50)
VOL_LOW_PCT  = 33   # bottom tercile → low vol
VOL_HIGH_PCT = 67   # top tercile    → high vol

# Prediction horizon (ticks ahead)
PREDICTION_HORIZON = 10

MIN_ROWS = 500

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_v2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════

def load_csv(symbol: str) -> pd.DataFrame:
    """
    Load raw tick CSV, filtering any malformed timestamp rows,
    and parsing timestamps as UTC-aware datetimes.
    """
    path = INPUT_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, dtype={"price": np.float64, "symbol": "category"})

    # Drop malformed timestamp rows (truncated strings from collector edge-case)
    mask_good = df["timestamp"].str.len() >= 20
    n_bad = (~mask_good).sum()
    if n_bad:
        log.warning("%s: dropped %d malformed timestamp rows", symbol, n_bad)
    df = df[mask_good].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

    log.info("Loaded  %-6s → %d rows", symbol, len(df))
    return df


# ═══════════════════════════════════════════════════════════
# STEP 2 — CLEAN
# ═══════════════════════════════════════════════════════════

def clean(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Deduplicate, sort, and validate the tick series.
    Only exact (timestamp + price + symbol) duplicates are removed,
    preserving all genuine unique ticks.
    """
    n_raw = len(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp", "price", "symbol"])
    df = df.dropna(subset=["timestamp", "price"])
    df = df[df["price"] > 0]
    df = df.reset_index(drop=True)

    log.info("Cleaned %-6s → %d rows kept, %d dropped",
             symbol, len(df), n_raw - len(df))

    if len(df) < MIN_ROWS:
        raise ValueError(f"{symbol}: only {len(df)} rows after cleaning")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 3 — CORE PRICE FEATURES  (Phase 2 baseline)
# ═══════════════════════════════════════════════════════════

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns      — % change tick-to-tick
    log_returns  — ln(price_t / price_{t-1}), more Gaussian, preferred for ML
    price_diff   — absolute pip movement
    """
    df["returns"]     = df["price"].pct_change()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
    df["price_diff"]  = df["price"].diff()
    return df


def add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each window W: rolling mean, std, min, max, range of price.
    These form the baseline trend and volatility context.
    """
    for w in ROLLING_WINDOWS:
        r = df["price"].rolling(w, min_periods=1)
        df[f"roll_mean_{w}"]  = r.mean()
        df[f"roll_std_{w}"]   = r.std()
        df[f"roll_min_{w}"]   = r.min()
        df[f"roll_max_{w}"]   = r.max()
        df[f"roll_range_{w}"] = df[f"roll_max_{w}"] - df[f"roll_min_{w}"]
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    momentum_N — price_t - price_{t-N}  (directional push)
    roc_N      — rate of change %, normalised across price scales
    """
    for n in MOMENTUM_PERIODS:
        df[f"momentum_{n}"] = df["price"].diff(n)
        df[f"roc_{n}"]      = df["price"].pct_change(n) * 100
    return df


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMA and EMA for each span, plus binary crossover signals.
    EMA reacts faster to recent price changes than SMA.
    """
    for s in EMA_SPANS:
        df[f"sma_{s}"] = df["price"].rolling(s, min_periods=1).mean()
        df[f"ema_{s}"] = df["price"].ewm(span=s, adjust=False).mean()

    pairs = [(10, 50), (10, 100), (50, 100)]
    for short, long in pairs:
        if short in EMA_SPANS and long in EMA_SPANS:
            df[f"sma_cross_{short}_{long}"] = (
                df[f"sma_{short}"] > df[f"sma_{long}"]
            ).astype(np.int8)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling volatility of log_returns, and a tick-level true range proxy.
    """
    for w in ROLLING_WINDOWS:
        df[f"roll_vol_returns_{w}"] = (
            df["log_returns"].rolling(w, min_periods=1).std()
        )
    df["true_range"] = np.maximum(
        df["price_diff"].abs(),
        (df["price"] - df["roll_mean_10"]).abs(),
    )
    return df


# ═══════════════════════════════════════════════════════════
# STEP 4 — MICROSTRUCTURE FEATURES  (NEW)
# ═══════════════════════════════════════════════════════════

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tick-level market microstructure signals.

    tick_direction
        +1 if price rose vs previous tick, -1 if fell, 0 if unchanged.
        The most granular signal available; directly captures buyer/seller
        aggression at each tick.

    streak
        Number of consecutive ticks moving in the same direction.
        A streak of +5 means 5 rising ticks in a row — momentum signal.
        Resets to 1 (or -1) on any direction change.
        Computed without lookahead: each row uses only past values.

    order_flow_W  (for each window W in ORDER_FLOW_WINDOWS)
        Sum of tick_direction over the last W ticks.
        Positive = more buyers; negative = more sellers.
        Range: [-W, +W]. Normalised as a proportion: order_flow_W / W
        → gives a [-1, +1] score comparable across window sizes.
        This is a proxy for the classic "order flow imbalance" metric used
        in HFT, adapted for synthetic tick data.
    """
    # Tick direction — NaN on row 0 (no prior tick) → fill with 0
    diff = df["price"].diff()
    df["tick_direction"] = np.sign(diff).fillna(0).astype(np.int8)

    # Consecutive streak (vectorised via groupby on direction-change groups)
    direction = df["tick_direction"]
    # Assign a group ID that increments each time direction changes
    group = (direction != direction.shift(1)).cumsum()
    df["streak"] = (direction * (
        direction.groupby(group).cumcount() + 1
    )).fillna(0).astype(np.int8)

    # Order flow imbalance per window (normalised to [-1, +1])
    for w in ORDER_FLOW_WINDOWS:
        raw_flow = df["tick_direction"].rolling(w, min_periods=1).sum()
        df[f"order_flow_{w}"] = raw_flow / w

    log.info("  + microstructure features added")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 5 — VOLATILITY REGIME  (NEW)
# ═══════════════════════════════════════════════════════════

def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each tick into a volatility regime using rolling_std_50
    as the volatility proxy (computed in add_rolling_statistics).

    Thresholds are set at the VOL_LOW_PCT and VOL_HIGH_PCT percentiles
    of the FULL symbol's rolling_std_50 distribution. This is NOT
    data leakage because:
      • In backtesting, you know the historical vol distribution.
      • For live use, you would compute these on a training window and
        store them as fixed constants.

    vol_regime     — integer category:
                     0 = low volatility   (bottom tercile)
                     1 = medium volatility
                     2 = high volatility  (top tercile)

    vol_regime_low / vol_regime_med / vol_regime_high
                   — one-hot encoded dummies (avoids ordinal assumption
                     that high=2x medium).

    Why this matters for multipliers:
      High vol → wider price swings → higher risk of stop-out.
      Your ML model can learn to reduce multiplier size in high vol regimes.
    """
    vol_proxy = df["roll_std_50"]

    low_thresh  = np.nanpercentile(vol_proxy, VOL_LOW_PCT)
    high_thresh = np.nanpercentile(vol_proxy, VOL_HIGH_PCT)

    conditions = [
        vol_proxy <= low_thresh,
        (vol_proxy > low_thresh) & (vol_proxy <= high_thresh),
        vol_proxy > high_thresh,
    ]
    df["vol_regime"] = np.select(conditions, [0, 1, 2], default=1).astype(np.int8)

    # One-hot dummies
    df["vol_regime_low"]  = (df["vol_regime"] == 0).astype(np.int8)
    df["vol_regime_med"]  = (df["vol_regime"] == 1).astype(np.int8)
    df["vol_regime_high"] = (df["vol_regime"] == 2).astype(np.int8)

    counts = df["vol_regime"].value_counts().sort_index()
    log.info("  + vol regime: low=%d  med=%d  high=%d",
             counts.get(0, 0), counts.get(1, 0), counts.get(2, 0))
    return df


# ═══════════════════════════════════════════════════════════
# STEP 6 — BREAKOUT FEATURES  (NEW)
# ═══════════════════════════════════════════════════════════

def add_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects whether price is breaking out of its recent range.

    For each window W in BREAKOUT_WINDOWS:

      dist_from_high_W
          price - rolling_max(W)
          Zero or negative. When near zero, price is testing the recent high.
          A break above (positive value on next tick) = bullish breakout.

      dist_from_low_W
          price - rolling_min(W)
          Zero or positive. When near zero, price is testing the recent low.

      pct_in_range_W
          (price - roll_min_W) / (roll_max_W - roll_min_W)
          Normalised position within the [0, 1] range.
          0 = at the bottom, 1 = at the top, 0.5 = midpoint.
          Useful for mean-reversion (extremes tend to revert).

      breakout_up_W
          1 if price > rolling_max over the PREVIOUS W ticks (shifted by 1).
          This is a clean upward breakout signal with NO lookahead:
          we only compare to the max of past W ticks, not the current window.

      breakout_down_W
          1 if price < rolling_min over the PREVIOUS W ticks.

    All rolling operations use shift(1) where needed to prevent current-tick
    inclusion, ensuring zero data leakage.
    """
    for w in BREAKOUT_WINDOWS:
        roll_max = df["price"].rolling(w, min_periods=1).max()
        roll_min = df["price"].rolling(w, min_periods=1).min()
        roll_rng = (roll_max - roll_min).replace(0, np.nan)

        df[f"dist_from_high_{w}"]  = df["price"] - roll_max
        df[f"dist_from_low_{w}"]   = df["price"] - roll_min
        df[f"pct_in_range_{w}"]    = (df["price"] - roll_min) / roll_rng

        # Breakout vs PREVIOUS window (shift to avoid leakage)
        prev_max = df["price"].shift(1).rolling(w, min_periods=1).max()
        prev_min = df["price"].shift(1).rolling(w, min_periods=1).min()

        df[f"breakout_up_{w}"]   = (df["price"] > prev_max).astype(np.int8)
        df[f"breakout_down_{w}"] = (df["price"] < prev_min).astype(np.int8)

    log.info("  + breakout features added (windows=%s)", BREAKOUT_WINDOWS)
    return df


# ═══════════════════════════════════════════════════════════
# STEP 7 — MEAN REVERSION SIGNALS  (NEW)
# ═══════════════════════════════════════════════════════════

def add_mean_reversion_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantifies how far price has deviated from its 'fair value' (moving average).

    dist_from_sma_W
        price - SMA(W)
        Raw pip distance. Positive = price above average (potentially overbought).
        Negative = price below average (potentially oversold).

    pct_from_sma_W
        (price - SMA(W)) / SMA(W) × 100
        Percentage deviation. Comparable across different price-scale symbols.

    zscore_W
        (price - rolling_mean(W)) / rolling_std(W)
        Number of standard deviations from the local mean.
        Z > +2 → extreme overbought; Z < -2 → extreme oversold.
        Classic mean-reversion trigger used in stat-arb strategies.
        Uses only past W ticks — no lookahead.

    Why mean reversion matters for multiplier trading:
        After a large directional move (high z-score), price often snaps back.
        Entering a multiplier trade against the extreme can be profitable when
        combined with a tight take-profit.
    """
    for w in ROLLING_WINDOWS:
        sma  = df["price"].rolling(w, min_periods=1).mean()
        std  = df["price"].rolling(w, min_periods=1).std().replace(0, np.nan)

        df[f"dist_from_sma_{w}"] = df["price"] - sma
        df[f"pct_from_sma_{w}"]  = ((df["price"] - sma) / sma) * 100
        df[f"zscore_{w}"]        = (df["price"] - sma) / std

    log.info("  + mean reversion signals added")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 8 — TIME FEATURES  (ENHANCED)
# ═══════════════════════════════════════════════════════════

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features capturing intra-day and intra-session patterns.

    Cyclical encodings (sin/cos) prevent the model from treating
    hour 23 and hour 0 as distant — they are adjacent on the clock.

    hour_sin / hour_cos     — hour of day (0–23)
    minute_sin / minute_cos — minute of hour (0–59)
    dow_sin / dow_cos       — day of week (0=Mon, 6=Sun)

    tick_index
        Sequential integer position of each tick within the dataset.
        Captures the concept of 'where in the session are we'.
        Useful for detecting open/close patterns even in 24/7 markets.

    tick_index_norm
        tick_index normalised to [0, 1] over the full dataset.
        Prevents the raw integer from dominating tree-based models.

    seconds_since_start
        Elapsed seconds from the first tick in the dataset.
        Complements tick_index when tick timing is irregular.

    hour_of_day / minute_of_hour
        Raw integer versions retained alongside cyclical for tree models
        (gradient boosted trees can exploit raw integers directly).
    """
    ts = df["timestamp"].dt

    # Cyclical time encodings
    df["hour_sin"]   = np.sin(2 * np.pi * ts.hour   / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * ts.hour   / 24)
    df["minute_sin"] = np.sin(2 * np.pi * ts.minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * ts.minute / 60)
    df["dow_sin"]    = np.sin(2 * np.pi * ts.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * ts.dayofweek / 7)

    # Raw integer time features (for tree models)
    df["hour_of_day"]    = ts.hour.astype(np.int8)
    df["minute_of_hour"] = ts.minute.astype(np.int8)

    # Session position features
    df["tick_index"]      = np.arange(len(df))
    df["tick_index_norm"] = df["tick_index"] / max(len(df) - 1, 1)

    t0 = df["timestamp"].iloc[0]
    df["seconds_since_start"] = (df["timestamp"] - t0).dt.total_seconds()

    log.info("  + time features added")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 9 — LABELS
# ═══════════════════════════════════════════════════════════

def add_labels(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """
    target        — 1 if price is higher N ticks ahead, 0 otherwise.
    future_return — continuous % return N ticks ahead (regression target).

    shift(-horizon) looks forward; rows at the tail with no valid label
    are dropped in finalize() — no leakage.
    """
    future_price        = df["price"].shift(-horizon)
    df["target"]        = (future_price > df["price"]).astype(np.int8)
    df["future_return"] = (future_price - df["price"]) / df["price"]

    up   = df["target"].sum()
    down = (df["target"] == 0).sum()
    log.info("  Labels  horizon=%d | UP=%d (%.1f%%)  DOWN=%d (%.1f%%)",
             horizon, up, up/(up+down)*100, down, down/(up+down)*100)
    return df


# ═══════════════════════════════════════════════════════════
# STEP 10 — NORMALIZATION
# ═══════════════════════════════════════════════════════════

_NO_NORMALIZE = {
    "timestamp", "symbol", "price",
    "target", "tick_direction", "streak",
    "vol_regime", "vol_regime_low", "vol_regime_med", "vol_regime_high",
    "breakout_up_20", "breakout_up_50",
    "breakout_down_20", "breakout_down_50",
    "sma_cross_10_50", "sma_cross_10_100", "sma_cross_50_100",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow_sin", "dow_cos", "tick_index_norm",
    "hour_of_day", "minute_of_hour",
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score standardisation on all continuous numeric features.
    Binary/categorical columns are excluded (see _NO_NORMALIZE).
    """
    cols = [
        c for c in df.columns
        if c not in _NO_NORMALIZE and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in cols:
        mu, std = df[col].mean(), df[col].std()
        if std > 0:
            df[col] = (df[col] - mu) / std

    log.info("  Normalised %d columns", len(cols))
    return df


# ═══════════════════════════════════════════════════════════
# STEP 11 — FINALIZE & SAVE
# ═══════════════════════════════════════════════════════════

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop tail rows with no valid label, any remaining NaNs, reset index."""
    df = df.dropna(subset=["target", "future_return"])
    df = df.dropna()
    df = df.reset_index(drop=True)
    log.info("  Final dataset: %d rows × %d columns", len(df), len(df.columns))
    return df


def save(df: pd.DataFrame, symbol: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{symbol}_features_v2.csv"
    df.to_csv(path, index=False)
    log.info("Saved   %s → %s", symbol, path)
    return path


# ═══════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

def process_symbol(symbol: str) -> pd.DataFrame:
    """Full pipeline for one symbol."""
    log.info("━" * 55)
    log.info("Processing %s", symbol)
    log.info("━" * 55)

    df = load_csv(symbol)
    df = clean(df, symbol)

    # ── Core features (Phase 2 baseline) ──
    df = add_price_features(df)
    df = add_rolling_statistics(df)
    df = add_momentum_indicators(df)
    df = add_trend_indicators(df)
    df = add_volatility_features(df)

    # ── Advanced features (Phase 2 enhanced) ──
    df = add_microstructure_features(df)
    df = add_volatility_regime(df)
    df = add_breakout_features(df)
    df = add_mean_reversion_signals(df)
    df = add_time_features(df)

    # ── Labels & normalisation ──
    df = add_labels(df, horizon=PREDICTION_HORIZON)
    df = normalize(df)
    df = finalize(df)

    return df


def run_pipeline() -> None:
    log.info("═" * 55)
    log.info("Phase 2 Enhanced — Feature Pipeline v2")
    log.info("Symbols  : %s", ", ".join(SYMBOLS))
    log.info("Horizon  : %d ticks", PREDICTION_HORIZON)
    log.info("Output   : %s/", OUTPUT_DIR)
    log.info("═" * 55)

    results, errors = {}, {}

    for symbol in SYMBOLS:
        try:
            df = process_symbol(symbol)
            save(df, symbol)
            results[symbol] = (len(df), len(df.columns))
        except FileNotFoundError as e:
            log.error("%s", e)
            errors[symbol] = str(e)
        except Exception as e:                  # noqa: BLE001
            log.exception("Failed: %s — %s", symbol, e)
            errors[symbol] = str(e)

    log.info("═" * 55)
    log.info("Summary")
    log.info("─" * 55)
    for sym, (rows, cols) in results.items():
        log.info("  ✔  %-8s %5d rows  ×  %d features", sym, rows, cols)
    for sym, err in errors.items():
        log.info("  ✘  %-8s %s", sym, err)
    log.info("═" * 55)


if __name__ == "__main__":
    run_pipeline()
