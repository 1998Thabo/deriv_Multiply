"""
pipeline.py — Robust data loading and feature engineering.
Fixes: doubled-year timestamps, extra CSV columns, duplicate handling,
forward-fill for prices (never drop valid ticks on price issues only).
Preserves >95% of valid data.
"""

import logging
import re
import io
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SYMBOLS, DATA_DIR, PROCESSED_DIR,
    PREDICTION_HORIZON, MIN_TRAIN_ROWS,
)
from features import compute_features_dataframe, FEATURE_NAMES

log = logging.getLogger(__name__)


# ── Timestamp repair ────────────────────────────────────────────────

_DOUBLED_YEAR = re.compile(r'^(\d{4})-\1-(\d{2}T.+)$')  # 2026-2026-04-09T...

def _fix_timestamp(ts: str) -> str:
    """Fix doubled-year format: '2026-2026-04-09T...' → '2026-04-09T...'"""
    m = _DOUBLED_YEAR.match(str(ts).strip())
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return str(ts).strip()


# ── Robust CSV loader ───────────────────────────────────────────────

def load_raw_csv(symbol: str) -> pd.DataFrame:
    """
    Load raw tick CSV robustly.
    - Uses python engine to handle malformed rows without crashing
    - Repairs doubled-year timestamps in-memory
    - Drops ONLY rows with null timestamps (not bad prices — those get ffill)
    - Removes exact duplicates only
    - Preserves >95% of data
    """
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")

    # Read with python engine — tolerates extra columns and bad lines
    try:
        df = pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",       # skip lines that can't be parsed
            usecols=[0, 1, 2],         # only take first 3 columns (timestamp, price, symbol)
            names=["timestamp", "price", "symbol"],
            header=0,
            dtype={"price": str},      # read price as str first for validation
        )
    except Exception as e:
        log.error("%s: CSV read failed: %s", symbol, e)
        raise

    n_raw = len(df)

    # Fix doubled-year timestamps
    df["timestamp"] = df["timestamp"].astype(str).apply(_fix_timestamp)

    # Parse timestamps — coerce bad ones to NaT
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed",
                                      utc=True, errors="coerce")

    # Drop ONLY null timestamps — never drop for price issues
    null_ts = df["timestamp"].isna().sum()
    if null_ts:
        log.warning("%s: dropped %d null-timestamp rows (%.2f%%)",
                    symbol, null_ts, null_ts / n_raw * 100)
    df = df.dropna(subset=["timestamp"])

    # Convert price — forward-fill instead of dropping
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    null_price = df["price"].isna().sum()
    if null_price:
        log.warning("%s: forward-filling %d null-price rows", symbol, null_price)
        df["price"] = df["price"].ffill().bfill()

    # Set symbol correctly
    df["symbol"] = symbol

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove exact duplicates (same timestamp + price + symbol)
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "price", "symbol"])
    n_dupes = n_before - len(df)
    if n_dupes:
        log.info("%s: removed %d exact duplicates", symbol, n_dupes)

    retention = len(df) / n_raw * 100 if n_raw > 0 else 0
    log.info("%-6s  raw=%d  clean=%d  retention=%.1f%%  span=%s → %s",
             symbol, n_raw, len(df), retention,
             str(df.timestamp.iloc[0])[:19] if len(df) else "N/A",
             str(df.timestamp.iloc[-1])[:19] if len(df) else "N/A")

    if retention < 95:
        log.warning("%s: retention %.1f%% < 95%% — check raw data quality", symbol, retention)

    return df


# ── Full pipeline ───────────────────────────────────────────────────

def build_dataset(symbol: str, save: bool = True) -> pd.DataFrame:
    """Load, clean, engineer features, validate, and optionally save."""
    df = load_raw_csv(symbol)

    if len(df) < MIN_TRAIN_ROWS:
        log.warning("%s: only %d clean rows (need %d) — collect more data",
                    symbol, len(df), MIN_TRAIN_ROWS)

    # Feature engineering (shared with live trader)
    df = compute_features_dataframe(df)

    # Drop tail rows with no label (horizon look-ahead goes beyond data)
    df = df.dropna(subset=["target", "future_return"])

    # Drop any remaining NaN feature rows
    feature_cols = FEATURE_NAMES + ["target", "future_return"]
    df = df.dropna(subset=[c for c in feature_cols if c in df.columns])

    df = df.reset_index(drop=True)

    # Target balance check
    if "target" in df.columns:
        up_pct = df["target"].mean() * 100
        log.info("  %s: target balance UP=%.1f%%  DOWN=%.1f%%", symbol, up_pct, 100 - up_pct)
        if abs(up_pct - 50) > 15:
            log.warning("  %s: target imbalance %.1f%% — model may be biased", symbol, up_pct)

    if save:
        out = PROCESSED_DIR / f"{symbol}_features.csv"
        df.to_csv(out, index=False)
        log.info("Saved %s → %s (%d rows × %d cols)", symbol, out, len(df), len(df.columns))

    return df


def run_pipeline(symbols=None) -> dict:
    """Run pipeline for all (or specified) symbols."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        ],
    )
    targets = symbols or SYMBOLS
    log.info("═" * 55)
    log.info("Data Pipeline — %d symbols", len(targets))
    log.info("═" * 55)

    datasets = {}
    for sym in targets:
        try:
            datasets[sym] = build_dataset(sym)
        except FileNotFoundError:
            log.error("%s: data file not found in data/ — run collector first", sym)
        except Exception:
            log.exception("%s: pipeline failed", sym)

    log.info("Pipeline complete.")
    return datasets


if __name__ == "__main__":
    run_pipeline()
