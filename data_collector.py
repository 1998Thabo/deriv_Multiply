"""
Deriv Volatility Index — Tick Data Collector
=============================================
Collects historical tick data from the Deriv WebSocket API
and stores it in per-symbol CSV files.

Usage:
    python data_collector.py

Symbols collected:
    R_10, R_25, R_50, R_75, R_100
"""

import asyncio
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

WS_URL = "wss://ws.derivws.com/websockets/v3"

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

TICKS_PER_BATCH = 5000          # Ticks per API request (max allowed)
BATCH_DELAY_SECONDS = 1.5       # Pause between batches (respect rate limits)
RECONNECT_DELAY_SECONDS = 5     # Wait before reconnecting on failure
MAX_RECONNECT_ATTEMPTS = 10     # Max retries per symbol per session

DATA_DIR = Path("data")         # Output folder for CSV files
LOG_LEVEL = logging.INFO

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("collector.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CSV Storage
# ─────────────────────────────────────────────

def get_csv_path(symbol: str) -> Path:
    """Return the CSV file path for a given symbol."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{symbol}.csv"


def ensure_csv_header(symbol: str) -> None:
    """Create the CSV with a header row if it does not already exist."""
    path = get_csv_path(symbol)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price", "symbol"])
        log.info("Created CSV for %s → %s", symbol, path)


def append_ticks_to_csv(symbol: str, ticks: list[dict]) -> int:
    """
    Append a list of tick dicts to the symbol's CSV.

    Each tick dict must have keys: epoch, quote.
    Returns the number of rows written.
    """
    if not ticks:
        return 0

    path = get_csv_path(symbol)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for tick in ticks:
            ts = datetime.fromtimestamp(tick["epoch"], tz=timezone.utc).isoformat()
            writer.writerow([ts, tick["quote"], symbol])

    return len(ticks)


def count_existing_rows(symbol: str) -> int:
    """Return the number of data rows already stored (excluding header)."""
    path = get_csv_path(symbol)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        # Subtract 1 for the header row
        return max(0, sum(1 for _ in f) - 1)


# ─────────────────────────────────────────────
# Deriv WebSocket Helpers
# ─────────────────────────────────────────────

def build_ticks_history_request(symbol: str, count: int = TICKS_PER_BATCH) -> dict:
    """Build the ticks_history API request payload."""
    return {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "start": 1,
        "style": "ticks",
    }


async def fetch_tick_batch(ws, symbol: str) -> list[dict] | None:
    """
    Send one ticks_history request and return the list of ticks.

    Returns None on API-level errors.
    """
    request = build_ticks_history_request(symbol)
    await ws.send(json.dumps(request))

    raw = await ws.recv()
    response = json.loads(raw)

    # API-level error check
    if "error" in response:
        log.error(
            "API error for %s — code=%s  message=%s",
            symbol,
            response["error"].get("code"),
            response["error"].get("message"),
        )
        return None

    # Extract ticks from the history payload
    history = response.get("history", {})
    prices = history.get("prices", [])
    times = history.get("times", [])

    if not prices or not times:
        log.warning("Empty history payload for %s", symbol)
        return None

    ticks = [{"epoch": int(t), "quote": float(p)} for t, p in zip(times, prices)]
    return ticks


# ─────────────────────────────────────────────
# Per-Symbol Collection Loop
# ─────────────────────────────────────────────

async def collect_symbol(symbol: str, target_ticks: int = 0) -> None:
    """
    Continuously collect tick data for one symbol.

    Args:
        symbol:       Deriv symbol, e.g. "R_50".
        target_ticks: Stop after collecting this many ticks total (0 = run forever).
    """
    ensure_csv_header(symbol)
    existing = count_existing_rows(symbol)
    log.info("▶  %s | existing rows: %d", symbol, existing)

    attempt = 0

    while True:
        # Check target
        if target_ticks > 0:
            current = count_existing_rows(symbol)
            if current >= target_ticks:
                log.info("✔  %s | target of %d ticks reached. Stopping.", symbol, target_ticks)
                return

        try:
            log.info("Connecting WebSocket for %s …", symbol)
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                attempt = 0  # Reset on successful connection
                log.info("Connected for %s", symbol)

                while True:
                    # Check target inside inner loop too
                    if target_ticks > 0:
                        current = count_existing_rows(symbol)
                        if current >= target_ticks:
                            log.info("✔  %s | target reached inside loop.", symbol)
                            return

                    ticks = await fetch_tick_batch(ws, symbol)

                    if ticks is None:
                        # API error — short pause then retry
                        await asyncio.sleep(BATCH_DELAY_SECONDS * 2)
                        continue

                    written = append_ticks_to_csv(symbol, ticks)
                    total = count_existing_rows(symbol)
                    log.info(
                        "  %s | batch: +%d ticks  |  total stored: %d",
                        symbol,
                        written,
                        total,
                    )

                    await asyncio.sleep(BATCH_DELAY_SECONDS)

        except websockets.exceptions.ConnectionClosedError as e:
            log.warning("Connection closed for %s: %s", symbol, e)
        except websockets.exceptions.WebSocketException as e:
            log.error("WebSocket error for %s: %s", symbol, e)
        except Exception as e:  # noqa: BLE001
            log.exception("Unexpected error for %s: %s", symbol, e)

        attempt += 1
        if attempt > MAX_RECONNECT_ATTEMPTS:
            log.error(
                "%s | exceeded max reconnect attempts (%d). Giving up.",
                symbol,
                MAX_RECONNECT_ATTEMPTS,
            )
            return

        wait = RECONNECT_DELAY_SECONDS * attempt
        log.info("%s | reconnecting in %ds (attempt %d)…", symbol, wait, attempt)
        await asyncio.sleep(wait)


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

async def main() -> None:
    """
    Launch concurrent data collection for all symbols.

    Set target_ticks > 0 to stop automatically after N ticks per symbol,
    or leave at 0 to run indefinitely.
    """
    target_ticks = 50_000  # Change to 0 to run forever

    log.info("=" * 60)
    log.info("Deriv Tick Data Collector — starting")
    log.info("Symbols : %s", ", ".join(SYMBOLS))
    log.info("Target  : %s ticks per symbol", target_ticks if target_ticks else "unlimited")
    log.info("Output  : %s/", DATA_DIR)
    log.info("=" * 60)

    tasks = [
        asyncio.create_task(collect_symbol(symbol, target_ticks=target_ticks))
        for symbol in SYMBOLS
    ]

    await asyncio.gather(*tasks)

    log.info("All collection tasks complete.")
    _print_summary()


def _print_summary() -> None:
    """Log a final row-count summary for all symbols."""
    log.info("─" * 40)
    log.info("Collection summary:")
    for symbol in SYMBOLS:
        rows = count_existing_rows(symbol)
        log.info("  %-6s → %d rows", symbol, rows)
    log.info("─" * 40)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user. Exiting cleanly.")
        _print_summary()
