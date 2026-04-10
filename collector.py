"""
collector.py — Robust WebSocket tick collector.
Switches to live ticks stream (better for synthetic indices).
Rate-limited, auto-reconnecting, validates ticks before writing.
"""

import asyncio
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import (
    WS_URL, SYMBOLS, DATA_DIR,
    COLLECTION_DELAY_SEC, RECONNECT_BASE_SEC, MAX_RECONNECT_RETRIES,
)

log = logging.getLogger(__name__)


def _csv_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}.csv"


def _ensure_header(symbol: str) -> None:
    p = _csv_path(symbol)
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "price", "symbol"])


def _append_tick(symbol: str, epoch: int, price: float) -> None:
    ts = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    with open(_csv_path(symbol), "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, price, symbol])


# Price sanity ranges per symbol (roughly)
PRICE_RANGES = {
    "R_10":  (4000, 7000),
    "R_25":  (1000, 10000),
    "R_50":  (50, 200),
    "R_75":  (5000, 100000),
    "R_100": (100, 2000),
}


def _is_valid_tick(symbol: str, price: float) -> bool:
    lo, hi = PRICE_RANGES.get(symbol, (0, 1e9))
    return lo <= price <= hi


async def collect_symbol(symbol: str) -> None:
    """Collect live ticks for one symbol with auto-reconnect."""
    _ensure_header(symbol)
    attempt  = 0
    seen     = set()           # dedup cache (last 1000 epochs)

    while True:
        wait = min(RECONNECT_BASE_SEC * (2 ** attempt), 60)
        if attempt > 0:
            log.info("%s: reconnecting in %.0fs (attempt %d)…", symbol, wait, attempt + 1)
            await asyncio.sleep(wait)

        if attempt > MAX_RECONNECT_RETRIES:
            log.error("%s: max retries exceeded — stopping", symbol)
            return

        try:
            import websockets
            async with websockets.connect(
                WS_URL,
                ping_interval=30,
                ping_timeout=15,
                close_timeout=5,
            ) as ws:
                attempt = 0   # reset on successful connect
                log.info("%s: connected", symbol)

                # Subscribe to live tick stream
                await ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))

                async for raw in ws:
                    msg = json.loads(raw)

                    if "error" in msg:
                        log.warning("%s: API error: %s", symbol, msg["error"].get("message"))
                        await asyncio.sleep(COLLECTION_DELAY_SEC)
                        continue

                    if msg.get("msg_type") != "tick":
                        continue

                    tick  = msg.get("tick", {})
                    epoch = tick.get("epoch")
                    price = tick.get("quote")

                    if epoch is None or price is None:
                        continue

                    price = float(price)
                    epoch = int(epoch)

                    # Validate
                    if not _is_valid_tick(symbol, price):
                        log.debug("%s: price %.5f out of range — skipped", symbol, price)
                        continue

                    # Deduplicate
                    if epoch in seen:
                        continue
                    seen.add(epoch)
                    if len(seen) > 1000:
                        seen.pop()

                    _append_tick(symbol, epoch, price)

        except Exception as e:
            log.warning("%s: connection error: %s", symbol, e)
            attempt += 1


async def run_collector(symbols=None) -> None:
    targets = symbols or SYMBOLS
    log.info("═" * 50)
    log.info("Tick Collector — %s", ", ".join(targets))
    log.info("Press Ctrl+C to stop")
    log.info("═" * 50)
    await asyncio.gather(*[collect_symbol(s) for s in targets])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/collector.log", encoding="utf-8"),
        ],
    )
    try:
        asyncio.run(run_collector())
    except KeyboardInterrupt:
        log.info("Collector stopped.")
