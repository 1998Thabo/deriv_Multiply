"""
trader.py — Live demo trading bot.
Fixes all WebSocket issues from the log:
  - Class-based DerivAPI with req_id routing (never steals messages)
  - Background recv loop routes messages to waiting coroutines
  - Proper ConnectionClosed handling
  - Correct Deriv multiplier contract parameters
  - Hybrid signal: rule-based + XGBoost filter
  - Rolling feature buffer using shared features.py
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    WS_URL, API_TOKEN, SYMBOLS,
    STAKE, MULTIPLIER, STOP_LOSS, TAKE_PROFIT,
    MAX_TRADES_SESSION, MIN_CONFIDENCE,
    RESULTS_DIR, LIVE_BUFFER_SIZE,
    RECONNECT_BASE_SEC, MAX_RECONNECT_RETRIES,
)
from features import compute_features_from_series, FEATURE_NAMES
from model import load_model, make_signal_fn
from backtest import strategy_mean_reversion, strategy_momentum

log = logging.getLogger(__name__)


# ── Deriv API class with req_id routing ─────────────────────────────

class DerivAPI:
    """
    Manages a single WebSocket connection.
    Uses req_id to route responses to the correct caller.
    Background recv loop prevents message-stealing.
    """

    def __init__(self, ws):
        self._ws       = ws
        self._req_id   = 0
        self._pending  = {}          # req_id → asyncio.Future
        self._tick_cbs = {}          # symbol → callback(price, epoch)
        self._recv_task = None

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def start(self) -> None:
        """Start the background receive loop."""
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def stop(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

    async def _recv_loop(self) -> None:
        """Route incoming messages by type or req_id. Never block callers."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                msg_type = msg.get("msg_type", "")
                req_id   = msg.get("req_id")

                # Route tick messages to registered callbacks
                if msg_type == "tick":
                    tick   = msg.get("tick", {})
                    symbol = tick.get("symbol")
                    if symbol and symbol in self._tick_cbs:
                        price = tick.get("quote")
                        epoch = tick.get("epoch")
                        if price and epoch:
                            try:
                                self._tick_cbs[symbol](float(price), int(epoch))
                            except Exception as e:
                                log.debug("Tick callback error: %s", e)
                    continue

                # Route responses to waiting requests
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)

        except Exception as e:
            log.debug("Recv loop ended: %s", e)
            # Resolve all pending futures with error
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WebSocket closed"))
            self._pending.clear()

    async def request(self, payload: dict, timeout: float = 10.0) -> dict:
        """Send a request and await its response via req_id."""
        rid = self._next_req_id()
        payload["req_id"] = rid
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise TimeoutError(f"Request timed out: {payload.get('msg_type', payload)}")

    def register_tick_callback(self, symbol: str, callback) -> None:
        self._tick_cbs[symbol] = callback

    async def authorize(self) -> dict:
        if not API_TOKEN:
            log.warning("No API_TOKEN set — signal-only mode (no real trades)")
            return {}
        resp = await self.request({"authorize": API_TOKEN})
        if "error" in resp:
            raise PermissionError(f"Auth failed: {resp['error']['message']}")
        log.info("Authenticated as: %s", resp.get("authorize", {}).get("loginid", "?"))
        return resp

    async def subscribe_ticks(self, symbol: str) -> None:
        await self.request({"ticks": symbol, "subscribe": 1})
        log.info("  Subscribed to %s", symbol)

    async def buy_multiplier(self, symbol: str, direction: str) -> Optional[str]:
        """
        Place a multiplier contract. Returns contract_id or None on failure.
        Fixed contract_type: MULTUP/MULTDOWN (not direction string).
        Fixed parameters structure that Deriv API expects.
        """
        contract_type = "MULTUP" if direction == "BUY" else "MULTDOWN"
        try:
            resp = await self.request({
                "buy": 1,
                "price": STAKE,
                "parameters": {
                    "contract_type": contract_type,
                    "symbol":        symbol,
                    "amount":        STAKE,
                    "currency":      "USD",
                    "multiplier":    MULTIPLIER,
                    "limit_order": {
                        "stop_loss":   {"order_amount": STOP_LOSS,   "order_type": "stop_loss"},
                        "take_profit": {"order_amount": TAKE_PROFIT, "order_type": "take_profit"},
                    },
                },
            }, timeout=15.0)
        except Exception as e:
            log.error("Buy request failed [%s %s]: %s", direction, symbol, e)
            return None

        if "error" in resp:
            log.error("Trade rejected [%s %s]: %s", direction, symbol, resp["error"]["message"])
            return None

        cid = str(resp.get("buy", {}).get("contract_id", ""))
        log.info("✔ TRADE PLACED | %s %s | id=%s", direction, symbol, cid)
        return cid


# ── Symbol state ─────────────────────────────────────────────────────

@dataclass
class SymbolState:
    symbol:        str
    price_buf:     deque = field(default_factory=lambda: deque(maxlen=250))
    ts_buf:        deque = field(default_factory=lambda: deque(maxlen=250))
    signal_fn:     object = None
    active_entry:  float = 0.0
    active_dir:    str   = ""
    active_cid:    str   = ""
    has_trade:     bool  = False


# ── Demo Trader ───────────────────────────────────────────────────────

class DemoTrader:

    def __init__(self, symbols: list[str]):
        self.symbols       = symbols
        self.states        = {}
        self.total_trades  = 0
        self._api: Optional[DerivAPI] = None

    def _load_models(self) -> None:
        for sym in self.symbols:
            state = SymbolState(symbol=sym)
            try:
                model, feat_cols = load_model(sym)
                state.signal_fn  = make_signal_fn(model, feat_cols)
                log.info("  ✔ Model loaded: %s (%d features)", sym, len(feat_cols))
            except FileNotFoundError:
                log.warning("  ✘ No model for %s — will run without signals", sym)
            except Exception as e:
                log.error("  ✘ Model load error for %s: %s", sym, e)
            self.states[sym] = state

    def _on_tick(self, symbol: str, price: float, epoch: int) -> None:
        """Synchronous tick handler — runs in the recv loop context."""
        state = self.states.get(symbol)
        if state is None:
            return

        state.price_buf.append(price)
        state.ts_buf.append(epoch)

        # Check stop-loss / take-profit on active trade
        if state.has_trade:
            chg = (price - state.active_entry) / state.active_entry
            if state.active_dir == "SELL":
                chg = -chg
            pnl = chg * STAKE * MULTIPLIER

            close_reason = None
            if pnl <= -STOP_LOSS:
                close_reason = "STOP_LOSS"
            elif pnl >= TAKE_PROFIT:
                close_reason = "TAKE_PROFIT"

            if close_reason:
                log.info("CLOSE [%s] %s | PnL=$%.4f | %s",
                         symbol, state.active_dir, pnl, close_reason)
                self._log_trade(state, price, pnl, close_reason)
                state.has_trade  = False
                state.active_cid = ""
            return   # don't look for new entry while in trade

        # Generate signal when buffer is warm
        if len(state.price_buf) < LIVE_BUFFER_SIZE:
            return
        if self.total_trades >= MAX_TRADES_SESSION:
            return
        if state.signal_fn is None:
            return

        features = compute_features_from_series(
            list(state.price_buf), list(state.ts_buf))
        if not features:
            return

        direction, confidence = state.signal_fn(features)
        if direction is None:
            return

        log.info("SIGNAL [%s] %s | conf=%.3f | price=%.5f",
                 symbol, direction, confidence, price)

        # Schedule trade placement (can't await in sync callback)
        asyncio.create_task(self._place_trade(symbol, direction, price))

    async def _place_trade(self, symbol: str, direction: str, price: float) -> None:
        state = self.states[symbol]
        if state.has_trade or self.total_trades >= MAX_TRADES_SESSION:
            return

        cid = ""
        if self._api and API_TOKEN:
            cid = await self._api.buy_multiplier(symbol, direction) or ""

        state.has_trade    = True
        state.active_entry = price
        state.active_dir   = direction
        state.active_cid   = cid
        self.total_trades  += 1
        log.info("ENTERED [%s] %s @ %.5f | SL=$%.2f TP=$%.2f (trade %d/%d)",
                 symbol, direction, price, STOP_LOSS, TAKE_PROFIT,
                 self.total_trades, MAX_TRADES_SESSION)

    def _log_trade(self, state: SymbolState, exit_price: float,
                   pnl: float, reason: str) -> None:
        path = RESULTS_DIR / "demo_trades.csv"
        header = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if header:
                f.write("timestamp,symbol,direction,entry,exit,pnl,reason,contract_id\n")
            f.write(f"{datetime.now(timezone.utc).isoformat()},"
                    f"{state.symbol},{state.active_dir},"
                    f"{state.active_entry:.5f},{exit_price:.5f},"
                    f"{pnl:.4f},{reason},{state.active_cid}\n")

    async def run(self) -> None:
        import websockets

        self._load_models()

        log.info("═" * 65)
        log.info("DEMO TRADER — %s", ", ".join(self.symbols))
        log.info("  Stake $%.2f × %dx | SL $%.2f | TP $%.2f | MaxTrades %d",
                 STAKE, MULTIPLIER, STOP_LOSS, TAKE_PROFIT, MAX_TRADES_SESSION)
        if not API_TOKEN:
            log.warning("  MODE: SIGNAL-ONLY (no API_TOKEN set)")
        log.info("═" * 65)

        attempt = 0

        while True:
            wait = min(RECONNECT_BASE_SEC * (2 ** attempt), 60)
            if attempt > 0:
                log.info("Reconnecting in %.0fs…", wait)
                await asyncio.sleep(wait)

            if attempt > MAX_RECONNECT_RETRIES:
                log.error("Max reconnect attempts reached — stopping")
                break

            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=25,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    attempt = 0
                    api     = DerivAPI(ws)
                    self._api = api
                    await api.start()

                    try:
                        await api.authorize()
                    except PermissionError as e:
                        log.error("%s", e)

                    # Register tick callbacks and subscribe
                    for sym in self.symbols:
                        api.register_tick_callback(
                            sym,
                            lambda p, e, s=sym: self._on_tick(s, p, e)
                        )
                        try:
                            await api.subscribe_ticks(sym)
                        except Exception as e:
                            log.error("Subscribe failed for %s: %s", sym, e)

                    log.info("Listening for ticks… (Ctrl+C to stop)")

                    # Keep alive until connection drops
                    while not ws.closed:
                        await asyncio.sleep(5)
                        if self.total_trades >= MAX_TRADES_SESSION:
                            log.info("Session cap reached (%d trades). Stopping.",
                                     MAX_TRADES_SESSION)
                            return

                    await api.stop()

            except websockets.exceptions.ConnectionClosedError as e:
                log.warning("Connection closed: %s", e)
                attempt += 1
            except Exception as e:
                log.error("Unexpected error: %s", e)
                attempt += 1


async def run_trader(symbols=None) -> None:
    targets = symbols or SYMBOLS
    t = DemoTrader(targets)
    await t.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/trader.log", encoding="utf-8"),
        ],
    )
    try:
        asyncio.run(run_trader())
    except KeyboardInterrupt:
        log.info("Trader stopped.")
