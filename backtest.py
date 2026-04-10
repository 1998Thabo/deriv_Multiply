"""
backtest.py — Tick-by-tick backtest engine with transaction costs.
Works with both rule-based signals and ML signal functions.
"""

import csv
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import (
    SYMBOLS, PROCESSED_DIR, RESULTS_DIR,
    STAKE, MULTIPLIER, STOP_LOSS, TAKE_PROFIT,
    TRANSACTION_COST, MAX_TRADE_TICKS, MIN_CONFIDENCE,
)

log = logging.getLogger(__name__)


@dataclass
class Trade:
    trade_id:       int
    symbol:         str
    strategy:       str
    direction:      str       # "BUY" or "SELL"
    entry_idx:      int
    entry_time:     str
    entry_price:    float
    exit_idx:       int
    exit_time:      str
    exit_price:     float
    stake:          float
    multiplier:     int
    gross_pnl:      float     # before transaction costs
    tx_cost:        float     # transaction cost
    net_pnl:        float     # after transaction costs
    duration_ticks: int
    close_reason:   str
    confidence:     float
    vol_regime:     int


def calc_pnl(direction: str, entry: float, current: float,
             stake: float, mult: int) -> float:
    """P&L formula. BUY profits on rise, SELL profits on fall."""
    chg = (current - entry) / entry if entry != 0 else 0
    if direction == "SELL":
        chg = -chg
    return chg * stake * mult


def calc_tx_cost(stake: float, mult: int, rate: float = TRANSACTION_COST) -> float:
    """Transaction cost = rate × notional value × 2 (entry + exit)."""
    return rate * stake * mult * 2


# ── Built-in rule-based strategies ──────────────────────────────────

def strategy_mean_reversion(row: dict) -> tuple:
    """
    Buy when price is BELOW the mean (negative z-score = oversold).
    Sell when price is ABOVE the mean (positive z-score = overbought).
    Fixed from previous broken version.
    """
    z = row.get("zscore_50", 0)
    if z < -0.5:   # price below mean → expect bounce UP → BUY
        return "BUY",  min(abs(z) / 3.0, 1.0)
    if z > 0.5:    # price above mean → expect pullback DOWN → SELL
        return "SELL", min(abs(z) / 3.0, 1.0)
    return None, 0.0


def strategy_momentum(row: dict) -> tuple:
    """Buy on positive momentum, sell on negative."""
    m = row.get("momentum_10", 0)
    if m > 0:
        return "BUY",  min(abs(m) / 5.0, 1.0)
    if m < 0:
        return "SELL", min(abs(m) / 5.0, 1.0)
    return None, 0.0


BUILTIN_STRATEGIES = {
    "mean_reversion": strategy_mean_reversion,
    "momentum":       strategy_momentum,
}


# ── Engine ───────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    strategy_name: str,
    signal_fn: Callable,
    stake: float      = STAKE,
    multiplier: int   = MULTIPLIER,
    stop_loss: float  = STOP_LOSS,
    take_profit: float = TAKE_PROFIT,
    max_ticks: int    = MAX_TRADE_TICKS,
    min_confidence: float = 0.0,
) -> list[Trade]:
    """
    Tick-by-tick simulation with transaction costs.
    signal_fn(row_dict) → ("BUY"|"SELL"|None, confidence)
    """
    prices   = df["price"].values
    n        = len(prices)
    trades   = []
    trade_id = 0

    in_trade    = False
    direction   = None
    entry_idx   = None
    entry_price = None
    entry_time  = None
    confidence  = 0.0
    vol_at_entry = 1

    rows = df.to_dict("records")

    for i, row in enumerate(rows):
        price = prices[i]

        if not in_trade:
            sig, conf = signal_fn(row)
            if sig is not None and conf >= min_confidence:
                in_trade     = True
                direction    = sig
                entry_idx    = i
                entry_price  = price
                entry_time   = str(row.get("timestamp", ""))
                confidence   = conf
                vol_at_entry = int(row.get("vol_regime_high", 0))
            continue

        gross       = calc_pnl(direction, entry_price, price, stake, multiplier)
        ticks_held  = i - entry_idx
        tx          = calc_tx_cost(stake, multiplier)
        net         = gross - tx
        close_reason = None

        if net <= -stop_loss:
            close_reason = "STOP_LOSS"
        elif net >= take_profit:
            close_reason = "TAKE_PROFIT"
        elif ticks_held >= max_ticks:
            close_reason = "MAX_TICKS"
        elif i == n - 1:
            close_reason = "END_OF_DATA"

        if close_reason:
            trade_id += 1
            trades.append(Trade(
                trade_id       = trade_id,
                symbol         = symbol,
                strategy       = strategy_name,
                direction      = direction,
                entry_idx      = entry_idx,
                entry_time     = entry_time,
                entry_price    = round(entry_price, 5),
                exit_idx       = i,
                exit_time      = str(row.get("timestamp", "")),
                exit_price     = round(price, 5),
                stake          = stake,
                multiplier     = multiplier,
                gross_pnl      = round(gross, 5),
                tx_cost        = round(tx, 5),
                net_pnl        = round(net, 5),
                duration_ticks = ticks_held,
                close_reason   = close_reason,
                confidence     = round(confidence, 4),
                vol_regime     = vol_at_entry,
            ))
            in_trade = False

    return trades


# ── Metrics ─────────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade], symbol: str, strategy: str) -> dict:
    if not trades:
        return {"symbol": symbol, "strategy": strategy, "total_trades": 0,
                "win_rate_pct": 0, "total_pnl": 0, "profit_factor": 0,
                "max_drawdown": 0, "sharpe_approx": 0}

    pnls   = [t.net_pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gp = sum(wins)
    gl = abs(sum(losses)) if losses else 0
    pf = round(gp / gl, 4) if gl > 0 else float("inf")

    equity = np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity))

    pnl_std = float(np.std(pnls)) or 1e-9
    sharpe  = float(np.mean(pnls)) / pnl_std * math.sqrt(len(pnls))

    return {
        "symbol":          symbol,
        "strategy":        strategy,
        "multiplier":      trades[0].multiplier,
        "total_trades":    len(trades),
        "win_rate_pct":    round(len(wins) / len(pnls) * 100, 2),
        "total_net_pnl":   round(sum(pnls), 4),
        "total_gross_pnl": round(sum(t.gross_pnl for t in trades), 4),
        "total_tx_cost":   round(sum(t.tx_cost for t in trades), 4),
        "avg_pnl":         round(float(np.mean(pnls)), 4),
        "profit_factor":   pf,
        "max_drawdown":    round(max_dd, 4),
        "sharpe_approx":   round(sharpe, 4),
        "avg_duration":    round(float(np.mean([t.duration_ticks for t in trades])), 1),
    }


# ── Output ───────────────────────────────────────────────────────────

def save_trades(trades: list[Trade], tag: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"trades_{tag}.csv"
    if not trades:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=asdict(trades[0]).keys())
        w.writeheader()
        w.writerows(asdict(t) for t in trades)


def print_summary(all_metrics: list[dict]) -> None:
    log.info("═" * 85)
    log.info("BACKTEST SUMMARY  |  Stake $%.2f  Mult %dx  SL $%.2f  TP $%.2f  TxCost %.1f%%",
             STAKE, MULTIPLIER, STOP_LOSS, TAKE_PROFIT, TRANSACTION_COST * 100)
    log.info("─" * 85)
    log.info("  %-8s %-20s %6s %8s %10s %8s %8s %7s",
             "Symbol","Strategy","Trades","WinRate","Net PnL","PF","MaxDD","Sharpe")
    log.info("─" * 85)
    for m in all_metrics:
        if not m.get("total_trades"):
            log.info("  %-8s %-20s  NO TRADES", m["symbol"], m["strategy"])
            continue
        log.info("  %-8s %-20s %6d %7.1f%% %+10.2f %8.3f %8.2f %7.3f",
                 m["symbol"], m["strategy"],
                 m["total_trades"], m["win_rate_pct"],
                 m["total_net_pnl"], m["profit_factor"],
                 m["max_drawdown"], m["sharpe_approx"])
    log.info("═" * 85)


def run_all_backtests(symbols=None) -> list[dict]:
    """Run built-in rule-based strategies across all symbols."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/backtest.log", encoding="utf-8"),
        ],
    )
    targets = symbols or SYMBOLS
    all_metrics = []

    for sym in targets:
        path = PROCESSED_DIR / f"{sym}_features.csv"
        if not path.exists():
            log.error("No feature file for %s. Run: python run.py pipeline", sym)
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])

        for name, fn in BUILTIN_STRATEGIES.items():
            log.info("  %s / %s …", sym, name)
            trades  = run_backtest(df, sym, name, fn)
            metrics = compute_metrics(trades, sym, name)
            all_metrics.append(metrics)
            save_trades(trades, f"{name}_{sym}")
            if trades:
                log.info("    → %d trades | WR=%.1f%% | Net=$%.2f | PF=%.3f",
                         metrics["total_trades"], metrics["win_rate_pct"],
                         metrics["total_net_pnl"], metrics["profit_factor"])

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(RESULTS_DIR / "summary.csv", index=False)

    print_summary(all_metrics)
    return all_metrics


if __name__ == "__main__":
    run_all_backtests()
