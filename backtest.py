"""
Deriv Multiplier Trading — Backtesting Engine
=============================================
Simulates multiplier trades tick-by-tick on engineered feature datasets.

Strategies implemented:
  1. Momentum      — trade in the direction of recent price momentum
  2. Mean Reversion — fade extreme deviations from the rolling mean

Usage:
    python backtest.py

Input:  processed/R_*_features_v2.csv
Output: results/trades_<strategy>_<symbol>.csv
        results/summary.csv
        results/equity_<strategy>_<symbol>.csv
"""

import csv
import logging
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SYMBOLS          = ["R_10", "R_25", "R_50", "R_75", "R_100"]
INPUT_DIR        = Path("processed")
RESULTS_DIR      = Path("results")

STAKE            = 10.0          # Fixed stake per trade ($)
MULTIPLIER       = 10            # Leverage multiplier (try 10, 50, 100)
STOP_LOSS        = 3.0           # Max loss per trade ($)
TAKE_PROFIT      = 6.0           # Max gain per trade ($)
MAX_TRADE_TICKS  = 200           # Force-close after N ticks (avoid infinite holds)

# Strategy signal thresholds (z-score / normalised units)
MOMENTUM_THRESHOLD      = 0.3    # momentum_10 > +threshold → BUY, < -threshold → SELL
MEAN_REV_THRESHOLD      = 0.5    # zscore_50 > +threshold → SELL (overbought), < -threshold → BUY

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class Direction(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class CloseReason(str, Enum):
    TAKE_PROFIT  = "TAKE_PROFIT"
    STOP_LOSS    = "STOP_LOSS"
    MAX_TICKS    = "MAX_TICKS"
    END_OF_DATA  = "END_OF_DATA"


@dataclass
class Trade:
    """Represents a single completed trade."""
    trade_id:      int
    symbol:        str
    strategy:      str
    direction:     Direction
    entry_idx:     int
    entry_time:    str
    entry_price:   float
    exit_idx:      int
    exit_time:     str
    exit_price:    float
    stake:         float
    multiplier:    int
    pnl:           float
    duration_ticks: int
    close_reason:  CloseReason
    vol_regime:    int           # market regime at entry (0=low,1=med,2=high)


@dataclass
class BacktestConfig:
    symbol:           str
    strategy_name:    str
    stake:            float = STAKE
    multiplier:       int   = MULTIPLIER
    stop_loss:        float = STOP_LOSS
    take_profit:      float = TAKE_PROFIT
    max_trade_ticks:  int   = MAX_TRADE_TICKS


# ─────────────────────────────────────────────
# PnL Calculation
# ─────────────────────────────────────────────

def calc_pnl(
    direction: Direction,
    entry_price: float,
    current_price: float,
    stake: float,
    multiplier: int,
) -> float:
    """
    Multiplier P&L formula:
        pnl = ((current - entry) / entry) * stake * multiplier   [BUY]
        pnl = ((entry - current) / entry) * stake * multiplier   [SELL]

    A BUY profits when price rises; a SELL profits when price falls.
    """
    price_change_pct = (current_price - entry_price) / entry_price
    if direction == Direction.SELL:
        price_change_pct = -price_change_pct
    return price_change_pct * stake * multiplier


# ─────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────

def signal_momentum(row: pd.Series) -> Optional[Direction]:
    """
    Strategy 1 — Momentum

    Logic:
      If momentum_10 > +threshold  → BUY  (price rising, ride the wave)
      If momentum_10 < -threshold  → SELL (price falling, ride the wave)
      Otherwise                    → no trade

    Uses the z-score normalised momentum_10 column.
    Threshold filters out weak signals close to zero.

    Rationale:
      Deriv synthetic indices have genuine tick-level autocorrelation.
      A sequence of rising ticks (positive momentum) has a marginally
      higher probability of continuing for a few more ticks — enough
      to be captured with a tight take-profit.
    """
    m = row.get("momentum_10", 0)
    if m > MOMENTUM_THRESHOLD:
        return Direction.BUY
    if m < -MOMENTUM_THRESHOLD:
        return Direction.SELL
    return None


def signal_mean_reversion(row: pd.Series) -> Optional[Direction]:
    """
    Strategy 2 — Mean Reversion

    Logic:
      If zscore_50 < -threshold  → BUY  (price far below mean, expect bounce)
      If zscore_50 > +threshold  → SELL (price far above mean, expect pullback)
      Otherwise                  → no trade

    Uses the z-score of price vs its 50-tick rolling mean.
    Z-score is normalised (via the feature pipeline), so threshold
    is in standard-deviation units.

    Rationale:
      Volatility indices are mean-reverting over short horizons — their
      prices oscillate around a moving equilibrium. Extreme deviations
      (z > 0.5 SD) tend to snap back within 5–20 ticks, making them
      natural entry points for faded trades.
    """
    z = row.get("zscore_50", 0)
    if z < -MEAN_REV_THRESHOLD:
        return Direction.BUY
    if z > MEAN_REV_THRESHOLD:
        return Direction.SELL
    return None


STRATEGIES = {
    "momentum":      signal_momentum,
    "mean_reversion": signal_mean_reversion,
}


# ─────────────────────────────────────────────
# Core Backtesting Engine
# ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> list[Trade]:
    """
    Tick-by-tick backtest simulation.

    Rules:
      • Only ONE trade open at a time (no pyramiding).
      • Entry signal is checked on every tick when flat.
      • Once in a trade, each subsequent tick checks stop-loss,
        take-profit, and max-duration exit conditions.
      • We do NOT trade on the same tick we exit (wait for next signal).
      • Entry price = price at the signal tick (no slippage model;
        synthetic indices have near-zero spread).

    Returns a list of completed Trade objects.
    """
    signal_fn = STRATEGIES[cfg.strategy_name]
    prices    = df["price"].values
    n         = len(prices)
    trades: list[Trade] = []
    trade_id  = 0

    in_trade      = False
    direction     = None
    entry_idx     = None
    entry_price   = None
    entry_time    = None
    vol_at_entry  = 0

    for i, row in enumerate(df.itertuples(index=False)):
        price = prices[i]

        if not in_trade:
            # ── Look for entry signal ──
            sig = signal_fn(row._asdict())
            if sig is not None:
                in_trade    = True
                direction   = sig
                entry_idx   = i
                entry_price = price
                entry_time  = str(row.timestamp)
                vol_at_entry = int(getattr(row, "vol_regime", 1))
            continue   # Always skip to next tick after entry

        # ── We are in a trade — evaluate exit conditions ──
        pnl           = calc_pnl(direction, entry_price, price, cfg.stake, cfg.multiplier)
        ticks_held    = i - entry_idx
        close_reason  = None

        if pnl <= -cfg.stop_loss:
            close_reason = CloseReason.STOP_LOSS
        elif pnl >= cfg.take_profit:
            close_reason = CloseReason.TAKE_PROFIT
        elif ticks_held >= cfg.max_trade_ticks:
            close_reason = CloseReason.MAX_TICKS
        elif i == n - 1:
            close_reason = CloseReason.END_OF_DATA

        if close_reason is not None:
            trade_id += 1
            trades.append(Trade(
                trade_id      = trade_id,
                symbol        = cfg.symbol,
                strategy      = cfg.strategy_name,
                direction     = direction,
                entry_idx     = entry_idx,
                entry_time    = entry_time,
                entry_price   = entry_price,
                exit_idx      = i,
                exit_time     = str(row.timestamp),
                exit_price    = price,
                stake         = cfg.stake,
                multiplier    = cfg.multiplier,
                pnl           = round(pnl, 6),
                duration_ticks = ticks_held,
                close_reason  = close_reason,
                vol_regime    = vol_at_entry,
            ))
            in_trade = False

    return trades


# ─────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────

def compute_metrics(trades: list[Trade], symbol: str, strategy: str) -> dict:
    """
    Compute standard quantitative trading performance metrics.

    total_trades    — total number of completed trades
    win_rate        — % of trades with pnl > 0
    total_pnl       — sum of all trade pnl ($)
    avg_pnl         — mean pnl per trade
    avg_win         — mean pnl of winning trades
    avg_loss        — mean pnl of losing trades
    profit_factor   — gross_profit / |gross_loss| (>1 = edge exists)
    max_drawdown    — largest peak-to-trough equity drop ($)
    sharpe_approx   — mean(pnl) / std(pnl) * sqrt(trades) [simplified]
    avg_duration    — average trade duration in ticks
    pnl_by_regime   — breakdown of avg pnl per volatility regime
    """
    if not trades:
        return {"symbol": symbol, "strategy": strategy, "total_trades": 0}

    pnls       = [t.pnl for t in trades]
    wins       = [p for p in pnls if p > 0]
    losses     = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)   if wins   else 0.0
    gross_loss   = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown via equity curve
    equity = np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    # Approximate Sharpe (per-trade basis, not annualised)
    pnl_std = float(np.std(pnls)) if len(pnls) > 1 else 1.0
    sharpe  = (float(np.mean(pnls)) / pnl_std) * math.sqrt(len(pnls)) if pnl_std > 0 else 0.0

    # Breakdown by exit reason
    reasons = {}
    for r in CloseReason:
        count = sum(1 for t in trades if t.close_reason == r)
        reasons[f"exits_{r.value.lower()}"] = count

    # P&L by volatility regime
    for regime in [0, 1, 2]:
        subset = [t.pnl for t in trades if t.vol_regime == regime]
        label  = ["low", "med", "high"][regime]
        reasons[f"avg_pnl_regime_{label}"] = round(np.mean(subset), 4) if subset else 0.0

    return {
        "symbol":          symbol,
        "strategy":        strategy,
        "multiplier":      trades[0].multiplier if trades else MULTIPLIER,
        "stake":           trades[0].stake      if trades else STAKE,
        "stop_loss":       STOP_LOSS,
        "take_profit":     TAKE_PROFIT,
        "total_trades":    len(trades),
        "win_rate_pct":    round(len(wins) / len(pnls) * 100, 2),
        "total_pnl":       round(sum(pnls), 4),
        "avg_pnl":         round(float(np.mean(pnls)), 4),
        "avg_win":         round(float(np.mean(wins)),   4) if wins   else 0.0,
        "avg_loss":        round(float(np.mean(losses)), 4) if losses else 0.0,
        "gross_profit":    round(gross_profit, 4),
        "gross_loss":      round(gross_loss,   4),
        "profit_factor":   round(profit_factor, 4),
        "max_drawdown":    round(max_dd, 4),
        "sharpe_approx":   round(sharpe, 4),
        "avg_duration_ticks": round(float(np.mean([t.duration_ticks for t in trades])), 1),
        **reasons,
    }


# ─────────────────────────────────────────────
# Output Helpers
# ─────────────────────────────────────────────

def save_trades(trades: list[Trade], strategy: str, symbol: str) -> Path:
    """Write the trade log to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"trades_{strategy}_{symbol}.csv"
    if not trades:
        path.touch()
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(trades[0]).keys())
        writer.writeheader()
        for t in trades:
            writer.writerow(asdict(t))
    return path


def save_equity_curve(trades: list[Trade], strategy: str, symbol: str) -> Path:
    """Write cumulative equity curve to CSV."""
    path = RESULTS_DIR / f"equity_{strategy}_{symbol}.csv"
    if not trades:
        path.touch()
        return path
    equity = 0.0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trade_id", "exit_time", "pnl", "cumulative_pnl"])
        for t in trades:
            equity += t.pnl
            writer.writerow([t.trade_id, t.exit_time, round(t.pnl, 4), round(equity, 4)])
    return path


def save_summary(all_metrics: list[dict]) -> Path:
    """Write the consolidated performance summary CSV."""
    path = RESULTS_DIR / "summary.csv"
    if not all_metrics:
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    return path


def print_summary_table(all_metrics: list[dict]) -> None:
    """Pretty-print results to console."""
    log.info("═" * 80)
    log.info("BACKTEST RESULTS SUMMARY")
    log.info("  Stake: $%.2f  |  Multiplier: %dx  |  SL: $%.2f  |  TP: $%.2f",
             STAKE, MULTIPLIER, STOP_LOSS, TAKE_PROFIT)
    log.info("─" * 80)
    log.info("  %-15s %-17s %6s %8s %9s %8s %9s %8s",
             "Symbol", "Strategy", "Trades", "WinRate", "TotalPnL", "ProfFact", "MaxDD", "Sharpe")
    log.info("─" * 80)
    for m in all_metrics:
        if m.get("total_trades", 0) == 0:
            log.info("  %-15s %-17s  NO TRADES", m["symbol"], m["strategy"])
            continue
        log.info(
            "  %-15s %-17s %6d %7.1f%% %+9.2f %8.3f %9.2f %8.3f",
            m["symbol"], m["strategy"],
            m["total_trades"],
            m["win_rate_pct"],
            m["total_pnl"],
            m["profit_factor"],
            m["max_drawdown"],
            m["sharpe_approx"],
        )
    log.info("═" * 80)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_features(symbol: str) -> pd.DataFrame:
    """Load the v2 feature CSV (falls back to v1 if v2 not present)."""
    v2 = INPUT_DIR / f"{symbol}_features_v2.csv"
    v1 = INPUT_DIR / f"{symbol}_features.csv"
    path = v2 if v2.exists() else v1
    if not path.exists():
        raise FileNotFoundError(f"No feature file found for {symbol} in {INPUT_DIR}/")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    log.info("Loaded %-6s → %d rows × %d cols from %s", symbol, len(df), len(df.columns), path.name)
    return df


# ─────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────

def run_all() -> None:
    log.info("═" * 80)
    log.info("Deriv Multiplier Backtesting Engine")
    log.info("Symbols    : %s", ", ".join(SYMBOLS))
    log.info("Strategies : %s", ", ".join(STRATEGIES.keys()))
    log.info("Stake $%.2f  |  Multiplier %dx  |  SL $%.2f  |  TP $%.2f  |  MaxTicks %d",
             STAKE, MULTIPLIER, STOP_LOSS, TAKE_PROFIT, MAX_TRADE_TICKS)
    log.info("═" * 80)

    all_metrics: list[dict] = []

    for symbol in SYMBOLS:
        try:
            df = load_features(symbol)
        except FileNotFoundError as e:
            log.error("%s", e)
            continue

        for strategy_name in STRATEGIES:
            cfg = BacktestConfig(
                symbol        = symbol,
                strategy_name = strategy_name,
                stake         = STAKE,
                multiplier    = MULTIPLIER,
                stop_loss     = STOP_LOSS,
                take_profit   = TAKE_PROFIT,
                max_trade_ticks = MAX_TRADE_TICKS,
            )

            log.info("  Running %-17s on %s …", strategy_name, symbol)
            trades  = run_backtest(df, cfg)
            metrics = compute_metrics(trades, symbol, strategy_name)
            all_metrics.append(metrics)

            save_trades(trades, strategy_name, symbol)
            save_equity_curve(trades, strategy_name, symbol)

            if trades:
                log.info(
                    "    → %d trades | WR=%.1f%% | PnL=$%.2f | PF=%.3f | MaxDD=$%.2f",
                    metrics["total_trades"],
                    metrics["win_rate_pct"],
                    metrics["total_pnl"],
                    metrics["profit_factor"],
                    metrics["max_drawdown"],
                )
            else:
                log.info("    → No trades generated")

    save_summary(all_metrics)
    print_summary_table(all_metrics)
    log.info("Output saved to: %s/", RESULTS_DIR)


if __name__ == "__main__":
    run_all()
