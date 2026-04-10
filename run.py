"""
run.py — Master entry point for the Deriv Trading System.

Usage:
    python run.py collect    Collect live tick data (run overnight)
    python run.py pipeline   Process raw CSVs into features
    python run.py backtest   Rule-based strategy backtests
    python run.py train      Train XGBoost models
    python run.py trade      Start live demo trading
    python run.py full       pipeline + backtest + train (no trading)
"""

import logging
import sys

USAGE = """
╔═══════════════════════════════════════════════════════╗
║         DERIV AI TRADING SYSTEM v2                   ║
╠═══════════════════════════════════════════════════════╣
║  python run.py collect    Collect tick data          ║
║  python run.py pipeline   Process data → features    ║
║  python run.py backtest   Rule-based backtests       ║
║  python run.py train      Train XGBoost models       ║
║  python run.py trade      Live demo trading          ║
║  python run.py full       Pipeline + Backtest + Train║
╠═══════════════════════════════════════════════════════╣
║  RECOMMENDED FIRST-TIME ORDER:                       ║
║    1. collect (run overnight for 10k+ ticks)         ║
║    2. pipeline                                       ║
║    3. backtest                                       ║
║    4. train                                          ║
║    5. trade                                          ║
╚═══════════════════════════════════════════════════════╝
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/system.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def cmd_collect():
    import asyncio
    from collector import run_collector
    log.info("MODE: Collect")
    asyncio.run(run_collector())


def cmd_pipeline():
    from pipeline import run_pipeline
    log.info("MODE: Pipeline")
    run_pipeline()


def cmd_backtest():
    from backtest import run_all_backtests
    log.info("MODE: Backtest")
    run_all_backtests()


def cmd_train():
    from model import run_training
    log.info("MODE: Train XGBoost")
    run_training()


def cmd_trade():
    import asyncio
    from trader import run_trader
    log.info("MODE: Demo Trading")
    asyncio.run(run_trader())


def cmd_full():
    log.info("MODE: Full (pipeline + backtest + train)")
    cmd_pipeline()
    cmd_backtest()
    cmd_train()


COMMANDS = {
    "collect":  cmd_collect,
    "pipeline": cmd_pipeline,
    "backtest": cmd_backtest,
    "train":    cmd_train,
    "trade":    cmd_trade,
    "full":     cmd_full,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(USAGE)
        sys.exit(0)
    try:
        COMMANDS[sys.argv[1]]()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
