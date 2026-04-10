"""
config.py — Central configuration for the Deriv Trading System.
ALL settings live here. Edit this file only.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR   = BASE_DIR / "results"
MODELS_DIR    = BASE_DIR / "models"
LOGS_DIR      = BASE_DIR / "logs"

for _d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Symbols ────────────────────────────────────────────────────────
SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

# ── Deriv API ──────────────────────────────────────────────────────
APP_ID    = "1089"
WS_URL    = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
API_TOKEN = ""           # ← paste your DEMO token here

# ── Trading Parameters ─────────────────────────────────────────────
STAKE              = 10.0
MULTIPLIER         = 10
STOP_LOSS          = 3.0
TAKE_PROFIT        = 5.0
TRANSACTION_COST   = 0.001   # 0.1% spread per trade (both sides)
MAX_TRADE_TICKS    = 150
MAX_TRADES_SESSION = 20
MIN_CONFIDENCE     = 0.60    # raised from 0.55 — only high-conviction trades

# ── Feature Engineering ────────────────────────────────────────────
PREDICTION_HORIZON = 10          # ticks ahead to predict
LIVE_BUFFER_SIZE   = 200         # minimum ticks before live signals fire

# Essential feature windows only (avoids overfitting)
SMA_WINDOWS     = [20, 50]
MOMENTUM_PERIOD = 10
ZSCORE_WINDOW   = 50
VOL_WINDOW      = 20

# ── XGBoost hyperparameters ────────────────────────────────────────
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        4,           # conservative — avoids overfitting on small data
    "learning_rate":    0.05,
    "n_estimators":     100,
    "min_child_weight": 50,          # high — forces large leaf support
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

# Model quality gate — reject models below this AUC (don't save, don't trade)
MIN_AUC_THRESHOLD  = 0.55
EARLY_STOPPING_ROUNDS = 20

# ── Training split ─────────────────────────────────────────────────
TEST_SIZE      = 0.20            # last 20% of data is test set
MIN_TRAIN_ROWS = 10_000          # minimum clean rows required for training

# ── Data Collection ────────────────────────────────────────────────
COLLECTION_DELAY_SEC  = 2.0      # rate-limit pause between API requests
RECONNECT_BASE_SEC    = 2.0      # base reconnect wait (doubles each attempt)
MAX_RECONNECT_RETRIES = 10
