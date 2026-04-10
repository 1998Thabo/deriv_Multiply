"""
model.py — XGBoost training with validation gate.
Models rejected if test AUC < MIN_AUC_THRESHOLD.
Uses early stopping, temporal CV, and feature importance logging.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from config import (
    SYMBOLS, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    XGB_PARAMS, TEST_SIZE, MIN_AUC_THRESHOLD, EARLY_STOPPING_ROUNDS,
    MIN_CONFIDENCE,
)
from features import FEATURE_NAMES

log = logging.getLogger(__name__)

# Columns never used as features
_EXCLUDE = {"timestamp", "symbol", "price", "target", "future_return"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns present in df that are numeric and not excluded."""
    return [c for c in FEATURE_NAMES if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])]


def temporal_split(df: pd.DataFrame):
    """Chronological 80/20 split. NEVER shuffle time-series data."""
    n_test  = int(len(df) * TEST_SIZE)
    n_train = len(df) - n_test
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def train(symbol: str, df: pd.DataFrame) -> tuple:
    """
    Train XGBoost with TimeSeriesSplit CV and early stopping.
    Returns (model, feature_cols, test_df, metrics) or raises ValueError if AUC too low.
    """
    feat_cols = get_feature_cols(df)
    train_df, test_df = temporal_split(df)

    X_train = train_df[feat_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df["target"].values

    log.info("  %s | train=%d test=%d features=%d", symbol, len(train_df), len(test_df), len(feat_cols))

    # ── TimeSeriesSplit CV ──
    tscv      = TimeSeriesSplit(n_splits=3)
    cv_aucs   = []

    for fold, (tr_i, val_i) in enumerate(tscv.split(X_train)):
        m = XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        m.fit(
            X_train[tr_i], y_train[tr_i],
            eval_set=[(X_train[val_i], y_train[val_i])],
            verbose=False,
        )
        proba = m.predict_proba(X_train[val_i])[:, 1]
        auc   = roc_auc_score(y_train[val_i], proba)
        cv_aucs.append(auc)
        log.info("    Fold %d AUC: %.4f (best iter=%d)", fold + 1, auc, m.best_iteration)

    mean_cv = float(np.mean(cv_aucs))
    log.info("  Mean CV AUC: %.4f", mean_cv)

    # ── Final model on full training set ──
    # Use median best_iteration from CV folds for final n_estimators
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # ── Test set evaluation ──
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred  = (test_proba >= 0.5).astype(int)
    test_auc   = roc_auc_score(y_test, test_proba)
    test_acc   = accuracy_score(y_test, test_pred)

    log.info("  Test AUC=%.4f  Acc=%.4f", test_auc, test_acc)
    log.info("\n%s", classification_report(
        y_test, test_pred, target_names=["DOWN", "UP"], zero_division=0))

    # ── Feature importance ──
    importances = pd.Series(model.feature_importances_, index=feat_cols)
    log.info("  Top 10 features:\n%s", importances.nlargest(10).to_string())

    # ── AUC gate: reject models below threshold ──
    if test_auc < MIN_AUC_THRESHOLD:
        log.warning(
            "  %s REJECTED: test AUC %.4f < threshold %.2f — model not saved, will not trade",
            symbol, test_auc, MIN_AUC_THRESHOLD,
        )
        raise ValueError(f"{symbol}: AUC {test_auc:.4f} below threshold {MIN_AUC_THRESHOLD}")

    test_df = test_df.copy()
    test_df["ml_proba"] = test_proba

    metrics = {
        "symbol":     symbol,
        "cv_auc":     round(mean_cv, 4),
        "test_auc":   round(test_auc, 4),
        "test_acc":   round(test_acc, 4),
        "n_train":    len(train_df),
        "n_test":     len(test_df),
        "n_features": len(feat_cols),
        "passed_gate": True,
    }

    return model, feat_cols, test_df, metrics


def save_model(model, feat_cols: list[str], symbol: str) -> Path:
    """Save model to models/{symbol}_xgb.pkl using joblib."""
    path = MODELS_DIR / f"{symbol}_xgb.pkl"
    joblib.dump({"model": model, "feature_cols": feat_cols}, path)
    log.info("  Saved → %s", path)
    return path


def load_model(symbol: str) -> tuple:
    """Load XGBoost model. Returns (model, feature_cols)."""
    path = MODELS_DIR / f"{symbol}_xgb.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No model for {symbol}. Run: python run.py train")
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def make_signal_fn(model, feat_cols: list[str], min_conf: float = MIN_CONFIDENCE):
    """
    Returns signal_fn(row_dict) → ("BUY"|"SELL"|None, confidence).
    Only fires when model confidence >= min_conf on the correct side.
    """
    margin = min_conf - 0.5  # e.g. 0.60 → margin=0.10

    def signal_fn(row: dict) -> tuple:
        try:
            x     = np.array([[row.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
            proba = float(model.predict_proba(x)[0, 1])
        except Exception as e:
            log.debug("Signal inference failed: %s", e)
            return None, 0.0

        if proba >= (0.5 + margin):
            return "BUY",  round(proba, 4)
        if proba <= (0.5 - margin):
            return "SELL", round(1.0 - proba, 4)
        return None, 0.0

    return signal_fn


def run_training(symbols=None) -> dict:
    """Train models for all symbols. Skip symbols that fail the AUC gate."""
    from backtest import run_backtest, compute_metrics, save_trades, print_summary

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/model.log", encoding="utf-8"),
        ],
    )
    targets = symbols or SYMBOLS
    train_metrics_all  = []
    bt_metrics_all     = []

    for sym in targets:
        path = PROCESSED_DIR / f"{sym}_features.csv"
        if not path.exists():
            log.error("%s: no feature file. Run: python run.py pipeline", sym)
            continue

        log.info("━" * 55)
        log.info("Training XGBoost: %s", sym)
        log.info("━" * 55)

        df = pd.read_csv(path, parse_dates=["timestamp"])

        try:
            model, feat_cols, test_df, metrics = train(sym, df)
        except ValueError as e:
            log.warning("Skipping %s: %s", sym, e)
            continue
        except Exception:
            log.exception("Training failed: %s", sym)
            continue

        save_model(model, feat_cols, sym)
        train_metrics_all.append(metrics)

        # Backtest on test set only
        sig_fn  = make_signal_fn(model, feat_cols)
        trades  = run_backtest(test_df, sym, "xgb", sig_fn)
        bt_m    = compute_metrics(trades, sym, "xgb")
        bt_metrics_all.append(bt_m)
        save_trades(trades, f"xgb_{sym}")

        log.info("  Backtest → trades=%d WR=%.1f%% PnL=$%.2f PF=%.3f",
                 bt_m["total_trades"], bt_m["win_rate_pct"],
                 bt_m["total_pnl"], bt_m["profit_factor"])

    # Save training metrics
    if train_metrics_all:
        pd.DataFrame(train_metrics_all).to_csv(
            RESULTS_DIR / "train_metrics.csv", index=False)

    log.info("\n" + "═" * 55)
    log.info("TRAINING COMPLETE — BACKTEST RESULTS")
    print_summary(bt_metrics_all)

    return {"train": train_metrics_all, "backtest": bt_metrics_all}


if __name__ == "__main__":
    run_training()
