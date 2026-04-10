# Deriv AI Trading System v2

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API token in config.py
#    API_TOKEN = "your_demo_token_here"
#    Get it from: app.deriv.com/account/api-token (Read + Trade scope)

# 3. Collect data (run overnight for best results)
python run.py collect

# 4. Process into features
python run.py pipeline

# 5. Run backtests (rule-based)
python run.py backtest

# 6. Train XGBoost models
python run.py train

# 7. Start live demo trading
python run.py trade
```

## Key Fixes from v1

| Issue | v1 | v2 |
|-------|----|----|
| CSV crashes on R_10/R_50/R_75/R_100 | C parser crashes on malformed rows | python engine + on_bad_lines='skip' |
| Doubled-year timestamps (2026-2026-...) | Crash | Auto-repaired |
| Data dropped on bad prices | Drop entire row | Forward-fill price |
| Trade placement fails | Wrong parameters format | Fixed limit_order structure |
| WebSocket crashes every 5 min | Single recv loop | Background loop with req_id routing |
| Mean reversion losing money | Signal direction inverted | Fixed: z<0 → BUY, z>0 → SELL |
| Model saves as _lgbm.pkl | Wrong filename | Saves as _xgb.pkl |
| No transaction costs | Ignoring spread | 0.1% spread included |
| Model AUC < 0.55 still saved | Always saves | Rejects + warns if AUC < 0.55 |
| tick_index_norm leakage | Included | Removed |
| Normalisation applied | Yes | No (trees don't need it) |

## File Structure

```
deriv_trading_bot/
├── config.py        All settings
├── features.py      Shared feature calc (pipeline + live)
├── pipeline.py      Data loading + feature engineering
├── backtest.py      Tick-by-tick simulation
├── model.py         XGBoost training + validation
├── trader.py        Live WebSocket trading bot
├── collector.py     Tick data collection
├── run.py           Entry point
├── requirements.txt
├── data/            Raw tick CSVs
├── processed/       Feature CSVs
├── models/          Trained models (*_xgb.pkl)
├── results/         Trade logs and summaries
└── logs/            System logs
```
