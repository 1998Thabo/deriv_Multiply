# Deriv Tick Data Collector

Collects historical tick data from the Deriv WebSocket API for Volatility indices (R_10, R_25, R_50, R_75, R_100) and saves it to CSV files.

---

## Project Structure

```
deriv_collector/
├── data_collector.py   ← Main script
├── requirements.txt    ← Python dependencies
├── collector.log       ← Auto-created when script runs
└── data/               ← Auto-created; one CSV per symbol
    ├── R_10.csv
    ├── R_25.csv
    ├── R_50.csv
    ├── R_75.csv
    └── R_100.csv
```

---

## Setup

### 1. Python version
Requires **Python 3.11+** (uses `list[dict] | None` type hint syntax).

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Collector

```bash
python data_collector.py
```

The script will:
- Connect to `wss://ws.derivws.com/websockets/v3`
- Collect all 5 symbols **concurrently** (parallel WebSocket connections)
- Fetch 5,000 ticks per batch
- Append data to `data/<SYMBOL>.csv`
- Reconnect automatically if the connection drops
- Stop when each symbol reaches the configured tick target

---

## Configuration

All settings are at the top of `data_collector.py`:

| Variable | Default | Description |
|---|---|---|
| `TICKS_PER_BATCH` | `5000` | Ticks fetched per API call |
| `BATCH_DELAY_SECONDS` | `1.5` | Pause between batches (rate limit safety) |
| `RECONNECT_DELAY_SECONDS` | `5` | Base wait before reconnecting |
| `MAX_RECONNECT_ATTEMPTS` | `10` | Max retries per symbol before giving up |
| `DATA_DIR` | `data/` | Output folder |
| `target_ticks` (in `main()`) | `50_000` | Set to `0` to run indefinitely |

### Run indefinitely (build a large dataset over time)
In `data_collector.py`, find `main()` and change:
```python
target_ticks = 0   # 0 = run forever
```

Then run it as a background process:
```bash
# Linux / macOS
nohup python data_collector.py &

# Windows (PowerShell)
Start-Process python -ArgumentList "data_collector.py" -NoNewWindow
```

---

## Output Format

Each CSV file has 3 columns:

```
timestamp,price,symbol
2024-01-15T08:30:00+00:00,1234.56,R_50
2024-01-15T08:30:01+00:00,1234.61,R_50
...
```

- `timestamp` — ISO 8601 UTC
- `price` — tick price (float)
- `symbol` — Deriv symbol name

---

## Logs

- Console output (live)
- `collector.log` file (persisted)

Sample output:
```
2024-01-15 08:30:00  [INFO    ]  ▶  R_50 | existing rows: 0
2024-01-15 08:30:00  [INFO    ]  Connecting WebSocket for R_50 …
2024-01-15 08:30:01  [INFO    ]  Connected for R_50
2024-01-15 08:30:02  [INFO    ]    R_50 | batch: +5000 ticks  |  total stored: 5000
```

---

## Notes

- The Deriv API does **not** require authentication for `ticks_history` on synthetic indices.
- Each batch request returns the latest N ticks — overlapping ticks between batches are possible. For deduplication in Phase 2, sort by timestamp and drop duplicates.
- The API rate-limits aggressive polling; `BATCH_DELAY_SECONDS = 1.5` is safe.
- No trading logic is implemented here — this is data collection only.
