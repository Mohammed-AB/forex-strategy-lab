# 📂 Input Data Format

The price data is **not included** in this repository — only the research engine is.
Both backtesters read a single CSV/TXT file from this folder (or the repo root) and
write a plain-text report back out. Nothing touches the network.

## File location

| Script | Default file it looks for | How to override |
| --- | --- | --- |
| `forex_backtester.py` | `forex_5min_combined.txt` | edit the path in `main()` |
| `smart_backtester.py` | `all_5pairs_daily.txt` | `--data <yourfile.txt>` |

`smart_backtester.py` **auto-detects the timeframe**: if the median gap between bars
is ≥ 12 hours it treats the feed as daily (and drops session/hour logic), otherwise
it runs the full intraday session machinery.

## Expected columns

One long (stacked) CSV containing **all pairs in the same file**, with a header row:

```csv
datetime,pair,open,high,low,close,volume
2024-01-02 08:00:00,EURUSD,1.10412,1.10455,1.10398,1.10440,1832
2024-01-02 08:05:00,EURUSD,1.10440,1.10472,1.10431,1.10468,1567
2024-01-02 08:00:00,USDJPY,143.812,143.905,143.790,143.880,2410
...
```

| Column | Type | Notes |
| --- | --- | --- |
| `datetime` | timestamp | Parsed with `pandas.to_datetime`. UTC-style 5-minute candles for the intraday file. Used to derive `hour`, `minute`, `dow`, `date`, and the trading-session masks. |
| `pair` | string | One of the five supported symbols (see below). Rows are grouped by this column. |
| `open` / `high` / `low` / `close` | float | OHLC for the bar. Raw price (e.g. `1.1044`, not pips). |
| `volume` | float/int | Tick or real volume. Zeros are safely coerced to `1.0` so VWAP never divides by zero. |

## Supported pairs (pip sizes are hard-coded)

```python
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "AUDUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
```

So the file should contain candles for **EURUSD, GBPUSD, AUDUSD, EURGBP, USDJPY**.
(USDJPY uses a `0.01` pip; the other four use `0.0001`.) The cross-pair robustness
gate in `smart_backtester.py` expects all five so it can demand a strategy works on
**at least 3 of the 5**.

## Timeframe

- **`forex_backtester.py`** is built for **5-minute** candles. Indicators such as the
  London/NY session-open ranges assume intraday M5 bars, and the train/test split is
  fixed at `2026-01-01` (data before = in-sample, on/after = out-of-sample).
- **`smart_backtester.py`** accepts either intraday or daily and adapts automatically.

## Bring your own data

Any broker/exporter that can produce OHLCV per pair works — export each of the five
symbols, add the `pair` column, concatenate into one CSV with the header above, and
point the script at it. Roughly two-plus years of history is recommended so the
rolling walk-forward (3-year train / 1-year test windows) has room to slide.
