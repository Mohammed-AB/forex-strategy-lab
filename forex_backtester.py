"""
Deep Forex Multi-Strategy Backtester V2 — Vectorized + Walk-Forward + Monte Carlo
10 strategies × fine grids × 5 pairs × 5 sessions × 16 filter combos.
Indicators: VWAP, Ichimoku, H1 EMA, MACD, ATR, RSI, Bollinger, etc.
Dependencies: pandas, numpy only.

Run:
  python3 forex_backtester.py              # full grid (long run; sweeps all pairs/sessions)
  python3 forex_backtester.py --quick      # smaller train slice for smoke tests (~10–12 min)
"""

from __future__ import annotations

import ast
import os
import sys
import warnings
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ACCOUNT = 550.0
LEVERAGE = 33
MAX_CONCURRENT = 2
DAILY_LOSS_PCT = 0.28
LOT = 100_000

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "AUDUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}

SESSIONS = {
    "Asian": (0, 8),
    "London": (8, 16),
    "NY": (13, 21),
    "Overlap": (13, 16),
    "All": (0, 24),
}

TRAIN_END = pd.Timestamp("2026-01-01 00:00:00")
WF_TOP_N = 1000
MC_SHUFFLES = 10_000
MC_TOP = 10
QUICK_RUN = False  # set True in main() when --quick
MIN_TRADES_REPORT = 20
MIN_TRADES_SWEEP = 5

# Fine grids (plan)
GRID_EMA_FAST = [8, 10, 12, 15, 18, 20, 25, 30]
GRID_EMA_SLOW = [35, 40, 50, 60, 80, 100]
GRID_ATR_MULT = [0.3, 0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
GRID_RR = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
GRID_RSI_HIGH = [65, 70, 72, 75, 78, 80, 85]
GRID_RSI_LOW = [15, 20, 22, 25, 28, 30, 35]


def session_mask(hours: np.ndarray, sess: str) -> np.ndarray:
    s, e = SESSIONS[sess]
    if s < e:
        return (hours >= s) & (hours < e)
    return (hours >= s) | (hours < e)


# ═══════════════════════════════════════════════════════════════════════════════
# Indicators (VWAP, Ichimoku, H1 EMA, MACD, etc.)
# ═══════════════════════════════════════════════════════════════════════════════


def add_indicators(df: pd.DataFrame) -> None:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    vol = df["volume"].replace(0, 1.0)

    for p in (8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100):
        df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()
    df["atr_median"] = df["atr"].rolling(500, min_periods=14).median()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_up"] = sma20 + 2 * std20
    df["bb_lo"] = sma20 - 2 * std20
    df["bb_mid"] = sma20
    df["bb_w"] = (df["bb_up"] - df["bb_lo"]) / sma20.replace(0, np.nan)

    df["body"] = (c - o).abs()
    df["uwick"] = h - pd.concat([c, o], axis=1).max(axis=1)
    df["lwick"] = pd.concat([c, o], axis=1).min(axis=1) - l
    df["crange"] = h - l
    df["hour"] = df["datetime"].dt.hour.astype(np.int32)
    df["minute"] = df["datetime"].dt.minute.astype(np.int32)
    df["dow"] = df["datetime"].dt.dayofweek.astype(np.int32)
    df["date"] = df["datetime"].dt.date

    # MACD (histogram for filters)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - sig

    # Daily VWAP: cumulative (close * vol) / vol per calendar day
    df["_cv"] = c * vol
    df["_cs"] = df.groupby(df["date"])["_cv"].cumsum()
    df["_vs"] = vol.groupby(df["date"]).cumsum()
    df["vwap"] = df["_cs"] / df["_vs"]
    df["vwap_dev"] = c - df["vwap"]
    df["vwap_std"] = df.groupby(df["date"])["vwap_dev"].transform(lambda x: x.rolling(20, min_periods=5).std())

    df.drop(columns=["_cv", "_cs", "_vs"], inplace=True, errors="ignore")

    # Ichimoku (standard periods)
    hi9 = h.rolling(9).max()
    lo9 = l.rolling(9).min()
    df["ichi_tenkan"] = (hi9 + lo9) / 2
    hi26 = h.rolling(26).max()
    lo26 = l.rolling(26).min()
    df["ichi_kijun"] = (hi26 + lo26) / 2
    hi52 = h.rolling(52).max()
    lo52 = l.rolling(52).min()
    df["ichi_senkou_b"] = (hi52 + lo52) / 2
    df["ichi_senkou_a"] = (df["ichi_tenkan"] + df["ichi_kijun"]) / 2
    cloud_top = pd.concat([df["ichi_senkou_a"], df["ichi_senkou_b"]], axis=1).max(axis=1)
    cloud_bot = pd.concat([df["ichi_senkou_a"], df["ichi_senkou_b"]], axis=1).min(axis=1)
    df["ichi_cloud_top"] = cloud_top
    df["ichi_cloud_bot"] = cloud_bot

    # H1 resample → EMA alignment (multi-timeframe)
    g = df.sort_values("datetime").copy()
    g = g.set_index("datetime")
    h1 = pd.DataFrame()
    h1["close"] = g["close"].resample("1h").last()
    h1["ema8_h1"] = h1["close"].ewm(span=8, adjust=False).mean()
    h1["ema21_h1"] = h1["close"].ewm(span=21, adjust=False).mean()
    h1 = h1.dropna(subset=["close"]).reset_index()
    base = df.sort_values("datetime").reset_index(drop=True)
    merged = pd.merge_asof(base[["datetime"]], h1[["datetime", "ema8_h1", "ema21_h1"]], on="datetime", direction="backward")
    df["ema8_h1"] = merged["ema8_h1"].values
    df["ema21_h1"] = merged["ema21_h1"].values

    # Session open ranges (London 08:00–08:14 first 3 M5 bars; NY 13:00–13:14)
    _add_session_open_ranges(df)


def _add_session_open_ranges(df: pd.DataFrame) -> None:
    n = len(df)
    lon_hi = np.full(n, np.nan)
    lon_lo = np.full(n, np.nan)
    ny_hi = np.full(n, np.nan)
    ny_lo = np.full(n, np.nan)

    hi = df["high"].values
    lo = df["low"].values
    ddates = df["date"].values
    hrs = df["hour"].values
    mins = df["minute"].values

    # group by date
    from collections import defaultdict

    by_date: Dict[Any, List[int]] = defaultdict(list)
    for i in range(n):
        by_date[ddates[i]].append(i)

    for d, idxs in by_date.items():
        # London 8:00–8:14 → first 3 five-minute bars
        lidx = [i for i in idxs if hrs[i] == 8 and mins[i] <= 10]
        if len(lidx) >= 1:
            take = lidx[:3]
            rh, rl = hi[take].max(), lo[take].min()
            for i in idxs:
                if hrs[i] > 8 or (hrs[i] == 8 and mins[i] >= 15):
                    lon_hi[i], lon_lo[i] = rh, rl
        # NY 13:00–13:14
        nidx = [i for i in idxs if hrs[i] == 13 and mins[i] <= 10]
        if len(nidx) >= 1:
            take = nidx[:3]
            rh, rl = hi[take].max(), lo[take].min()
            for i in idxs:
                if hrs[i] > 13 or (hrs[i] == 13 and mins[i] >= 15):
                    ny_hi[i], ny_lo[i] = rh, rl

    df["lon_open_hi"] = lon_hi
    df["lon_open_lo"] = lon_lo
    df["ny_open_hi"] = ny_hi
    df["ny_open_lo"] = ny_lo


def load_data(filepath: str) -> Dict[str, pd.DataFrame]:
    print(f"Loading {filepath}...")
    raw = pd.read_csv(filepath, parse_dates=["datetime"])
    pairs: Dict[str, pd.DataFrame] = {}
    for name, g in raw.groupby("pair"):
        g = g.sort_values("datetime").reset_index(drop=True).copy()
        add_indicators(g)
        pairs[name] = g
        print(f"  {name}: {len(g):,} candles  {g['datetime'].iloc[0]} → {g['datetime'].iloc[-1]}")
    return pairs


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[df["datetime"] < TRAIN_END].copy().reset_index(drop=True)
    te = df[df["datetime"] >= TRAIN_END].copy().reset_index(drop=True)
    return tr, te


# ═══════════════════════════════════════════════════════════════════════════════
# Filter masks (4 bits → 16 combos). Bit ON = apply that filter.
# ═══════════════════════════════════════════════════════════════════════════════


def build_base_filter_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    rsi = df["rsi"].values
    cr = df["crange"].values
    atr = df["atr"].values
    mh = df["macd_hist"].values
    med = df["atr_median"].values

    rsi_ok = (rsi >= 35) & (rsi <= 65) & np.isfinite(rsi)
    macd_long_ok = mh > 0
    macd_short_ok = mh < 0
    min_atr_ok = atr >= med
    spread_ok = cr >= (0.3 * atr)

    return {
        "rsi_ok": rsi_ok.astype(np.bool_),
        "macd_long_ok": macd_long_ok.astype(np.bool_),
        "macd_short_ok": macd_short_ok.astype(np.bool_),
        "min_atr_ok": min_atr_ok.astype(np.bool_),
        "spread_ok": spread_ok.astype(np.bool_),
    }


def apply_filter_bits(
    direction: np.ndarray,
    filt: Dict[str, np.ndarray],
    filter_bits: int,
) -> np.ndarray:
    """filter_bits: bit0=RSI, bit1=MACD direction, bit2=min ATR, bit3=spread proxy."""
    out = direction.copy()
    if filter_bits == 0:
        return out

    if filter_bits & 1:
        rsi_ok = filt["rsi_ok"]
        out = np.where(rsi_ok | (out == 0), out, 0)
    if filter_bits & 2:
        ml, ms = filt["macd_long_ok"], filt["macd_short_ok"]
        good = (out == 0) | ((out == 1) & ml) | ((out == -1) & ms)
        out = np.where(good, out, 0)
    if filter_bits & 4:
        ma = filt["min_atr_ok"]
        out = np.where(ma | (out == 0), out, 0)
    if filter_bits & 8:
        sp = filt["spread_ok"]
        out = np.where(sp | (out == 0), out, 0)
    return out


def filter_bits_label(bits: int) -> str:
    parts = []
    if bits & 1:
        parts.append("RSI")
    if bits & 2:
        parts.append("MACD")
    if bits & 4:
        parts.append("MinATR")
    if bits & 8:
        parts.append("Spr")
    return "+".join(parts) if parts else "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Signal generators (10 strategies)
# ═══════════════════════════════════════════════════════════════════════════════


def sig_ema_scalper(df: pd.DataFrame, sess: str, ema_fast=20, ema_slow=50, atr_mult=1.5, rr=2.0):
    f, s = f"ema{ema_fast}", f"ema{ema_slow}"
    if f not in df.columns or s not in df.columns:
        return np.zeros(len(df), dtype=np.int8), np.zeros(len(df)), np.zeros(len(df))
    fast, slow, c, atr = df[f].values, df[s].values, df["close"].values, df["atr"].values
    prev_c = np.roll(c, 1)
    prev_f = np.roll(fast, 1)
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    pullback = np.abs(c - fast)
    near = pullback < atr * 0.5
    long_sig = valid & (fast > slow) & near & (c > fast) & (prev_c < prev_f)
    short_sig = valid & (fast < slow) & near & (c < fast) & (prev_c > prev_f)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_breakout(df: pd.DataFrame, sess: str, bb_width_thresh=0.004, atr_mult=2.0, rr=2.0):
    bw, atr = df["bb_w"].values, df["atr"].values
    prev_bw = np.roll(bw, 1)
    c, up, lo = df["close"].values, df["bb_up"].values, df["bb_lo"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    compressed = prev_bw < bb_width_thresh
    long_sig = valid & compressed & (c > up)
    short_sig = valid & compressed & (c < lo)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_mean_reversion(df: pd.DataFrame, sess: str, rsi_high=75, rsi_low=25, atr_mult=1.5):
    rsi, atr = df["rsi"].values, df["atr"].values
    c, mid = df["close"].values, df["bb_mid"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    short_sig = valid & (rsi > rsi_high) & (np.abs(c - mid) > atr * 0.5)
    long_sig = valid & (rsi < rsi_low) & (np.abs(mid - c) > atr * 0.5)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp_long = np.abs(mid - c)
    tp_short = np.abs(c - mid)
    tp = np.where(direction == 1, tp_long, np.where(direction == -1, tp_short, 0))
    return direction, sl, tp


def sig_news_momentum(df: pd.DataFrame, sess: str, atr_mult=3.0, rr=1.5, vol_thresh=2.0):
    atr, cr = df["atr"].values, df["crange"].values
    c = df["close"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    big_move = cr > atr * vol_thresh
    c1, c2, c3 = np.roll(c, 1), np.roll(c, 2), np.roll(c, 3)
    bullish = (c > c1) & (c > c2) & (c > c3)
    bearish = (c < c1) & (c < c2) & (c < c3)
    long_sig = valid & big_move & bullish
    short_sig = valid & big_move & bearish
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_market_maker(df: pd.DataFrame, sess: str, atr_mult_sl=0.5, atr_mult_tp=0.8, spike_thresh=1.8):
    atr, cr = df["atr"].values, df["crange"].values
    c, o = df["close"].values, df["open"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    spike = cr > atr * spike_thresh
    bullish_candle = c > o
    direction = np.where(valid & spike & bullish_candle, -1, np.where(valid & spike & ~bullish_candle, 1, 0)).astype(
        np.int8
    )
    sl = atr * atr_mult_sl
    tp = atr * atr_mult_tp
    return direction, sl, tp


def sig_failed_breakout(df: pd.DataFrame, sess: str, wick_ratio=0.6, atr_mult=1.5, rr=1.5):
    atr = df["atr"].values
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    up, lo, mid = df["bb_up"].values, df["bb_lo"].values, df["bb_mid"].values
    uw, lw, cr, body = df["uwick"].values, df["lwick"].values, df["crange"].values, df["body"].values
    valid = (
        session_mask(df["hour"].values, sess)
        & (df["dow"].values < 5)
        & (atr > 0)
        & (cr > atr * 0.5)
        & (body > 0)
    )
    fail_up = valid & ((uw / cr) > wick_ratio) & (h > up) & (c < up)
    fail_lo = valid & ((lw / cr) > wick_ratio) & (l < lo) & (c > lo)
    tp_short = np.abs(c - mid)
    tp_long = np.abs(mid - c)
    direction = np.where(fail_up & (tp_short > atr * 0.3), -1, np.where(fail_lo & (tp_long > atr * 0.3), 1, 0)).astype(
        np.int8
    )
    sl = atr * atr_mult
    tp = np.where(direction == -1, tp_short, np.where(direction == 1, tp_long, 0))
    return direction, sl, tp


def sig_vwap_reversion(df: pd.DataFrame, sess: str, std_mult=1.0, atr_mult=1.5, rr=2.0):
    c, atr = df["close"].values, df["atr"].values
    dev, vs = df["vwap_dev"].values, df["vwap_std"].values
    vs = np.where(np.isfinite(vs) & (vs > 0), vs, atr)
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    long_sig = valid & (dev < -std_mult * vs)
    short_sig = valid & (dev > std_mult * vs)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_session_open_breakout(df: pd.DataFrame, sess: str, which="London", atr_mult=1.5, rr=2.0):
    """Breakout of first 15m range for London or NY session."""
    c, atr = df["close"].values, df["atr"].values
    h, l = df["high"].values, df["low"].values
    hrs, mins = df["hour"].values, df["minute"].values
    if which == "London":
        rh, rl = df["lon_open_hi"].values, df["lon_open_lo"].values
        in_window = (hrs == 8) & (mins <= 10)
        after = (hrs > 8) | ((hrs == 8) & (mins >= 15))
    else:
        rh, rl = df["ny_open_hi"].values, df["ny_open_lo"].values
        in_window = (hrs == 13) & (mins <= 10)
        after = (hrs > 13) | ((hrs == 13) & (mins >= 15))

    base_sess = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    valid = base_sess & after & np.isfinite(rh) & np.isfinite(rl)
    long_sig = valid & (c > rh) & ~in_window
    short_sig = valid & (c < rl) & ~in_window
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_ichimoku(df: pd.DataFrame, sess: str, atr_mult=1.5, rr=2.0):
    c, atr = df["close"].values, df["atr"].values
    ten, kij = df["ichi_tenkan"].values, df["ichi_kijun"].values
    ct, cb = df["ichi_cloud_top"].values, df["ichi_cloud_bot"].values
    pt, pk = np.roll(ten, 1), np.roll(kij, 1)
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    cross_up = (pt <= pk) & (ten > kij)
    cross_dn = (pt >= pk) & (ten < kij)
    above_cloud = c > np.maximum(ct, cb)
    below_cloud = c < np.minimum(ct, cb)
    long_sig = valid & cross_up & above_cloud
    short_sig = valid & cross_dn & below_cloud
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_mtf_ema(df: pd.DataFrame, sess: str, ema_fast=20, ema_slow=50, atr_mult=1.5, rr=2.0):
    """M5 signals from EMA scalper logic only when H1 EMA8 > EMA21 (long) or opposite (short)."""
    f, s = f"ema{ema_fast}", f"ema{ema_slow}"
    if f not in df.columns or s not in df.columns:
        return np.zeros(len(df), dtype=np.int8), np.zeros(len(df)), np.zeros(len(df))
    fast, slow, c, atr = df[f].values, df[s].values, df["close"].values, df["atr"].values
    e8, e21 = df["ema8_h1"].values, df["ema21_h1"].values
    prev_c = np.roll(c, 1)
    prev_f = np.roll(fast, 1)
    h1_bull = np.isfinite(e8) & np.isfinite(e21) & (e8 > e21)
    h1_bear = np.isfinite(e8) & np.isfinite(e21) & (e8 < e21)
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    pullback = np.abs(c - fast)
    near = pullback < atr * 0.5
    long_sig = valid & h1_bull & (fast > slow) & near & (c > fast) & (prev_c < prev_f)
    short_sig = valid & h1_bear & (fast < slow) & near & (c < fast) & (prev_c > prev_f)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def _session_open_fn(df, sess, which="London", atr_mult=1.5, rr=2.0):
    return sig_session_open_breakout(df, sess, which=which, atr_mult=atr_mult, rr=rr)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation + deep metrics
# ═══════════════════════════════════════════════════════════════════════════════


def fast_simulate(
    df: pd.DataFrame,
    direction: np.ndarray,
    sl_arr: np.ndarray,
    tp_arr: np.ndarray,
    pair: str,
    max_hold: int = 12,
    return_details: bool = False,
):
    pip_val = PIP[pair]
    idxs = np.where(direction != 0)[0]
    if len(idxs) == 0:
        if return_details:
            return [], 0, 0, 0, 0, 0, 0, {}
        return [], 0, 0, 0, 0, 0, 0

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    dates = df["date"].values
    n = len(df)

    trades_pips: List[float] = []
    trade_days: List[Any] = []
    wins = 0
    last_exit = -1
    daily_loss: Dict[Any, float] = {}

    for idx in idxs:
        if idx <= last_exit or idx >= n - 2:
            continue
        d = int(direction[idx])
        entry = closes[idx]
        sl_dist = sl_arr[idx]
        tp_dist = tp_arr[idx]
        if sl_dist <= 0 or tp_dist <= 0:
            continue

        date_key = dates[idx]
        daily_loss.setdefault(date_key, 0.0)
        if daily_loss[date_key] >= ACCOUNT * DAILY_LOSS_PCT:
            continue

        if d == 1:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        exit_price = None
        hold = 0
        end_idx = min(idx + max_hold + 1, n)

        for j in range(idx + 1, end_idx):
            hold = j - idx
            if d == 1:
                if lows[j] <= sl_price:
                    exit_price = sl_price
                    break
                if highs[j] >= tp_price:
                    exit_price = tp_price
                    break
            else:
                if highs[j] >= sl_price:
                    exit_price = sl_price
                    break
                if lows[j] <= tp_price:
                    exit_price = tp_price
                    break

        if exit_price is None:
            j = min(idx + max_hold, n - 1)
            exit_price = closes[j]
            hold = j - idx

        pips = (exit_price - entry) * d / pip_val
        trades_pips.append(pips)
        trade_days.append(date_key)
        if pips > 0:
            wins += 1

        loss_usd = max(0, -pips * pip_val * LOT * 0.01)
        daily_loss[date_key] += loss_usd
        last_exit = idx + hold

    total = len(trades_pips)
    if total == 0:
        if return_details:
            return [], 0, 0, 0, 0, 0, 0, {}
        return [], 0, 0, 0, 0, 0, 0

    net_pips = float(sum(trades_pips))
    win_rate = wins / total * 100

    equity = [ACCOUNT]
    for p in trades_pips:
        equity.append(equity[-1] + p * pip_val * LOT * 0.01)
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / np.where(peak > 0, peak, 1)).max() * 100)

    returns = np.array(trades_pips) * pip_val * LOT * 0.01 / ACCOUNT
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(min(total * 4, 12_000)))

    gp = sum(p for p in trades_pips if p > 0)
    gl = abs(sum(p for p in trades_pips if p < 0))
    pf = gp / gl if gl > 0 else 0.0

    if not return_details:
        return trades_pips, net_pips, win_rate, dd, sharpe, pf, total

    extra = deep_risk_from_trades(trades_pips, trade_days, eq, pip_val, pair)
    return trades_pips, net_pips, win_rate, dd, sharpe, pf, total, extra


def daily_pnl_series(trades_pips: List[float], trade_days: List[Any], pip_val: float) -> Tuple[np.ndarray, np.ndarray]:
    if not trades_pips:
        return np.array([]), np.array([])
    from collections import defaultdict

    dsum: Dict[Any, float] = defaultdict(float)
    for d, p in zip(trade_days, trades_pips):
        dsum[d] += p * pip_val * LOT * 0.01
    days = sorted(dsum.keys())
    pnl = np.array([dsum[x] for x in days], dtype=float)
    return np.array(days, dtype=object), pnl


def deep_risk_from_trades(
    trades_pips: List[float],
    trade_days: List[Any],
    equity: np.ndarray,
    pip_val: float,
    pair: str,
) -> Dict[str, Any]:
    """Sortino, Calmar, expectancy, consecutive streaks, recovery, risk of ruin proxy."""
    n = len(trades_pips)
    if n == 0:
        return {}

    usd = np.array([p * pip_val * LOT * 0.01 for p in trades_pips])
    wins = usd > 0
    win_rate = float(wins.sum() / n)
    avg_win = float(usd[wins].mean()) if wins.any() else 0.0
    avg_loss = float(abs(usd[~wins].mean())) if (~wins).any() else 0.0
    expectancy_usd = float(usd.mean())
    expectancy_pips = float(np.mean(trades_pips))

    # Consecutive wins/losses
    streak_w = streak_l = cur_w = cur_l = 0
    for p in trades_pips:
        if p > 0:
            cur_w += 1
            cur_l = 0
            streak_w = max(streak_w, cur_w)
        elif p < 0:
            cur_l += 1
            cur_w = 0
            streak_l = max(streak_l, cur_l)
        else:
            cur_w = cur_l = 0

    # Max drawdown recovery (bars in trade index space)
    peak = np.maximum.accumulate(equity)
    dd_series = peak - equity
    max_dd_idx = int(np.argmax(dd_series))
    recovery = 0
    if max_dd_idx > 0:
        peak_before = float(equity[:max_dd_idx].max())
        for j in range(max_dd_idx, len(equity)):
            if equity[j] >= peak_before:
                recovery = j - max_dd_idx
                break

    rets = usd / ACCOUNT
    downside = rets[rets < 0]
    sortino = 0.0
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(rets.mean() / downside.std() * np.sqrt(min(n * 4, 12_000)))

    total_ret = (equity[-1] - equity[0]) / max(equity[0], 1e-9)
    max_dd_pct = float((dd_series.max() / max(peak.max(), 1e-9)) * 100) if len(equity) else 0.0
    calmar = float((total_ret * 252 / max(n / 4, 1)) / max(max_dd_pct / 100, 1e-6)) if max_dd_pct > 0 else 0.0

    # Risk of ruin (simplified binary outcome approx)
    p = max(min(win_rate, 0.999), 0.001)
    q = 1 - p
    R = avg_loss / max(avg_win, 1e-9) if avg_win > 0 else 1.0
    # Gambler's ruin style upper bound
    edge = p * avg_win - q * avg_loss
    ror_approx = float(np.exp(-2 * edge * n / max(avg_win + avg_loss, 1e-9))) if n > 0 else 1.0

    days, dpnl = daily_pnl_series(trades_pips, trade_days, pip_val)

    return {
        "max_consecutive_wins": streak_w,
        "max_consecutive_losses": streak_l,
        "recovery_candles": recovery,
        "sortino": sortino,
        "calmar": calmar,
        "expectancy_pips": expectancy_pips,
        "expectancy_usd": expectancy_usd,
        "risk_of_ruin_approx": min(ror_approx, 1.0),
        "daily_pnl_days": days,
        "daily_pnl_values": dpnl,
        "max_dd_pct": max_dd_pct,
        "trade_days": trade_days,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════


def monte_carlo_shuffle(
    trades_pips: List[float],
    trade_days: List[Any],
    pair: str,
    n_shuffle: int = MC_SHUFFLES,
) -> Dict[str, Any]:
    if not trades_pips:
        return {}
    pip_val = PIP[pair]
    arr = np.array(trades_pips, dtype=float)
    rng = np.random.default_rng(42)
    finals = []
    max_dds = []
    ruin_hits = 0
    daily_lim = ACCOUNT * DAILY_LOSS_PCT

    for _ in range(n_shuffle):
        perm = rng.permutation(len(arr))
        sh = arr[perm]
        eq = ACCOUNT
        peak = ACCOUNT
        mdd = 0.0
        dloss: Dict[Any, float] = {}
        bad_days = 0
        # approximate daily bucket by permuted order — use original day order shuffled with trades
        days_perm = np.array(trade_days, dtype=object)[perm]
        for p, day in zip(sh, days_perm):
            eq += p * pip_val * LOT * 0.01
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak * 100 if peak > 0 else 0)
            if p < 0:
                dloss[day] = dloss.get(day, 0) + abs(p * pip_val * LOT * 0.01)
                if dloss[day] >= daily_lim:
                    bad_days += 1
        finals.append(eq)
        max_dds.append(mdd)
        # ruin: 3+ consecutive days at limit (approximate on shuffled sequence)
        # count consecutive calendar days hitting limit from daily_loss map — simplified:
        if bad_days >= 3:
            ruin_hits += 1

    finals = np.array(finals)
    return {
        "mc_median_final": float(np.median(finals)),
        "mc_p5_final": float(np.percentile(finals, 5)),
        "mc_p95_final": float(np.percentile(finals, 95)),
        "mc_median_mdd": float(np.median(max_dds)),
        "mc_p5_mdd": float(np.percentile(max_dds, 5)),
        "mc_p95_mdd": float(np.percentile(max_dds, 95)),
        "ruin_prob_proxy": float(ruin_hits / n_shuffle),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy registry — fine sweeps (plan)
# ═══════════════════════════════════════════════════════════════════════════════

STRATS: Dict[str, Dict[str, Any]] = {
    "EMA Scalper": {
        "fn": sig_ema_scalper,
        "hold": 12,
        "default": {"ema_fast": 20, "ema_slow": 50, "atr_mult": 1.5, "rr": 2.0},
        "sweep": {
            "ema_fast": GRID_EMA_FAST,
            "ema_slow": GRID_EMA_SLOW,
            "atr_mult": GRID_ATR_MULT,
            "rr": GRID_RR,
        },
    },
    "Breakout": {
        "fn": sig_breakout,
        "hold": 12,
        "default": {"bb_width_thresh": 0.004, "atr_mult": 2.0, "rr": 2.0},
        "sweep": {
            "bb_width_thresh": [0.002, 0.003, 0.004, 0.005, 0.006],
            "atr_mult": [0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
            "rr": GRID_RR,
        },
    },
    "Mean Reversion": {
        "fn": sig_mean_reversion,
        "hold": 8,
        "default": {"rsi_high": 75, "rsi_low": 25, "atr_mult": 1.5},
        "sweep": {"rsi_high": GRID_RSI_HIGH, "rsi_low": GRID_RSI_LOW, "atr_mult": GRID_ATR_MULT},
    },
    "News Momentum": {
        "fn": sig_news_momentum,
        "hold": 20,
        "default": {"atr_mult": 3.0, "rr": 1.5, "vol_thresh": 2.0},
        "sweep": {
            "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "rr": GRID_RR,
            "vol_thresh": [1.25, 1.5, 2.0, 2.5, 3.0],
        },
    },
    "Market Maker": {
        "fn": sig_market_maker,
        "hold": 6,
        "default": {"atr_mult_sl": 0.5, "atr_mult_tp": 0.8, "spike_thresh": 1.8},
        "sweep": {
            "atr_mult_sl": [0.3, 0.5, 0.7, 1.0],
            "atr_mult_tp": [0.5, 0.6, 0.8, 1.0, 1.2],
            "spike_thresh": [1.25, 1.5, 1.8, 2.0, 2.5, 3.0],
        },
    },
    "Failed Breakout": {
        "fn": sig_failed_breakout,
        "hold": 10,
        "default": {"wick_ratio": 0.6, "atr_mult": 1.5, "rr": 1.5},
        "sweep": {
            "wick_ratio": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
            "atr_mult": GRID_ATR_MULT,
            "rr": GRID_RR,
        },
    },
    "VWAP Reversion": {
        "fn": sig_vwap_reversion,
        "hold": 12,
        "default": {"std_mult": 1.0, "atr_mult": 1.5, "rr": 2.0},
        "sweep": {"std_mult": [0.5, 0.75, 1.0, 1.25, 1.5], "atr_mult": GRID_ATR_MULT, "rr": GRID_RR},
    },
    "Session Open Breakout": {
        "fn": _session_open_fn,
        "hold": 16,
        "default": {"which": "London", "atr_mult": 1.5, "rr": 2.0},
        "sweep": {
            "which": ["London", "NY"],
            "atr_mult": [0.7, 1.0, 1.25, 1.5, 2.0, 2.5],
            "rr": GRID_RR,
        },
    },
    "Ichimoku Cloud": {
        "fn": sig_ichimoku,
        "hold": 16,
        "default": {"atr_mult": 1.5, "rr": 2.0},
        "sweep": {"atr_mult": GRID_ATR_MULT, "rr": GRID_RR},
    },
    "Multi-TF EMA": {
        "fn": sig_mtf_ema,
        "hold": 12,
        "default": {"ema_fast": 20, "ema_slow": 50, "atr_mult": 1.5, "rr": 2.0},
        "sweep": {"ema_fast": GRID_EMA_FAST, "ema_slow": GRID_EMA_SLOW, "atr_mult": GRID_ATR_MULT, "rr": GRID_RR},
    },
}


def run_one_backtest(
    df: pd.DataFrame,
    sname: str,
    sess: str,
    params: Dict[str, Any],
    pair: str,
    filter_bits: int,
    filt_cache: Dict[str, np.ndarray],
    hold: int,
) -> Optional[Dict[str, Any]]:
    sinfo = STRATS[sname]
    fn = sinfo["fn"]
    try:
        direction, sl, tp = fn(df, sess, **params)
    except Exception:
        return None
    direction = apply_filter_bits(direction.astype(np.int8), filt_cache, filter_bits)
    res = fast_simulate(df, direction, sl, tp, pair, hold)
    if len(res) < 7:
        return None
    trades_pips, net_pips, wr, dd, sharpe, pf, ntr = res
    if ntr < MIN_TRADES_SWEEP:
        return None
    return {
        "Strategy": sname,
        "Pair": pair,
        "Session": sess,
        "Params": str(params),
        "FilterBits": filter_bits,
        "FilterLabel": filter_bits_label(filter_bits),
        "Trades": ntr,
        "WinRate": round(wr, 2),
        "NetPips": round(net_pips, 2),
        "MaxDD": round(dd, 2),
        "Sharpe": round(sharpe, 4),
        "PF": round(pf, 4),
    }


def estimate_total_sweep() -> int:
    sess_n = len(SESSIONS)
    pair_n = 5
    filt_n = 16
    tot = 0
    for sinfo in STRATS.values():
        keys = list(sinfo["sweep"].keys())
        n = 1
        for k in keys:
            n *= len(sinfo["sweep"][k])
        tot += n * sess_n * pair_n * filt_n
    return tot


def run_train_sweep(
    pairs_train: Dict[str, pd.DataFrame],
    quick: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if quick:
        # Smaller slice: 3 pairs, 2 sessions, 16 filters, ~50 param draws per strategy
        sessions = ["London", "All"]
        pair_names = sorted(pairs_train.keys())[:3]
        total = estimate_total_sweep()
        total = total // 200 + 500
    else:
        sessions = list(SESSIONS.keys())
        pair_names = list(pairs_train.keys())
        total = estimate_total_sweep()
    print(f"\n  Train sweep estimated combos: ~{total:,} (quick={quick})")

    count = 0
    for sname, sinfo in STRATS.items():
        keys = list(sinfo["sweep"].keys())
        vals = [sinfo["sweep"][k] for k in keys]
        combos = list(product(*vals))
        if quick:
            step = max(1, len(combos) // 50)
            combos = combos[::step][:50]

        for combo in combos:
            params = dict(zip(keys, combo))
            for pname in pair_names:
                df = pairs_train[pname]
                filt = build_base_filter_arrays(df)
                for sess in sessions:
                    try:
                        direction, sl, tp = sinfo["fn"](df, sess, **params)
                    except Exception:
                        continue
                    for fb in range(16):
                        count += 1
                        if count % 2000 == 0:
                            sys.stdout.write(f"\r    [{count:,}] {sname} | {pname} | {sess} | f{fb}      ")
                            sys.stdout.flush()
                        d2 = apply_filter_bits(direction.astype(np.int8), filt, fb)
                        res = fast_simulate(df, d2, sl, tp, pname, sinfo["hold"])
                        if len(res) < 7:
                            continue
                        _, net_pips, wr, dd, sharpe, pf, ntr = res
                        if ntr < MIN_TRADES_SWEEP:
                            continue
                        rows.append(
                            {
                                "Strategy": sname,
                                "Pair": pname,
                                "Session": sess,
                                "Params": str(params),
                                "FilterBits": fb,
                                "FilterLabel": filter_bits_label(fb),
                                "Trades": ntr,
                                "WinRate": round(wr, 2),
                                "NetPips": round(net_pips, 2),
                                "MaxDD": round(dd, 2),
                                "Sharpe": round(sharpe, 4),
                                "PF": round(pf, 4),
                            }
                        )

    print(f"\r    Train sweep done: {count:,} iterations, {len(rows):,} rows with {MIN_TRADES_SWEEP}+ trades")
    return pd.DataFrame(rows)


def walk_forward_validate(
    pairs_full: Dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    top_n: int = WF_TOP_N,
) -> pd.DataFrame:
    if train_df.empty:
        return pd.DataFrame()
    top = train_df.sort_values("Sharpe", ascending=False).head(top_n)
    wf_rows = []
    for _, row in top.iterrows():
        sname = row["Strategy"]
        pair = row["Pair"]
        sess = row["Session"]
        fb = int(row["FilterBits"])
        try:
            params = ast.literal_eval(row["Params"])
        except Exception:
            continue
        df_tr, df_te = split_train_test(pairs_full[pair])
        filt_tr = build_base_filter_arrays(df_tr)
        filt_te = build_base_filter_arrays(df_te)
        sinfo = STRATS[sname]
        fn = sinfo["fn"]
        # train metrics already in row
        try:
            d_tr, sl_tr, tp_tr = fn(df_tr, sess, **params)
        except Exception:
            continue
        d_tr = apply_filter_bits(d_tr.astype(np.int8), filt_tr, fb)
        r_tr = fast_simulate(df_tr, d_tr, sl_tr, tp_tr, pair, sinfo["hold"])
        try:
            d_te, sl_te, tp_te = fn(df_te, sess, **params)
        except Exception:
            continue
        d_te = apply_filter_bits(d_te.astype(np.int8), filt_te, fb)
        r_te = fast_simulate(df_te, d_te, sl_te, tp_te, pair, sinfo["hold"])
        if len(r_tr) < 7 or len(r_te) < 7:
            continue
        _, _, _, _, sharpe_is, _, n_is = r_tr
        _, _, _, _, sharpe_oos, _, n_oos = r_te
        drop = False
        if sharpe_is > 0 and sharpe_oos < 0.5 * sharpe_is:
            drop = True
        wf_rows.append(
            {
                "Strategy": sname,
                "Pair": pair,
                "Session": sess,
                "Params": row["Params"],
                "FilterBits": fb,
                "Sharpe_IS": round(sharpe_is, 4),
                "Sharpe_OOS": round(sharpe_oos, 4),
                "Trades_IS": n_is,
                "Trades_OOS": n_oos,
                "OverfitWarn": drop,
            }
        )
    return pd.DataFrame(wf_rows)


def run_defaults_all_filters(
    pairs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for sname, sinfo in STRATS.items():
        params = sinfo["default"].copy()
        for pname, df in pairs.items():
            filt = build_base_filter_arrays(df)
            for sess in SESSIONS:
                try:
                    direction, sl, tp = sinfo["fn"](df, sess, **params)
                except Exception:
                    continue
                for fb in range(16):
                    d2 = apply_filter_bits(direction.astype(np.int8), filt, fb)
                    res = fast_simulate(df, d2, sl, tp, pname, sinfo["hold"])
                    if len(res) < 7:
                        continue
                    _, net_pips, wr, dd, sharpe, pf, ntr = res
                    if ntr < MIN_TRADES_SWEEP:
                        continue
                    rows.append(
                        {
                            "Strategy": sname,
                            "Pair": pname,
                            "Session": sess,
                            "Params": str(params),
                            "FilterBits": fb,
                            "FilterLabel": filter_bits_label(fb),
                            "Trades": ntr,
                            "WinRate": round(wr, 2),
                            "NetPips": round(net_pips, 2),
                            "MaxDD": round(dd, 2),
                            "Sharpe": round(sharpe, 4),
                            "PF": round(pf, 4),
                        }
                    )
    return pd.DataFrame(rows)


def top10_mc_and_risk(
    pairs: Dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if train_df.empty:
        return pd.DataFrame(), []
    top = train_df[train_df["Trades"] >= MIN_TRADES_REPORT].sort_values("Sharpe", ascending=False).head(MC_TOP)
    mc_results = []
    risk_rows = []

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        sname = row["Strategy"]
        pair = row["Pair"]
        sess = row["Session"]
        fb = int(row["FilterBits"])
        try:
            params = ast.literal_eval(row["Params"])
        except Exception:
            continue
        df = pairs[pair]
        filt = build_base_filter_arrays(df)
        sinfo = STRATS[sname]
        fn = sinfo["fn"]
        try:
            direction, sl, tp = fn(df, sess, **params)
        except Exception:
            continue
        direction = apply_filter_bits(direction.astype(np.int8), filt, fb)
        res = fast_simulate(df, direction, sl, tp, pair, sinfo["hold"], return_details=True)
        if len(res) < 8:
            continue
        trades_pips = res[0]
        extra = res[7]
        if isinstance(extra, dict) and extra:
            extra["Strategy"] = sname
            extra["Pair"] = pair
            extra["Rank"] = rank
            risk_rows.append(extra)

        td = list(extra.get("trade_days", [])) if isinstance(extra, dict) else []
        if len(td) != len(trades_pips):
            td = [df["date"].iloc[0]] * len(trades_pips)

        mc_n = min(500, MC_SHUFFLES) if QUICK_RUN else MC_SHUFFLES
        mc = monte_carlo_shuffle(trades_pips, td, pair, mc_n)
        mc["Strategy"] = sname
        mc["Pair"] = pair
        mc["Rank"] = rank
        mc_results.append(mc)

    return pd.DataFrame(mc_results), risk_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Report V2 — 11 sections
# ═══════════════════════════════════════════════════════════════════════════════


def report_v2(
    df_def: pd.DataFrame,
    df_sweep: pd.DataFrame,
    wf: pd.DataFrame,
    mc: pd.DataFrame,
    risk_rows: List[Dict[str, Any]],
    path: str,
) -> None:
    L: List[str] = []
    w = L.append

    w("=" * 90)
    w("  DEEP FOREX BACKTESTER V2 — COMPREHENSIVE REPORT")
    w(f"  Account: ${ACCOUNT} | Leverage: {LEVERAGE}:1 | Daily loss cap: {DAILY_LOSS_PCT*100:.0f}% of account")
    w(f"  Train (IS): data before {TRAIN_END.date()} | Test (OOS): from {TRAIN_END.date()} onward")
    w("=" * 90)

    # 1 — Strategy ranking with/without filters
    w("\n" + "─" * 90)
    w("  1. STRATEGY RANKING (10 strategies; all vs no-filter vs all-filters-on)")
    w("─" * 90)
    if len(df_def):
        base = df_def[df_def["FilterBits"] == 0].groupby("Strategy").agg(
            Trades=("Trades", "sum"),
            NetPips=("NetPips", "sum"),
            Sharpe=("Sharpe", "mean"),
            MaxDD=("MaxDD", "mean"),
        )
        allf = df_def[df_def["FilterBits"] == 15].groupby("Strategy").agg(
            Trades=("Trades", "sum"),
            NetPips=("NetPips", "sum"),
            Sharpe=("Sharpe", "mean"),
            MaxDD=("MaxDD", "mean"),
        )
        w("\n  No-filter (bits=0):")
        w(f"  {'Strategy':<22}{'Trades':>8}{'NetPips':>12}{'Sharpe':>10}{'DD%':>8}")
        w("  " + "-" * 70)
        for name in STRATS:
            if name in base.index:
                r = base.loc[name]
                w(f"  {name:<22}{int(r.Trades):>8}{r.NetPips:>12.1f}{r.Sharpe:>10.2f}{r.MaxDD:>8.1f}")
        w("\n  All filters on (bits=15):")
        w(f"  {'Strategy':<22}{'Trades':>8}{'NetPips':>12}{'Sharpe':>10}{'DD%':>8}")
        w("  " + "-" * 70)
        for name in STRATS:
            if name in allf.index:
                r = allf.loc[name]
                w(f"  {name:<22}{int(r.Trades):>8}{r.NetPips:>12.1f}{r.Sharpe:>10.2f}{r.MaxDD:>8.1f}")

    # 2 — Best filter per strategy (from sweep)
    w("\n" + "─" * 90)
    w("  2. BEST FILTER COMBINATION PER STRATEGY (by mean Sharpe on train sweep)")
    w("─" * 90)
    if len(df_sweep):
        for sn in STRATS:
            sub = df_sweep[df_sweep["Strategy"] == sn]
            if sub.empty or sub["Trades"].max() < MIN_TRADES_REPORT:
                continue
            g = sub.groupby(["FilterBits", "FilterLabel"]).agg(Sharpe=("Sharpe", "mean"), NetPips=("NetPips", "sum")).reset_index()
            best = g.loc[g["Sharpe"].idxmax()]
            w(f"\n  {sn}: bits={int(best.FilterBits)} ({best.FilterLabel})  |  avg Sharpe {best.Sharpe:.3f}  |  sum pips {best.NetPips:.1f}")

    # 3 — Best pair & session per strategy
    w("\n" + "─" * 90)
    w("  3. BEST PAIR AND SESSION PER STRATEGY (train sweep, Sharpe≥ viable)")
    w("─" * 90)
    if len(df_sweep):
        for sn in STRATS:
            sub = df_sweep[(df_sweep["Strategy"] == sn) & (df_sweep["Trades"] >= MIN_TRADES_REPORT)]
            if sub.empty:
                continue
            best = sub.loc[sub["Sharpe"].idxmax()]
            w(f"\n  {sn}:")
            w(f"    Best pair: {best.Pair} | Session: {best.Session} | Filter: {best.FilterLabel}")
            w(f"    Params: {best.Params}")
            w(f"    Trades {int(best.Trades)} | Net {best.NetPips:+.1f} pips | Sharpe {best.Sharpe:.2f} | DD {best.MaxDD:.1f}%")

    # 4 — Optimal parameters (fine grid)
    w("\n" + "─" * 90)
    w("  4. OPTIMAL PARAMETERS (fine grid — best Sharpe per strategy on train)")
    w("─" * 90)
    if len(df_sweep):
        for sn in STRATS:
            sub = df_sweep[(df_sweep["Strategy"] == sn) & (df_sweep["Trades"] >= MIN_TRADES_REPORT)]
            if sub.empty:
                continue
            best = sub.loc[sub["Sharpe"].idxmax()]
            w(f"\n  {sn} — {best.Pair} / {best.Session} / {best.FilterLabel}")
            w(f"    {best.Params}")
            w(f"    Trades {int(best.Trades)} | Win {best.WinRate:.1f}% | Net {best.NetPips:+.1f} | Sharpe {best.Sharpe:.2f}")

    # 5 — Walk-forward
    w("\n" + "─" * 90)
    w("  5. WALK-FORWARD VALIDATION (optimize on IS Oct2023–Dec2025, test OOS Jan2026+)")
    w("─" * 90)
    if len(wf):
        w(f"\n  Top {len(wf)} configs re-evaluated on OOS slice.")
        w(f"  {'Strategy':<20}{'Pair':<8}{'Sharpe IS':>10}{'Sharpe OOS':>11}{'Tr IS':>7}{'Tr OOS':>8}{'Warn':>6}")
        w("  " + "-" * 80)
        for _, r in wf.head(30).iterrows():
            w(
                f"  {r.Strategy:<20}{r.Pair:<8}{r.Sharpe_IS:>10.3f}{r.Sharpe_OOS:>11.3f}{int(r.Trades_IS):>7}{int(r.Trades_OOS):>8}{str(r.OverfitWarn):>6}"
            )
    else:
        w("\n  (No walk-forward rows — insufficient train results.)")

    # 6 — Overfitting warnings
    w("\n" + "─" * 90)
    w("  6. OVERFITTING WARNINGS (OOS Sharpe < 50% of IS Sharpe)")
    w("─" * 90)
    if len(wf):
        bad = wf[wf["OverfitWarn"] == True]
        if len(bad):
            for _, r in bad.head(50).iterrows():
                w(f"  !! {r.Strategy} {r.Pair}  IS {r.Sharpe_IS:.3f} → OOS {r.Sharpe_OOS:.3f}")
        else:
            w("\n  No configs flagged under this rule (top slice).")
    else:
        w("\n  N/A")

    # 7 — Monte Carlo top 10
    w("\n" + "─" * 90)
    w("  7. MONTE CARLO — TOP 10 CONFIGS (10,000 shuffles each)")
    w("─" * 90)
    if len(mc):
        for _, r in mc.iterrows():
            w(
                f"\n  #{int(r.Rank)} {r.Strategy} | {r.Pair}"
                f"\n     median final ${r.get('mc_median_final', 0):.2f} | p5 {r.get('mc_p5_final', 0):.2f} | p95 {r.get('mc_p95_final', 0):.2f}"
                f"\n     median max DD% {r.get('mc_median_mdd', 0):.2f} | ruin-proxy P {r.get('ruin_prob_proxy', 0):.4f}"
            )
    else:
        w("\n  N/A")

    # 8 — Ruin probability
    w("\n" + "─" * 90)
    w("  8. PROBABILITY OF RUIN ANALYSIS (MC proxy + analytical approx on top configs)")
    w("─" * 90)
    if len(mc):
        w(f"\n  Mean ruin-proxy across top {len(mc)}: {mc['ruin_prob_proxy'].mean():.4f}")
    if risk_rows:
        w("\n  Analytical risk_of_ruin_approx (from trade stats):")
        for rr in risk_rows[:10]:
            w(f"    {rr.get('Strategy')} {rr.get('Pair')}: ROR≈ {rr.get('risk_of_ruin_approx', 0):.4f}")

    # 9 — Deep risk analytics
    w("\n" + "─" * 90)
    w("  9. DEEP RISK ANALYTICS (consecutive losses, recovery, Sortino, Calmar, expectancy)")
    w("─" * 90)
    if risk_rows:
        for rr in risk_rows[:15]:
            w(
                f"\n  {rr.get('Strategy')} | {rr.get('Pair')}"
                f"\n    max consec W/L: {rr.get('max_consecutive_wins', 0)}/{rr.get('max_consecutive_losses', 0)}"
                f" | recovery idx: {rr.get('recovery_candles', 0)}"
                f"\n    Sortino {rr.get('sortino', 0):.3f} | Calmar {rr.get('calmar', 0):.3f}"
                f" | E[USD]/trade {rr.get('expectancy_usd', 0):.4f} | E[pips] {rr.get('expectancy_pips', 0):.2f}"
            )
    else:
        w("\n  N/A")

    # 10 — Portfolio
    w("\n" + "─" * 90)
    w("  10. RECOMMENDED PORTFOLIO (2–3 strategies — max Sharpe blend, penalize DD)")
    w("─" * 90)
    if len(df_sweep):
        best_per = {}
        for sn in STRATS:
            sub = df_sweep[(df_sweep["Strategy"] == sn) & (df_sweep["Trades"] >= MIN_TRADES_REPORT)]
            if not sub.empty:
                best_per[sn] = sub.loc[sub["Sharpe"].idxmax()]
        best_combo = None
        best_score = -999.0
        items = list(best_per.items())
        for size in (2, 3):
            for combo in combinations(items, size):
                sharpe = float(np.mean([r.Sharpe for _, r in combo]))
                dd = max(r.MaxDD for _, r in combo)
                pips = sum(r.NetPips for _, r in combo)
                score = sharpe * 2 + pips / 100.0 - dd * 0.5
                if score > best_score:
                    best_score, best_combo = score, combo
        if best_combo:
            w("\n  Selected combo:")
            for name, r in best_combo:
                w(f"    → {name} | {r.Pair} | {r.Session} | {r.FilterLabel}")
                w(f"      {r.Params}")
                w(f"      Sharpe {r.Sharpe:.2f} | DD {r.MaxDD:.1f}% | Net {r.NetPips:+.1f} pips")

    # 11 — Bot configs top 3
    w("\n" + "─" * 90)
    w("  11. COPY-PASTE BOT CONFIG — TOP 3 STRATEGIES (by train Sharpe)")
    w("─" * 90)
    if len(df_sweep):
        top3 = df_sweep[df_sweep["Trades"] >= MIN_TRADES_REPORT].sort_values("Sharpe", ascending=False).head(3)
        for i, (_, r) in enumerate(top3.iterrows(), 1):
            w(
                f"""
  # --- Rank {i} ---
  STRATEGY = "{r.Strategy}"
  PAIR = "{r.Pair}"
  SESSION = "{r.Session}"
  FILTER_BITS = {int(r.FilterBits)}  # {r.FilterLabel}
  PARAMS = {r.Params}
  # Stats: trades={int(r.Trades)} win={r.WinRate:.1f}% net_pips={r.NetPips:+.1f} sharpe={r.Sharpe:.2f} max_dd={r.MaxDD:.1f}%
"""
            )

    txt = "\n".join(L)
    with open(path, "w") as f:
        f.write(txt)
    print(f"\n{'=' * 60}\n  V2 Report saved → {path}\n{'=' * 60}\n")
    print(txt[:8000] + ("\n\n... [truncated print] ...\n" if len(txt) > 8000 else ""))


def main() -> None:
    global QUICK_RUN
    quick = "--quick" in sys.argv
    QUICK_RUN = quick

    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forex_5min_combined.txt")
    if not os.path.exists(data_file):
        print(f"Not found: {data_file}")
        sys.exit(1)

    print("=" * 60)
    print("  DEEP FOREX BACKTESTER V2")
    print("=" * 60)

    pairs_full = load_data(data_file)
    pairs_train: Dict[str, pd.DataFrame] = {}
    for k, df in pairs_full.items():
        tr, _ = split_train_test(df)
        pairs_train[k] = tr

    print("\nPhase 1 — default params × 16 filters × train slice …")
    df_def = run_defaults_all_filters(pairs_train)
    print(f"  Rows with {MIN_TRADES_SWEEP}+ trades: {len(df_def):,}")

    print("\nPhase 2 — fine parameter sweep on train (Oct 2023 – Dec 2025) …")
    df_sweep = run_train_sweep(pairs_train, quick=quick)
    print(f"  Sweep rows: {len(df_sweep):,}")

    top_n = 50 if quick else WF_TOP_N
    print(f"\nPhase 3 — walk-forward (top {top_n} configs) …")
    wf = walk_forward_validate(pairs_full, df_sweep, top_n=top_n)
    print(f"  WF rows: {len(wf):,}")

    print(f"\nPhase 4 — Monte Carlo ({MC_SHUFFLES if not quick else min(500, MC_SHUFFLES)} shuffles) + deep risk on top {MC_TOP} …")
    mc, risk_rows = top10_mc_and_risk(pairs_full, df_sweep)
    print(f"  MC configs: {len(mc):,} | risk profiles: {len(risk_rows):,}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_report.txt")
    report_v2(df_def, df_sweep, wf, mc, risk_rows, out)


if __name__ == "__main__":
    main()
