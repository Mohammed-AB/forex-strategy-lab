"""
Smart Forex Backtester — Optuna-Powered Strategy Discovery
Uses Bayesian optimization to search millions of strategy configs in minutes.
Logs every trial so you see the "learning journey" from bad → good.

Run:
  python3 smart_backtester.py                          # default: daily data
  python3 smart_backtester.py --data forex_5min_combined.txt --trials 500
  python3 smart_backtester.py --trials 2000            # deeper search
"""

from __future__ import annotations

import ast
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ACCOUNT = 550.0
LEVERAGE = 33
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

DEFAULT_TRIALS = 100_000
MC_SHUFFLES = 3000
MC_TOP = 10
WF_SPLIT = 0.75
MIN_TRADES = 30
CROSS_PAIR_MIN_PAIRS = 3  # must work on at least 3 of 5 pairs
ROLLING_WF_TRAIN_YEARS = 3
ROLLING_WF_TEST_YEARS = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def session_mask(hours: np.ndarray, sess: str) -> np.ndarray:
    s, e = SESSIONS[sess]
    if s < e:
        return (hours >= s) & (hours < e)
    return (hours >= s) | (hours < e)


# ═══════════════════════════════════════════════════════════════════════════════
# Indicators
# ═══════════════════════════════════════════════════════════════════════════════


def add_indicators(df: pd.DataFrame, is_daily: bool = False) -> None:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    vol = df["volume"].replace(0, 1.0)

    for p in (5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200):
        df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()
    df["atr_median"] = df["atr"].rolling(100, min_periods=14).median()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss_s = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss_s.replace(0, np.nan))

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
    df["dow"] = df["datetime"].dt.dayofweek.astype(np.int32)
    df["date"] = df["datetime"].dt.date

    if not is_daily:
        df["hour"] = df["datetime"].dt.hour.astype(np.int32)
        df["minute"] = df["datetime"].dt.minute.astype(np.int32)
    else:
        df["hour"] = np.zeros(len(df), dtype=np.int32)
        df["minute"] = np.zeros(len(df), dtype=np.int32)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - sig

    df["_cv"] = c * vol
    df["_cs"] = df.groupby(df["date"])["_cv"].cumsum()
    df["_vs"] = vol.groupby(df["date"]).cumsum()
    df["vwap"] = df["_cs"] / df["_vs"].replace(0, 1)
    df["vwap_dev"] = c - df["vwap"]
    df.drop(columns=["_cv", "_cs", "_vs"], inplace=True, errors="ignore")

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
    ct = pd.concat([df["ichi_senkou_a"], df["ichi_senkou_b"]], axis=1).max(axis=1)
    cb = pd.concat([df["ichi_senkou_a"], df["ichi_senkou_b"]], axis=1).min(axis=1)
    df["ichi_cloud_top"] = ct
    df["ichi_cloud_bot"] = cb


def load_data(filepath: str) -> Tuple[Dict[str, pd.DataFrame], bool]:
    print(f"Loading {filepath}...")
    raw = pd.read_csv(filepath, parse_dates=["datetime"])
    is_daily = False
    if len(raw) > 0:
        diffs = raw.groupby("pair")["datetime"].diff().dropna()
        median_gap = diffs.median()
        if median_gap >= pd.Timedelta(hours=12):
            is_daily = True
    print(f"  Detected timeframe: {'daily' if is_daily else 'intraday'}")

    pairs: Dict[str, pd.DataFrame] = {}
    for name, g in raw.groupby("pair"):
        g = g.sort_values("datetime").reset_index(drop=True).copy()
        add_indicators(g, is_daily=is_daily)
        pairs[name] = g
        print(f"  {name}: {len(g):,} bars  {g['datetime'].iloc[0].date()} → {g['datetime'].iloc[-1].date()}")
    return pairs, is_daily


# ═══════════════════════════════════════════════════════════════════════════════
# Signal generators (10 strategies)
# ═══════════════════════════════════════════════════════════════════════════════


def sig_ema_crossover(df, sess, ema_fast=20, ema_slow=50, atr_mult=1.5, rr=2.0):
    f, s = f"ema{ema_fast}", f"ema{ema_slow}"
    if f not in df.columns or s not in df.columns:
        return np.zeros(len(df), dtype=np.int8), np.zeros(len(df)), np.zeros(len(df))
    fast, slow, c, atr = df[f].values, df[s].values, df["close"].values, df["atr"].values
    prev_f, prev_s = np.roll(fast, 1), np.roll(slow, 1)
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    long_sig = valid & (prev_f <= prev_s) & (fast > slow)
    short_sig = valid & (prev_f >= prev_s) & (fast < slow)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_ema_pullback(df, sess, ema_fast=20, ema_slow=50, atr_mult=1.5, rr=2.0):
    f, s = f"ema{ema_fast}", f"ema{ema_slow}"
    if f not in df.columns or s not in df.columns:
        return np.zeros(len(df), dtype=np.int8), np.zeros(len(df)), np.zeros(len(df))
    fast, slow, c, atr = df[f].values, df[s].values, df["close"].values, df["atr"].values
    prev_c = np.roll(c, 1)
    prev_f = np.roll(fast, 1)
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    near = np.abs(c - fast) < atr * 0.5
    long_sig = valid & (fast > slow) & near & (c > fast) & (prev_c < prev_f)
    short_sig = valid & (fast < slow) & near & (c < fast) & (prev_c > prev_f)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_breakout(df, sess, bb_width_thresh=0.004, atr_mult=2.0, rr=2.0):
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


def sig_mean_reversion(df, sess, rsi_high=75, rsi_low=25, atr_mult=1.5):
    rsi, atr = df["rsi"].values, df["atr"].values
    c, mid = df["close"].values, df["bb_mid"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    short_sig = valid & (rsi > rsi_high) & (np.abs(c - mid) > atr * 0.5)
    long_sig = valid & (rsi < rsi_low) & (np.abs(mid - c) > atr * 0.5)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = np.where(direction != 0, np.abs(c - mid), 0)
    return direction, sl, tp


def sig_momentum(df, sess, atr_mult=3.0, rr=1.5, vol_thresh=2.0):
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


def sig_fade(df, sess, atr_mult_sl=0.5, atr_mult_tp=0.8, spike_thresh=1.8):
    atr, cr = df["atr"].values, df["crange"].values
    c, o = df["close"].values, df["open"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    spike = cr > atr * spike_thresh
    bullish_candle = c > o
    direction = np.where(valid & spike & bullish_candle, -1, np.where(valid & spike & ~bullish_candle, 1, 0)).astype(np.int8)
    sl = atr * atr_mult_sl
    tp = atr * atr_mult_tp
    return direction, sl, tp


def sig_failed_breakout(df, sess, wick_ratio=0.6, atr_mult=1.5, rr=1.5):
    atr = df["atr"].values
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    up, lo, mid = df["bb_up"].values, df["bb_lo"].values, df["bb_mid"].values
    uw, lw, cr, body = df["uwick"].values, df["lwick"].values, df["crange"].values, df["body"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0) & (cr > atr * 0.5) & (body > 0)
    fail_up = valid & ((uw / np.where(cr > 0, cr, 1)) > wick_ratio) & (h > up) & (c < up)
    fail_lo = valid & ((lw / np.where(cr > 0, cr, 1)) > wick_ratio) & (l < lo) & (c > lo)
    tp_short = np.abs(c - mid)
    tp_long = np.abs(mid - c)
    direction = np.where(fail_up & (tp_short > atr * 0.3), -1, np.where(fail_lo & (tp_long > atr * 0.3), 1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = np.where(direction == -1, tp_short, np.where(direction == 1, tp_long, 0))
    return direction, sl, tp


def sig_vwap_reversion(df, sess, std_mult=1.0, atr_mult=1.5, rr=2.0):
    c, atr = df["close"].values, df["atr"].values
    dev = df["vwap_dev"].values
    threshold = atr * std_mult
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    long_sig = valid & (dev < -threshold)
    short_sig = valid & (dev > threshold)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


def sig_ichimoku(df, sess, atr_mult=1.5, rr=2.0):
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


def sig_trend_follow(df, sess, ema_period=50, atr_mult=2.0, rr=3.0):
    ek = f"ema{ema_period}"
    if ek not in df.columns:
        return np.zeros(len(df), dtype=np.int8), np.zeros(len(df)), np.zeros(len(df))
    ema, c, atr = df[ek].values, df["close"].values, df["atr"].values
    rsi = df["rsi"].values
    valid = session_mask(df["hour"].values, sess) & (df["dow"].values < 5) & (atr > 0)
    long_sig = valid & (c > ema) & (rsi > 50) & (rsi < 70)
    short_sig = valid & (c < ema) & (rsi < 50) & (rsi > 30)
    prev_c = np.roll(c, 1)
    long_sig = long_sig & (prev_c <= ema)
    short_sig = short_sig & (prev_c >= ema)
    direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)).astype(np.int8)
    sl = atr * atr_mult
    tp = sl * rr
    return direction, sl, tp


STRATEGY_FUNCS = {
    "EMA Crossover": sig_ema_crossover,
    "EMA Pullback": sig_ema_pullback,
    "Breakout": sig_breakout,
    "Mean Reversion": sig_mean_reversion,
    "Momentum": sig_momentum,
    "Fade": sig_fade,
    "Failed Breakout": sig_failed_breakout,
    "VWAP Reversion": sig_vwap_reversion,
    "Ichimoku": sig_ichimoku,
    "Trend Follow": sig_trend_follow,
}

STRATEGY_PARAMS = {
    "EMA Crossover": {
        "ema_fast": ("int", [5, 8, 10, 12, 15, 18, 20, 25, 30]),
        "ema_slow": ("int", [35, 40, 50, 60, 80, 100, 150, 200]),
        "atr_mult": ("float", 0.3, 4.0),
        "rr": ("float", 0.5, 5.0),
    },
    "EMA Pullback": {
        "ema_fast": ("int", [5, 8, 10, 12, 15, 18, 20, 25, 30]),
        "ema_slow": ("int", [35, 40, 50, 60, 80, 100, 150, 200]),
        "atr_mult": ("float", 0.3, 4.0),
        "rr": ("float", 0.5, 5.0),
    },
    "Breakout": {
        "bb_width_thresh": ("float", 0.001, 0.02),
        "atr_mult": ("float", 0.3, 4.0),
        "rr": ("float", 0.5, 5.0),
    },
    "Mean Reversion": {
        "rsi_high": ("int_range", 60, 90),
        "rsi_low": ("int_range", 10, 40),
        "atr_mult": ("float", 0.1, 4.0),
    },
    "Momentum": {
        "atr_mult": ("float", 1.0, 5.0),
        "rr": ("float", 0.5, 4.0),
        "vol_thresh": ("float", 1.0, 4.0),
    },
    "Fade": {
        "atr_mult_sl": ("float", 0.2, 2.0),
        "atr_mult_tp": ("float", 0.3, 3.0),
        "spike_thresh": ("float", 1.0, 4.0),
    },
    "Failed Breakout": {
        "wick_ratio": ("float", 0.3, 0.85),
        "atr_mult": ("float", 0.3, 4.0),
        "rr": ("float", 0.5, 5.0),
    },
    "VWAP Reversion": {
        "std_mult": ("float", 0.3, 3.0),
        "atr_mult": ("float", 0.3, 4.0),
        "rr": ("float", 0.5, 5.0),
    },
    "Ichimoku": {
        "atr_mult": ("float", 0.3, 4.0),
        "rr": ("float", 0.5, 5.0),
    },
    "Trend Follow": {
        "ema_period": ("int", [20, 30, 40, 50, 60, 80, 100, 150, 200]),
        "atr_mult": ("float", 0.5, 5.0),
        "rr": ("float", 1.0, 6.0),
    },
}

EMA_VALUES = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200]


# ═══════════════════════════════════════════════════════════════════════════════
# Filter masks
# ═══════════════════════════════════════════════════════════════════════════════


def build_filters(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    rsi = df["rsi"].values
    mh = df["macd_hist"].values
    atr = df["atr"].values
    med = df["atr_median"].values
    cr = df["crange"].values
    return {
        "rsi_ok": ((rsi >= 35) & (rsi <= 65) & np.isfinite(rsi)).astype(np.bool_),
        "macd_long": (mh > 0).astype(np.bool_),
        "macd_short": (mh < 0).astype(np.bool_),
        "min_atr": (atr >= med).astype(np.bool_),
        "spread": (cr >= 0.3 * atr).astype(np.bool_),
    }


def apply_filters(direction: np.ndarray, filt: Dict[str, np.ndarray], bits: int) -> np.ndarray:
    out = direction.copy()
    if bits == 0:
        return out
    if bits & 1:
        out = np.where(filt["rsi_ok"] | (out == 0), out, 0)
    if bits & 2:
        good = (out == 0) | ((out == 1) & filt["macd_long"]) | ((out == -1) & filt["macd_short"])
        out = np.where(good, out, 0)
    if bits & 4:
        out = np.where(filt["min_atr"] | (out == 0), out, 0)
    if bits & 8:
        out = np.where(filt["spread"] | (out == 0), out, 0)
    return out


def filter_label(bits: int) -> str:
    parts = []
    if bits & 1: parts.append("RSI")
    if bits & 2: parts.append("MACD")
    if bits & 4: parts.append("MinATR")
    if bits & 8: parts.append("Spread")
    return "+".join(parts) if parts else "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator
# ═══════════════════════════════════════════════════════════════════════════════


def simulate(
    df: pd.DataFrame,
    direction: np.ndarray,
    sl_arr: np.ndarray,
    tp_arr: np.ndarray,
    pair: str,
    max_hold: int = 20,
) -> Dict[str, Any]:
    pip_val = PIP[pair]
    idxs = np.where(direction != 0)[0]
    if len(idxs) == 0:
        return {"trades": 0}

    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
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
        sl_dist, tp_dist = sl_arr[idx], tp_arr[idx]
        if sl_dist <= 0 or tp_dist <= 0:
            continue

        date_key = dates[idx]
        daily_loss.setdefault(date_key, 0.0)
        if daily_loss[date_key] >= ACCOUNT * DAILY_LOSS_PCT:
            continue

        sl_price = entry - sl_dist * d if d == 1 else entry + sl_dist
        tp_price = entry + tp_dist * d if d == 1 else entry - tp_dist
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
                    exit_price = sl_price; break
                if highs[j] >= tp_price:
                    exit_price = tp_price; break
            else:
                if highs[j] >= sl_price:
                    exit_price = sl_price; break
                if lows[j] <= tp_price:
                    exit_price = tp_price; break

        if exit_price is None:
            j = min(idx + max_hold, n - 1)
            exit_price = closes[j]
            hold = j - idx

        pips = (exit_price - entry) * d / pip_val
        trades_pips.append(pips)
        trade_days.append(date_key)
        if pips > 0:
            wins += 1
        daily_loss[date_key] += max(0, -pips * pip_val * LOT * 0.01)
        last_exit = idx + hold

    total = len(trades_pips)
    if total == 0:
        return {"trades": 0}

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

    usd = np.array([p * pip_val * LOT * 0.01 for p in trades_pips])

    streak_w = streak_l = cur_w = cur_l = 0
    for p in trades_pips:
        if p > 0:
            cur_w += 1; cur_l = 0; streak_w = max(streak_w, cur_w)
        elif p < 0:
            cur_l += 1; cur_w = 0; streak_l = max(streak_l, cur_l)
        else:
            cur_w = cur_l = 0

    dd_series = peak - eq
    max_dd_idx = int(np.argmax(dd_series))
    recovery = 0
    if max_dd_idx > 0:
        pk_before = float(np.max(eq[:max_dd_idx]))
        for j in range(max_dd_idx, len(equity)):
            if equity[j] >= pk_before:
                recovery = j - max_dd_idx; break

    downside = returns[returns < 0]
    sortino = 0.0
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(returns.mean() / downside.std() * np.sqrt(min(total * 4, 12_000)))

    return {
        "trades": total,
        "net_pips": net_pips,
        "win_rate": win_rate,
        "max_dd": dd,
        "sharpe": sharpe,
        "pf": pf,
        "equity_final": eq[-1],
        "max_consec_wins": streak_w,
        "max_consec_losses": streak_l,
        "recovery": recovery,
        "sortino": sortino,
        "expectancy_pips": float(np.mean(trades_pips)),
        "expectancy_usd": float(usd.mean()),
        "trades_pips": trades_pips,
        "trade_days": trade_days,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════


def monte_carlo(trades_pips: List[float], trade_days: List[Any], pair: str, n_shuffle: int = MC_SHUFFLES) -> Dict[str, Any]:
    if not trades_pips:
        return {}
    pip_val = PIP[pair]
    arr = np.array(trades_pips, dtype=float)
    rng = np.random.default_rng(42)
    finals, max_dds = [], []
    for _ in range(n_shuffle):
        sh = arr[rng.permutation(len(arr))]
        eq = ACCOUNT
        peak = ACCOUNT
        mdd = 0.0
        for p in sh:
            eq += p * pip_val * LOT * 0.01
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak * 100 if peak > 0 else 0)
        finals.append(eq)
        max_dds.append(mdd)
    finals = np.array(finals)
    return {
        "mc_median": float(np.median(finals)),
        "mc_p5": float(np.percentile(finals, 5)),
        "mc_p95": float(np.percentile(finals, 95)),
        "mc_median_dd": float(np.median(max_dds)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna objective — the "brain"
# ═══════════════════════════════════════════════════════════════════════════════


def suggest_params(trial: optuna.Trial, strategy: str) -> Dict[str, Any]:
    spec = STRATEGY_PARAMS[strategy]
    params = {}
    for k, v in spec.items():
        if v[0] == "int":
            params[k] = trial.suggest_categorical(k, v[1])
        elif v[0] == "int_range":
            params[k] = trial.suggest_int(k, v[1], v[2])
        elif v[0] == "float":
            params[k] = trial.suggest_float(k, v[1], v[2])
    return params


def make_objective(pairs_train: Dict[str, pd.DataFrame], is_daily: bool):
    pair_names = list(pairs_train.keys())
    sess_list = ["All"] if is_daily else list(SESSIONS.keys())
    filter_cache = {pn: build_filters(df) for pn, df in pairs_train.items()}
    hold_choices = [5, 10, 15, 20, 30, 40, 60] if is_daily else [6, 10, 12, 16, 20]

    def objective(trial: optuna.Trial) -> float:
        max_hold = trial.suggest_categorical("max_hold", hold_choices)
        fixed = trial.user_attrs.get("_fixed_strategy")
        if fixed:
            strategy = fixed
            trial.set_user_attr("strategy_name", fixed)
        else:
            strategy = trial.suggest_categorical("strategy", list(STRATEGY_FUNCS.keys()))
        pair = trial.suggest_categorical("pair", pair_names)
        sess = trial.suggest_categorical("session", sess_list)
        filter_bits = trial.suggest_int("filter_bits", 0, 15)
        params = suggest_params(trial, strategy)

        df = pairs_train[pair]
        filt = filter_cache[pair]
        fn = STRATEGY_FUNCS[strategy]
        try:
            direction, sl, tp = fn(df, sess, **params)
        except Exception:
            return -100.0

        direction = apply_filters(direction.astype(np.int8), filt, filter_bits)
        result = simulate(df, direction, sl, tp, pair, max_hold=max_hold)

        if result["trades"] < MIN_TRADES:
            return -100.0

        sharpe = result["sharpe"]
        dd = result["max_dd"]
        n_trades = result["trades"]
        wr = result["win_rate"]
        pf = result["pf"]

        # Scale: 200 trades → 0.4x, 1000 → 1.0x, 5000 → 1.5x, 10000+ → 1.8x
        trade_mult = min(np.log10(max(n_trades, 10)) / np.log10(5000), 1.8)
        wr_bonus = max(0, (wr - 40) / 80.0)
        pf_bonus = min(max(pf - 1.0, 0) * 0.3, 0.5)
        score = (sharpe + wr_bonus + pf_bonus) * trade_mult - dd * 0.05

        trial.set_user_attr("net_pips", round(result["net_pips"], 2))
        trial.set_user_attr("win_rate", round(result["win_rate"], 1))
        trial.set_user_attr("max_dd", round(dd, 2))
        trial.set_user_attr("sharpe", round(sharpe, 3))
        trial.set_user_attr("pf", round(result["pf"], 3))
        trial.set_user_attr("trades", result["trades"])
        trial.set_user_attr("strategy_name", strategy)
        trial.set_user_attr("params", str(params))
        trial.set_user_attr("filter_label", filter_label(filter_bits))
        trial.set_user_attr("max_hold", max_hold)

        return score

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward on top configs
# ═══════════════════════════════════════════════════════════════════════════════


def _trial_strategy(t) -> str:
    return t.user_attrs.get("strategy_name", t.params.get("strategy", "?"))


def _get_unique_top_trials(study: optuna.Study, top_n: int = 100) -> list:
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
    seen = set()
    result = []
    for t in trials:
        if t.value is None or t.value <= -100:
            continue
        params_str = t.user_attrs.get("params", "{}")
        key = f"{_trial_strategy(t)}|{t.params.get('session', 'All')}|{t.params.get('filter_bits', 0)}|{params_str}"
        if key in seen:
            continue
        seen.add(key)
        result.append(t)
        if len(result) >= top_n:
            break
    return result


def cross_pair_validate(
    study: optuna.Study,
    pairs_full: Dict[str, pd.DataFrame],
    is_daily: bool,
    top_n: int = 50,
) -> pd.DataFrame:
    """Test each top config on ALL pairs. Keep only those that work on 3+ pairs."""
    top_trials = _get_unique_top_trials(study, top_n * 2)
    pair_names = list(pairs_full.keys())
    rows = []

    for t in top_trials:
        strategy = _trial_strategy(t)
        sess = t.params.get("session", "All")
        fb = t.params.get("filter_bits", 0)
        hold = t.params.get("max_hold", 20)
        params_str = t.user_attrs.get("params", "{}")
        try:
            params = ast.literal_eval(params_str)
        except Exception:
            continue

        fn = STRATEGY_FUNCS[strategy]
        pair_results = []

        for pname in pair_names:
            df = pairs_full[pname]
            filt = build_filters(df)
            try:
                direction, sl, tp = fn(df, sess, **params)
            except Exception:
                continue
            direction = apply_filters(direction.astype(np.int8), filt, fb)
            r = simulate(df, direction, sl, tp, pname, hold)
            if r["trades"] >= 10:
                pair_results.append({
                    "pair": pname,
                    "trades": r["trades"],
                    "net_pips": r["net_pips"],
                    "win_rate": r["win_rate"],
                    "sharpe": r["sharpe"],
                    "max_dd": r["max_dd"],
                    "pf": r["pf"],
                })

        profitable_pairs = [pr for pr in pair_results if pr["net_pips"] > 0 and pr["sharpe"] > 0]
        total_trades = sum(pr["trades"] for pr in pair_results)
        total_pips = sum(pr["net_pips"] for pr in pair_results)
        avg_sharpe = float(np.mean([pr["sharpe"] for pr in pair_results])) if pair_results else 0
        avg_wr = float(np.mean([pr["win_rate"] for pr in pair_results])) if pair_results else 0
        max_dd = max((pr["max_dd"] for pr in pair_results), default=0)

        rows.append({
            "Strategy": strategy, "Session": sess,
            "Filter": filter_label(fb), "Params": params_str,
            "Hold": hold,
            "PairsTested": len(pair_results),
            "PairsProfitable": len(profitable_pairs),
            "TotalTrades": total_trades,
            "TotalPips": round(total_pips, 1),
            "AvgSharpe": round(avg_sharpe, 3),
            "AvgWinRate": round(avg_wr, 1),
            "WorstDD": round(max_dd, 1),
            "PairDetail": str({pr["pair"]: f"{pr['net_pips']:+.0f}p S{pr['sharpe']:.1f}" for pr in pair_results}),
            "Universal": len(profitable_pairs) >= CROSS_PAIR_MIN_PAIRS,
        })
        if len(rows) >= top_n:
            break

    return pd.DataFrame(rows)


def rolling_walk_forward(
    study: optuna.Study,
    pairs_full: Dict[str, pd.DataFrame],
    is_daily: bool,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rolling walk-forward: slide a train/test window across the data."""
    top_trials = _get_unique_top_trials(study, top_n * 2)
    rows = []

    for t in top_trials[:top_n]:
        strategy = _trial_strategy(t)
        pair = t.params.get("pair", list(pairs_full.keys())[0])
        sess = t.params.get("session", "All")
        fb = t.params.get("filter_bits", 0)
        hold = t.params.get("max_hold", 20)
        params_str = t.user_attrs.get("params", "{}")
        try:
            params = ast.literal_eval(params_str)
        except Exception:
            continue

        fn = STRATEGY_FUNCS[strategy]
        df = pairs_full[pair]
        dates = df["datetime"]
        min_date, max_date = dates.min(), dates.max()
        total_days = (max_date - min_date).days

        train_days = ROLLING_WF_TRAIN_YEARS * 365
        test_days = ROLLING_WF_TEST_YEARS * 365
        window_days = train_days + test_days

        if total_days < window_days:
            continue

        window_results = []
        cursor = min_date
        while cursor + pd.Timedelta(days=window_days) <= max_date:
            train_end = cursor + pd.Timedelta(days=train_days)
            test_end = cursor + pd.Timedelta(days=window_days)

            df_tr = df[(dates >= cursor) & (dates < train_end)].copy().reset_index(drop=True)
            df_te = df[(dates >= train_end) & (dates < test_end)].copy().reset_index(drop=True)

            if len(df_tr) < 50 or len(df_te) < 20:
                cursor += pd.Timedelta(days=test_days)
                continue

            filt_te = build_filters(df_te)
            try:
                d_te, sl_te, tp_te = fn(df_te, sess, **params)
            except Exception:
                cursor += pd.Timedelta(days=test_days)
                continue
            d_te = apply_filters(d_te.astype(np.int8), filt_te, fb)
            r_te = simulate(df_te, d_te, sl_te, tp_te, pair, hold)

            period_label = f"{cursor.strftime('%Y')}→{train_end.strftime('%Y')}/{test_end.strftime('%Y')}"
            window_results.append({
                "period": period_label,
                "trades": r_te["trades"],
                "pips": round(r_te.get("net_pips", 0), 1),
                "sharpe": round(r_te.get("sharpe", 0), 3),
                "win_rate": round(r_te.get("win_rate", 0), 1),
                "dd": round(r_te.get("max_dd", 0), 1),
            })
            cursor += pd.Timedelta(days=test_days)

        if not window_results:
            continue

        n_positive = sum(1 for w in window_results if w["pips"] > 0)
        avg_sharpe = float(np.mean([w["sharpe"] for w in window_results]))
        total_trades = sum(w["trades"] for w in window_results)
        total_pips = sum(w["pips"] for w in window_results)

        rows.append({
            "Strategy": strategy, "Pair": pair, "Session": sess,
            "Filter": filter_label(fb), "Params": params_str,
            "Windows": len(window_results),
            "WindowsPositive": n_positive,
            "ConsistencyPct": round(n_positive / len(window_results) * 100, 0),
            "TotalTrades": total_trades,
            "TotalPips": round(total_pips, 1),
            "AvgSharpe": round(avg_sharpe, 3),
            "WindowDetail": window_results,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(
    study: optuna.Study,
    cp: pd.DataFrame,
    rwf: pd.DataFrame,
    mc_results: List[Dict[str, Any]],
    risk_results: List[Dict[str, Any]],
    elapsed: float,
    n_trials: int,
    path: str,
    mc_shuffles: int = MC_SHUFFLES,
) -> None:
    L: List[str] = []
    w = L.append

    w("=" * 95)
    w("  SMART FOREX BACKTESTER — AI-POWERED STRATEGY DISCOVERY REPORT")
    w(f"  Optuna trials: {n_trials:,} | Search time: {elapsed:.1f}s | Account: ${ACCOUNT}")
    w("=" * 95)

    # 1 — Learning journey
    w("\n" + "─" * 95)
    w("  1. LEARNING JOURNEY — how the AI improved over time")
    w("─" * 95)
    trials = [t for t in study.trials if t.value is not None and t.value > -100]
    if trials:
        best_so_far = -999
        milestones = []
        for t in trials:
            if t.value > best_so_far:
                best_so_far = t.value
                milestones.append(t)
        w(f"\n  {len(trials):,} valid trials out of {n_trials:,} total")
        w(f"  {len(milestones)} improvements found (shown below):\n")
        w(f"  {'Trial':>7}  {'Strategy':<18}{'Pair':<8}{'Trades':>7}{'Sharpe':>8}{'Pips':>10}{'Win%':>7}{'DD%':>7}  Filter")
        w("  " + "-" * 90)
        for t in milestones:
            ua = t.user_attrs
            w(f"  {t.number:>7}  {_trial_strategy(t):<18}{t.params['pair']:<8}"
              f"{ua.get('trades', 0):>7}{ua.get('sharpe', 0):>8.2f}{ua.get('net_pips', 0):>10.1f}"
              f"{ua.get('win_rate', 0):>7.1f}{ua.get('max_dd', 0):>7.1f}"
              f"  {ua.get('filter_label', 'none')}")

    # 2 — Top 20 configs found
    w("\n" + "─" * 95)
    w("  2. TOP 20 CONFIGS FOUND (by composite score: Sharpe × trade volume − DD penalty)")
    w("─" * 95)
    top_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -100],
        key=lambda t: t.value, reverse=True
    )[:20]
    if top_trials:
        w(f"\n  {'#':>3}  {'Strategy':<18}{'Pair':<8}{'Sess':<8}{'Trades':>7}{'Pips':>10}{'Win%':>7}{'Sharpe':>8}{'DD%':>7}{'PF':>7}  Filter")
        w("  " + "-" * 100)
        for i, t in enumerate(top_trials, 1):
            ua = t.user_attrs
            w(f"  {i:>3}  {_trial_strategy(t):<18}{t.params['pair']:<8}{t.params['session']:<8}"
              f"{ua.get('trades', 0):>7}{ua.get('net_pips', 0):>10.1f}"
              f"{ua.get('win_rate', 0):>7.1f}{ua.get('sharpe', 0):>8.2f}"
              f"{ua.get('max_dd', 0):>7.1f}{ua.get('pf', 0):>7.2f}"
              f"  {ua.get('filter_label', 'none')}")

    # 3 — Best per strategy
    w("\n" + "─" * 95)
    w("  3. BEST CONFIG PER STRATEGY")
    w("─" * 95)
    best_per_strat: Dict[str, Any] = {}
    for t in top_trials:
        s = _trial_strategy(t)
        if s not in best_per_strat:
            best_per_strat[s] = t
    for s, t in best_per_strat.items():
        ua = t.user_attrs
        w(f"\n  {s}:")
        w(f"    Pair: {t.params['pair']} | Session: {t.params['session']} | Filter: {ua.get('filter_label', 'none')}")
        w(f"    Params: {ua.get('params', '{}')}")
        w(f"    Trades {ua.get('trades', 0)} | Win {ua.get('win_rate', 0):.1f}% | Net {ua.get('net_pips', 0):+.1f} pips | Sharpe {ua.get('sharpe', 0):.2f} | DD {ua.get('max_dd', 0):.1f}%")

    # 4 — Cross-pair validation
    w("\n" + "─" * 95)
    w("  4. CROSS-PAIR VALIDATION — does the strategy work on ALL pairs?")
    w("─" * 95)
    if len(cp):
        universal = cp[cp["Universal"] == True]
        w(f"\n  {len(cp)} configs tested across all pairs. {len(universal)} are UNIVERSAL ({CROSS_PAIR_MIN_PAIRS}+ profitable pairs).\n")
        show = universal if len(universal) else cp.head(15)
        w(f"  {'Strategy':<18}{'Pairs OK':>9}{'Tot Trades':>11}{'Tot Pips':>10}{'Avg Sharpe':>11}{'Avg Win%':>9}{'Worst DD':>9}  Universal")
        w("  " + "-" * 95)
        for _, r in show.head(20).iterrows():
            w(f"  {r.Strategy:<18}{r.PairsProfitable:>5}/{r.PairsTested:<3}{r.TotalTrades:>11}{r.TotalPips:>10.0f}"
              f"{r.AvgSharpe:>11.3f}{r.AvgWinRate:>9.1f}{r.WorstDD:>9.1f}  {str(r.Universal)}")
        w("\n  Per-pair breakdown of best universal config:")
        if len(universal):
            best_u = universal.iloc[0]
            w(f"  {best_u.Strategy} | {best_u.Params}")
            w(f"  {best_u.PairDetail}")
    else:
        w("\n  No cross-pair results.")

    # 5 — Rolling walk-forward
    w("\n" + "─" * 95)
    w(f"  5. ROLLING WALK-FORWARD ({ROLLING_WF_TRAIN_YEARS}yr train / {ROLLING_WF_TEST_YEARS}yr test, sliding)")
    w("─" * 95)
    if len(rwf):
        w(f"\n  {'Strategy':<18}{'Pair':<8}{'Windows':>8}{'Positive':>9}{'Consist%':>9}{'Tot Trades':>11}{'Tot Pips':>10}{'Avg Sharpe':>11}")
        w("  " + "-" * 95)
        for _, r in rwf.head(20).iterrows():
            w(f"  {r.Strategy:<18}{r.Pair:<8}{r.Windows:>8}{r.WindowsPositive:>9}{r.ConsistencyPct:>8.0f}%"
              f"{r.TotalTrades:>11}{r.TotalPips:>10.1f}{r.AvgSharpe:>11.3f}")
        # Show window detail for #1
        if len(rwf):
            best_rwf = rwf.iloc[0]
            w(f"\n  Window detail for {best_rwf.Strategy} on {best_rwf.Pair}:")
            for wd in best_rwf.WindowDetail:
                status = "+" if wd["pips"] > 0 else "-"
                w(f"    {status} {wd['period']}: {wd['trades']} trades, {wd['pips']:+.1f} pips, Sharpe {wd['sharpe']:.2f}, Win {wd['win_rate']:.0f}%, DD {wd['dd']:.1f}%")
    else:
        w("\n  No rolling walk-forward results.")

    # 6 — Monte Carlo
    w("\n" + "─" * 95)
    w(f"  6. MONTE CARLO — TOP {len(mc_results)} CONFIGS ({mc_shuffles:,} shuffles each)")
    w("─" * 95)
    for mc in mc_results:
        w(f"\n  {mc['strategy']} | {mc['pair']}")
        w(f"    median final ${mc.get('mc_median', 0):.2f} | p5 ${mc.get('mc_p5', 0):.2f} | p95 ${mc.get('mc_p95', 0):.2f}")
        w(f"    median max DD% {mc.get('mc_median_dd', 0):.1f}")

    # 7 — Risk
    w("\n" + "─" * 95)
    w("  7. DEEP RISK ANALYTICS")
    w("─" * 95)
    for rr in risk_results[:10]:
        w(f"\n  {rr['strategy']} | {rr['pair']}")
        w(f"    consec W/L: {rr['max_consec_wins']}/{rr['max_consec_losses']} | recovery: {rr['recovery']}")
        w(f"    Sortino {rr['sortino']:.3f} | E[pips] {rr['expectancy_pips']:.2f} | E[USD] {rr['expectancy_usd']:.4f}")

    # 8 — Portfolio
    w("\n" + "─" * 95)
    w("  8. RECOMMENDED PORTFOLIO (universal + consistent across years)")
    w("─" * 95)
    if len(cp):
        viable = cp[cp["Universal"] == True].head(10) if len(cp[cp["Universal"] == True]) >= 2 else cp.head(10)
        if len(viable) >= 2:
            best_combo = None
            best_score = -999.0
            items = list(viable.iterrows())
            for size in (2, 3):
                for combo in combinations(items[:min(8, len(items))], size):
                    strats = set(r.Strategy for _, r in combo)
                    if len(strats) < size:
                        continue
                    sharpe = float(np.mean([r.AvgSharpe for _, r in combo]))
                    dd = max(r.WorstDD for _, r in combo)
                    pips = sum(r.TotalPips for _, r in combo)
                    pairs_ok = min(r.PairsProfitable for _, r in combo)
                    score = sharpe * 2 + pips / 200.0 - dd * 0.3 + pairs_ok * 0.5
                    if score > best_score:
                        best_score, best_combo = score, combo
            if best_combo:
                w("\n  Selected combo (universal across pairs):")
                for _, r in best_combo:
                    w(f"    → {r.Strategy} | {r.Filter} | hold={r.Hold}")
                    w(f"      {r.PairsProfitable}/{r.PairsTested} pairs | {r.TotalTrades} trades | {r.TotalPips:+.0f} pips | Sharpe {r.AvgSharpe:.2f}")

    # 9 — Bot configs
    w("\n" + "─" * 95)
    w("  9. COPY-PASTE BOT CONFIG — TOP 3")
    w("─" * 95)
    for i, t in enumerate(top_trials[:3], 1):
        ua = t.user_attrs
        w(f"""
  # --- Rank {i} ---
  STRATEGY = "{_trial_strategy(t)}"
  PAIR = "{t.params['pair']}"
  SESSION = "{t.params['session']}"
  FILTER_BITS = {t.params['filter_bits']}  # {ua.get('filter_label', 'none')}
  PARAMS = {ua.get('params', '{}')}
  # Stats: trades={ua.get('trades', 0)} win={ua.get('win_rate', 0):.1f}% pips={ua.get('net_pips', 0):+.1f} sharpe={ua.get('sharpe', 0):.2f} dd={ua.get('max_dd', 0):.1f}%
""")

    # 10 — Strategy distribution
    w("─" * 95)
    w("  10. STRATEGY DISCOVERY STATS")
    w("─" * 95)
    strat_counts: Dict[str, int] = defaultdict(int)
    strat_best: Dict[str, float] = defaultdict(lambda: -999)
    for t in study.trials:
        if t.value is not None and t.value > -100:
            s = _trial_strategy(t)
            strat_counts[s] += 1
            strat_best[s] = max(strat_best[s], t.value)
    w(f"\n  {'Strategy':<22}{'Trials':>8}{'Best Score':>12}")
    w("  " + "-" * 42)
    for s in sorted(strat_counts, key=lambda x: strat_best[x], reverse=True):
        w(f"  {s:<22}{strat_counts[s]:>8}{strat_best[s]:>12.3f}")

    txt = "\n".join(L)
    with open(path, "w") as f:
        f.write(txt)
    print(f"\n{'=' * 60}\n  Report saved → {path}\n{'=' * 60}\n")
    print(txt[:10000] + ("\n\n... [truncated] ...\n" if len(txt) > 10000 else ""))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="all_5pairs_daily.txt")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--mc-shuffles", type=int, default=MC_SHUFFLES)
    args = parser.parse_args()

    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data)
    if not os.path.exists(data_file):
        print(f"Not found: {data_file}")
        sys.exit(1)

    mc_shuffles = args.mc_shuffles

    print("=" * 60)
    print("  SMART FOREX BACKTESTER — AI Strategy Discovery")
    print(f"  Trials: {args.trials:,} | MC shuffles: {args.mc_shuffles:,}")
    print("=" * 60)

    pairs_full, is_daily = load_data(data_file)

    # Split train/test — use 75% by row count
    pairs_train: Dict[str, pd.DataFrame] = {}
    for k, df in pairs_full.items():
        split_idx = int(len(df) * WF_SPLIT)
        pairs_train[k] = df.iloc[:split_idx].copy().reset_index(drop=True)
    train_bars = sum(len(df) for df in pairs_train.values())
    print(f"  Train slice: {train_bars:,} total bars across {len(pairs_train)} pairs")

    print(f"\nPhase 1 — Optuna search ({args.trials:,} trials) …")
    t0 = time.time()

    # Run per-strategy studies so every strategy gets explored deeply
    strat_names = list(STRATEGY_FUNCS.keys())
    n_per_strat = max(args.trials // len(strat_names), 500)
    all_studies: Dict[str, optuna.Study] = {}
    objective_fn = make_objective(pairs_train, is_daily)

    print(f"    {len(strat_names)} strategies × {n_per_strat:,} trials each = {len(strat_names) * n_per_strat:,} total")

    for si, sname in enumerate(strat_names):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42 + si, n_startup_trials=50),
        )

        def fixed_objective(trial, _sname=sname):
            # Inject strategy name directly, bypassing suggest_categorical conflict
            trial.set_user_attr("_fixed_strategy", _sname)
            return objective_fn(trial)

        def progress_cb(study, trial, _si=si, _sname=sname):
            n = trial.number + 1
            if n % 500 == 0 or n == n_per_strat:
                best = study.best_value if study.best_value is not None else -999
                elapsed = time.time() - t0
                sys.stdout.write(f"\r    [{_si+1}/{len(strat_names)}] {_sname:<20} trial {n:,}/{n_per_strat:,} | best {best:.3f}    ")
                sys.stdout.flush()

        study.optimize(fixed_objective, n_trials=n_per_strat, callbacks=[progress_cb])
        all_studies[sname] = study

    # Merge all trials into one master study for reporting
    study = optuna.create_study(direction="maximize")
    for sname, sub in all_studies.items():
        for t in sub.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                study.add_trial(t)
    actual_trials = sum(len(s.trials) for s in all_studies.values())
    elapsed_search = time.time() - t0
    print(f"\r    Search done: {actual_trials:,} trials in {elapsed_search:.1f}s ({actual_trials / max(elapsed_search, 0.1):.0f}/sec)          ")

    print("\nPhase 2 — Cross-pair validation (top configs tested on ALL pairs) …")
    cp = cross_pair_validate(study, pairs_full, is_daily, top_n=50)
    n_universal = len(cp[cp["Universal"] == True]) if len(cp) else 0
    print(f"  Configs tested: {len(cp):,} | Universal (work on {CROSS_PAIR_MIN_PAIRS}+ pairs): {n_universal}")

    print("\nPhase 3 — Rolling walk-forward (sliding year windows) …")
    rwf = rolling_walk_forward(study, pairs_full, is_daily, top_n=30)
    print(f"  Configs with rolling WF: {len(rwf):,}")

    print(f"\nPhase 4 — Monte Carlo + deep risk on top {MC_TOP} ({mc_shuffles:,} shuffles) …")
    top_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -100],
        key=lambda t: t.value, reverse=True
    )[:MC_TOP]

    mc_results = []
    risk_results = []
    for t in top_trials:
        strategy = _trial_strategy(t)
        pair = t.params["pair"]
        sess = t.params["session"]
        fb = t.params["filter_bits"]
        try:
            params = ast.literal_eval(t.user_attrs.get("params", "{}"))
        except Exception:
            continue
        df = pairs_full[pair]
        filt = build_filters(df)
        fn = STRATEGY_FUNCS[strategy]
        hold = t.params.get("max_hold", 20)
        try:
            direction, sl, tp = fn(df, sess, **params)
        except Exception:
            continue
        direction = apply_filters(direction.astype(np.int8), filt, fb)
        result = simulate(df, direction, sl, tp, pair, hold)
        if result["trades"] < MIN_TRADES:
            continue

        mc = monte_carlo(result["trades_pips"], result["trade_days"], pair, mc_shuffles)
        mc["strategy"] = strategy
        mc["pair"] = pair
        mc_results.append(mc)

        risk_results.append({
            "strategy": strategy, "pair": pair,
            "max_consec_wins": result["max_consec_wins"],
            "max_consec_losses": result["max_consec_losses"],
            "recovery": result["recovery"],
            "sortino": result["sortino"],
            "expectancy_pips": result["expectancy_pips"],
            "expectancy_usd": result["expectancy_usd"],
        })

    print(f"  MC configs: {len(mc_results)} | risk profiles: {len(risk_results)}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_report.txt")
    generate_report(study, cp, rwf, mc_results, risk_results, elapsed_search, actual_trials, out, mc_shuffles)


if __name__ == "__main__":
    main()
