"""
SOXL Swing Trading Agent
Strategy: EMA(5,20) + FGI momentum gate (mom_buy=-3, mom_sell=8)
          + FGI velocity gate (vel_thresh=-3, vel_win=5)
          + 11% trailing stop from peak

Backtest: ~5909% return, Sharpe 2.83, -34% max drawdown (Oct 2023–Jun 2026)

Run every hour during market hours. State persists in trade_state.json.
"""

import json
import math
import os
import sys
from datetime import datetime, date, timedelta

import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import fear_and_greed
import warnings
warnings.filterwarnings("ignore")

# ── Strategy params ───────────────────────────────────────────────────────────
TICKER       = "SOXL"
EMA_FAST     = 5
EMA_SLOW     = 20
MOM_WINDOW   = 5    # days for FGI momentum
VEL_WINDOW   = 5    # days for velocity (d/dt of momentum)
MOM_BUY      = -3   # suppress buy if 5d-mom < this
MOM_SELL     = 8    # suppress sell if 5d-mom > this (still rising hard)
VEL_THRESH   = -3   # suppress buy if FGI acceleration < this
TRAIL_PCT    = 0.11 # 11% trailing stop from peak

STATE_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_state.json")
ET           = pytz.timezone("America/New_York")

# ── State persistence ─────────────────────────────────────────────────────────

def load_state() -> dict:
    defaults = {"position": 0, "entry_price": 0.0, "peak_price": 0.0, "entry_date": None}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            return {**defaults, **s}
        except Exception:
            pass
    return defaults

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ── Market hours ──────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_ <= now < close_

# ── Fear & Greed Index (live, no CSV) ────────────────────────────────────────

def fetch_fgi_series() -> pd.Series:
    """
    Pull the last ~250 days of FGI from CNN via the fear_and_greed library.
    Returns a daily pd.Series indexed by date, values are FGI scores.
    """
    from fear_and_greed.cnn import Fetcher
    raw  = Fetcher()()
    hist = raw["fear_and_greed_historical"]["data"]
    df   = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["x"], unit="ms").dt.normalize()
    df   = df.rename(columns={"y": "fgi"})[["date", "fgi"]]
    df   = df.drop_duplicates("date").sort_values("date").set_index("date")
    # Also pin today's live reading (may differ from historical)
    live = fear_and_greed.get()
    today = pd.Timestamp(date.today())
    df.loc[today] = live.value
    return df["fgi"].sort_index()

# ── Price data ────────────────────────────────────────────────────────────────

def fetch_price_data(lookback_days: int = 60) -> pd.DataFrame:
    df = yf.download(TICKER, period=f"{lookback_days}d", interval="1h",
                     auto_adjust=True, progress=False)
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[df.index.to_series().dt.time.between(
        pd.Timestamp("09:30").time(), pd.Timestamp("15:59").time()
    )]
    return df

# ── Merge FGI onto hourly bars ────────────────────────────────────────────────

def merge_fgi(price_df: pd.DataFrame, fgi_series: pd.Series) -> pd.DataFrame:
    df    = price_df.copy()
    naive = df.index.tz_localize(None) if df.index.tzinfo else df.index
    norm  = naive.normalize()
    df["fgi"] = fgi_series.reindex(norm, method="ffill").values
    return df.dropna(subset=["fgi"])

# ── Indicators ────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return df.dropna()

def add_fgi_features(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    naive = df.index.tz_localize(None) if df.index.tzinfo else df.index
    fgi   = pd.Series(df["fgi"].values, index=naive)
    daily = fgi.resample("D").first().dropna()
    mom   = daily.diff(MOM_WINDOW)
    vel   = mom.diff(VEL_WINDOW)
    norm  = naive.normalize()
    def ff(s): return s.reindex(norm, method="ffill").values
    df["fgi_mom"] = ff(mom)
    df["fgi_vel"] = ff(vel)
    return df.dropna()

# ── Signal ────────────────────────────────────────────────────────────────────

def compute_signal(df: pd.DataFrame) -> tuple[int, int]:
    """
    Returns (signal_bar_signal, latest_bar_index).
    Uses iloc[-2] for crossover detection to avoid acting on an in-progress bar.
    Returns signal: 1=BUY, -1=SELL, 0=HOLD
    """
    ef = df["ema_fast"]
    es = df["ema_slow"]

    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[cross_up] =  1
    sig[cross_dn] = -1

    # FGI gates
    sig[(sig ==  1) & (df["fgi_mom"] < MOM_BUY)]  = 0
    sig[(sig == -1) & (df["fgi_mom"] > MOM_SELL)] = 0
    sig[(sig ==  1) & (df["fgi_vel"] < VEL_THRESH)] = 0

    # Use the last COMPLETED bar (-2) so an in-progress bar never fires a signal.
    # The very last bar (-1) is used only for the current price / trailing stop.
    return int(sig.iloc[-2])

def check_trailing_stop(current_price: float, peak_price: float) -> bool:
    return peak_price > 0 and current_price <= peak_price * (1 - TRAIL_PCT)

# ── Reconcile state with actual holdings (via robin_stocks if available) ──────

def reconcile_state(state: dict) -> dict:
    """
    Check actual SOXL position via robin_stocks and patch trade_state if out of sync.
    Silently skips if robin_stocks isn't installed or credentials aren't set.
    """
    try:
        import robin_stocks.robinhood as rh
        email    = os.environ.get("email", "")
        password = os.environ.get("password", "")
        if not email or not password:
            return state

        rh.login(email, password, store_session=True, mfa_code=None)
        positions = rh.get_open_stock_positions() or []
        actual_qty = 0.0
        avg_price  = 0.0
        for pos in positions:
            inst = rh.get_instrument_by_url(pos["instrument"])
            if inst and inst.get("symbol") == TICKER:
                actual_qty = float(pos["quantity"])
                avg_price  = float(pos.get("average_buy_price", 0))
                break

        if actual_qty > 0.01 and state["position"] == 0:
            # Holding SOXL but state says we're out — sync it
            print(f"[SYNC] State out of sync: holding {actual_qty:.4f} shares @ ${avg_price:.2f}. Updating state.")
            state["position"]    = 1
            state["entry_price"] = avg_price
            state["peak_price"]  = max(avg_price, state.get("peak_price", avg_price))
            state["entry_date"]  = date.today().isoformat()
            save_state(state)

        elif actual_qty < 0.01 and state["position"] == 1:
            # No SOXL held but state says we're in — reset
            print(f"[SYNC] State out of sync: no SOXL held but state says 'in position'. Resetting.")
            state["position"]    = 0
            state["entry_price"] = 0.0
            state["peak_price"]  = 0.0
            state["entry_date"]  = None
            save_state(state)

    except Exception as e:
        print(f"[SYNC] Skipped position reconciliation: {e}")

    return state

# ── Notify ────────────────────────────────────────────────────────────────────

def notify(message: str):
    """Send a push notification via ntfy.sh (free, no signup, just curl).
    Set NTFY_TOPIC env var to your private topic name to enable."""
    topic = os.environ.get("NTFY_TOPIC", "")
    if not topic:
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode(),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict | None:
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    print(f"\n[{now_str}] Running SOXL trade agent...")

    if not is_market_open():
        print("Market is closed. No action.")
        return None

    # Load and optionally reconcile state
    state = load_state()
    state = reconcile_state(state)

    # Fetch data
    try:
        price_df   = fetch_price_data(lookback_days=60)
        fgi_series = fetch_fgi_series()
    except Exception as e:
        msg = f"ERROR fetching data: {e}"
        print(msg)
        notify(f"SOXL agent ERROR: {msg}")
        return None

    if len(price_df) < 25:
        print("Not enough price bars. Skipping.")
        return None

    df = merge_fgi(price_df, fgi_series)
    df = add_indicators(df)
    df = add_fgi_features(df)

    if len(df) < 3:
        print("Not enough bars after features. Skipping.")
        return None

    # Current price from latest bar; signal from last COMPLETED bar
    current_price = float(df["Close"].iloc[-1])
    signal_price  = float(df["Close"].iloc[-2])
    fgi_val       = float(df["fgi"].iloc[-1])
    fgi_mom       = float(df["fgi_mom"].iloc[-1])
    fgi_vel       = float(df["fgi_vel"].iloc[-1])
    signal        = compute_signal(df)

    action = "HOLD"

    if state["position"] == 0:
        if signal == 1:
            action = "BUY"
            state["position"]    = 1
            state["entry_price"] = current_price
            state["peak_price"]  = current_price
            state["entry_date"]  = date.today().isoformat()
    else:
        # Update peak
        state["peak_price"] = max(float(state["peak_price"]), current_price)

        if check_trailing_stop(current_price, float(state["peak_price"])):
            action = "SELL (trailing stop)"
            pnl = (current_price - float(state["entry_price"])) / float(state["entry_price"]) * 100
            print(f"TRAILING STOP HIT: entry=${state['entry_price']:.2f} peak=${state['peak_price']:.2f} now=${current_price:.2f} PnL={pnl:+.1f}%")
            state["position"]    = 0
            state["entry_price"] = 0.0
            state["peak_price"]  = 0.0
            state["entry_date"]  = None
        elif signal == -1:
            action = "SELL (EMA signal)"
            pnl = (current_price - float(state["entry_price"])) / float(state["entry_price"]) * 100
            print(f"EMA SELL: entry=${state['entry_price']:.2f} now=${current_price:.2f} PnL={pnl:+.1f}%")
            state["position"]    = 0
            state["entry_price"] = 0.0
            state["peak_price"]  = 0.0
            state["entry_date"]  = None

    # Print report
    print("=" * 55)
    print(f"  Price   : ${current_price:.2f}  (signal bar: ${signal_price:.2f})")
    print(f"  FGI     : {fgi_val:.1f}  mom={fgi_mom:.2f}  vel={fgi_vel:.2f}")
    print(f"  Signal  : {signal:+d}")
    print(f"  Position: {'IN' if state['position'] else 'OUT'}", end="")
    if state["position"]:
        peak   = float(state["peak_price"])
        entry  = float(state["entry_price"])
        pnl    = (current_price - entry) / entry * 100
        stop_at = peak * (1 - TRAIL_PCT)
        print(f"  entry=${entry:.2f}  peak=${peak:.2f}  stop=${stop_at:.2f}  PnL={pnl:+.1f}%")
    else:
        print()
    print(f"\n  >>> ACTION: {action} <<<")
    print("=" * 55)

    save_state(state)

    if action != "HOLD":
        notify(f"SOXL {action} @ ${current_price:.2f}  FGI={fgi_val:.0f}  mom={fgi_mom:.1f}")

    result = {
        "action":       action,
        "ticker":       TICKER,
        "price":        current_price,
        "signal":       signal,
        "position":     state["position"],
        "fgi":          fgi_val,
        "fgi_mom":      fgi_mom,
        "fgi_vel":      fgi_vel,
        "peak_price":   state.get("peak_price", 0.0),
        "entry_price":  state.get("entry_price", 0.0),
        "trail_trigger": float(state.get("peak_price", 0) or 0) * (1 - TRAIL_PCT),
    }
    return result


if __name__ == "__main__":
    result = run()
    if result:
        import json as _json
        print(f"\nJSON:\n{_json.dumps(result, indent=2)}")
