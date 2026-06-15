#!/usr/bin/env python3
"""
SOXL Live Trading Signal — hourly cron job (10:30am–3:30pm ET)

Pure algorithmic execution — no LLM at runtime.

Strategy: EMA(5/20) + FGI momentum gate (mom_buy=-3, mom_sell=8)
          + FGI velocity gate (vel_w=6, vt=-3)
          + 11% trailing stop from peak

Signal pipeline replicates backtest.py exactly:
  merge_fgi_lagged → add_fgi_features_ext(mom_w=5, vel_w=6)
  → ema_cross(5,20) → apply_mom_gate(-3,8) → apply_vel_gate(vt=-3)

FGI is always lagged 1 calendar day before merge (prevents look-ahead leakage).
Trades execute directly via the Robinhood MCP Python SDK — no Claude in the loop.

Usage:
  python run_signal.py              # normal hourly run
  python run_signal.py --dry-run   # full logic + logging, no actual order
  python run_signal.py --config /path/to/config.json

Auth: reads email/password from env vars:
  export email="your@email.com"
  export password="yourpassword"
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf
import pandas_market_calendars as mcal
import warnings

warnings.filterwarnings("ignore")

LIVE_DIR = Path(__file__).resolve().parent
PROJ_DIR = LIVE_DIR.parent
ET = pytz.timezone("America/New_York")
RH_MCP_URL = "https://agent.robinhood.com/mcp/trading"


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    for key in ("fgi_data_csv", "fgi_data_2_csv", "state_file", "log_file"):
        if key in cfg:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str((LIVE_DIR / p).resolve())
    return cfg


# ── Market calendar ────────────────────────────────────────────────────────────

def is_trading_day() -> bool:
    nyse = mcal.get_calendar("NYSE")
    today = date.today().isoformat()
    return not nyse.schedule(start_date=today, end_date=today).empty

def is_market_open() -> bool:
    """Regular session only: 9:30am–4pm ET, Mon–Fri."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_ = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_ <= now < close_


# ── State persistence ──────────────────────────────────────────────────────────

def load_state(state_file: str) -> dict:
    defaults = {
        "position": 0,
        "entry_price": 0.0,
        "peak_price": 0.0,
        "entry_date": None,
        "last_timestamp": None,
    }
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                return {**defaults, **json.load(f)}
        except Exception:
            pass
    return defaults

def save_state(state: dict, state_file: str) -> None:
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ── FGI loading — exact replica of backtest.py ────────────────────────────────

def load_fgi(fgi_data_csv: str, fgi_data_2_csv: str) -> pd.DataFrame:
    dfs = []
    for path, col in [(fgi_data_csv, "fear_greed"), (fgi_data_2_csv, "Fear Greed")]:
        d = pd.read_csv(path, parse_dates=["Date"])
        d = d.rename(columns={col: "fgi", "Date": "date"})
        d = d[["date", "fgi"]].dropna()
        dfs.append(d)
    fgi = pd.concat(dfs).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return fgi.set_index("date")

def _merge_fgi_raw(price_df: pd.DataFrame, fgi_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    naive = df.index.tz_localize(None) if df.index.tzinfo else df.index
    norm = naive.normalize()
    df["fgi"] = fgi_df["fgi"].reindex(norm, method="ffill").values
    return df.dropna(subset=["fgi"])

def merge_fgi_lagged(price_df: pd.DataFrame, fgi_df: pd.DataFrame) -> pd.DataFrame:
    """1-day lag: each hourly bar uses yesterday's FGI close. No look-ahead."""
    fgi_shifted = fgi_df.copy()
    fgi_shifted.index = fgi_shifted.index + pd.Timedelta(days=1)
    return _merge_fgi_raw(price_df, fgi_shifted)


# ── Indicators — exact replica of backtest.py ─────────────────────────────────

def add_fgi_features_ext(df: pd.DataFrame, mom_w: int = 5, vel_w: int = 3) -> pd.DataFrame:
    naive = df.index.tz_localize(None) if df.index.tzinfo else df.index
    fgi = pd.Series(df["fgi"].values, index=naive)
    daily = fgi.resample("D").first().dropna()
    mom = daily.diff(mom_w)
    vel = mom.diff(vel_w)
    norm = naive.normalize()
    def ff(s): return s.reindex(norm, method="ffill").values
    df["fgi_mom"] = ff(mom)
    df["fgi_vel"] = ff(vel)
    return df.dropna()

def ema_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    sig = pd.Series(0, index=df.index)
    ef = df["Close"].ewm(span=fast, adjust=False).mean()
    es = df["Close"].ewm(span=slow, adjust=False).mean()
    sig[(ef > es) & (ef.shift(1) <= es.shift(1))] = 1
    sig[(ef < es) & (ef.shift(1) >= es.shift(1))] = -1
    return sig

def apply_mom_gate(sig: pd.Series, df: pd.DataFrame, mb: float, ms: float) -> pd.Series:
    s = sig.copy()
    s[(sig == 1) & (df["fgi_mom"] < mb)] = 0
    s[(sig == -1) & (df["fgi_mom"] > ms)] = 0
    return s

def apply_vel_gate(sig: pd.Series, df: pd.DataFrame, vt: float) -> pd.Series:
    s = sig.copy()
    s[(sig == 1) & (df["fgi_vel"] < vt)] = 0
    return s


# ── Price data ─────────────────────────────────────────────────────────────────

def fetch_price_data(ticker: str, lookback_days: int = 90) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{lookback_days}d", interval="1h",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[df.index.to_series().dt.time.between(
        pd.Timestamp("09:30").time(), pd.Timestamp("15:59").time()
    )]
    return df


# ── CSV log ────────────────────────────────────────────────────────────────────

LOG_COLUMNS = [
    "timestamp", "action", "signal", "price", "fgi", "fgi_mom", "fgi_vel",
    "position_before", "position_after", "entry_price", "peak_price",
    "trail_stop", "entry_date", "dry_run",
]

def append_log(log_file: str, row: dict) -> None:
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


# ── Push notification (ntfy.sh, optional) ─────────────────────────────────────

def notify(message: str, topic: str) -> None:
    if not topic:
        return
    try:
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(f"https://ntfy.sh/{topic}", data=message.encode(), method="POST"),
            timeout=5,
        )
    except Exception:
        pass


# ── Robinhood MCP client (async) ──────────────────────────────────────────────

def _mcp_session_context(auth_headers: dict):
    """Return an async context manager for an MCP ClientSession."""
    # Try Streamable HTTP first (newer protocol), fall back to SSE
    try:
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession

        class _StreamableCtx:
            async def __aenter__(self):
                self._outer = streamablehttp_client(RH_MCP_URL, headers=auth_headers)
                r, w, _ = await self._outer.__aenter__()
                self._session = ClientSession(r, w)
                sess = await self._session.__aenter__()
                await sess.initialize()
                return sess

            async def __aexit__(self, *a):
                await self._session.__aexit__(*a)
                await self._outer.__aexit__(*a)

        return _StreamableCtx()

    except ImportError:
        pass

    from mcp.client.sse import sse_client
    from mcp import ClientSession

    class _SseCtx:
        async def __aenter__(self):
            self._outer = sse_client(RH_MCP_URL, headers=auth_headers)
            r, w = await self._outer.__aenter__()
            self._session = ClientSession(r, w)
            sess = await self._session.__aenter__()
            await sess.initialize()
            return sess

        async def __aexit__(self, *a):
            await self._session.__aexit__(*a)
            await self._outer.__aexit__(*a)

    return _SseCtx()


async def execute_buy(session, rh_account: str, ticker: str) -> str:
    """Deploy all buying power into ticker. Dollar-based fractional fill, regular hours."""
    # Response shape: {"buying_power": {"buying_power": "100.00", ...}, ...}
    portfolio_raw = await session.call_tool("get_portfolio", {"account_number": rh_account})
    portfolio = _parse_tool_result(portfolio_raw)
    bp_field = portfolio.get("buying_power", {})
    buying_power = float(bp_field.get("buying_power", 0) if isinstance(bp_field, dict) else bp_field)

    if buying_power < 1.0:
        return f"SKIPPED: buying_power=${buying_power:.2f} below $1 minimum"

    result = await session.call_tool("place_equity_order", {
        "account_number": rh_account,
        "symbol": ticker,
        "side": "buy",
        "type": "market",
        "dollar_amount": f"{buying_power:.2f}",
        "time_in_force": "gfd",
    })
    return f"BUY ${buying_power:.2f} → {_parse_tool_result(result)}"


async def execute_sell(session, rh_account: str, ticker: str) -> str:
    """Sell full position (fractional shares supported in regular hours)."""
    positions_raw = await session.call_tool("get_equity_positions", {"account_number": rh_account})
    positions_data = _parse_tool_result(positions_raw)

    qty = 0.0
    results = positions_data if isinstance(positions_data, list) else [positions_data]
    for pos in results:
        if isinstance(pos, dict) and (pos.get("symbol") == ticker or pos.get("ticker") == ticker):
            qty = float(pos.get("quantity", 0))
            break

    if qty < 0.0001:
        return f"SKIPPED: no {ticker} position found"

    result = await session.call_tool("place_equity_order", {
        "account_number": rh_account,
        "symbol": ticker,
        "side": "sell",
        "type": "market",
        "quantity": f"{qty:.6f}",
        "time_in_force": "gfd",
    })
    return f"SELL {qty:.6f} shares → {_parse_tool_result(result)}"


def _parse_tool_result(result) -> object:
    """Extract JSON data from an MCP tool call result."""
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return text
    return {}


async def _place_order(action: str, cfg: dict) -> str:
    email = os.environ.get("email", "")
    password = os.environ.get("password", "")
    if not email or not password:
        raise RuntimeError("email and password env vars not set")

    auth_headers = {"X-Email": email, "X-Password": password}
    rh_account = cfg.get("rh_account", "")
    ticker = cfg["ticker"]

    async with _mcp_session_context(auth_headers) as session:
        if action == "BUY":
            return await execute_buy(session, rh_account, ticker)
        elif action == "SELL":
            return await execute_sell(session, rh_account, ticker)
        else:
            return "HOLD — no order placed"


def place_order(action: str, cfg: dict, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY RUN] Would place {action} order via Robinhood MCP")
        return
    try:
        summary = asyncio.run(_place_order(action, cfg))
        print(f"[MCP] {summary}")
    except Exception as e:
        print(f"[MCP ERROR] {type(e).__name__}: {e}")
        print("[MCP ERROR] Trade NOT executed. Check credentials and MCP connectivity.")
        raise


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SOXL hourly signal check and trade executor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute signal and log, but do not place orders")
    parser.add_argument("--config", default=str(LIVE_DIR / "config.json"),
                        help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    tag = " (DRY RUN)" if args.dry_run else ""
    print(f"\n[{now_str}] SOXL signal check{tag}")

    # ── Market gate ────────────────────────────────────────────────────────────
    if not is_trading_day():
        print("NYSE holiday — no action.")
        return
    if not is_market_open():
        print("Market is closed — no action.")
        return

    # ── Load state ─────────────────────────────────────────────────────────────
    state = load_state(cfg["state_file"])
    pre_position = state["position"]
    pre_entry = float(state.get("entry_price") or 0)
    pre_peak = float(state.get("peak_price") or 0)

    # ── Fetch data ─────────────────────────────────────────────────────────────
    try:
        fgi_df = load_fgi(cfg["fgi_data_csv"], cfg["fgi_data_2_csv"])
    except Exception as e:
        print(f"ERROR loading FGI CSV: {e}")
        sys.exit(1)

    try:
        price_df = fetch_price_data(cfg["ticker"], cfg.get("lookback_days", 90))
    except Exception as e:
        print(f"ERROR fetching price data: {e}")
        sys.exit(1)

    if len(price_df) < 30:
        print(f"Insufficient price history ({len(price_df)} bars). Skipping.")
        sys.exit(1)

    # ── Signal pipeline (exact backtest.py replication) ────────────────────────
    df = merge_fgi_lagged(price_df, fgi_df)
    df = add_fgi_features_ext(df, mom_w=cfg["mom_w"], vel_w=cfg["vel_w"])

    if len(df) < 5:
        print(f"Insufficient bars after feature computation ({len(df)}). Skipping.")
        sys.exit(1)

    sig_series = apply_vel_gate(
        apply_mom_gate(
            ema_cross(df, cfg["ema_fast"], cfg["ema_slow"]),
            df, cfg["mom_buy"], cfg["mom_sell"],
        ),
        df, cfg["vel_thresh"],
    )

    signal = int(sig_series.iloc[-2])          # last COMPLETED bar — never in-progress
    current_price = float(df["Close"].iloc[-1])
    fgi_val  = float(df["fgi"].iloc[-1])
    fgi_mom  = float(df["fgi_mom"].iloc[-1])
    fgi_vel  = float(df["fgi_vel"].iloc[-1])

    # ── State machine: trailing stop checked before signal ─────────────────────
    action = "HOLD"
    trail_stop = 0.0

    if pre_position == 1:
        peak = max(pre_peak, current_price)
        state["peak_price"] = peak
        trail_stop = peak * (1 - cfg["trail_pct"])

        if current_price <= trail_stop:
            pnl = (current_price - pre_entry) / pre_entry * 100 if pre_entry else 0
            action = "SELL"
            print(f"TRAILING STOP: entry=${pre_entry:.2f}  peak=${peak:.2f}  now=${current_price:.2f}  PnL={pnl:+.1f}%")
            state.update(position=0, entry_price=0.0, peak_price=0.0, entry_date=None)
        elif signal == -1:
            pnl = (current_price - pre_entry) / pre_entry * 100 if pre_entry else 0
            action = "SELL"
            print(f"EMA SELL: entry=${pre_entry:.2f}  now=${current_price:.2f}  PnL={pnl:+.1f}%")
            state.update(position=0, entry_price=0.0, peak_price=0.0, entry_date=None)
    else:
        if signal == 1:
            action = "BUY"
            state.update(
                position=1,
                entry_price=current_price,
                peak_price=current_price,
                entry_date=date.today().isoformat(),
            )

    state["last_timestamp"] = str(df.index[-1])

    # ── Print report ───────────────────────────────────────────────────────────
    print("=" * 58)
    print(f"  {cfg['ticker']}   ${current_price:.2f}")
    print(f"  FGI     {fgi_val:.1f}   mom={fgi_mom:.2f}   vel={fgi_vel:.2f}")
    print(f"  Signal  {signal:+d}  (bar -2 of {len(df)} bars)")
    if pre_position == 1:
        pnl = (current_price - pre_entry) / pre_entry * 100 if pre_entry else 0
        peak_disp = state.get("peak_price") or pre_peak
        print(f"  Pos     IN   entry=${pre_entry:.2f}  peak=${peak_disp:.2f}  stop=${trail_stop:.2f}  PnL={pnl:+.1f}%")
    else:
        print(f"  Pos     OUT")
    print(f"\n  >>> {action} <<<")
    print("=" * 58)

    # ── Log ────────────────────────────────────────────────────────────────────
    append_log(cfg["log_file"], {
        "timestamp":       datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
        "action":          action,
        "signal":          signal,
        "price":           f"{current_price:.4f}",
        "fgi":             f"{fgi_val:.2f}",
        "fgi_mom":         f"{fgi_mom:.4f}",
        "fgi_vel":         f"{fgi_vel:.4f}",
        "position_before": pre_position,
        "position_after":  state["position"],
        "entry_price":     f"{pre_entry:.4f}",
        "peak_price":      f"{pre_peak:.4f}",
        "trail_stop":      f"{trail_stop:.4f}",
        "entry_date":      state.get("entry_date", ""),
        "dry_run":         1 if args.dry_run else 0,
    })

    save_state(state, cfg["state_file"])

    # ── Execute ────────────────────────────────────────────────────────────────
    if action != "HOLD":
        notify(
            f"SOXL {action} @ ${current_price:.2f}  FGI={fgi_val:.0f}  mom={fgi_mom:.1f}",
            cfg.get("ntfy_topic", ""),
        )
        try:
            place_order(action, cfg, args.dry_run)
        except Exception:
            # Error already printed inside place_order; don't crash cron
            sys.exit(1)


if __name__ == "__main__":
    main()
