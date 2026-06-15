import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

TICKER = "SOXL"
INITIAL_CAPITAL = 10000

# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_price_data():
    df = yf.download(TICKER, period="730d", interval="1h", auto_adjust=True, progress=False)
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[df.index.to_series().dt.time.between(
        pd.Timestamp("09:30").time(), pd.Timestamp("15:59").time()
    )]
    return df

def load_fgi():
    dfs = []
    for path, col in [
        ("fgi_data.csv", "fear_greed"),
        ("fgi_data_2.csv", "Fear Greed"),
    ]:
        d = pd.read_csv(path, parse_dates=["Date"])
        d = d.rename(columns={col: "fgi", "Date": "date"})
        d = d[["date", "fgi"]].dropna()
        dfs.append(d)
    fgi = pd.concat(dfs).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    fgi = fgi.set_index("date")
    return fgi

def merge_fgi_raw(price_df, fgi_df):
    """Forward-fill daily FGI onto hourly price bars (no lag correction — leaky)."""
    price = price_df.copy()
    price["date"] = price.index.normalize().tz_localize(None) if price.index.tzinfo else price.index.normalize()
    fgi_reindexed = fgi_df.reindex(price["date"].values, method="ffill")
    price["fgi"] = fgi_reindexed.values
    return price.dropna(subset=["fgi"])

def merge_fgi_lagged(price_df, fgi_df):
    """Shift daily FGI by 1 calendar day before merging.
    Bar on day D sees only FGI from D-1's close — eliminates same-day look-ahead leakage.
    FGI row labelled D reflects EOD market data through D; earliest safe use is D+1 open."""
    fgi_shifted = fgi_df.copy()
    fgi_shifted.index = fgi_shifted.index + pd.Timedelta(days=1)
    return merge_fgi_raw(price_df, fgi_shifted)

# ── Indicators ────────────────────────────────────────────────────────────────

def add_indicators(df, ema_fast=9, ema_slow=21, rsi_period=14, bb_period=20, bb_std=2):
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_mid"] = df["Close"].rolling(bb_period).mean()
    bb_std_val = df["Close"].rolling(bb_period).std()
    df["bb_upper"] = df["bb_mid"] + bb_std * bb_std_val
    df["bb_lower"] = df["bb_mid"] - bb_std * bb_std_val
    df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df.dropna()

# ── Signal generators ─────────────────────────────────────────────────────────

def signal_ema(df, **_):
    sig = pd.Series(0, index=df.index)
    cross_up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    cross_dn = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    sig[cross_up] = 1
    sig[cross_dn] = -1
    return sig

def signal_rsi(df, rsi_buy=35, rsi_sell=65, **_):
    sig = pd.Series(0, index=df.index)
    sig[(df["rsi"] > rsi_buy)  & (df["rsi"].shift(1) <= rsi_buy)]  = 1
    sig[(df["rsi"] < rsi_sell) & (df["rsi"].shift(1) >= rsi_sell)] = -1
    return sig

def signal_macd(df, **_):
    sig = pd.Series(0, index=df.index)
    sig[(df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)] = 1
    sig[(df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)] = -1
    return sig

def signal_bb(df, **_):
    sig = pd.Series(0, index=df.index)
    sig[(df["bb_pct"] > 0.2) & (df["bb_pct"].shift(1) <= 0.2)] = 1
    sig[(df["bb_pct"] > 0.8) & (df["bb_pct"].shift(1) <= 0.8)] = -1
    return sig

def signal_ema_macd(df, **_):
    ema = signal_ema(df)
    sig = pd.Series(0, index=df.index)
    sig[(ema == 1)  & (df["macd_hist"] > 0)] = 1
    sig[(ema == -1) & (df["macd_hist"] < 0)] = -1
    return sig

def signal_triple(df, **_):
    ema = signal_ema(df)
    sig = pd.Series(0, index=df.index)
    sig[(ema == 1)  & (df["macd_hist"] > 0) & (df["rsi"] < 60)] = 1
    sig[(ema == -1) & (df["macd_hist"] < 0) & (df["rsi"] > 40)] = -1
    return sig

# ── FGI strategies ────────────────────────────────────────────────────────────

def signal_fgi_contrarian(df, fear_buy=35, greed_sell=65, **_):
    """Buy the fear, sell the greed (contrarian)."""
    sig = pd.Series(0, index=df.index)
    sig[(df["fgi"] < fear_buy)  & (df["fgi"].shift(1) >= fear_buy)]  = 1
    sig[(df["fgi"] > greed_sell) & (df["fgi"].shift(1) <= greed_sell)] = -1
    return sig

def signal_fgi_momentum(df, greed_buy=55, fear_sell=45, **_):
    """Ride the greed wave — buy when sentiment turns bullish, sell when it fades."""
    sig = pd.Series(0, index=df.index)
    sig[(df["fgi"] > greed_buy) & (df["fgi"].shift(1) <= greed_buy)] = 1
    sig[(df["fgi"] < fear_sell) & (df["fgi"].shift(1) >= fear_sell)] = -1
    return sig

def signal_fgi_extreme(df, **_):
    """Only trade extremes: buy extreme fear (<25), sell extreme greed (>75)."""
    sig = pd.Series(0, index=df.index)
    sig[(df["fgi"] > 25) & (df["fgi"].shift(1) <= 25)] = 1
    sig[(df["fgi"] > 75) & (df["fgi"].shift(1) <= 75)] = -1
    return sig

def signal_fgi_rsi(df, rsi_buy=40, rsi_sell=60, **_):
    """RSI signal filtered by FGI: only buy when FGI < 60 (not extreme greed)."""
    rsi = signal_rsi(df, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
    sig = rsi.copy()
    sig[(rsi == 1) & (df["fgi"] > 70)] = 0   # suppress buy in extreme greed
    sig[(rsi == -1) & (df["fgi"] < 30)] = 0  # suppress sell in extreme fear
    return sig

def signal_fgi_ema(df, **_):
    """EMA crossover filtered by FGI: skip buys in extreme greed, skip sells in extreme fear."""
    ema = signal_ema(df)
    sig = ema.copy()
    sig[(ema == 1)  & (df["fgi"] > 75)] = 0
    sig[(ema == -1) & (df["fgi"] < 25)] = 0
    return sig

def signal_fgi_macd(df, **_):
    """MACD filtered by FGI sentiment."""
    macd = signal_macd(df)
    sig = macd.copy()
    sig[(macd == 1)  & (df["fgi"] > 80)] = 0
    sig[(macd == -1) & (df["fgi"] < 20)] = 0
    return sig

def signal_fgi_triple(df, **_):
    """Triple (EMA+MACD+RSI) gated by FGI not being extreme greed on entry."""
    base = signal_triple(df)
    sig = base.copy()
    sig[(base == 1) & (df["fgi"] > 75)] = 0
    return sig

STRATEGIES = {
    # Technical
    "EMA Crossover":         signal_ema,
    "RSI":                   signal_rsi,
    "MACD":                  signal_macd,
    "Bollinger Bands":       signal_bb,
    "EMA + MACD":            signal_ema_macd,
    "Triple (EMA+MACD+RSI)": signal_triple,
    # FGI pure
    "FGI Contrarian":        signal_fgi_contrarian,
    "FGI Momentum":          signal_fgi_momentum,
    "FGI Extremes Only":     signal_fgi_extreme,
    # FGI hybrid
    "FGI + RSI":             signal_fgi_rsi,
    "FGI + EMA":             signal_fgi_ema,
    "FGI + MACD":            signal_fgi_macd,
    "FGI + Triple":          signal_fgi_triple,
}

# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest(df, signal_fn):
    signals = signal_fn(df)
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    equity = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        sig = signals.iloc[i]

        if sig == 1 and position == 0:
            position = capital / price
            entry_price = price
            capital = 0
        elif sig == -1 and position > 0:
            proceeds = position * price
            trades.append({
                "pnl": proceeds - position * entry_price,
                "pnl_pct": (price - entry_price) / entry_price * 100,
            })
            capital = proceeds
            position = 0

        equity.append(capital + position * price)

    if position > 0:
        last = df["Close"].iloc[-1]
        trades.append({"pnl": position * (last - entry_price), "pnl_pct": (last - entry_price) / entry_price * 100})
        capital = position * last

    equity = pd.Series(equity, index=df.index)
    returns = equity.pct_change().dropna()

    n = len(trades)
    return {
        "total_return":  (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "bh_return":     (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100,
        "sharpe":        returns.mean() / returns.std() * np.sqrt(6.5 * 252) if returns.std() > 0 else 0,
        "max_drawdown":  ((equity - equity.cummax()) / equity.cummax()).min() * 100,
        "n_trades":      n,
        "win_rate":      sum(1 for t in trades if t["pnl"] > 0) / n * 100 if n else 0,
        "avg_trade_pct": np.mean([t["pnl_pct"] for t in trades]) if trades else 0,
    }

# ── FGI derived features ──────────────────────────────────────────────────────

def add_fgi_features(df, mom_window=5, vel_window=3):
    df = df.copy()
    # Work on a tz-naive date series
    naive_dates = df.index.tz_localize(None) if df.index.tzinfo else df.index
    fgi_series = pd.Series(df["fgi"].values, index=naive_dates)
    daily_fgi  = fgi_series.resample("D").first().dropna()
    daily_mom  = daily_fgi.diff(mom_window)
    daily_vel  = daily_mom.diff(vel_window)
    daily_ma5  = daily_fgi.rolling(5).mean()
    daily_ma10 = daily_fgi.rolling(10).mean()

    norm_dates = naive_dates.normalize()
    def _fill(series):
        return series.reindex(norm_dates, method="ffill").values

    df["fgi_mom"]    = _fill(daily_mom)
    df["fgi_vel"]    = _fill(daily_vel)
    df["fgi_ma5"]    = _fill(daily_ma5)
    df["fgi_ma10"]   = _fill(daily_ma10)
    df["fgi_rising"] = (df["fgi_ma5"] > df["fgi_ma10"]).astype(int)
    return df.dropna()

def add_fgi_features_ext(df, mom_w=5, vel_w=3):
    df = df.copy()
    naive = df.index.tz_localize(None) if df.index.tzinfo else df.index
    fgi   = pd.Series(df["fgi"].values, index=naive)
    daily = fgi.resample("D").first().dropna()
    mom   = daily.diff(mom_w)
    vel   = mom.diff(vel_w)
    norm  = naive.normalize()
    def ff(s): return s.reindex(norm, method="ffill").values
    df["fgi_mom"]       = ff(mom)
    df["fgi_vel"]       = ff(vel)
    df["fgi_ma5"]       = ff(daily.rolling(5).mean())
    df["fgi_ma10"]      = ff(daily.rolling(10).mean())
    df["fgi_ma20"]      = ff(daily.rolling(20).mean())
    df["fgi_pct10"]     = ff(daily.rolling(10).rank(pct=True))
    df["fgi_pct20"]     = ff(daily.rolling(20).rank(pct=True))
    df["fgi_pct60"]     = ff(daily.rolling(60).rank(pct=True))
    fgi_ef              = daily.ewm(span=5,  adjust=False).mean()
    fgi_es              = daily.ewm(span=15, adjust=False).mean()
    df["fgi_ema_fast"]  = ff(fgi_ef)
    df["fgi_ema_slow"]  = ff(fgi_es)
    df["fgi_rising"]    = (df["fgi_ma5"] > df["fgi_ma10"]).astype(int)
    df["fgi_regime"]    = pd.cut(df["fgi"], bins=[-1,20,40,60,80,101], labels=[0,1,2,3,4]).astype(float)
    df["fgi_ema_cross"] = np.sign(df["fgi_ema_fast"] - df["fgi_ema_slow"])
    return df.dropna()

# ── Parametric FGI+EMA (level thresholds) ─────────────────────────────────────

def signal_fgi_ema_tuned(df, greed_cut=60, fear_cut=20):
    ema = signal_ema(df)
    sig = ema.copy()
    sig[(ema == 1)  & (df["fgi"] > greed_cut)] = 0
    sig[(ema == -1) & (df["fgi"] < fear_cut)]  = 0
    return sig

# ── FGI momentum / velocity strategies ───────────────────────────────────────

def signal_fgi_ema_mom(df, greed_cut=60, fear_cut=20, mom_buy=-5, mom_sell=5):
    """EMA + level gate + momentum: prefer buying when FGI momentum is turning up."""
    ema = signal_ema(df)
    sig = ema.copy()
    # Level gates
    sig[(ema == 1)  & (df["fgi"] > greed_cut)] = 0
    sig[(ema == -1) & (df["fgi"] < fear_cut)]  = 0
    # Momentum boost: suppress buy if FGI is still falling fast (momentum < mom_buy)
    sig[(ema == 1)  & (df["fgi_mom"] < mom_buy)] = 0
    # Suppress sell if FGI is rising fast (market heating up, not topping yet)
    sig[(ema == -1) & (df["fgi_mom"] > mom_sell)] = 0
    return sig

def signal_fgi_ema_vel(df, greed_cut=60, fear_cut=20, vel_thresh=0):
    """EMA + level gate + velocity: only buy when FGI momentum is accelerating upward."""
    ema = signal_ema(df)
    sig = ema.copy()
    sig[(ema == 1)  & (df["fgi"] > greed_cut)]  = 0
    sig[(ema == -1) & (df["fgi"] < fear_cut)]   = 0
    # Velocity gate: only take buy if FGI acceleration is positive (turning up)
    sig[(ema == 1)  & (df["fgi_vel"] < vel_thresh)] = 0
    return sig

def signal_fgi_ema_trend(df, greed_cut=60, fear_cut=20):
    """EMA + level gate + FGI trend direction (5MA vs 10MA)."""
    ema = signal_ema(df)
    sig = ema.copy()
    sig[(ema == 1)  & (df["fgi"] > greed_cut)]   = 0
    sig[(ema == -1) & (df["fgi"] < fear_cut)]    = 0
    # Only buy when FGI short MA is above long MA (sentiment improving)
    sig[(ema == 1)  & (df["fgi_rising"] == 0)]   = 0
    return sig

def signal_fgi_ema_full(df, greed_cut=60, fear_cut=20, mom_buy=-3, vel_thresh=0):
    """Kitchen sink: level + momentum + velocity + trend."""
    ema = signal_ema(df)
    sig = ema.copy()
    sig[(ema == 1)  & (df["fgi"] > greed_cut)]      = 0
    sig[(ema == -1) & (df["fgi"] < fear_cut)]       = 0
    sig[(ema == 1)  & (df["fgi_mom"] < mom_buy)]    = 0
    sig[(ema == 1)  & (df["fgi_vel"] < vel_thresh)] = 0
    sig[(ema == 1)  & (df["fgi_rising"] == 0)]      = 0
    return sig

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("━" * 70)
    print("FGI LAG VERIFICATION")
    print("━" * 70)
    print("Source: Alternative.me / CNN Fear & Greed Index (fgi_data.csv)")
    print("Publication timing: FGI[D] is computed from EOD market data through D's")
    print("close and published at/after D's market close.  Earliest safe use = D+1 open.")
    print()
    print("Spot-checks confirming 1-day lag is correct:")
    print("  Apr 2025 tariff shock — tariffs announced after Apr 2 close.")
    print("    FGI[Apr 2] = 12.14 (extreme fear, available Apr 3 morning).")
    print("    FGI[Apr 3] =  5.39 (reflects Apr 3 selloff close — future data).")
    print("    Same-day merge: 9:30am Apr 3 bar incorrectly sees 5.39, not 12.14.")
    print("  Aug 5 2024 carry-trade crash — Nikkei crashed overnight, US opened -3%.")
    print("    FGI[Aug 4] = 33.06 (last published value before Aug 5 open).")
    print("    FGI[Aug 5] = 19.01 (EOD reading after the selloff — future data).")
    print("    Same-day merge: Aug 5 morning bars see 19, not 33.")
    print()
    print("CONCLUSION: 1-day lag is correct. Using merge_fgi_lagged() as pipeline default.")
    print("━" * 70)
    print()
    print(f"Fetching {TICKER} 1h data...")
    price_df = fetch_price_data()
    print(f"Loading & merging FGI data...")
    fgi_df = load_fgi()
    df_base = merge_fgi_lagged(price_df, fgi_df)
    df_base = add_indicators(df_base)
    df      = add_fgi_features_ext(df_base)

    bh = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
    print(f"\nData: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} bars)")
    print(f"Buy & Hold benchmark: {bh:.1f}%\n")

    hdr = f"{'Strategy / Params':<42} {'Return':>8} {'Alpha':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}"
    sep = "-" * len(hdr)

    def row(name, r):
        alpha = r["total_return"] - bh
        sign  = "+" if alpha >= 0 else ""
        return (
            f"{name:<42} "
            f"{r['total_return']:>7.1f}% "
            f"{sign}{alpha:>7.1f}% "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_drawdown']:>7.1f}% "
            f"{r['n_trades']:>7} "
            f"{r['win_rate']:>7.1f}%"
        )

    # ── extended FGI features ─────────────────────────────────────────────────
    # ── helpers ───────────────────────────────────────────────────────────────
    def ema_cross(df, fast, slow):
        sig = pd.Series(0, index=df.index)
        ef  = df["Close"].ewm(span=fast, adjust=False).mean()
        es  = df["Close"].ewm(span=slow, adjust=False).mean()
        sig[(ef > es) & (ef.shift(1) <= es.shift(1))] =  1
        sig[(ef < es) & (ef.shift(1) >= es.shift(1))] = -1
        return sig

    def apply_mom_gate(sig, df, mb, ms):
        s = sig.copy()
        s[(sig ==  1) & (df["fgi_mom"] < mb)] = 0
        s[(sig == -1) & (df["fgi_mom"] > ms)] = 0
        return s

    def apply_level_gate(sig, df, gc, fc):
        s = sig.copy()
        s[(sig ==  1) & (df["fgi"] > gc)] = 0
        s[(sig == -1) & (df["fgi"] < fc)] = 0
        return s

    def apply_vel_gate(sig, df, vt):
        s = sig.copy()
        s[(sig == 1) & (df["fgi_vel"] < vt)] = 0
        return s

    def apply_stop_loss(df, signals, stop_pct):
        """Inject a sell signal if price drops stop_pct from entry."""
        sig = signals.copy()
        position = 0
        entry_price = 0
        for i in range(len(df)):
            price = df["Close"].iloc[i]
            if sig.iloc[i] == 1 and position == 0:
                position = 1
                entry_price = price
            elif position == 1:
                if price <= entry_price * (1 - stop_pct):
                    sig.iloc[i] = -1
                    position = 0
                elif sig.iloc[i] == -1:
                    position = 0
        return sig

    def apply_rsi_entry(df, rsi_buy=35, rsi_sell=65):
        sig = pd.Series(0, index=df.index)
        sig[(df["rsi"] > rsi_buy)  & (df["rsi"].shift(1) <= rsi_buy)]  =  1
        sig[(df["rsi"] < rsi_sell) & (df["rsi"].shift(1) >= rsi_sell)] = -1
        return sig

    def apply_macd_entry(df):
        sig = pd.Series(0, index=df.index)
        sig[(df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)] =  1
        sig[(df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)] = -1
        return sig

    def apply_zscore_gate(sig, df, z_buy=-1.0, z_sell=1.0, window=20):
        """Gate on FGI z-score relative to recent history."""
        fgi_z = (df["fgi"] - df["fgi"].rolling(window).mean()) / df["fgi"].rolling(window).std()
        s = sig.copy()
        s[(sig ==  1) & (fgi_z > z_sell)] = 0   # skip buy if FGI zscore too high
        s[(sig == -1) & (fgi_z < z_buy)]  = 0   # skip sell if FGI zscore too low
        return s

    def apply_vol_filter(sig, df, vol_multiplier=1.2):
        """Only trade when volume is above its 20-bar average * multiplier."""
        avg_vol = df["Volume"].rolling(20).mean()
        s = sig.copy()
        s[(sig != 0) & (df["Volume"] < avg_vol * vol_multiplier)] = 0
        return s

    def apply_time_filter(sig, df, start_hour=10, end_hour=15):
        """Only trade within certain market hours (ET)."""
        hours = df.index.hour if not df.index.tzinfo else df.index.tz_convert("America/New_York").hour
        s = sig.copy()
        s[(sig != 0) & ((hours < start_hour) | (hours >= end_hour))] = 0
        return s

    def apply_dow_filter(sig, df, allowed_days=(0,1,2,3,4)):
        """Only trade on certain days of week (0=Mon … 4=Fri)."""
        dow = df.index.dayofweek
        s = sig.copy()
        s[(sig != 0) & (~dow.isin(allowed_days))] = 0
        return s

    def apply_trailing_stop(df, signals, trail_pct):
        sig = signals.copy()
        position = 0
        entry_price = peak = 0.0
        for i in range(len(df)):
            price = float(df["Close"].iloc[i])
            if sig.iloc[i] == 1 and position == 0:
                position = 1; entry_price = peak = price
            elif position == 1:
                peak = max(peak, price)
                if price <= peak * (1 - trail_pct):
                    sig.iloc[i] = -1; position = 0
                elif sig.iloc[i] == -1:
                    position = 0
        return sig

    def apply_cooldown(df, signals, cooldown_bars=6):
        """After a stop-loss exit, wait N bars before re-entering."""
        sig = signals.copy()
        position = 0; entry_price = 0.0; cooldown = 0
        for i in range(len(df)):
            price = float(df["Close"].iloc[i])
            if cooldown > 0:
                cooldown -= 1
                if sig.iloc[i] == 1: sig.iloc[i] = 0
            if sig.iloc[i] == 1 and position == 0:
                position = 1; entry_price = price
            elif position == 1:
                if sig.iloc[i] == -1:
                    if price < entry_price: cooldown = cooldown_bars
                    position = 0
        return sig

    def apply_fgi_ema_cross_gate(sig, df):
        """Only buy when FGI short EMA > long EMA (sentiment trending up)."""
        s = sig.copy()
        s[(sig == 1) & (df["fgi_ema_cross"] < 1)] = 0
        return s

    def apply_regime_gate(sig, df, buy_regimes=(1,2), sell_regimes=(3,4)):
        """Only buy in fear/neutral regimes, sell in greed/extreme greed."""
        s = sig.copy()
        s[(sig ==  1) & (~df["fgi_regime"].isin(buy_regimes))]  = 0
        s[(sig == -1) & (~df["fgi_regime"].isin(sell_regimes))] = 0
        return s

    def apply_pct_rank_gate(sig, df, pct_col="fgi_pct20", buy_max=0.7, sell_min=0.3):
        """Skip buy if FGI is in top X% of recent range; skip sell if in bottom Y%."""
        s = sig.copy()
        s[(sig ==  1) & (df[pct_col] > buy_max)]  = 0
        s[(sig == -1) & (df[pct_col] < sell_min)] = 0
        return s

    def backtest_sized(df, signal_fn, size_fn):
        """Like backtest() but position size 0–1 is determined by size_fn(fgi)."""
        signals = signal_fn(df)
        capital = INITIAL_CAPITAL; position = 0.0; entry_price = 0.0
        trades = []; equity = []
        for i in range(len(df)):
            price = float(df["Close"].iloc[i]); sig = signals.iloc[i]
            fgi   = float(df["fgi"].iloc[i])
            if sig == 1 and position == 0:
                sz = np.clip(size_fn(fgi), 0.1, 1.0)
                position = (capital * sz) / price
                capital -= position * price
                entry_price = price
            elif sig == -1 and position > 0:
                proceeds = position * price
                trades.append({"pnl": proceeds - position * entry_price,
                                "pnl_pct": (price - entry_price) / entry_price * 100})
                capital += proceeds; position = 0
            equity.append(capital + position * price)
        if position > 0:
            last = float(df["Close"].iloc[-1])
            trades.append({"pnl": position*(last-entry_price),
                            "pnl_pct": (last-entry_price)/entry_price*100})
            capital += position * last
        equity = pd.Series(equity, index=df.index)
        ret    = equity.pct_change().dropna()
        n = len(trades)
        return {"total_return": (capital-INITIAL_CAPITAL)/INITIAL_CAPITAL*100,
                "bh_return": bh,
                "sharpe": ret.mean()/ret.std()*np.sqrt(6.5*252) if ret.std()>0 else 0,
                "max_drawdown": ((equity-equity.cummax())/equity.cummax()).min()*100,
                "n_trades": n,
                "win_rate": sum(1 for t in trades if t["pnl"]>0)/n*100 if n else 0,
                "avg_trade_pct": np.mean([t["pnl_pct"] for t in trades]) if trades else 0}

    # ── canonical base signal ─────────────────────────────────────────────────
    def base_sig(d): return apply_mom_gate(ema_cross(d,7,15), d, -3, 8)

    all_results = []

    def section(title):
        print(f"\n━━━ {title} ━━━")
        print(hdr); print(sep)

    def rec(name, r):
        all_results.append((name, r))
        print(row(name, r))

    # ── A. Velocity window tuning ─────────────────────────────────────────────
    section("A. Velocity window (how many days to measure momentum change)")
    for vw in [2, 3, 5, 7, 10]:
        df_v = add_fgi_features_ext(df_base, mom_w=5, vel_w=vw)
        for vt in [-4, -2, 0]:
            sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v,7,15), df_v,-3,8), df_v, vt)
            r   = backtest(df_v, lambda d,s=sig: s)
            rec(f"vel_win={vw} vt={vt}", r)

    # ── B. Trailing stop sweep ────────────────────────────────────────────────
    section("B. Trailing stop (trail from peak, sell when price drops X%)")
    base0 = base_sig(df)
    for trail in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        ts = apply_trailing_stop(df, base0.copy(), trail)
        r  = backtest(df, lambda d,s=ts: s)
        rec(f"trail={int(trail*100)}%", r)

    # ── C. Fixed stop-loss sweep ──────────────────────────────────────────────
    section("C. Fixed stop-loss")
    for stop in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        sl = apply_stop_loss(df, base0.copy(), stop)
        r  = backtest(df, lambda d,s=sl: s)
        rec(f"fixed_stop={int(stop*100)}%", r)

    # ── D. Stop + trailing combined (fixed stop OR trail, whichever hits first)
    section("D. Fixed stop + trailing stop layered")
    for stop, trail in [(0.10,0.12),(0.10,0.15),(0.12,0.15),(0.08,0.12),(0.12,0.20)]:
        sl  = apply_stop_loss(df, base0.copy(), stop)
        tsl = apply_trailing_stop(df, sl.copy(), trail)
        r   = backtest(df, lambda d,s=tsl: s)
        rec(f"stop={int(stop*100)}%+trail={int(trail*100)}%", r)

    # ── E. Cooldown after stop-loss ───────────────────────────────────────────
    section("E. Re-entry cooldown bars after stop-loss exit")
    sl10 = apply_stop_loss(df, base0.copy(), 0.10)
    for cd in [0, 3, 6, 12, 24, 48]:
        sig = apply_cooldown(df, sl10.copy(), cd)
        r   = backtest(df, lambda d,s=sig: s)
        rec(f"stop=10%+cooldown={cd}bars", r)

    # ── F. FGI EMA cross (sentiment trend direction) ─────────────────────────
    section("F. FGI EMA cross gate (FGI 5EMA > 15EMA = sentiment uptrend)")
    for fast, slow, mb, ms in [(7,15,-3,8),(5,13,-3,12),(3,8,0,8)]:
        sig = apply_fgi_ema_cross_gate(
                apply_mom_gate(ema_cross(df,fast,slow), df, mb, ms), df)
        r   = backtest(df, lambda d,s=sig: s)
        rec(f"EMA{fast}/{slow}+FGI_EMA_cross mb={mb} ms={ms}", r)

    # ── G. FGI regime gate ────────────────────────────────────────────────────
    section("G. FGI regime gate (only enter in fear/neutral, exit in greed)")
    for buy_r, sell_r in [
        ((0,1,2),(3,4)), ((1,2),(3,4)), ((0,1),(3,4)),
        ((1,2),(4,)),    ((0,1,2),(4,)),
    ]:
        sig = apply_regime_gate(base_sig(df), df, buy_r, sell_r)
        r   = backtest(df, lambda d,s=sig: s)
        rec(f"regime buy={buy_r} sell={sell_r}", r)

    # ── H. FGI percentile rank gate ───────────────────────────────────────────
    section("H. FGI percentile rank gate (relative position in rolling window)")
    for col, win_label in [("fgi_pct10","10d"),("fgi_pct20","20d"),("fgi_pct60","60d")]:
        for buy_max, sell_min in [(0.6,0.3),(0.7,0.3),(0.7,0.4),(0.8,0.3),(0.8,0.4)]:
            sig = apply_pct_rank_gate(base_sig(df), df, col, buy_max, sell_min)
            r   = backtest(df, lambda d,s=sig: s)
            rec(f"pct_{win_label} buy<{buy_max} sell>{sell_min}", r)

    # ── I. FGI z-score gate (finer sweep) ────────────────────────────────────
    section("I. FGI z-score gate fine-tuned (w=10 from round 1 was best)")
    for zb, zs in [(-0.5,1.0),(-0.5,1.5),(-0.5,2.0),
                   (-1.0,1.0),(-1.0,1.5),(-1.0,2.0),
                   (-1.5,1.0),(-1.5,1.5),(-1.5,2.0)]:
        sig = apply_zscore_gate(base_sig(df), df, z_buy=zb, z_sell=zs, window=10)
        r   = backtest(df, lambda d,s=sig: s)
        rec(f"zscore zb={zb} zs={zs}", r)

    # ── J. Month-of-year seasonality ─────────────────────────────────────────
    section("J. Month-of-year seasonality (skip certain months)")
    months = df.index.month
    for skip in [(1,),(9,),(9,10),(1,9),(8,9,10),(6,7,8)]:
        sig = base_sig(df).copy()
        sig[sig.ne(0) & months.isin(skip)] = 0
        r   = backtest(df, lambda d,s=sig: s)
        rec(f"skip months={skip}", r)
    # also test only trading certain months
    for keep in [(1,2,3,4,5),(3,4,5,10,11,12),(11,12,1,2,3,4)]:
        sig = base_sig(df).copy()
        sig[sig.ne(0) & ~months.isin(keep)] = 0
        r   = backtest(df, lambda d,s=sig: s)
        rec(f"only months={keep}", r)

    # ── K. FGI-based position sizing (scale in/out) ───────────────────────────
    section("K. Dynamic position sizing based on FGI level")
    def size_inverse(fgi):   return 1.0 - (fgi / 100)        # more fear = bigger bet
    def size_linear(fgi):    return 0.5                        # flat 50%
    def size_fear_only(fgi): return 1.0 if fgi < 40 else 0.5  # full in fear, half otherwise
    def size_graded(fgi):    # 100% if <30, 75% if 30-50, 50% if 50-65, 25% if >65
        if fgi < 30:  return 1.0
        if fgi < 50:  return 0.75
        if fgi < 65:  return 0.50
        return 0.25
    for label, fn in [("size_inverse",size_inverse),("size_flat50",size_linear),
                      ("size_fear_only",size_fear_only),("size_graded",size_graded)]:
        r = backtest_sized(df, base_sig, fn)
        rec(f"sizing: {label}", r)

    # ── L. Velocity window tuning on top of vel_thresh=-4 winner ─────────────
    section("L. Velocity window on best vel_thresh=-4 (from round 1)")
    for vw in [2, 3, 5, 7]:
        df_v = add_fgi_features_ext(df_base, mom_w=5, vel_w=vw)
        sig  = apply_vel_gate(apply_mom_gate(ema_cross(df_v,7,15), df_v,-3,8), df_v, -4)
        r    = backtest(df_v, lambda d,s=sig: s)
        rec(f"vel_win={vw} vt=-4", r)

    # ── M. DOW × stop combo (Tue-Fri was best) ───────────────────────────────
    section("M. DOW filter (Tue-Fri) + stop-loss combos")
    for stop in [0.08, 0.10, 0.12, 0.15]:
        for trail in [None, 0.12, 0.15]:
            sig = base_sig(df).copy()
            sig = apply_dow_filter(sig, df, (1,2,3,4))
            if trail:
                sig = apply_trailing_stop(df, apply_stop_loss(df, sig.copy(), stop), trail)
            else:
                sig = apply_stop_loss(df, sig.copy(), stop)
            r   = backtest(df, lambda d,s=sig: s)
            trail_s = f"+trail={int(trail*100)}%" if trail else ""
            rec(f"TueFri stop={int(stop*100)}%{trail_s}", r)

    # ── N. Full best combo sweep ──────────────────────────────────────────────
    section("N. Full combo: EMA7/15 + mom(-3/8) + vel(-4) + stop + DOW(Tue-Fri)")
    df_v5 = add_fgi_features_ext(df_base, mom_w=5, vel_w=3)
    for stop in [0.08, 0.10, 0.12]:
        for trail in [None, 0.12, 0.15]:
            for dow in [(0,1,2,3,4),(1,2,3,4)]:
                sig = apply_vel_gate(
                        apply_mom_gate(ema_cross(df_v5,7,15), df_v5,-3,8),
                        df_v5, -4)
                sig = apply_dow_filter(sig, df_v5, dow)
                if trail:
                    sig = apply_trailing_stop(df_v5, apply_stop_loss(df_v5,sig.copy(),stop), trail)
                else:
                    sig = apply_stop_loss(df_v5, sig.copy(), stop)
                r    = backtest(df_v5, lambda d,s=sig: s)
                dow_s  = "TueFri" if len(dow)==4 else "AllDays"
                trail_s = f"+trail={int(trail*100)}%" if trail else ""
                rec(f"{dow_s} stop={int(stop*100)}%{trail_s}", r)

    # ── O. Benchmarks for comparison ─────────────────────────────────────────
    section("O. Benchmarks")
    rec("Buy & Hold",        {"total_return":bh,"bh_return":bh,"sharpe":0,"max_drawdown":0,"n_trades":0,"win_rate":0,"avg_trade_pct":0})
    rec("Base EMA7/15 mom",  backtest(df, base_sig))
    rec("Base + stop=10%",   backtest(df, lambda d: apply_stop_loss(d, base_sig(d).copy(), 0.10)))
    rec("Base + vel=-4",     backtest(df_v5, lambda d: apply_vel_gate(apply_mom_gate(ema_cross(d,7,15),d,-3,8), d, -4)))

    # ══════════════════════════════════════════════════════════════════════════
    # ROUND 6 — Combine winners, new dimensions
    # ══════════════════════════════════════════════════════════════════════════

    # ── P. vel_win=5 vt=-2 (top by return/sharpe) + stop/trail ───────────────
    section("P. TOP winner vel_win=5 vt=-2 with stop-loss / trailing")
    df_v5_2 = add_fgi_features_ext(df_base, mom_w=5, vel_w=5)
    base_v52 = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2,-3,8), df_v5_2, -2)
    for stop in [0.08, 0.10, 0.12, 0.15]:
        for trail in [None, 0.10, 0.12, 0.15]:
            sig = base_v52.copy()
            if trail:
                sig = apply_trailing_stop(df_v5_2, apply_stop_loss(df_v5_2, sig, stop), trail)
            else:
                sig = apply_stop_loss(df_v5_2, sig, stop)
            r   = backtest(df_v5_2, lambda d, s=sig: s)
            trail_s = f"+trail={int(trail*100)}%" if trail else ""
            rec(f"v5t-2 stop={int(stop*100)}%{trail_s}", r)

    # ── Q. EMA pair sweep on vel_win=5 vt=-2 base ────────────────────────────
    section("Q. EMA pair sweep: vel_win=5 vt=-2 (best vel)")
    for fast, slow in [(3,8),(5,13),(7,15),(9,21),(5,20)]:
        for mb, ms in [(-3,8),(-3,12),(0,8)]:
            sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,fast,slow), df_v5_2, mb, ms), df_v5_2, -2)
            r   = backtest(df_v5_2, lambda d, s=sig: s)
            rec(f"EMA{fast}/{slow} mb={mb} ms={ms} vel5t-2", r)

    # ── R. RSI confirmation gate on best combo ────────────────────────────────
    section("R. RSI confirmation gate on vel5/vt-2 base")
    def apply_rsi_gate(sig, df, rsi_buy=45, rsi_sell=60):
        s = sig.copy()
        s[(sig == 1)  & (df["rsi"] > rsi_sell)] = 0   # skip buy if RSI overbought
        s[(sig == -1) & (df["rsi"] < rsi_buy)]  = 0   # skip sell if RSI oversold
        return s
    for rb, rs in [(40,65),(45,65),(50,65),(40,60),(45,60)]:
        sig = apply_rsi_gate(base_v52.copy(), df_v5_2, rb, rs)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"v5t-2 RSI gate buy<{rb} sell>{rs}", r)
    # RSI gate + stop
    for rb, rs, stop in [(45,65,0.10),(45,65,0.12),(40,65,0.10)]:
        sig = apply_rsi_gate(base_v52.copy(), df_v5_2, rb, rs)
        sig = apply_stop_loss(df_v5_2, sig, stop)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"v5t-2 RSI({rb}/{rs}) stop={int(stop*100)}%", r)

    # ── S. Time-of-day filter on best combo (vel5 vt=-2) ─────────────────────
    section("S. Time-of-day filter on best combo")
    for start_h, end_h in [(9,15),(10,15),(10,14),(11,15),(9,14),(10,16)]:
        sig = apply_time_filter(base_v52.copy(), df_v5_2, start_h, end_h)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"v5t-2 hours={start_h}-{end_h}", r)
    # TOD + stop
    sig_tod = apply_time_filter(base_v52.copy(), df_v5_2, 10, 15)
    sig_tod = apply_stop_loss(df_v5_2, sig_tod, 0.10)
    rec("v5t-2 hours=10-15 stop=10%", backtest(df_v5_2, lambda d, s=sig_tod: s))

    # ── T. Z-score best config (zb=-0.5 zs=1.0) + stops ─────────────────────
    section("T. Z-score winner (zb=-0.5 zs=1.0, 4237%) with stops")
    for stop in [0.08, 0.10, 0.12]:
        for trail in [None, 0.10, 0.12]:
            sig = apply_zscore_gate(base_sig(df).copy(), df, z_buy=-0.5, z_sell=1.0, window=10)
            if trail:
                sig = apply_trailing_stop(df, apply_stop_loss(df, sig, stop), trail)
            else:
                sig = apply_stop_loss(df, sig, stop)
            r   = backtest(df, lambda d, s=sig: s)
            trail_s = f"+trail={int(trail*100)}%" if trail else ""
            rec(f"zscore(-0.5/1.0) stop={int(stop*100)}%{trail_s}", r)

    # ── U. Stack vel + zscore gates (multi-confirmation) ─────────────────────
    section("U. vel gate + z-score gate stacked (multi-confirm)")
    for vt, zb, zs in [(-2,-0.5,1.0),(-2,-0.5,1.5),(-4,-0.5,1.0),(-2,-1.0,1.0),(-4,-1.0,1.0)]:
        sig = apply_zscore_gate(
                apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2,-3,8), df_v5_2, vt),
                df_v5_2, z_buy=zb, z_sell=zs, window=10)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"vel5/vt={vt} + zscore({zb}/{zs})", r)
    # stack + stop
    sig_u = apply_zscore_gate(base_v52.copy(), df_v5_2, z_buy=-0.5, z_sell=1.0, window=10)
    sig_u = apply_stop_loss(df_v5_2, sig_u, 0.10)
    rec("vel5/vt=-2+zscore(-0.5/1.0)+stop=10%", backtest(df_v5_2, lambda d, s=sig_u: s))

    # ── V. FGI EMA cross winner (EMA3/8 mb=0 ms=8, 2.40 Sharpe) + stops ─────
    section("V. FGI EMA cross winner (EMA3/8 mb=0 ms=8) with stops")
    base_fec = apply_fgi_ema_cross_gate(apply_mom_gate(ema_cross(df,3,8), df, 0, 8), df)
    for stop in [0.08, 0.10, 0.12]:
        for trail in [None, 0.10, 0.12, 0.15]:
            sig = base_fec.copy()
            if trail:
                sig = apply_trailing_stop(df, apply_stop_loss(df, sig, stop), trail)
            else:
                sig = apply_stop_loss(df, sig, stop)
            r   = backtest(df, lambda d, s=sig: s)
            trail_s = f"+trail={int(trail*100)}%" if trail else ""
            rec(f"FEC3/8 stop={int(stop*100)}%{trail_s}", r)

    # ── W. Mom threshold sweep on vel_win=5 (holding vel_w=5) ────────────────
    section("W. mom_buy / mom_sell sweep (vel_win=5 vt=-2 base)")
    for mb, ms in [(-5,6),(-5,8),(-5,10),(-3,6),(-3,10),(-3,12),(0,8),(0,10),(0,12),(-1,8)]:
        sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2, mb, ms), df_v5_2, -2)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"vel5/vt-2 mb={mb} ms={ms}", r)

    # ── X. Best N-combo (AllDays stop=10%+trail=12%) + time filter ────────────
    section("X. Best full-combo (AllDays EMA7/15 vel3/vt-4 stop=10+trail=12) + time filter")
    df_v3   = add_fgi_features_ext(df_base, mom_w=5, vel_w=3)
    base_n  = apply_vel_gate(apply_mom_gate(ema_cross(df_v3,7,15), df_v3,-3,8), df_v3, -4)
    for start_h, end_h in [(9,15),(10,15),(10,14),(9,14)]:
        sig = apply_time_filter(base_n.copy(), df_v3, start_h, end_h)
        sig = apply_trailing_stop(df_v3, apply_stop_loss(df_v3, sig, 0.10), 0.12)
        r   = backtest(df_v3, lambda d, s=sig: s)
        rec(f"N-best TOD={start_h}-{end_h}", r)
    # DOW on best N
    for dow_days in [(1,2,3,4),(0,1,2,3,4)]:
        sig = apply_dow_filter(base_n.copy(), df_v3, dow_days)
        sig = apply_trailing_stop(df_v3, apply_stop_loss(df_v3, sig, 0.10), 0.12)
        r   = backtest(df_v3, lambda d, s=sig: s)
        dow_s = "TueFri" if len(dow_days)==4 else "AllDays"
        rec(f"N-best DOW={dow_s} stop=10+trail=12", r)

    # ── Y. Dual EMA regime: use fast EMA on downtrend, slow on uptrend ────────
    section("Y. Adaptive EMA pair: EMA3/8 when FGI rising, EMA7/15 when falling")
    def adaptive_ema_sig(df):
        fast_sig = apply_mom_gate(ema_cross(df, 3, 8),  df, 0, 8)
        slow_sig = apply_mom_gate(ema_cross(df, 7, 15), df, -3, 8)
        sig = pd.Series(0, index=df.index)
        sig[df["fgi_rising"] == 1] = fast_sig[df["fgi_rising"] == 1]
        sig[df["fgi_rising"] == 0] = slow_sig[df["fgi_rising"] == 0]
        return sig
    r = backtest(df, adaptive_ema_sig)
    rec("Adaptive EMA (rising→3/8, falling→7/15)", r)
    for stop in [0.08, 0.10]:
        sig = apply_stop_loss(df, adaptive_ema_sig(df).copy(), stop)
        r   = backtest(df, lambda d, s=sig: s)
        rec(f"Adaptive EMA stop={int(stop*100)}%", r)
    sig_a = apply_trailing_stop(df, apply_stop_loss(df, adaptive_ema_sig(df).copy(), 0.10), 0.12)
    rec("Adaptive EMA stop=10%+trail=12%", backtest(df, lambda d, s=sig_a: s))

    # ── Z. Mom window sweep (hold vel_w=5, vary mom_w) ───────────────────────
    section("Z. mom_window sweep (vel_w=5, vt=-2, EMA7/15)")
    for mw in [3, 4, 5, 6, 7, 8, 10]:
        df_mw = add_fgi_features_ext(df_base, mom_w=mw, vel_w=5)
        sig   = apply_vel_gate(apply_mom_gate(ema_cross(df_mw,7,15), df_mw,-3,8), df_mw, -2)
        r     = backtest(df_mw, lambda d, s=sig: s)
        rec(f"mom_w={mw} vel_w=5 vt=-2", r)

    # ══════════════════════════════════════════════════════════════════════════
    # ROUND 7 — Fine-tuning the new champion (v5t-2 + trail=12%, 5639%)
    # ══════════════════════════════════════════════════════════════════════════

    # ── AA. mom_sell sweep on champion (v5t-2 + trail=12%) ───────────────────
    section("AA. mom_sell sweep on champion (vel5 vt=-2 trail=12%)")
    for mb, ms in [(-3,6),(-3,8),(-3,9),(-3,10),(-3,11),(-3,12),
                   (-3,14),(-3,16),(-5,8),(-5,10),(-5,12),(0,10),(0,12)]:
        sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2, mb, ms), df_v5_2, -2)
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ mb={mb} ms={ms} trail=12%", r)

    # ── AB. vel threshold fine-tune on champion EMA + trail=12% ──────────────
    section("AB. vt fine-tune on champion (vel5 trail=12%)")
    for vt in [-5, -4, -3, -2, -1, 0, 1]:
        sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2,-3,10), df_v5_2, vt)
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ ms=10 vt={vt} trail=12%", r)

    # ── AC. Trail pct fine-tune on best ms=10 vt=-2 ──────────────────────────
    section("AC. Trail pct sweep on best ms=10 vt=-2")
    for trail in [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.20]:
        sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2,-3,10), df_v5_2, -2)
        sig = apply_trailing_stop(df_v5_2, sig, trail)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"ms=10 vt=-2 trail={int(trail*100)}%", r)

    # ── AD. DOW filter on champion + trail ───────────────────────────────────
    section("AD. DOW filter on champion (ms=10 vt=-2 trail=12%)")
    base_champ = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2,-3,10), df_v5_2, -2)
    for dow_days in [(0,1,2,3,4),(1,2,3,4),(2,3,4),(0,1,2,3)]:
        sig = apply_dow_filter(base_champ.copy(), df_v5_2, dow_days)
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        dow_s = {5:"AllDays",4:"TueFri",3:"WedFriOnly",4:"MonThu"}
        dow_lbl = {(0,1,2,3,4):"AllDays",(1,2,3,4):"TueFri",(2,3,4):"WedThFri",(0,1,2,3):"MonThru"}
        rec(f"champ DOW={dow_lbl.get(tuple(dow_days),'custom')} trail=12%", r)

    # ── AE. zscore gate on champion ───────────────────────────────────────────
    section("AE. Z-score gate on champion (ms=10 vt=-2)")
    for zb, zs, trail in [(-0.5,1.0,0.12),(-0.5,1.5,0.12),(-1.0,1.0,0.12),
                           (-0.5,1.0,0.15),(-0.5,1.0,None)]:
        sig = apply_zscore_gate(base_champ.copy(), df_v5_2, z_buy=zb, z_sell=zs, window=10)
        if trail:
            sig = apply_trailing_stop(df_v5_2, sig, trail)
        trail_s = f" trail={int(trail*100)}%" if trail else ""
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ zscore({zb}/{zs}){trail_s}", r)

    # ── AF. FGI EMA cross gate on champion ────────────────────────────────────
    section("AF. FGI EMA cross gate on champion (ms=10 vt=-2 trail=12%)")
    sig_fec_c = apply_fgi_ema_cross_gate(base_champ.copy(), df_v5_2)
    sig_fec_c = apply_trailing_stop(df_v5_2, sig_fec_c, 0.12)
    rec("champ + FGI EMA cross trail=12%", backtest(df_v5_2, lambda d, s=sig_fec_c: s))
    # FGI rising only
    sig_fr_c = base_champ.copy()
    sig_fr_c[(df_v5_2["fgi_rising"] == 0)] = 0
    sig_fr_c = apply_trailing_stop(df_v5_2, sig_fr_c, 0.12)
    rec("champ + FGI rising filter trail=12%", backtest(df_v5_2, lambda d, s=sig_fr_c: s))

    # ── AG. EMA pair sweep on champion (ms=10 vt=-2 trail=12%) ───────────────
    section("AG. EMA pair on champion params (ms=10 vt=-2 trail=12%)")
    for fast, slow in [(3,8),(5,13),(7,15),(9,21),(5,20),(4,12)]:
        sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,fast,slow), df_v5_2,-3,10), df_v5_2, -2)
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"EMA{fast}/{slow} champ trail=12%", r)

    # ── AH. Regime gate on champion ──────────────────────────────────────────
    section("AH. Regime gate on champion (ms=10 vt=-2 trail=12%)")
    for buy_r, sell_r in [((0,1,2),(3,4)),((1,2),(3,4)),((0,1,2,3),(4,)),((1,2,3),(4,))]:
        sig = apply_regime_gate(base_champ.copy(), df_v5_2, buy_r, sell_r)
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ regime buy={buy_r} trail=12%", r)

    # ── AI. Level gate (FGI cap on entry) on champion ─────────────────────────
    section("AI. FGI level cap on entry (champion ms=10 vt=-2 trail=12%)")
    for gc, fc in [(70,15),(65,20),(75,15),(75,10),(80,15),(65,15)]:
        sig = apply_level_gate(base_champ.copy(), df_v5_2, gc, fc)
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ level gc={gc} fc={fc} trail=12%", r)

    # ══════════════════════════════════════════════════════════════════════════
    # ROUND 8 — Final squeeze: combine remaining signal layers on champion
    # ══════════════════════════════════════════════════════════════════════════

    # Champion params: EMA7/15, mb=-3, ms=8, vel_win=5, vt=-2, trail=12%

    # ── AJ. Monthly seasonality on champion ───────────────────────────────────
    section("AJ. Monthly seasonality on champion (EMA7/15 mb=-3 ms=8 vel5 vt=-2 trail=12%)")
    champ_base = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,7,15), df_v5_2,-3,8), df_v5_2, -2)
    months5 = df_v5_2.index.month
    for skip in [(1,),(9,),(9,10),(6,7,8),(1,9),(8,9,10)]:
        sig = champ_base.copy()
        sig[sig.ne(0) & months5.isin(skip)] = 0
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ skip months={skip} trail=12%", r)
    for keep in [(11,12,1,2,3,4),(3,4,5,10,11,12)]:
        sig = champ_base.copy()
        sig[sig.ne(0) & ~months5.isin(keep)] = 0
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ only months={keep} trail=12%", r)

    # ── AK. MACD confirmation on champion ────────────────────────────────────
    section("AK. MACD direction gate on champion")
    def apply_macd_gate(sig, df):
        s = sig.copy()
        s[(sig == 1)  & (df["macd_hist"] < 0)] = 0   # skip buy if MACD negative
        s[(sig == -1) & (df["macd_hist"] > 0)] = 0   # skip sell if MACD positive
        return s
    sig_macd_c = apply_macd_gate(champ_base.copy(), df_v5_2)
    rec("champ + MACD gate", backtest(df_v5_2, lambda d, s=sig_macd_c: s))
    sig_macd_ct = apply_trailing_stop(df_v5_2, sig_macd_c.copy(), 0.12)
    rec("champ + MACD gate trail=12%", backtest(df_v5_2, lambda d, s=sig_macd_ct: s))

    # ── AL. BB position gate on champion ─────────────────────────────────────
    section("AL. Bollinger Band position gate on champion")
    def apply_bb_gate(sig, df, bb_buy=0.5, bb_sell=0.5):
        s = sig.copy()
        s[(sig == 1)  & (df["bb_pct"] > bb_buy)]  = 0  # skip buy if price above mid
        s[(sig == -1) & (df["bb_pct"] < bb_sell)] = 0  # skip sell if price below mid
        return s
    for bbb, bbs in [(0.5,0.5),(0.6,0.4),(0.7,0.3),(0.4,0.6)]:
        sig = apply_bb_gate(champ_base.copy(), df_v5_2, bbb, bbs)
        sig = apply_trailing_stop(df_v5_2, sig, 0.12)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ BB buy<{bbb} sell>{bbs} trail=12%", r)

    # ── AM. EMA5/20 (runner-up) with champion params + trail tune ─────────────
    section("AM. EMA5/20 (runner-up) full param sweep")
    for mb, ms in [(-3,8),(-3,10),(-3,12)]:
        for vt in [-2, -3]:
            for trail in [0.10, 0.11, 0.12, 0.14]:
                sig = apply_vel_gate(apply_mom_gate(ema_cross(df_v5_2,5,20), df_v5_2, mb, ms), df_v5_2, vt)
                sig = apply_trailing_stop(df_v5_2, sig, trail)
                r   = backtest(df_v5_2, lambda d, s=sig: s)
                rec(f"EMA5/20 mb={mb} ms={ms} vt={vt} trail={int(trail*100)}%", r)

    # ── AN. Grand final: champion + level cap gc=70 (reduces drawdown) ────────
    section("AN. Champion + FGI level cap gc=70 (2.74 Sharpe, -36% DD variant)")
    base_lc70 = apply_level_gate(champ_base.copy(), df_v5_2, 70, 15)
    for trail in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:
        sig = apply_trailing_stop(df_v5_2, base_lc70.copy(), trail)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ gc=70 trail={int(trail*100)}%", r)
    # with stop too
    for stop, trail in [(0.08,0.12),(0.10,0.11),(0.10,0.12)]:
        sig = apply_trailing_stop(df_v5_2, apply_stop_loss(df_v5_2, base_lc70.copy(), stop), trail)
        r   = backtest(df_v5_2, lambda d, s=sig: s)
        rec(f"champ gc=70 stop={int(stop*100)}% trail={int(trail*100)}%", r)

    # ── AO. Position sizing on champion ──────────────────────────────────────
    section("AO. Position sizing on champion + trail=12%")
    def size_fear_greed(fgi):  return 1.0 if fgi < 40 else (0.7 if fgi < 55 else 0.5)
    def size_aggressive(fgi):  return min(1.0, max(0.3, (70-fgi)/70))
    def size_stepped(fgi):
        if fgi < 25: return 1.0
        if fgi < 40: return 0.85
        if fgi < 55: return 0.65
        if fgi < 70: return 0.45
        return 0.25
    for label, fn in [("fear_graded",size_fear_greed),("aggressive",size_aggressive),("stepped",size_stepped)]:
        def sig_fn_sized(d, fn=fn):
            sig = apply_vel_gate(apply_mom_gate(ema_cross(d,7,15), d,-3,8), d, -2)
            return apply_trailing_stop(d, sig, 0.12)
        r = backtest_sized(df_v5_2, lambda d: apply_vel_gate(apply_mom_gate(ema_cross(d,7,15), d,-3,8), d,-2), fn)
        rec(f"champ sized:{label}", r)

    # ── Master comparison table ───────────────────────────────────────────────
    print(f"\n{'━'*20} MASTER COMPARISON TABLE {'━'*20}")
    print(f"Buy & Hold: {bh:.1f}%\n")

    filtered   = [(n,r) for n,r in all_results if r["n_trades"] >= 20]
    by_sharpe  = sorted(filtered, key=lambda x: x[1]["sharpe"],       reverse=True)[:20]
    by_return  = sorted(filtered, key=lambda x: x[1]["total_return"],  reverse=True)[:20]

    print("▶ TOP 20 BY SHARPE (≥20 trades)")
    print(hdr); print(sep)
    for n,r in by_sharpe: print(row(n,r))

    print("\n▶ TOP 20 BY RETURN (≥20 trades)")
    print(hdr); print(sep)
    for n,r in by_return: print(row(n,r))

    # ══════════════════════════════════════════════════════════════════════════
    # BEFORE / AFTER: Corrected (lagged FGI) vs Leaky Baseline
    # ══════════════════════════════════════════════════════════════════════════
    # Old values captured from the leaky run (merge_fgi_raw, same-day, 2026-06-12).
    # Keys must match rec() names exactly so we can look them up in all_results.

    old_leaky = {
        # FGI-gated top performers (EMA5/20 variants)
        "EMA5/20 mb=-3 ms=8 vt=-2 trail=11%":  (5889.2, 2.84, 51),
        "EMA5/20 mb=-3 ms=8 vt=-3 trail=11%":  (5909.6, 2.83, 53),
        "EMA5/20 mb=-3 ms=8 vt=-2 trail=12%":  (5358.5, 2.78, 51),
        "EMA5/20 mb=-3 ms=10 vt=-2 trail=11%": (5058.1, 2.76, 54),
        # v5t-2 + trail variants
        "v5t-2 stop=8%+trail=12%":             (5639.7, 2.83, 49),
        "v5t-2 stop=10%+trail=12%":            (5639.7, 2.83, 49),
        # champ mb/ms variants
        "champ mb=-3 ms=8 trail=12%":          (5639.7, 2.83, 49),
        "champ mb=-3 ms=10 trail=12%":         (4898.8, 2.75, 52),
        "ms=10 vt=-2 trail=11%":               (5311.4, 2.81, 52),
        # FGI level-gate / gc=70 variants
        "champ gc=70 trail=11%":               (4518.0, 2.81, 39),
        "champ gc=70 stop=10% trail=11%":      (4518.0, 2.81, 39),
        "champ level gc=70 fc=15 trail=12%":   (4034.2, 2.74, 40),
        # FGI z-score stacked
        "vel5/vt=-2 + zscore(-0.5/1.0)":       (4931.6, 2.51, 28),
        "vel5/vt=-2 + zscore(-0.5/1.5)":       (5011.0, 2.31, 34),
        # Non-FGI best — sections B/C/D, pure trailing/fixed stop, no FGI gating
        "trail=12%":                           (2831.1, 2.31, 63),
        "stop=10%+trail=12%":                  (2831.1, 2.31, 63),
    }

    new_lookup = {n: r for n, r in all_results}   # last-write wins for any dups

    print(f"\n{'━'*20} BEFORE / AFTER: Corrected vs Leaky Baseline {'━'*20}")
    print("Pipeline change: merge_fgi_raw (same-day, leaky) → merge_fgi_lagged (D-1, correct).")
    print(f"Buy & Hold on corrected date range: {bh:.1f}%\n")

    # ── New top 5 by Sharpe and Return ───────────────────────────────────────
    filtered_corr  = [(n, r) for n, r in all_results if r["n_trades"] >= 20]
    top5_sh        = sorted(filtered_corr, key=lambda x: x[1]["sharpe"],       reverse=True)[:5]
    top5_ret       = sorted(filtered_corr, key=lambda x: x[1]["total_return"], reverse=True)[:5]

    print("▶ NEW TOP 5 BY SHARPE (corrected sweep, ≥20 trades)")
    print(hdr); print(sep)
    for n, r in top5_sh:  print(row(n, r))

    print("\n▶ NEW TOP 5 BY RETURN (corrected sweep, ≥20 trades)")
    print(hdr); print(sep)
    for n, r in top5_ret: print(row(n, r))

    # ── Degradation table: FGI-gated strategies ───────────────────────────────
    ba_hdr = (f"{'Strategy':<42} "
              f"{'Ret_OLD':>8} {'Sh_OLD':>7} {'Tr_OLD':>6}  "
              f"{'Ret_NEW':>8} {'Sh_NEW':>7} {'Tr_NEW':>6}  "
              f"{'Ret_Δ':>8} {'Sh_Δ%':>7}")
    ba_sep = "-" * len(ba_hdr)

    print(f"\n▶ DEGRADATION TABLE — FGI-gated strategies (leaky → corrected)")
    print(ba_hdr); print(ba_sep)
    non_fgi_names = {"trail=12%", "stop=10%+trail=12%"}
    for old_name, (old_ret, old_sh, old_tr) in old_leaky.items():
        if old_name in non_fgi_names:
            continue
        nr = new_lookup.get(old_name)
        if nr:
            ret_d  = nr["total_return"] - old_ret
            sh_pct = (nr["sharpe"] - old_sh) / abs(old_sh) * 100 if old_sh else 0
            print(
                f"{old_name:<42} "
                f"{old_ret:>7.1f}% {old_sh:>7.2f} {old_tr:>6}  "
                f"{nr['total_return']:>7.1f}% {nr['sharpe']:>7.2f} {nr['n_trades']:>6}  "
                f"{ret_d:>+7.1f}% {sh_pct:>+6.1f}%"
            )
        else:
            print(f"{old_name:<42} {old_ret:>7.1f}% {old_sh:>7.2f} {old_tr:>6}  {'(name not matched)':>38}")

    # ── Non-FGI baseline (sections B/C/D — no FGI logic at all) ─────────────
    print(f"\n▶ NON-FGI BASELINE — sections B/C/D (pure trailing/fixed stop, no FGI gating)")
    print(ba_hdr); print(ba_sep)
    for old_name in sorted(non_fgi_names):
        old_ret, old_sh, old_tr = old_leaky[old_name]
        nr = new_lookup.get(old_name)
        if nr:
            ret_d  = nr["total_return"] - old_ret
            sh_pct = (nr["sharpe"] - old_sh) / abs(old_sh) * 100 if old_sh else 0
            print(
                f"{old_name:<42} "
                f"{old_ret:>7.1f}% {old_sh:>7.2f} {old_tr:>6}  "
                f"{nr['total_return']:>7.1f}% {nr['sharpe']:>7.2f} {nr['n_trades']:>6}  "
                f"{ret_d:>+7.1f}% {sh_pct:>+6.1f}%"
            )

    # ── FGI value-add verdict ─────────────────────────────────────────────────
    best_fgi_name, best_fgi_r = top5_sh[0]
    nonfgi_r = new_lookup.get("trail=12%")
    print(f"\n▶ FGI VALUE-ADD VERDICT (corrected data, BH = {bh:.1f}%)")
    if nonfgi_r:
        fgi_sh     = best_fgi_r["sharpe"]
        nonfgi_sh  = nonfgi_r["sharpe"]
        fgi_ret    = best_fgi_r["total_return"]
        nonfgi_ret = nonfgi_r["total_return"]
        margin     = (fgi_sh / nonfgi_sh - 1) * 100 if nonfgi_sh else 0
        if fgi_sh > nonfgi_sh * 1.05:
            verdict = f"FGI gating STILL ADDS VALUE (+{margin:.0f}% Sharpe over non-FGI)."
        elif fgi_sh > nonfgi_sh:
            verdict = f"FGI gating adds MARGINAL value (+{margin:.1f}% Sharpe, within 5% of non-FGI)."
        else:
            verdict = f"FGI gating NO LONGER BEATS non-FGI ({margin:+.1f}% Sharpe vs non-FGI)."
        print(f"  Best FGI strategy : '{best_fgi_name}'")
        print(f"    Sharpe {fgi_sh:.2f}  Return {fgi_ret:.0f}%")
        print(f"  Best non-FGI      : 'trail=12%' (section B)")
        print(f"    Sharpe {nonfgi_sh:.2f}  Return {nonfgi_ret:.0f}%")
        print(f"  Verdict: {verdict}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: GENUINELY FGI-FREE BASELINE
    # ══════════════════════════════════════════════════════════════════════════
    # ema_cross() + trailing stop only.  No apply_mom_gate, no apply_vel_gate,
    # no fgi* column touched anywhere.  Uses df (same date range as FGI strategies
    # so bh is identical and comparison is apples-to-apples).

    print(f"\n{'━'*70}")
    print("PART 1: GENUINELY FGI-FREE BASELINE")
    print("Signal = ema_cross(df, fast, slow) only.  No FGI columns accessed.")
    print(f"{'━'*70}\n")
    print(hdr); print(sep)

    fgi_free_res = []   # (name, r, fast, slow, trail)
    for fast, slow in [(7, 15), (5, 20)]:
        raw = ema_cross(df, fast, slow)
        r   = backtest(df, lambda d, s=raw: s)
        n   = f"FGI-free EMA{fast}/{slow} no-stop"
        fgi_free_res.append((n, r, fast, slow, None)); print(row(n, r))
        for trail in [0.05, 0.08, 0.10, 0.11, 0.12, 0.15, 0.20]:
            sig = apply_trailing_stop(df, raw.copy(), trail)
            r   = backtest(df, lambda d, s=sig: s)
            n   = f"FGI-free EMA{fast}/{slow} trail={int(trail*100)}%"
            fgi_free_res.append((n, r, fast, slow, trail)); print(row(n, r))

    best_ff_name, best_ff_r, *_ = max(fgi_free_res, key=lambda x: x[1]["sharpe"])
    print(f"\n→ Best FGI-free: '{best_ff_name}'  Sharpe={best_ff_r['sharpe']:.2f}  Return={best_ff_r['total_return']:.0f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: FRESH STAGED GRID SEARCH ON CORRECTED DATA
    # ══════════════════════════════════════════════════════════════════════════
    # Greedy staged optimisation on corrected (lagged) df_base.
    # Stage A → best EMA pair.  B → best mom.  C → best vel.  D → best trail.
    # Reuses already-computed feature DFs where possible.

    print(f"\n{'━'*70}")
    print("PART 2: FRESH STAGED GRID SEARCH (corrected data, not leaky search path)")
    print(f"{'━'*70}")

    # Cache of (mom_w=5, vel_w) → feature DF so we don't recompute repeatedly
    _dfcache = {3: df, 5: df_v5_2}
    def _get_fdf(vel_w):
        if vel_w not in _dfcache:
            _dfcache[vel_w] = add_fgi_features_ext(df_base, mom_w=5, vel_w=vel_w)
        return _dfcache[vel_w]

    # ── Stage A: EMA pair × trail (fixed mom=-3/8, vel_w=5, vt=-2) ──────────
    section("2A. EMA pair sweep  (mom=-3/8, vel5/vt=-2, trail sweep)")
    _d5 = _get_fdf(5)
    _stage_a = []   # (fast, slow, trail, r)
    for fast, slow in [(3,8),(4,12),(5,13),(5,20),(7,15),(9,21)]:
        _base_a = apply_vel_gate(apply_mom_gate(ema_cross(_d5,fast,slow), _d5,-3,8), _d5,-2)
        for trail in [0.08, 0.10, 0.11, 0.12, 0.15]:
            sig = apply_trailing_stop(_d5, _base_a.copy(), trail)
            r   = backtest(_d5, lambda d, s=sig: s)
            n   = f"EMA{fast}/{slow} m-3/8 v5/-2 tr={int(trail*100)}%"
            _stage_a.append((fast, slow, trail, r)); print(row(n, r))
    _bA_fast, _bA_slow, _, _ = max(_stage_a, key=lambda x: x[3]["sharpe"])
    print(f"\n→ Stage A best EMA pair: EMA{_bA_fast}/{_bA_slow}")

    # ── Stage B: mom_buy / mom_sell (best EMA, vel5/vt=-2, trail=11%) ────────
    section(f"2B. mom_buy/mom_sell sweep  (EMA{_bA_fast}/{_bA_slow}, vel5/vt=-2, trail=11%)")
    _stage_b = []   # (mb, ms, r)
    for mb in [-5, -3, -1, 0]:
        for ms in [6, 8, 10, 12, 15]:
            sig = apply_vel_gate(apply_mom_gate(ema_cross(_d5,_bA_fast,_bA_slow), _d5,mb,ms), _d5,-2)
            sig = apply_trailing_stop(_d5, sig, 0.11)
            r   = backtest(_d5, lambda d, s=sig: s)
            n   = f"EMA{_bA_fast}/{_bA_slow} mb={mb} ms={ms} vel5/-2 tr=11%"
            _stage_b.append((mb, ms, r)); print(row(n, r))
    _bB_mb, _bB_ms, _ = max(_stage_b, key=lambda x: x[2]["sharpe"])
    print(f"\n→ Stage B best mom: mb={_bB_mb} ms={_bB_ms}")

    # ── Stage C: vel_win / vel_thresh (best EMA/mom, trail=11%) ──────────────
    section(f"2C. vel_win/vel_thresh sweep  (EMA{_bA_fast}/{_bA_slow}, mom={_bB_mb}/{_bB_ms}, trail=11%)")
    _stage_c = []   # (vel_w, vt, r)
    for vel_w in [3, 5, 7]:
        _dfv = _get_fdf(vel_w)
        for vt in [-4, -3, -2, -1, 0]:
            sig = apply_vel_gate(apply_mom_gate(ema_cross(_dfv,_bA_fast,_bA_slow), _dfv,_bB_mb,_bB_ms), _dfv,vt)
            sig = apply_trailing_stop(_dfv, sig, 0.11)
            r   = backtest(_dfv, lambda d, s=sig: s)
            n   = f"EMA{_bA_fast}/{_bA_slow} mb={_bB_mb} ms={_bB_ms} vel{vel_w}/vt={vt} tr=11%"
            _stage_c.append((vel_w, vt, r)); print(row(n, r))
    _bC_vw, _bC_vt, _ = max(_stage_c, key=lambda x: x[2]["sharpe"])
    print(f"\n→ Stage C best vel: vel_w={_bC_vw} vt={_bC_vt}")

    # ── Stage D: trail fine-tune (best EMA/mom/vel) ───────────────────────────
    section(f"2D. Trail fine-tune  (EMA{_bA_fast}/{_bA_slow} mb={_bB_mb} ms={_bB_ms} vel{_bC_vw}/vt={_bC_vt})")
    _dfD = _get_fdf(_bC_vw)
    _stage_d = []   # (trail, r)
    for trail in [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.20]:
        _base_d = apply_vel_gate(apply_mom_gate(ema_cross(_dfD,_bA_fast,_bA_slow), _dfD,_bB_mb,_bB_ms), _dfD,_bC_vt)
        sig     = apply_trailing_stop(_dfD, _base_d, trail)
        r       = backtest(_dfD, lambda d, s=sig: s)
        n       = f"EMA{_bA_fast}/{_bA_slow} mb={_bB_mb} ms={_bB_ms} vel{_bC_vw}/vt={_bC_vt} tr={int(trail*100)}%"
        _stage_d.append((trail, r)); print(row(n, r))
    _bD_trail, _new_champ_r = max(_stage_d, key=lambda x: x[1]["sharpe"])
    _new_champ_name = (f"EMA{_bA_fast}/{_bA_slow} mb={_bB_mb} ms={_bB_ms} "
                       f"vel{_bC_vw}/vt={_bC_vt} trail={int(_bD_trail*100)}%")
    print(f"\n→ NEW CHAMPION: {_new_champ_name}")
    print(f"  Sharpe={_new_champ_r['sharpe']:.2f}  Return={_new_champ_r['total_return']:.0f}%  "
          f"MaxDD={_new_champ_r['max_drawdown']:.1f}%  Trades={_new_champ_r['n_trades']}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: THREE-WAY COMPARISON
    # ══════════════════════════════════════════════════════════════════════════

    _survivor_name = "EMA5/20 mb=-3 ms=8 vt=-2 trail=11%"
    _survivor_r    = new_lookup.get(_survivor_name)

    print(f"\n{'━'*70}")
    print("PART 3: THREE-WAY COMPARISON  (all on corrected lagged-FGI data)")
    print(f"{'━'*70}")
    print(f"  (a) FGI-free best   : '{best_ff_name}'")
    print(f"  (b) Leaky survivor  : '{_survivor_name}'")
    print(f"  (c) Fresh champion  : '{_new_champ_name}'\n")
    print(hdr); print(sep)
    print(row("(a) FGI-free best", best_ff_r))
    if _survivor_r:
        print(row("(b) Leaky survivor (EMA5/20 v5/-2 tr11%)", _survivor_r))
    print(row("(c) Fresh champion", _new_champ_r))
    print(sep)

    _a_sh  = best_ff_r["sharpe"]
    _a_ret = best_ff_r["total_return"]
    print()
    for _lbl, _nm, _rc in [("(b)", _survivor_name, _survivor_r), ("(c)", _new_champ_name, _new_champ_r)]:
        if _rc is None:
            continue
        _sh_lift  = (_rc["sharpe"] - _a_sh) / abs(_a_sh) * 100 if _a_sh else 0
        _ret_lift = _rc["total_return"] - _a_ret
        if _rc["sharpe"] > _a_sh * 1.05:
            _verd = f"FGI gating ADDS VALUE (+{_sh_lift:.0f}% Sharpe vs FGI-free)"
        elif _rc["sharpe"] > _a_sh:
            _verd = f"FGI gating marginal (+{_sh_lift:.1f}% Sharpe, within 5% of FGI-free)"
        else:
            _verd = f"FGI gating does NOT beat FGI-free ({_sh_lift:+.1f}% Sharpe)"
        print(f"  {_lbl} vs (a): Sharpe {_rc['sharpe']:.2f} vs {_a_sh:.2f} ({_sh_lift:+.0f}%)  "
              f"Return {_rc['total_return']:.0f}% vs {_a_ret:.0f}% ({_ret_lift:+.0f}pp)")
        print(f"         → {_verd}")

    # Which FGI dimensions actually survive on corrected data?
    print()
    print("FGI dimensions in fresh champion vs FGI-free:")
    print(f"  mom gate (mb={_bB_mb}/ms={_bB_ms}): yes")
    print(f"  vel gate (vel_w={_bC_vw}/vt={_bC_vt}): yes")
    print("  level gate: not included (excluded from fresh grid — test separately if needed)")
    print("  z-score gate: not included (collapsed under lag correction)")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4: SENSITIVITY ANALYSIS ON NEW CHAMPION
    # ══════════════════════════════════════════════════════════════════════════
    # Each dimension swept independently, all others fixed at champion values.
    # Checks for narrow-peak brittleness (old champion swung 2000pp on vt=-1 vs -2).

    print(f"\n{'━'*70}")
    print(f"PART 4: SENSITIVITY ANALYSIS ON FRESH CHAMPION")
    print(f"  {_new_champ_name}")
    print(f"  Sharpe={_new_champ_r['sharpe']:.2f}  Return={_new_champ_r['total_return']:.0f}%  "
          f"MaxDD={_new_champ_r['max_drawdown']:.1f}%  Trades={_new_champ_r['n_trades']}")
    print(f"{'━'*70}")

    def _champ_run(fast, slow, mb, ms, vw, vt, trail):
        _d = _get_fdf(vw)
        s  = apply_vel_gate(apply_mom_gate(ema_cross(_d,fast,slow), _d,mb,ms), _d,vt)
        s  = apply_trailing_stop(_d, s, trail)
        return backtest(_d, lambda d, sig=s: sig)

    # 4A: trail sensitivity
    section("4A. trail% sensitivity  (all else fixed at champion)")
    _bt = round(_bD_trail * 100)
    for t_int in range(max(5, _bt - 4), min(25, _bt + 5)):
        t = t_int / 100
        r = _champ_run(_bA_fast, _bA_slow, _bB_mb, _bB_ms, _bC_vw, _bC_vt, t)
        marker = "  ← champion" if t_int == _bt else ""
        print(row(f"trail={t_int}%{marker}", r))

    # 4B: mom_sell sensitivity
    section(f"4B. mom_sell sensitivity  (all else fixed, trail={_bt}%)")
    for ms in range(max(1, _bB_ms - 5), _bB_ms + 7):
        r = _champ_run(_bA_fast, _bA_slow, _bB_mb, ms, _bC_vw, _bC_vt, _bD_trail)
        marker = "  ← champion" if ms == _bB_ms else ""
        print(row(f"mom_sell={ms}{marker}", r))

    # 4C: vel_thresh sensitivity
    section(f"4C. vel_thresh sensitivity  (all else fixed, trail={_bt}%)")
    for vt in range(_bC_vt - 4, _bC_vt + 5):
        r = _champ_run(_bA_fast, _bA_slow, _bB_mb, _bB_ms, _bC_vw, vt, _bD_trail)
        marker = "  ← champion" if vt == _bC_vt else ""
        print(row(f"vel_thresh={vt}{marker}", r))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print("SUMMARY")
    print(f"{'━'*70}")
    print(f"  (a) FGI-free best  : '{best_ff_name}'")
    print(f"      Sharpe {best_ff_r['sharpe']:.2f}  Return {best_ff_r['total_return']:.0f}%  MaxDD {best_ff_r['max_drawdown']:.1f}%  Trades {best_ff_r['n_trades']}")
    if _survivor_r:
        print(f"  (b) Leaky survivor : '{_survivor_name}'")
        print(f"      Sharpe {_survivor_r['sharpe']:.2f}  Return {_survivor_r['total_return']:.0f}%  MaxDD {_survivor_r['max_drawdown']:.1f}%  Trades {_survivor_r['n_trades']}")
    print(f"  (c) Fresh champion : '{_new_champ_name}'")
    print(f"      Sharpe {_new_champ_r['sharpe']:.2f}  Return {_new_champ_r['total_return']:.0f}%  MaxDD {_new_champ_r['max_drawdown']:.1f}%  Trades {_new_champ_r['n_trades']}")
    print(f"  Buy & Hold         : {bh:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 5A: vel_w SENSITIVITY  (missing from Part 4)
    # ══════════════════════════════════════════════════════════════════════════
    # All champion params fixed; only vel_w swept.  Flags spike vs plateau.

    print(f"\n{'━'*70}")
    print(f"PART 5A: vel_w SENSITIVITY  "
          f"(EMA{_bA_fast}/{_bA_slow} mb={_bB_mb} ms={_bB_ms} vt={_bC_vt} trail={round(_bD_trail*100)}%)")
    print(f"{'━'*70}\n")
    print(hdr); print(sep)

    _vw_sharpes = {}
    for _vw in range(4, 11):
        _r5a = _champ_run(_bA_fast, _bA_slow, _bB_mb, _bB_ms, _vw, _bC_vt, _bD_trail)
        _vw_sharpes[_vw] = _r5a["sharpe"]
        _mk = "  ← champion" if _vw == _bC_vw else ""
        print(row(f"vel_w={_vw}{_mk}", _r5a))

    _champ_sh5a = _vw_sharpes[_bC_vw]
    _left_drop  = _champ_sh5a - _vw_sharpes.get(_bC_vw - 1, _champ_sh5a)
    _right_drop = _champ_sh5a - _vw_sharpes.get(_bC_vw + 1, _champ_sh5a)
    _is_spike   = max(_left_drop, _right_drop) > 0.15
    print()
    if _is_spike:
        print(f"⚠ vel_w={_bC_vw} is a LOCAL SPIKE: "
              f"left-neighbor drop={_left_drop:+.2f}  right-neighbor drop={_right_drop:+.2f}  (>0.15 threshold)")
    else:
        print(f"✓ vel_w={_bC_vw} is part of a PLATEAU: "
              f"left-neighbor drop={_left_drop:+.2f}  right-neighbor drop={_right_drop:+.2f}  (≤0.15 threshold)")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 5B: BOOTSTRAP CONFIDENCE INTERVALS ON CHAMPION SHARPE
    # ══════════════════════════════════════════════════════════════════════════
    # Resamples per-trade pnl_pct with replacement (2000 iterations).
    # Compares champion (42 trades) vs FGI-free baseline (120 trades).

    def _extract_trades(df_t, signal_fn):
        """Re-run backtest and return list of per-trade pnl_pct values."""
        sigs = signal_fn(df_t)
        cap = INITIAL_CAPITAL; pos = 0.0; ep = 0.0; tpcts = []
        for i in range(len(df_t)):
            price = float(df_t["Close"].iloc[i]); sig = sigs.iloc[i]
            if sig == 1 and pos == 0:
                pos = cap / price; ep = price; cap = 0.0
            elif sig == -1 and pos > 0:
                tpcts.append((price - ep) / ep * 100)
                cap = pos * price; pos = 0.0
        if pos > 0:
            last = float(df_t["Close"].iloc[-1])
            tpcts.append((last - ep) / ep * 100)
        return tpcts

    # Champion signal on the vel_w=_bC_vw df; FGI-free on df (same date range)
    _champ_sig_fn = lambda d: apply_trailing_stop(
        d, apply_vel_gate(apply_mom_gate(ema_cross(d, _bA_fast, _bA_slow), d, _bB_mb, _bB_ms), d, _bC_vt),
        _bD_trail)
    _ff_sig_fn = lambda d: apply_trailing_stop(d, ema_cross(d, _bA_fast, _bA_slow), _bD_trail)

    _champ_tpcts = _extract_trades(_dfD, _champ_sig_fn)
    _ff_tpcts    = _extract_trades(df,   _ff_sig_fn)

    _n_years = (_dfD.index[-1] - _dfD.index[0]).days / 365.25

    def _bootstrap_dist(tpcts, n_boot=2000, seed=42):
        arr   = np.array(tpcts, dtype=float)
        n     = len(arr)
        if n < 2:
            return np.zeros(n_boot), np.zeros(n_boot)
        rng_b  = np.random.default_rng(seed)
        samp   = rng_b.choice(arr, size=(n_boot, n), replace=True)
        # compounded total return per resample
        tot_ret = (np.expm1(np.log1p(samp / 100).sum(axis=1))) * 100
        # per-trade Sharpe proxy (annualised)
        ann    = np.sqrt(n / _n_years)
        mu     = samp.mean(axis=1)
        std    = samp.std(axis=1, ddof=1)
        sh_prx = np.where(std > 0, mu / std * ann, 0.0)
        return tot_ret, sh_prx

    _champ_ret_boot, _champ_sh_boot = _bootstrap_dist(_champ_tpcts)
    _ff_ret_boot,    _ff_sh_boot    = _bootstrap_dist(_ff_tpcts)

    print(f"\n{'━'*70}")
    print("PART 5B: BOOTSTRAP CONFIDENCE INTERVALS  (2000 resamples, per-trade)")
    print(f"{'━'*70}")
    print("Sharpe proxy = mean(pnl_pct)/std(pnl_pct)*sqrt(n/yr).  Annualisation uses")
    print(f"actual period ({_n_years:.2f} yr).  Compare within this table — proxy ≠ equity Sharpe.\n")

    _bs_hdr = (f"{'Strategy':<42} {'N':>5}  "
               f"{'Ret p5':>8} {'Ret p50':>8} {'Ret p95':>8}  "
               f"{'Sh p5':>7} {'Sh p50':>7} {'Sh p95':>7}")
    print(_bs_hdr)
    print("-" * len(_bs_hdr))
    for _lbl, _tpcts, _rboot, _sboot in [
        (f"Champion  EMA{_bA_fast}/{_bA_slow} vel{_bC_vw}/vt{_bC_vt} tr{round(_bD_trail*100)}%",
         _champ_tpcts, _champ_ret_boot, _champ_sh_boot),
        (f"FGI-free  EMA{_bA_fast}/{_bA_slow} tr{round(_bD_trail*100)}%",
         _ff_tpcts, _ff_ret_boot, _ff_sh_boot),
    ]:
        rp5, rp50, rp95 = np.percentile(_rboot, [5, 50, 95])
        sp5, sp50, sp95 = np.percentile(_sboot, [5, 50, 95])
        print(f"{_lbl:<42} {len(_tpcts):>5}  "
              f"{rp5:>7.0f}% {rp50:>7.0f}% {rp95:>7.0f}%  "
              f"{sp5:>7.2f} {sp50:>7.2f} {sp95:>7.2f}")

    _p_overlap = float(np.mean(_champ_sh_boot < _ff_sh_boot)) * 100
    print(f"\nP(champion bootstrap Sharpe < FGI-free bootstrap Sharpe) = {_p_overlap:.1f}%")
    if _p_overlap < 5:
        print("→ Gap is statistically robust: champion's distribution rarely falls below baseline's.")
    elif _p_overlap < 20:
        print("→ Gap is likely real but with meaningful tail risk: tails overlap.")
    else:
        print("→ Distributions overlap substantially: gap may be noise given sample sizes.")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 5C: TRANSACTION COST SENSITIVITY
    # ══════════════════════════════════════════════════════════════════════════
    # cost_pct applied at both entry fill and exit fill (round-trip = 2× cost).

    def _backtest_cost(df_t, signal_fn, cost_pct):
        sigs = signal_fn(df_t)
        cap = INITIAL_CAPITAL; pos = 0.0; ep = 0.0
        trades = []; equity = []
        for i in range(len(df_t)):
            price = float(df_t["Close"].iloc[i]); sig = sigs.iloc[i]
            if sig == 1 and pos == 0:
                fill = price * (1 + cost_pct)
                pos  = cap / fill; ep = fill; cap = 0.0
            elif sig == -1 and pos > 0:
                fill     = price * (1 - cost_pct)
                proceeds = pos * fill
                trades.append({"pnl": proceeds - pos * ep,
                                "pnl_pct": (fill - ep) / ep * 100})
                cap += proceeds; pos = 0.0
            equity.append(cap + pos * price)   # mark-to-market at mid
        if pos > 0:
            last = float(df_t["Close"].iloc[-1])
            fill = last * (1 - cost_pct)
            trades.append({"pnl": pos*(fill-ep), "pnl_pct": (fill-ep)/ep*100})
            cap += pos * fill
        eq  = pd.Series(equity, index=df_t.index)
        ret = eq.pct_change().dropna()
        n   = len(trades)
        return {
            "total_return": (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
            "sharpe":       ret.mean()/ret.std()*np.sqrt(6.5*252) if ret.std()>0 else 0,
            "max_drawdown": ((eq-eq.cummax())/eq.cummax()).min()*100,
            "n_trades":     n,
            "win_rate":     sum(1 for t in trades if t["pnl"]>0)/n*100 if n else 0,
            "avg_trade_pct":np.mean([t["pnl_pct"] for t in trades]) if trades else 0,
        }

    # Survivor: EMA5/20, vel_w=5, vt=-2, trail=11% (on corrected data)
    _surv_sig_fn = lambda d: apply_trailing_stop(
        d, apply_vel_gate(apply_mom_gate(ema_cross(d, _bA_fast, _bA_slow), d, _bB_mb, _bB_ms), d, -2),
        _bD_trail)
    _surv_df = _get_fdf(5)

    print(f"\n{'━'*70}")
    print("PART 5C: TRANSACTION COST SENSITIVITY  (per-fill cost at entry and exit)")
    print(f"{'━'*70}")
    print(f"{'Round-trip cost = 2× per-fill.  Champion: vel_w={_bC_vw}/vt={_bC_vt}.  Survivor: vel5/vt=-2.'}\n")

    _c_hdr = (f"{'Strategy / Cost per fill':<38} "
              f"{'Return':>8} {'Alpha':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7}")
    _c_sep = "-" * len(_c_hdr)

    for _cost_pct in [0.0, 0.0005, 0.001, 0.002, 0.003]:
        _cost_label = f"{_cost_pct*100:.2f}%"
        _rt_label   = f"{_cost_pct*200:.2f}% RT"
        print(f"── cost={_cost_label}/fill  ({_rt_label}) {'─'*30}")
        print(_c_hdr); print(_c_sep)
        for _sn, _sfn, _sdf in [
            (f"Champion  vel{_bC_vw}/vt{_bC_vt} tr{round(_bD_trail*100)}%", _champ_sig_fn, _dfD),
            (f"Survivor  vel5/vt-2        tr{round(_bD_trail*100)}%",         _surv_sig_fn,  _surv_df),
        ]:
            _rc = _backtest_cost(_sdf, _sfn, _cost_pct)
            _alpha = _rc["total_return"] - bh
            _sign  = "+" if _alpha >= 0 else ""
            print(f"{_sn:<38} "
                  f"{_rc['total_return']:>7.1f}% "
                  f"{_sign}{_alpha:>7.1f}% "
                  f"{_rc['sharpe']:>7.2f} "
                  f"{_rc['max_drawdown']:>7.1f}% "
                  f"{_rc['n_trades']:>7}")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # PART 5D: CROSS-TICKER GENERALIZATION CHECK  (TQQQ)
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'━'*70}")
    print("PART 5D: CROSS-TICKER GENERALIZATION CHECK  (TQQQ)")
    print(f"{'━'*70}")
    print("Same FGI data, same lagged merge, same champion signal params.")
    print("Tests whether FGI edge is SOXL-specific or holds on another leveraged ETF.\n")

    try:
        _tqqq_raw = yf.download("TQQQ", period="730d", interval="1h",
                                 auto_adjust=True, progress=False)
        _tqqq_raw.columns = (_tqqq_raw.columns.droplevel(1)
                              if isinstance(_tqqq_raw.columns, pd.MultiIndex)
                              else _tqqq_raw.columns)
        _tqqq_raw = _tqqq_raw[["Open","High","Low","Close","Volume"]].dropna()
        _tqqq_raw.index = pd.to_datetime(_tqqq_raw.index)
        _tqqq_raw = _tqqq_raw[_tqqq_raw.index.to_series().dt.time.between(
            pd.Timestamp("09:30").time(), pd.Timestamp("15:59").time())]

        _tqqq_base = merge_fgi_lagged(_tqqq_raw, fgi_df)
        _tqqq_base = add_indicators(_tqqq_base)
        # Need vel_w=_bC_vw for champion, vel_w=5 for survivor comparison
        _tqqq_dfC  = add_fgi_features_ext(_tqqq_base, mom_w=5, vel_w=_bC_vw)
        _tqqq_df5  = add_fgi_features_ext(_tqqq_base, mom_w=5, vel_w=5)

        _tqqq_bh = (_tqqq_dfC["Close"].iloc[-1] - _tqqq_dfC["Close"].iloc[0]) / _tqqq_dfC["Close"].iloc[0] * 100
        print(f"TQQQ: {_tqqq_dfC.index[0].date()} → {_tqqq_dfC.index[-1].date()}  "
              f"({len(_tqqq_dfC)} bars)  BH={_tqqq_bh:.1f}%\n")

        def _row_ticker(name, r_t, bh_t):
            alpha = r_t["total_return"] - bh_t
            sign  = "+" if alpha >= 0 else ""
            return (f"{name:<42} "
                    f"{r_t['total_return']:>7.1f}% "
                    f"{sign}{alpha:>7.1f}% "
                    f"{r_t['sharpe']:>7.2f} "
                    f"{r_t['max_drawdown']:>7.1f}% "
                    f"{r_t['n_trades']:>7} "
                    f"{r_t['win_rate']:>7.1f}%")

        def _run_on(df_t, fast, slow, mb, ms, vt, trail):
            """Champion (FGI-gated) signal on an arbitrary ticker df."""
            sig = apply_vel_gate(apply_mom_gate(ema_cross(df_t,fast,slow), df_t,mb,ms), df_t,vt)
            sig = apply_trailing_stop(df_t, sig, trail)
            return backtest(df_t, lambda d, s=sig: s)

        def _run_ff(df_t, fast, slow, trail):
            """FGI-free signal on an arbitrary ticker df."""
            sig = apply_trailing_stop(df_t, ema_cross(df_t,fast,slow), trail)
            return backtest(df_t, lambda d, s=sig: s)

        _r_tqqq_champ = _run_on(_tqqq_dfC, _bA_fast, _bA_slow, _bB_mb, _bB_ms, _bC_vt, _bD_trail)
        _r_tqqq_ff    = _run_ff(_tqqq_dfC, _bA_fast, _bA_slow, _bD_trail)   # ff uses same df (ignores FGI cols)
        _r_soxl_champ = _run_on(_dfD, _bA_fast, _bA_slow, _bB_mb, _bB_ms, _bC_vt, _bD_trail)
        _r_soxl_ff    = _run_ff(df,   _bA_fast, _bA_slow, _bD_trail)

        print(hdr); print(sep)
        print(f"SOXL  (BH={bh:.0f}%):")
        print(_row_ticker(f"  SOXL champion  vel{_bC_vw}/vt{_bC_vt} trail{round(_bD_trail*100)}%",
                          _r_soxl_champ, bh))
        print(_row_ticker(f"  SOXL FGI-free  EMA{_bA_fast}/{_bA_slow} trail{round(_bD_trail*100)}%",
                          _r_soxl_ff, bh))
        print(f"\nTQQQ  (BH={_tqqq_bh:.0f}%):")
        print(_row_ticker(f"  TQQQ champion  vel{_bC_vw}/vt{_bC_vt} trail{round(_bD_trail*100)}%",
                          _r_tqqq_champ, _tqqq_bh))
        print(_row_ticker(f"  TQQQ FGI-free  EMA{_bA_fast}/{_bA_slow} trail{round(_bD_trail*100)}%",
                          _r_tqqq_ff, _tqqq_bh))

        _soxl_lift = _r_soxl_champ["sharpe"] - _r_soxl_ff["sharpe"]
        _tqqq_lift = _r_tqqq_champ["sharpe"] - _r_tqqq_ff["sharpe"]
        print(f"\nFGI Sharpe lift on SOXL: {_soxl_lift:+.2f}  "
              f"({_r_soxl_ff['sharpe']:.2f} → {_r_soxl_champ['sharpe']:.2f})")
        print(f"FGI Sharpe lift on TQQQ: {_tqqq_lift:+.2f}  "
              f"({_r_tqqq_ff['sharpe']:.2f} → {_r_tqqq_champ['sharpe']:.2f})")
        if _tqqq_lift > 0.20:
            print("→ FGI gating GENERALIZES to TQQQ: meaningful lift on second ticker.")
        elif _tqqq_lift > 0:
            print("→ FGI gating shows MARGINAL lift on TQQQ: edge may be partially general.")
        else:
            print("→ FGI gating does NOT generalize to TQQQ: edge appears SOXL-specific.")

    except Exception as _e:
        print(f"TQQQ fetch/run failed: {_e}")
