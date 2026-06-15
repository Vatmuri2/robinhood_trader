# Raspberry Pi Setup

## 1. Prerequisites

Raspberry Pi OS (Bookworm / Bullseye). Set the system timezone to Eastern:

```bash
sudo timedatectl set-timezone America/New_York
timedatectl   # confirm
```

## 2. Python venv + packages

```bash
cd ~/robinhood_trader
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install \
  yfinance \
  pandas \
  numpy \
  pytz \
  pandas_market_calendars \
  mcp \
  robin-stocks    # optional — used only by reconcile_state
```

## 3. Credentials

Add to `~/.profile` (sourced by cron via `. $HOME/.profile`):

```bash
export email="your@robinhood.com"
export password="yourpassword"
export NTFY_TOPIC="soxl-trades"   # optional — ntfy.sh push notifications
```

Then reload: `source ~/.profile`

## 4. Step 0 — confirm MCP connectivity

```bash
cd ~/robinhood_trader
source .venv/bin/activate
python live/test_mcp.py
```

Expected output: list of all Robinhood MCP tools and schemas, plus one
read-only call result (get_accounts). If it fails, check credentials and
that the Pi has internet access to agent.robinhood.com.

## 5. Sanity-check the signal

Verify the signal computation matches a known date from the backtest:

```bash
# Check current signal (dry run — no order placed)
python live/run_signal.py --dry-run
```

Cross-reference the printed signal, FGI values, and position against what
backtest.py would produce for the same date range.

## 6. Crontab

```bash
crontab -e
```

Pi timezone must be `America/New_York` (Step 1 above). Cron times are local ET.

```
# SOXL — hourly 10:30am–3:30pm ET, Mon–Fri
30 10-15 * * 1-5 . $HOME/.profile && /home/pi/robinhood_trader/.venv/bin/python /home/pi/robinhood_trader/live/run_signal.py >> /home/pi/robinhood_trader/live/cron.log 2>&1
```

The :30 past-the-hour timing means each run sees a freshly closed bar at -2.
The script exits immediately on weekends and NYSE holidays.

### Dry-run phase (recommended 1–2 weeks before going live)

Add `--dry-run` to the cron line above. `trade_log.csv` records every run;
confirm signals and intended actions match expectations before removing it.

## 7. Monitor

```bash
# Tail live cron output
tail -f ~/robinhood_trader/live/cron.log

# Review trade log
column -t -s, ~/robinhood_trader/trade_log.csv | less -S

# Check current state
cat ~/robinhood_trader/trade_state.json
```

## 8. Testing sequence

1. `python live/test_mcp.py`           — MCP connectivity + tool listing
2. `python live/run_signal.py --dry-run`  — full signal run, no order
3. Let dry-run cron run 1–2 weeks; compare trade_log.csv vs backtest
4. Remove `--dry-run` from crontab to go live
