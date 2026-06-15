# Raspberry Pi Setup

## 1. System timezone

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
pip install yfinance pandas numpy pytz pandas_market_calendars mcp
```

## 3. Credentials file

systemd reads environment variables from a plain `key=value` file (no `export`).
Create it and lock down permissions:

```bash
cat > ~/robinhood_trader/.env << 'EOF'
email=your@robinhood.com
password=yourpassword
EOF
chmod 600 ~/robinhood_trader/.env
```

Optional — add your ntfy.sh topic to get push notifications on trades:

```bash
echo "NTFY_TOPIC=soxl-trades" >> ~/robinhood_trader/.env
```

## 4. Confirm MCP connectivity (run once during market hours)

```bash
source .venv/bin/activate
python live/test_mcp.py
```

Expected: list of Robinhood MCP tools + a read-only account call result.

## 5. Install systemd service

```bash
# If your Pi username isn't "pi", edit User= and the paths first:
# nano live/soxl-trader.service

sudo cp live/soxl-trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable soxl-trader    # auto-start on boot
```

## 6. Manage the service

```bash
sudo systemctl start   soxl-trader   # start now
sudo systemctl stop    soxl-trader   # stop cleanly
sudo systemctl restart soxl-trader   # restart (e.g. after a config change)
sudo systemctl status  soxl-trader   # one-line status + last few log lines
```

### Shorthand aliases (add to ~/.bashrc, then `source ~/.bashrc`)

```bash
alias tt-start='sudo systemctl start soxl-trader'
alias tt-stop='sudo systemctl stop soxl-trader'
alias tt-restart='sudo systemctl restart soxl-trader'
alias tt-status='sudo systemctl status soxl-trader'
alias tt-logs='sudo journalctl -u soxl-trader -f'
```

Then just type `tt-start`, `tt-stop`, `tt-logs`, etc.

## 7. How the daemon behaves

- **Start any time** — if started mid-hour (e.g. 11:42am), it fires immediately
  if past :30 in a market hour, then sleeps to the next :30 mark.
- **Check schedule** — 10:30 11:30 12:30 13:30 14:30 15:30 ET, Mon–Fri.
- **Nights / weekends / holidays** — the daemon keeps running but skips the
  signal check; `run_signal.py` exits immediately via the NYSE calendar check.
- **Crash recovery** — systemd restarts the daemon automatically after 30s
  (`Restart=on-failure`).
- **Logs** — all output goes to journald. Tail live:

```bash
sudo journalctl -u soxl-trader -f
```

Or review today's runs:

```bash
sudo journalctl -u soxl-trader --since today
```

## 8. Monitor trades

```bash
# Trade log (every run, win or lose)
column -t -s, ~/robinhood_trader/trade_log.csv | less -S

# Current position state
cat ~/robinhood_trader/trade_state.json
```

## 9. Testing sequence

1. `python live/test_mcp.py`                — MCP connectivity (during market hours)
2. `python live/run_signal.py`              — single manual signal check (live)
3. `tt-start` then `tt-logs`               — watch the daemon run hourly
4. Review `trade_log.csv` after a few days  — confirm signal/action cadence

## 10. Updating config or code

```bash
# Pull latest changes
git pull

# Restart to pick them up
tt-restart
```

No need to re-enable or re-copy the service file unless `soxl-trader.service` itself changed.
