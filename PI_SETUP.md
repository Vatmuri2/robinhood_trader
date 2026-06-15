# Raspberry Pi Setup Guide

## 1. Copy project to Pi
```bash
# Run from your Mac:
rsync -avz /Users/vikramatmuri/Downloads/Projects/robinhood_trader/ pi@raspberrypi.local:~/robinhood_trader/
```

## 2. Install Python deps on Pi
```bash
ssh pi@raspberrypi.local
cd ~/robinhood_trader
python3 -m venv .venv
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install yfinance pandas numpy pytz fear-and-greed robin-stocks
```

## 3. Install Claude Code CLI on Pi
```bash
# Requires Node.js 18+
sudo apt-get install -y nodejs npm
npm install -g @anthropic-ai/claude-code

# Authenticate once (follow the prompt):
claude
```

## 4. Set environment variables
Add to `~/.profile` on the Pi:
```bash
export email="your-robinhood@email.com"
export password="your-robinhood-password"
export NTFY_TOPIC="pick-any-private-name"   # for push notifications (optional)
# export RH_ACCOUNT="979859477"             # already defaulted in run_agent.sh
```

## 5. Set Pi timezone to ET
```bash
sudo timedatectl set-timezone America/New_York
```

## 6. Add cron job
```bash
crontab -e
```
Add:
```
7 9-15 * * 1-5 . $HOME/.profile && /home/pi/robinhood_trader/run_agent.sh >> /home/pi/robinhood_trader/agent.log 2>&1
```
Fires at :07 past each hour, 9am–3pm ET, Mon–Fri. `trade.py` enforces the 9:30 open and 4pm close internally.

## 7. Optional: push notifications
Install the **ntfy** app on your phone. Subscribe to whatever you set as `NTFY_TOPIC`.
You'll get a push every time a BUY or SELL fires.

## 8. Verify
```bash
# Signal test (market closed — should print "Market is closed"):
cd ~/robinhood_trader && .venv/bin/python trade.py

# Full agent dry-run:
bash run_agent.sh

# Watch live:
tail -f agent.log
```

## File layout
```
~/robinhood_trader/
├── trade.py          ← signal engine (fetches live FGI + price, outputs JSON)
├── execute.py        ← fallback executor (robin_stocks, used if claude CLI fails)
├── run_agent.sh      ← cron entry point
├── trade_state.json  ← position/peak/entry (auto-managed)
├── agent.log         ← all output
├── .mcp.json         ← Robinhood MCP config (claude reads this automatically)
├── backtest.py       ← strategy research only, not used in live trading
├── fgi_data.csv      ← historical FGI (backtest only)
└── fgi_data_2.csv    ← historical FGI supplement (backtest only)
```

## How it works
```
cron (every hour on weekdays)
  └── run_agent.sh
        ├── trade.py          → fetches SOXL price + live FGI → computes signal
        │     └── trade_state.json  (tracks position, entry, peak for trailing stop)
        └── claude --print    → reads signal → calls Robinhood MCP tools → places order
              └── execute.py  (fallback if claude CLI unavailable)
```
