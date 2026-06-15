#!/bin/bash
# SOXL Trading Agent — runs every hour during market hours
#
# Crontab (crontab -e on Pi, adjust hours for your timezone):
#   7 14-20 * * 1-5 . $HOME/.profile && /home/pi/robinhood_trader/run_agent.sh >> /home/pi/robinhood_trader/agent.log 2>&1
#
# Required env vars (add to ~/.profile):
#   export email="your@email.com"
#   export password="yourpassword"
#   export NTFY_TOPIC="your-topic"   # optional push notifications

set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJ/.venv/bin/python"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

# Robinhood agentic account number (agentic_allowed=true)
RH_ACCOUNT="${RH_ACCOUNT:-979859477}"

echo ""
echo "[$TS] ═══ SOXL Agent ═══"

# ── Step 1: Run signal script ─────────────────────────────────────────────────
cd "$PROJ"
SIGNAL_OUTPUT="$("$VENV" trade.py 2>&1)"
echo "$SIGNAL_OUTPUT"

if echo "$SIGNAL_OUTPUT" | grep -q "Market is closed"; then
    echo "[$TS] Market closed — done."
    exit 0
fi

# ── Step 2: Parse action from JSON output ─────────────────────────────────────
ACTION="$(echo "$SIGNAL_OUTPUT" | "$VENV" -c "
import sys, json, re
txt = sys.stdin.read()
m = re.search(r'\{[\s\S]+\}', txt)
if m:
    try: print(json.loads(m.group()).get('action', 'HOLD'))
    except: print('HOLD')
else: print('HOLD')
" 2>/dev/null || echo "HOLD")"

echo "[$TS] Action: $ACTION"

if [ "$ACTION" = "HOLD" ]; then
    echo "[$TS] Holding — done."
    exit 0
fi

# ── Step 3: Execute via Claude Code + Robinhood MCP ──────────────────────────
CLAUDE_BIN="$(command -v claude 2>/dev/null || echo '')"

if [ -z "$CLAUDE_BIN" ]; then
    echo "[$TS] claude CLI not found — using execute.py fallback."
    "$VENV" "$PROJ/execute.py" "$ACTION"
    exit 0
fi

echo "[$TS] Invoking Claude..."

PROMPT="You are the SOXL trading agent. Execute this trade immediately, no confirmation needed.

Signal: $ACTION
Account: $RH_ACCOUNT (agentic account, already confirmed)

Signal details:
$SIGNAL_OUTPUT

Steps:
1. Call get_equity_positions to check current SOXL holdings.
2. Call get_portfolio to get buying power.
3. Execute:
   - BUY: if no SOXL held, call place_equity_order(account_number=$RH_ACCOUNT, symbol=SOXL, side=buy, type=market, dollar_amount=<all buying power>, time_in_force=gfd).
   - SELL: if SOXL held, call place_equity_order(account_number=$RH_ACCOUNT, symbol=SOXL, side=sell, type=market, quantity=<full position>, time_in_force=gfd).
4. Call get_equity_orders to confirm the order was accepted.
5. Print one line: what you did and the order ID."

if timeout 90 "$CLAUDE_BIN" --print "$PROMPT" 2>&1; then
    echo "[$TS] Done."
else
    echo "[$TS] Claude failed — using execute.py fallback."
    "$VENV" "$PROJ/execute.py" "$ACTION"
fi
