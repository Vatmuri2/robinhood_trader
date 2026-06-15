"""
SOXL direct executor via robin_stocks.
Fallback when Claude Code CLI is unavailable.

Usage:
  python execute.py BUY
  python execute.py SELL
  python execute.py HOLD   (no-op)
"""

import json, os, sys

try:
    import robin_stocks.robinhood as rh
except ImportError:
    print("ERROR: pip install robin-stocks")
    sys.exit(1)

TICKER   = "SOXL"
EMAIL    = os.environ.get("email", "")
PASSWORD = os.environ.get("password", "")


def login():
    if not EMAIL or not PASSWORD:
        print("ERROR: export email and password env vars.")
        sys.exit(1)
    rh.login(EMAIL, PASSWORD, store_session=True)


def get_cash():
    return float(rh.load_account_profile().get("buying_power", 0))


def get_price():
    q = rh.get_latest_price(TICKER, includeExtendedHours=False)
    return float(q[0]) if q else 0.0


def get_position():
    for pos in (rh.get_open_stock_positions() or []):
        inst = rh.get_instrument_by_url(pos["instrument"])
        if inst and inst.get("symbol") == TICKER:
            return float(pos["quantity"]), float(pos.get("average_buy_price", 0))
    return 0.0, 0.0


def buy():
    cash = get_cash()
    price = get_price()
    if price <= 0 or cash < 1:
        print(f"Cannot buy: cash=${cash:.2f}, price=${price:.2f}")
        return
    print(f"BUY ${cash:.2f} of {TICKER} @ ~${price:.2f}")
    order = rh.order_buy_fractional_by_price(TICKER, cash, timeInForce="gfd")
    print(f"Order: {json.dumps(order, indent=2)}")


def sell():
    qty, avg = get_position()
    if qty < 0.0001:
        print(f"No {TICKER} position to sell.")
        return
    price = get_price()
    pnl = (price - avg) / avg * 100 if avg else 0
    print(f"SELL {qty:.6f} shares of {TICKER} @ ~${price:.2f}  PnL={pnl:+.1f}%")
    order = rh.order_sell_fractional_by_quantity(TICKER, qty, timeInForce="gfd")
    print(f"Order: {json.dumps(order, indent=2)}")


if __name__ == "__main__":
    action = sys.argv[1].upper() if len(sys.argv) > 1 else "HOLD"
    if action == "HOLD":
        print("HOLD — no action.")
        sys.exit(0)
    login()
    if action == "BUY":
        buy()
    elif action.startswith("SELL"):
        sell()
    else:
        print(f"Unknown action: {action}")
