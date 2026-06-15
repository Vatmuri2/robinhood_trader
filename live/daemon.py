#!/usr/bin/env python3
"""
SOXL trader daemon — fires run_signal.py at :30 past each market hour.

Start whenever you like; the daemon works out whether to run now or wait.
NYSE holidays and weekends are handled by run_signal.py's own calendar check.

Managed via systemd:
  sudo systemctl start   soxl-trader
  sudo systemctl stop    soxl-trader
  sudo systemctl restart soxl-trader
  sudo systemctl status  soxl-trader
  sudo journalctl -u soxl-trader -f        # live log stream

Or run directly for testing:
  python live/daemon.py
"""

import os
import subprocess
import sys
import time
import signal as _signal
from datetime import datetime, timedelta
from pathlib import Path

import pytz

ET       = pytz.timezone("America/New_York")
PYTHON   = sys.executable
SCRIPT   = str(Path(__file__).parent / "run_signal.py")

# Signal checks fire at :30 past each hour from OPEN_HOUR to CLOSE_HOUR (inclusive).
# 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 — six checks per trading day.
OPEN_HOUR  = 10
CLOSE_HOUR = 15

_running = True


def _handle_signal(signum, frame):
    global _running
    print(f"[daemon] Signal {signum} — will stop after current check.", flush=True)
    _running = False


_signal.signal(_signal.SIGTERM, _handle_signal)
_signal.signal(_signal.SIGINT,  _handle_signal)


def _in_window(now: datetime) -> bool:
    """True if now is a weekday within the check-hour range."""
    return now.weekday() < 5 and OPEN_HOUR <= now.hour <= CLOSE_HOUR


def _next_half_hour(now: datetime) -> datetime:
    """Return the next :30:00 mark (ET) strictly after now."""
    if now.minute < 30:
        return now.replace(minute=30, second=0, microsecond=0)
    return (now + timedelta(hours=1)).replace(minute=30, second=0, microsecond=0)


def _run_signal() -> None:
    now = datetime.now(ET)
    print(f"[daemon] {now.strftime('%Y-%m-%d %H:%M ET')} — invoking run_signal.py", flush=True)
    try:
        result = subprocess.run([PYTHON, SCRIPT], timeout=300)
        print(f"[daemon] run_signal.py finished (exit {result.returncode})", flush=True)
    except subprocess.TimeoutExpired:
        print("[daemon] run_signal.py timed out after 300s", flush=True)
    except Exception as e:
        print(f"[daemon] run_signal.py error: {e}", flush=True)


def _sleep_interruptible(seconds: float) -> None:
    """Sleep for `seconds`, waking every 10s to check _running."""
    deadline = time.monotonic() + seconds
    while _running:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 10))


def main() -> None:
    print(f"[daemon] SOXL trader daemon started (pid {os.getpid()})", flush=True)

    now = datetime.now(ET)

    # If started mid-hour past :30 and within the window, run immediately
    # rather than making the user wait until the next :30 mark.
    if _in_window(now) and now.minute >= 30:
        _run_signal()

    while _running:
        now      = datetime.now(ET)
        next_run = _next_half_hour(now)
        wait_sec = (next_run - now).total_seconds()

        print(
            f"[daemon] Sleeping {wait_sec/60:.1f} min → next check at "
            f"{next_run.strftime('%H:%M ET')}",
            flush=True,
        )
        _sleep_interruptible(wait_sec)

        if not _running:
            break

        now = datetime.now(ET)
        if _in_window(now):
            _run_signal()
        else:
            print(
                f"[daemon] {now.strftime('%H:%M ET')} — outside check window, skipping",
                flush=True,
            )

    print("[daemon] Stopped.", flush=True)


if __name__ == "__main__":
    main()
