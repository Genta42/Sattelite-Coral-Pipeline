#!/usr/bin/env python3
"""
Pipeline notification helper — Telegram bot + terminal.

Three notification levels:
  - notify()          → progress/status (normal message)
  - notify_error()    → 🚨 ERROR — clearly marked
  - notify_and_wait() → 🔴 INPUT NEEDED — clearly marked, waits for terminal
"""
import argparse
import sys
import time
from urllib import request, parse

# ── Telegram config ───────────────────────────────────────────────────
TELEGRAM_TOKEN = "8750286923:AAE6kVzyO7-ggHnf_YwzfT_xG5ny3rEJWmg"
TELEGRAM_CHAT_ID = "8331347781"
# ──────────────────────────────────────────────────────────────────────


def _send_telegram(message: str) -> bool:
    """Send a message via Telegram bot API."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        with request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"  [telegram] Could not send: {exc}", file=sys.stderr)
        return False


def notify(message: str, title: str = "Coral Pipeline") -> None:
    """Progress/status update — sent to Telegram + terminal."""
    _send_telegram(f"*{title}*\n{message}")
    print(f"[{title}] {message}")


def notify_error(message: str) -> None:
    """Error notification — clearly marked in Telegram."""
    text = (
        "🚨🚨🚨 *ERROR* 🚨🚨🚨\n\n"
        f"{message}\n\n"
        "_Check terminal for details_"
    )
    _send_telegram(text)
    print(f"\a\n{'!'*60}\n  ERROR: {message}\n{'!'*60}\n")


def notify_and_wait(prompt: str) -> str:
    """Input needed — clearly marked in Telegram, waits for terminal input."""
    text = (
        "🔴🔴🔴 *INPUT NEEDED* 🔴🔴🔴\n\n"
        f"{prompt}\n\n"
        "_Waiting for your response in terminal..._"
    )
    _send_telegram(text)
    print(f"\a\n{'!'*60}\n  INPUT NEEDED: {prompt}\n{'!'*60}")
    return input(f">>> ")


# ── Progress Tracker ──────────────────────────────────────────────────


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 360_000:
        return "??:??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _make_bar(pct: float, width: int = 16) -> str:
    filled = int(width * pct)
    empty = width - filled
    return "\u2588" * filled + "\u2591" * empty


class ProgressTracker:
    """
    Sends periodic progress updates to Telegram.

    Usage:
        tracker = ProgressTracker("Build Table", total=2640, unit="CSVs")
        for i in range(2640):
            process(i)
            tracker.update(i + 1, extra={"rows": total_rows})
        tracker.finish()
    """

    def __init__(
        self,
        stage_name: str,
        total: int,
        unit: str = "items",
        interval_sec: float = 300.0,  # every 5 min
        pct_step: float = 0.05,       # or every 5%
    ):
        self.stage_name = stage_name
        self.total = total
        self.unit = unit
        self.interval_sec = interval_sec
        self.pct_step = pct_step

        self._start_time = time.time()
        self._last_send_time = 0.0
        self._last_send_pct = -1.0

    def _format_extra(self, extra: dict | None) -> str:
        if not extra:
            return ""
        parts = []
        for k, v in extra.items():
            if isinstance(v, (int, float)) and v >= 1000:
                parts.append(f"{k}: {v:,.0f}")
            else:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def update(self, completed: int, extra: dict | None = None) -> None:
        if self.total <= 0:
            return

        now = time.time()
        pct = completed / self.total
        elapsed = now - self._start_time

        time_ok = (now - self._last_send_time) >= self.interval_sec
        pct_ok = pct >= (self._last_send_pct + self.pct_step)

        if not (time_ok or pct_ok):
            return

        if completed > 0 and pct < 1.0:
            eta_sec = (elapsed / completed) * (self.total - completed)
            eta_str = _format_eta(eta_sec)
        else:
            eta_str = "--"

        bar = _make_bar(pct)
        lines = [
            f"*{self.stage_name}*",
            f"`{bar}` {pct * 100:.0f}%",
            f"{completed:,} / {self.total:,} {self.unit} | ETA: {eta_str}",
        ]
        extra_str = self._format_extra(extra)
        if extra_str:
            lines.append(extra_str)

        _send_telegram("\n".join(lines))

        self._last_send_time = now
        self._last_send_pct = pct

    def finish(self, extra: dict | None = None) -> None:
        elapsed = time.time() - self._start_time
        elapsed_str = _format_eta(elapsed)

        lines = [
            f"*{self.stage_name}* — Done",
            f"`{_make_bar(1.0)}` 100%",
            f"{self.total:,} {self.unit} | took {elapsed_str}",
        ]
        extra_str = self._format_extra(extra)
        if extra_str:
            lines.append(extra_str)

        _send_telegram("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pipeline notification helper")
    p.add_argument("message", help="Notification message")
    p.add_argument("--title", default="Coral Pipeline")
    p.add_argument("--error", action="store_true")
    p.add_argument("--input", action="store_true")
    args = p.parse_args()

    if args.input:
        response = notify_and_wait(args.message)
        print(f"User response: {response}")
    elif args.error:
        notify_error(args.message)
    else:
        notify(args.message, title=args.title)
