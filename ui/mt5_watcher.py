"""
ui/mt5_watcher.py

Watches MT5's report folders for new HTML backtest reports.
When a new report appears, parses it and stores results automatically.

Usage:
    watcher = MT5ReportWatcher()
    watcher.start()          # begins watching in background thread
    watcher.stop()           # stops watching
    watcher.scan_once()      # one-shot scan for any new reports
"""

import os
import time
import threading
import glob
from pathlib import Path
from typing import Callable


# ── Known MT5 report locations ────────────────────────────────────────────────
# MT5 can save reports in several places depending on version and settings.
# We watch all of them.

def get_watch_dirs() -> list[Path]:
    """Return all directories where MT5 might save backtest reports."""
    home     = Path.home()
    wine_c   = home / ".wine" / "drive_c"
    username = os.environ.get("USER", "user")

    # ── Primary: StrategyBlender's own reports folder ─────────────────────────
    # This is the simplest workflow — user saves reports here manually.
    sb_reports = Path(__file__).parent.parent / "reports"
    sb_reports.mkdir(exist_ok=True)  # create if not exists

    candidates = [
        sb_reports,  # ← checked first

        # MT5 install dir
        wine_c / "Program Files" / "MetaTrader 5" / "Tester",
        wine_c / "Program Files" / "MetaTrader 5",
        wine_c / "Program Files (x86)" / "MetaTrader 5" / "Tester",

        # AppData
        wine_c / "users" / username / "AppData" / "Roaming" / "MetaQuotes",

        # Common save locations
        wine_c / "users" / username / "Documents",
        wine_c / "users" / username / "My Documents",
        wine_c / "users" / username / "Desktop",

        # Windows native
        Path(os.path.expandvars(r"%APPDATA%\MetaQuotes")) if os.name == "nt" else Path("/nonexistent"),
        Path(os.path.expandvars(r"%USERPROFILE%\Documents")) if os.name == "nt" else Path("/nonexistent"),
    ]

    return [d for d in candidates if d.exists()]


def find_all_reports() -> list[Path]:
    """Find all HTML backtest reports in all watch directories."""
    reports = []
    for watch_dir in get_watch_dirs():
        for pattern in ["**/*.html", "**/*.htm"]:
            for f in watch_dir.glob(pattern):
                # Filter out non-MT5 files by checking size (reports are >10KB)
                try:
                    if f.stat().st_size > 10_000:
                        reports.append(f)
                except OSError:
                    continue
    return reports


def is_mt5_report(path: Path) -> bool:
    """Quick check if an HTML file looks like an MT5 backtest report."""
    try:
        # Read first 2KB to check
        with open(path, "rb") as f:
            header = f.read(2048)
        # MT5 reports are UTF-16-LE encoded
        try:
            text = header.decode("utf-16-le", errors="ignore")
        except Exception:
            text = header.decode("utf-8", errors="ignore")
        # MT5 reports always contain these strings
        return "Strategy Tester" in text or "StrategyTester" in text or "Tester Report" in text
    except Exception:
        return False


# ── Watcher ───────────────────────────────────────────────────────────────────

class MT5ReportWatcher:
    """
    Background thread that watches for new MT5 backtest reports.
    Calls on_new_report(path) whenever a new report is detected.
    """

    def __init__(self, on_new_report: Callable[[Path], None] = None,
                 poll_interval: int = 5):
        self.on_new_report  = on_new_report
        self.poll_interval  = poll_interval
        self._known_files: set[str] = set()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        """Start watching in a background thread."""
        # Snapshot existing files so we don't re-import old reports
        self._known_files = {str(p) for p in find_all_reports()}
        self._running = True
        self._thread  = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def scan_once(self) -> list[Path]:
        """
        Scan for new reports right now (blocking).
        Returns list of new report paths found.
        """
        current   = {str(p): p for p in find_all_reports()}
        new_paths = []

        for path_str, path in current.items():
            if path_str not in self._known_files and is_mt5_report(path):
                new_paths.append(path)
                self._known_files.add(path_str)
                if self.on_new_report:
                    self.on_new_report(path)

        return new_paths

    def _watch_loop(self):
        while self._running:
            self.scan_once()
            time.sleep(self.poll_interval)


# ── Report metadata extractor ─────────────────────────────────────────────────

def extract_report_metadata(path: Path) -> dict:
    """
    Read just enough of the report to extract symbol, timeframe, EA name.
    Returns dict with keys: symbol, timeframe, ea_name, date_from, date_to
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-16-le", errors="replace")
        except Exception:
            text = raw.decode("utf-8", errors="replace")

        import re
        meta = {}

        # EA name
        m = re.search(r'Expert[:\s]+([^\n<|]+)', text)
        if m:
            meta["ea_name"] = m.group(1).strip()

        # Symbol
        m = re.search(r'Symbol[:\s]+([A-Z0-9.]+)', text)
        if m:
            meta["symbol"] = m.group(1).strip()

        # Period/timeframe
        m = re.search(r'Period[:\s]+(M\d+|H\d+|D1|W1|MN)', text)
        if m:
            meta["timeframe"] = m.group(1).strip()

        return meta
    except Exception:
        return {}
