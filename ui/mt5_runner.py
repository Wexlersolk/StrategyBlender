"""
ui/mt5_runner.py

Triggers MT5 Strategy Tester via INI config.

Folder structure created under MT5's Experts/StrategyBlender/:
    Experts/StrategyBlender/
        {Symbol}/
            {Timeframe}/
                {StrategyName}/
                    source/
                        StrategyName.mq5
                    reports/
                        {date_from}_{date_to}/
                            Report.html
"""

import os
import re
import time
import platform
import subprocess
import glob
from pathlib import Path


# ── Root paths ────────────────────────────────────────────────────────────────

MT5_ROOT = Path.home() / ".wine/drive_c/Program Files/MetaTrader 5"
MQL5_ROOT = MT5_ROOT / "MQL5"
SB_ROOT   = MQL5_ROOT / "Experts" / "StrategyBlender"


def get_strategy_dir(symbol: str, timeframe: str, strategy_name: str) -> Path:
    """Return the StrategyBlender folder for this strategy."""
    safe_sym  = re.sub(r'[^\w.]', '_', symbol)
    safe_tf   = timeframe.upper()
    safe_name = re.sub(r'[^\w\-. ]', '_', strategy_name).strip()
    return SB_ROOT / safe_sym / safe_tf / safe_name


def get_report_dir(symbol: str, timeframe: str, strategy_name: str,
                   date_from: str, date_to: str) -> Path:
    period = f"{date_from.replace('.', '')}_{date_to.replace('.', '')}"
    return get_strategy_dir(symbol, timeframe, strategy_name) / "reports" / period


def _linux_to_wine(path: str) -> str:
    return "Z:" + path.replace("/", "\\")


# ── EA deployment ─────────────────────────────────────────────────────────────

def deploy_ea(symbol: str, timeframe: str, strategy_name: str,
              ea_source: str) -> Path:
    """
    Write EA source into StrategyBlender folder structure.
    Returns path to the .mq5 file.
    """
    source_dir = get_strategy_dir(symbol, timeframe, strategy_name) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r'[^\w\-. ]', '_', strategy_name).strip()
    ea_path   = source_dir / f"{safe_name}.mq5"
    ea_path.write_text(ea_source, encoding="utf-8")
    return ea_path


# ── INI config ────────────────────────────────────────────────────────────────

TF_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408,
    "W1": 32769, "MN": 49153,
}


def _write_ini(ea_wine_path: str, symbol: str, timeframe: str,
               date_from: str, date_to: str, login: int, password: str,
               server: str, report_wine_path: str, model: int = 1,
               deposit: float = 100000.0) -> Path:
    """
    Write tester INI into the tester/ folder inside MT5 root.
    Both EA path and report path must be Windows-style paths.
    """
    ini_dir  = MT5_ROOT / "tester"
    ini_dir.mkdir(parents=True, exist_ok=True)
    ini_path = ini_dir / f"sb_{int(time.time())}.ini"
    tf_value = TF_MAP.get(timeframe.upper(), 16385)

    content = (
        f"[Tester]\n"
        f"Expert={ea_wine_path}\n"
        f"Symbol={symbol}\n"
        f"Period={tf_value}\n"
        f"Deposit={deposit:.0f}\n"
        f"Currency=USD\n"
        f"ProfitInPips=0\n"
        f"Model={model}\n"
        f"FromDate={date_from}\n"
        f"ToDate={date_to}\n"
        f"Report={report_wine_path}\n"
        f"ReplaceReport=1\n"
        f"ShutdownTerminal=1\n"
        f"Login={login}\n"
        f"Password={password}\n"
        f"Server={server}\n"
    )
    ini_path.write_text(content, encoding="utf-8")
    return ini_path


# ── Report finder ─────────────────────────────────────────────────────────────

def _find_report(report_dir: Path, timeout: int) -> str | None:
    """Poll for HTML report in the expected directory."""
    start = time.time()
    while time.time() - start < timeout:
        for ext in ("*.html", "*.htm"):
            matches = list(report_dir.glob(ext))
            if matches:
                time.sleep(1)
                return str(matches[0])
        # Also search broadly under MT5 root in case MT5 saves elsewhere
        broad = glob.glob(str(MT5_ROOT / "**" / "*.html"), recursive=True)
        fresh = [f for f in broad
                 if os.path.getmtime(f) > start and "StrategyBlender" in f]
        if fresh:
            return fresh[0]
        time.sleep(2)
    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def run_backtest(
    ea_source_path: str = "",
    symbol: str = "",
    timeframe: str = "H1",
    date_from: str = "2020.01.01",
    date_to: str   = "2026.01.01",
    login: int     = 0,
    password: str  = "",
    server: str    = "",
    progress_callback=None,
    timeout: int   = 300,
    model: int     = 1,
    ea_source: str = "",
    ea_name: str   = "",
) -> tuple[bool, str, str]:

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    terminal = MT5_ROOT / "terminal64.exe"
    if not terminal.exists():
        return False, f"MT5 terminal not found at {terminal}", ""

    log(f"MT5 root: {MT5_ROOT}")

    # ── 1. Deploy EA source ───────────────────────────────────────────────────
    if not ea_source:
        return False, "No EA source code provided.", ""

    try:
        ea_path = deploy_ea(symbol, timeframe, ea_name, ea_source)
        log(f"EA deployed: {ea_path}")
    except Exception as e:
        return False, f"Failed to deploy EA: {e}", ""

    # EA path as Windows path relative to MT5 root for the INI
    # MT5 expects Expert= to be relative from MQL5/Experts or absolute Wine path
    ea_wine_path = _linux_to_wine(str(ea_path))
    log(f"EA Wine path: {ea_wine_path}")

    # ── 2. Prepare report directory ───────────────────────────────────────────
    report_dir = get_report_dir(symbol, timeframe, ea_name, date_from, date_to)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file_wine = _linux_to_wine(str(report_dir / "Report"))
    log(f"Report will be saved to: {report_dir}")

    # ── 3. Write INI ──────────────────────────────────────────────────────────
    try:
        ini_path = _write_ini(
            ea_wine_path=ea_wine_path,
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
            login=login,
            password=password,
            server=server,
            report_wine_path=report_file_wine,
            model=model,
        )
        log(f"INI config: {ini_path}")
    except Exception as e:
        return False, f"Failed to write INI: {e}", ""

    # ── 4. Launch MT5 ─────────────────────────────────────────────────────────
    wine_ini = _linux_to_wine(str(ini_path))
    cmd = ["wine", str(terminal), f"/config:{wine_ini}"]
    log(f"Launching MT5: {' '.join(cmd)}")

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        return False, f"Failed to launch MT5: {e}", ""

    # ── 5. Wait for report ────────────────────────────────────────────────────
    log(f"Waiting for backtest to complete (timeout: {timeout}s)...")
    report_path = _find_report(report_dir, timeout)

    try:
        ini_path.unlink()
    except Exception:
        pass

    if report_path:
        log(f"Report saved: {report_path}")
        return True, "Backtest completed.", report_path

    return False, (
        f"Timeout after {timeout}s — report not found.\n"
        "Likely causes:\n"
        "• EA failed to compile — check it's valid MQL5\n"
        "• Symbol not available on your broker\n"
        "• MT5 opened but didn't run the tester\n\n"
        f"Check MT5 Experts/Journal tab for errors.\n"
        f"Expected report at: {report_dir}"
    ), ""


# ── Utility — list all saved reports ─────────────────────────────────────────

def list_saved_reports() -> list[dict]:
    """
    Return all saved backtest reports as a list of dicts:
    { symbol, timeframe, strategy, period, path }
    """
    reports = []
    if not SB_ROOT.exists():
        return reports

    for html in SB_ROOT.glob("**/reports/**/*.html"):
        parts = html.relative_to(SB_ROOT).parts
        if len(parts) >= 4:
            reports.append({
                "symbol":    parts[0],
                "timeframe": parts[1],
                "strategy":  parts[2],
                "period":    parts[4] if len(parts) > 4 else "",
                "path":      str(html),
            })
    return reports
