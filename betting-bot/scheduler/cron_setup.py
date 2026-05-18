import os
import subprocess
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CRON_JOBS = [
    # ── Data collection: 2× per day to stay within free-tier quota ───────────
    # Budget: 8 sport_keys × 2 collects/day × 31 days = 496 req/month (<500).
    # 06:00  morning baseline — feeds morning + midday predict windows
    # 15:50  pre-afternoon refresh — fresh odds for evening matches
    # The 11:50 midday collect was removed to preserve the free-tier budget.
    ("0 6 * * *",    "--collect",       "Collect data — morning baseline"),
    ("50 15 * * *",  "--collect",       "Collect data — pre-afternoon"),

    # ── Predict: 3 windows, window auto-detected from clock ───────────────────
    # morning   07:30 → matches kicking off before 13:00
    # midday    12:00 → matches kicking off 13:00-18:30
    # afternoon 16:00 → matches kicking off 18:30-midnight
    ("30 7 * * *",   "--predict",       "Predict — morning window (kickoff before 13:00)"),
    ("0 12 * * *",   "--predict",       "Predict — midday window (kickoff 13:00-18:30)"),
    ("0 16 * * *",   "--predict",       "Predict — afternoon window (kickoff 18:30-midnight)"),

    # ── Brief: runs 30 min after each predict, sends only new (unbriefed) bets ─
    ("30 8 * * *",   "--brief",         "Brief — morning window"),
    ("30 12 * * *",  "--brief",         "Brief — midday window"),
    ("30 16 * * *",  "--brief",         "Brief — afternoon window"),

    # ── Fixed daily jobs ──────────────────────────────────────────────────────
    ("0 */6 * * *",  "--odds_snapshot", "Save odds snapshot every 6h"),
    ("30 18 * * *",  "--goalie_check",  "Check goalie status 90min pre-game"),
    ("0 23 * * *",   "--summarize",     "Send evening summary"),
    ("0 10 * * *",   "--results",       "Settle yesterday's results"),
    ("0 9 * * *",    "--healthcheck",   "Daily health check"),

    # ── Weekly jobs ───────────────────────────────────────────────────────────
    ("0 3 * * 0",    "--retrain",       "Retrain models (Sunday 03:00)"),
    ("0 20 * * 0",   "--weekly_report", "Send weekly report (Sunday 20:00)"),
]


def install_cron_jobs(main_script: Path):
    """Install all cron jobs. On Windows: prints manual instructions instead."""
    import platform
    python = sys.executable
    main_path = str(main_script.resolve())

    if platform.system() == "Windows":
        print("✅ Scheduler-Hinweis für Windows:")
        print("   Cron gibt es auf Windows nicht. Starte den Bot täglich manuell:")
        print()
        for _, arg, comment in CRON_JOBS:
            print(f"   python \"{main_path}\" {arg}   # {comment}")
        print()
        print("   Oder nutze den Windows Task-Scheduler (optional).")
        return 0

    # Linux/macOS: use crontab
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        current = result.stdout if result.returncode == 0 else ""
    except Exception:
        current = ""

    # Strip all existing betting-bot entries so re-runs are always idempotent.
    # Substring matching is fragile when Python path changes (e.g. venv upgrade);
    # a full replace-and-rewrite is simpler and guarantees no duplicates.
    lines = [l for l in current.splitlines() if "# betting-bot:" not in l]

    # Build new job lines with quoted paths to handle spaces in paths/executables
    new_jobs = []
    for schedule, arg, comment in CRON_JOBS:
        job_line = f'{schedule} "{python}" "{main_path}" {arg}  # betting-bot: {comment}'
        lines.append(job_line)
        new_jobs.append(job_line)

    new_crontab = "\n".join(lines) + "\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
    if proc.returncode == 0:
        logger.info(f"Installed/refreshed {len(new_jobs)} cron jobs")
        print(f"✅ Installed {len(new_jobs)} cron jobs")
    else:
        logger.error(f"crontab install failed: {proc.stderr}")
        print(f"❌ crontab install failed: {proc.stderr}")

    return added


def show_cron_jobs():
    """Print current betting-bot cron jobs."""
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if result.returncode != 0:
            print("No crontab installed")
            return
        betting_jobs = [l for l in result.stdout.splitlines() if "betting-bot:" in l]
        if betting_jobs:
            print("Current betting-bot cron jobs:")
            for job in betting_jobs:
                print(f"  {job}")
        else:
            print("No betting-bot cron jobs found")
    except Exception as e:
        print(f"Error reading crontab: {e}")
