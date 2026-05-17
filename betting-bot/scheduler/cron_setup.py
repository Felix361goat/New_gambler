import os
import subprocess
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CRON_JOBS = [
    ("0 6 * * *",    "--collect",       "Collect all data"),
    ("30 7 * * *",   "--predict",       "Run prediction engine"),
    ("30 8 * * *",   "--brief",         "Send morning briefing"),
    ("0 */6 * * *",  "--odds_snapshot", "Save odds snapshot"),
    ("0 23 * * *",   "--summarize",     "Send evening summary"),
    ("0 3 * * 0",    "--retrain",       "Retrain models"),
    ("0 20 * * 0",   "--weekly_report", "Send weekly report"),
    ("0 10 * * *",   "--results",       "Settle yesterday's results"),
    ("0 9 * * *",    "--healthcheck",   "Daily health check"),
    ("30 18 * * *",  "--goalie_check",  "Check goalie status pre-game"),
]


def install_cron_jobs(main_script: Path):
    """Install all cron jobs. Idempotent — won't duplicate existing jobs."""
    python = sys.executable
    main_path = str(main_script.resolve())

    # Read current crontab
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        current = result.stdout if result.returncode == 0 else ""
    except Exception:
        current = ""

    lines = current.splitlines()
    added = 0

    for schedule, arg, comment in CRON_JOBS:
        job_line = f"{schedule} {python} {main_path} {arg}  # betting-bot: {comment}"
        # Check if this arg is already scheduled
        if any(f"main.py {arg}" in line or f"main.py {arg}" in line for line in lines):
            logger.info(f"Cron job already exists: {arg}")
            continue
        lines.append(job_line)
        added += 1

    if added > 0:
        new_crontab = "\n".join(lines) + "\n"
        proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
        if proc.returncode == 0:
            logger.info(f"Installed {added} new cron jobs")
            print(f"✅ Installed {added} cron jobs")
        else:
            logger.error(f"crontab install failed: {proc.stderr}")
            print(f"❌ crontab install failed: {proc.stderr}")
    else:
        print("✅ All cron jobs already installed")

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
