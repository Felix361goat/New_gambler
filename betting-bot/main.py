#!/usr/bin/env python3
"""
Autonomous Sports Betting Prediction Bot — Main Entry Point

Usage:
  python main.py --setup          First-time setup
  python main.py --collect        Fetch all data sources
  python main.py --predict        Run prediction engine
  python main.py --brief          Send morning briefing via Telegram
  python main.py --odds_snapshot  Save current odds snapshot to DB
  python main.py --summarize      Send evening summary via Telegram
  python main.py --retrain        Retrain all models
  python main.py --weekly_report  Send weekly report
  python main.py --results        Settle yesterday's results
  python main.py --test           Run all tests
  python main.py --paper_status   Print current paper mode statistics
"""
import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import yaml

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "data" / "betting.log", mode="a"),
    ],
)
logger = logging.getLogger("main")


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        logger.error("config.yaml not found")
        sys.exit(1)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    # Resolve env vars in string values
    import re
    def resolve(obj):
        if isinstance(obj, str):
            return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(1)), obj)
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj
    return resolve(raw)


def cmd_setup(config: dict):
    """First-time setup: DB, verify API keys, install cron jobs."""
    print("🔧 Running first-time setup...")

    # Create directories
    for d in ["data", "data/sources"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    # Init DB
    from tracking.database import DatabaseHandler
    db = DatabaseHandler()
    print("✅ Database initialized")

    # Verify API keys
    required_env = ["FOOTBALL_API_KEY", "ODDS_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing = [k for k in required_env if not os.environ.get(k) or os.environ.get(k, "").startswith("your_")]
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("   Fill in your .env file and run --setup again")
    else:
        print("✅ All API keys found")

    # Sheets setup (optional)
    sheets_id = os.environ.get("GOOGLE_SHEET_ID", "")
    creds_file = ROOT / config.get("google_sheets", {}).get("credentials_file", "google_credentials.json")
    if sheets_id and creds_file.exists():
        from tracking.sheets import SheetsHandler
        sheets = SheetsHandler(config)
        if sheets.connect():
            print("✅ Google Sheets connected")
        else:
            print("⚠️  Google Sheets connection failed (optional — continuing)")
    else:
        print("⚠️  Google Sheets skipped (no credentials or sheet ID)")

    # Install cron jobs
    from scheduler.cron_setup import install_cron_jobs
    install_cron_jobs(ROOT / "main.py")

    print("\n✅ Setup complete! Paper mode starts tomorrow at 08:30.")


def cmd_collect(config: dict):
    """Fetch all data sources."""
    from data.collector import DataCollector
    from tracking.database import DatabaseHandler

    db = DatabaseHandler()
    collector = DataCollector(config, db)
    result = collector.collect_all()
    print(result.summary())
    return result


def cmd_predict(config: dict):
    """Run prediction engine and store daily bets."""
    from tracking.database import DatabaseHandler
    from data.collector import DataCollector
    from features.builder import FeatureBuilder
    from models.poisson_model import PoissonModel
    from models.elo_model import EloModel
    from models.xgboost_model import XGBoostModel
    from models.ensemble import EnsembleModel
    from selection.ev_calculator import calculate_ev, calculate_ev_betfair, find_best_odds, get_platform_odds
    from selection.kelly import kelly_stake
    from selection.filter import select_daily_bets
    from tracking.performance import PerformanceTracker

    db = DatabaseHandler()
    collector = DataCollector(config, db)

    # Load historical data for models
    logger.info("Loading training data...")
    training_data = db.get_bets_for_retraining()

    # Initialize models
    poisson = PoissonModel()
    elo = EloModel()
    xgboost = XGBoostModel()

    # Try to load pre-trained XGBoost
    if not xgboost.load():
        logger.info("No pre-trained XGBoost found — using Poisson + ELO only")

    # Fit on historical match data if available
    collection = collector.collect_all()
    matches_df = getattr(collection, "matches", None)

    if matches_df is not None and not matches_df.empty:
        logger.info("Fitting Poisson and ELO models...")
        poisson.fit(matches_df)
        elo.fit(matches_df)

    ensemble = EnsembleModel(poisson, xgboost, elo, config)
    feature_builder = FeatureBuilder(db, config)
    tracker = PerformanceTracker(db, config)

    # CLV auto-gate: warn if rolling average CLV is negative (model overpaying)
    clv_ok, avg_clv = tracker.check_clv_gate(db)
    if avg_clv is not None:
        if clv_ok:
            logger.info(f"CLV gate PASSED: rolling avg CLV = {avg_clv:+.4f} ({avg_clv*100:+.2f}%)")
        else:
            logger.warning(
                f"CLV gate FAILED: rolling avg CLV = {avg_clv:+.4f} ({avg_clv*100:+.2f}%). "
                "Model may be consistently overpaying — review model calibration."
            )

    bankroll = tracker.get_current_bankroll()

    # Get upcoming matches
    upcoming = getattr(collection, "upcoming", None)
    if upcoming is None or (hasattr(upcoming, "empty") and upcoming.empty):
        logger.warning("No upcoming matches found")
        print("No upcoming matches to predict")
        return []

    # Generate predictions
    predictions = []
    odds_data = getattr(collection, "odds", []) or []

    for _, match in (upcoming.iterrows() if hasattr(upcoming, "iterrows") else []):
        match_dict = dict(match)
        match_id = match_dict.get("match_id", "")
        home = match_dict.get("home_team", "")
        away = match_dict.get("away_team", "")

        if not home or not away:
            continue

        try:
            features = feature_builder.build_match_features(
                match_dict,
                historical_matches=matches_df,
                xg_data=getattr(collection, "xg", None),
                injury_data=getattr(collection, "injuries", None),
                market_values=getattr(collection, "squad_values", None),
                team_schedule=getattr(collection, "schedules", None),
                news_sentiment=getattr(collection, "news", {}).get(f"{home}_{away}"),
            )

            sport = match_dict.get("sport", "soccer")
            surface = features.get("surface") or match_dict.get("surface")
            prediction = ensemble.predict(home, away, features, sport=sport, surface=surface)
            if prediction is None:
                continue

            # Check all markets
            markets = {
                "1x2_home": prediction.get("home_win_prob", 0),
                "1x2_draw": prediction.get("draw_prob", 0),
                "1x2_away": prediction.get("away_win_prob", 0),
                "over_2.5": prediction.get("over_25_prob", 0),
                "under_2.5": prediction.get("under_25_prob", 0),
                "btts": prediction.get("btts_prob", 0),
            }

            safety_margin = config.get("betting", {}).get("probability_safety_margin", 0.10)
            min_odds = config.get("betting", {}).get("min_odds", 1.50)

            primary_bookmaker = config.get("betting", {}).get("primary_bookmaker", "bet365")
            for market, our_prob in markets.items():
                # Only use odds from the configured primary platform.
                # Line-shopping across bookmakers looks better on paper but is
                # unreliable in practice: odds move, accounts get limited, and
                # mixing platforms makes CLV tracking meaningless.
                platform_odds, bookmaker = get_platform_odds(match_id, market, odds_data, primary_bookmaker)
                if platform_odds < min_odds:
                    continue

                # Apply safety margin: 10% haircut before EV + Kelly
                our_prob_adj = our_prob * (1.0 - safety_margin)
                if primary_bookmaker.lower() == "betfair":
                    ev = calculate_ev_betfair(our_prob_adj, platform_odds)
                else:
                    ev = calculate_ev(our_prob_adj, platform_odds)
                if ev <= 0:
                    continue

                stake = kelly_stake(our_prob_adj, platform_odds, bankroll, config)
                kelly_frac = (platform_odds - 1) * our_prob_adj - (1 - our_prob_adj)
                kelly_frac = kelly_frac / (platform_odds - 1) if platform_odds > 1 else 0

                bet_dict = {
                    "match_date": match_dict.get("date", str(date.today())),
                    "match_id": match_id,
                    "league": match_dict.get("league", ""),
                    "home_team": home,
                    "away_team": away,
                    "market": market,
                    "our_probability": our_prob,
                    "bookmaker_odds": platform_odds,
                    "bookmaker_name": bookmaker,
                    "ev_score": ev,
                    "confidence_score": prediction.get("confidence_score", 50),
                    "stake_recommended": stake,
                    "kelly_fraction": kelly_frac,
                    "kickoff_time": match_dict.get("kickoff_time"),
                }
                predictions.append(bet_dict)

        except Exception as e:
            logger.error(f"Prediction failed for {home} vs {away}: {e}")
            continue

    # Apply selection filter
    week_watchable = _get_week_watchable_count(db)
    selected = select_daily_bets(predictions, config, week_watchable)

    # Store in DB
    for bet in selected:
        bet_copy = {k: v for k, v in bet.items() if k not in ("kickoff_time",)}
        db.insert_bet(bet_copy)

    logger.info(f"Generated {len(selected)} bets from {len(predictions)} candidates")
    print(f"✅ {len(selected)} bets generated (from {len(predictions)} candidates with positive EV)")

    if len(selected) == 0:
        logger.warning("0 bets generated today — sending Telegram alert")
        try:
            from notifications.telegram_bot import TelegramBotHandler
            bot = TelegramBotHandler(config, db)
            bot.send_message_sync("⚠️ 0 Bets heute generiert — prüfe Daten & Modell")
        except Exception as _tg_err:
            logger.error(f"Failed to send zero-bets alert: {_tg_err}")

    return selected


def _get_week_watchable_count(db) -> int:
    """Count watchable bets placed this week."""
    try:
        with db._get_conn() as conn:
            count = conn.execute(
                """SELECT COUNT(*) FROM bets
                   WHERE is_watchable = 1
                   AND match_date >= date('now', 'weekday 0', '-7 days')"""
            ).fetchone()[0]
            return count or 0
    except Exception:
        return 0


def cmd_brief(config: dict):
    """Send morning briefing via Telegram."""
    from tracking.database import DatabaseHandler
    from tracking.performance import PerformanceTracker
    from notifications.morning_briefing import format_morning_briefing
    from notifications.telegram_bot import TelegramBotHandler

    db = DatabaseHandler()
    tracker = PerformanceTracker(db, config)
    bets = db.get_pending_bets(date.today())
    perf = tracker.get_full_summary()
    perf["starting_bankroll"] = config.get("betting", {}).get("bankroll_paper", 1000.0)

    message = format_morning_briefing(bets, perf)
    bot = TelegramBotHandler(config, db)
    success = bot.send_message_sync(message)

    if success:
        print("✅ Morning briefing sent")
    else:
        print("❌ Failed to send morning briefing")
    return success


def cmd_odds_snapshot(config: dict):
    """Save current odds snapshot to DB."""
    from tracking.database import DatabaseHandler
    from data.sources.odds_api import OddsAPISource

    db = DatabaseHandler()
    source = OddsAPISource(config)
    data = source.fetch_and_normalize()

    if data is None or (hasattr(data, "empty") and data.empty):
        print("⚠️  No odds data retrieved")
        return

    count = 0
    for _, row in (data.iterrows() if hasattr(data, "iterrows") else []):
        snapshot = {
            "match_id": row.get("match_id", ""),
            "bookmaker": row.get("bookmaker", ""),
            "market": row.get("market", ""),
            "odds_home": row.get("odds_home"),
            "odds_draw": row.get("odds_draw"),
            "odds_away": row.get("odds_away"),
            "odds_over": row.get("odds_over"),
            "odds_under": row.get("odds_under"),
        }
        if db.insert_odds_snapshot(snapshot):
            count += 1

    print(f"✅ Saved {count} odds snapshots")


def cmd_summarize(config: dict):
    """Send evening summary via Telegram."""
    from tracking.database import DatabaseHandler
    from tracking.performance import PerformanceTracker
    from notifications.evening_summary import format_evening_summary
    from notifications.telegram_bot import TelegramBotHandler

    db = DatabaseHandler()
    tracker = PerformanceTracker(db, config)
    bets = db.get_pending_bets(date.today())
    # Include placed bets too
    try:
        with db._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM bets WHERE match_date = ?", (str(date.today()),)
            ).fetchall()
            all_bets = [dict(r) for r in rows]
    except Exception:
        all_bets = bets

    db.update_daily_performance(date.today())
    perf_today = {"pnl_daily": 0.0}
    perf_total = tracker.get_full_summary()

    message = format_evening_summary(all_bets, perf_today, perf_total)
    bot = TelegramBotHandler(config, db)
    success = bot.send_message_sync(message)

    if success:
        print("✅ Evening summary sent")
    else:
        print("❌ Failed to send evening summary")


def cmd_retrain(config: dict):
    """Retrain all models."""
    from tracking.database import DatabaseHandler
    from models.poisson_model import PoissonModel
    from models.elo_model import EloModel
    from models.xgboost_model import XGBoostModel
    from models.ensemble import EnsembleModel
    from models.retrainer import ModelRetrainer
    from tracking.sheets import SheetsHandler

    db = DatabaseHandler()
    poisson = PoissonModel()
    elo = EloModel()
    xgboost = XGBoostModel()
    xgboost.load()
    ensemble = EnsembleModel(poisson, xgboost, elo, config)

    sheets = None
    try:
        sheets = SheetsHandler(config)
        sheets.connect()
    except Exception:
        pass

    retrainer = ModelRetrainer(db, ensemble, sheets, config)
    success = retrainer.retrain_weekly()
    print("✅ Retraining complete" if success else "⚠️  Retraining skipped (insufficient data)")


def cmd_weekly_report(config: dict):
    """Send weekly report via Telegram."""
    from tracking.database import DatabaseHandler
    from tracking.performance import PerformanceTracker
    from notifications.weekly_report import format_weekly_report
    from notifications.telegram_bot import TelegramBotHandler

    db = DatabaseHandler()
    tracker = PerformanceTracker(db, config)
    perf = tracker.get_full_summary()
    message = format_weekly_report(perf)
    bot = TelegramBotHandler(config, db)
    success = bot.send_message_sync(message)
    print("✅ Weekly report sent" if success else "❌ Failed to send weekly report")


def cmd_results(config: dict):
    """Fetch actual match outcomes and settle all pending/placed bets."""
    from tracking.database import DatabaseHandler
    from data.sources.results_source import ResultsSource

    db = DatabaseHandler()
    source = ResultsSource()

    # Settle bets with match_date up to and including yesterday.
    # Today's matches may still be in progress.
    yesterday = date.today() - timedelta(days=1)

    logger.info(f"Settling bets with match_date <= {yesterday}")
    summary = source.fetch_results(db, match_date=yesterday)

    settled = summary["settled"]
    voided = summary["voided"]
    skipped = summary["skipped"]

    print(
        f"Results settled: {settled} bet(s) settled, "
        f"{voided} voided, {skipped} still pending."
    )

    # Refresh daily performance snapshot for yesterday
    if settled + voided > 0:
        try:
            db.update_daily_performance(yesterday)
        except Exception as exc:
            logger.warning(f"Could not update daily performance: {exc}")

    return summary


def cmd_healthcheck(config: dict) -> dict:
    """
    Health check: verifies DB, data freshness, predictions, API keys.
    Sends a Telegram alert if no bets were generated today on a potential game day.
    Returns a health report dict and prints a summary.
    """
    import time

    report = {
        "db_ok": False,
        "db_tables_ok": False,
        "collect_within_24h": False,
        "predict_ran_today": False,
        "bets_today": 0,
        "api_keys_ok": False,
        "missing_api_keys": [],
        "alerts": [],
    }

    db = None

    # 1. DB accessible and has tables
    try:
        from tracking.database import DatabaseHandler
        db = DatabaseHandler()
        with db._get_conn() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        report["db_ok"] = True
        expected_tables = {"bets", "results", "performance"}
        report["db_tables_ok"] = expected_tables.issubset(set(tables))
        if not report["db_tables_ok"]:
            missing = expected_tables - set(tables)
            report["alerts"].append(f"Missing DB tables: {', '.join(sorted(missing))}")
        logger.info(f"[healthcheck] DB OK, tables: {tables}")
    except Exception as e:
        report["alerts"].append(f"DB error: {e}")
        logger.error(f"[healthcheck] DB error: {e}")

    # 2. Last --collect within 24h (check betting.log mtime)
    log_path = ROOT / "data" / "betting.log"
    try:
        if log_path.exists():
            age_seconds = time.time() - log_path.stat().st_mtime
            report["collect_within_24h"] = age_seconds < 86400
            if not report["collect_within_24h"]:
                hours_ago = age_seconds / 3600
                report["alerts"].append(f"Last --collect was {hours_ago:.1f}h ago (>24h)")
        else:
            report["alerts"].append("betting.log not found — --collect may never have run")
    except Exception as e:
        report["alerts"].append(f"Log mtime check failed: {e}")

    # 3. Last --predict ran today (check bets table for today's date)
    today_str = str(date.today())
    try:
        if report["db_ok"]:
            with db._get_conn() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM bets WHERE match_date = ?", (today_str,)
                ).fetchone()[0]
            report["bets_today"] = count or 0
            report["predict_ran_today"] = count > 0
            if not report["predict_ran_today"]:
                report["alerts"].append("No bets generated today — --predict may not have run")
    except Exception as e:
        report["alerts"].append(f"Bets-today check failed: {e}")

    # 4. API keys set in environment
    required_keys = [
        "FOOTBALL_API_KEY",
        "ODDS_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ]
    missing_keys = [
        k for k in required_keys
        if not os.environ.get(k) or os.environ.get(k, "").startswith("your_")
    ]
    report["missing_api_keys"] = missing_keys
    report["api_keys_ok"] = len(missing_keys) == 0
    if missing_keys:
        report["alerts"].append(f"Missing/unconfigured API keys: {', '.join(missing_keys)}")

    # 5. If 0 bets today and predict didn't run, send Telegram alert
    if report["bets_today"] == 0 and not report["predict_ran_today"]:
        alert_msg = (
            f"[HEALTHCHECK] No bets generated today ({today_str}). "
            "Run --predict or check data pipeline."
        )
        try:
            from notifications.telegram_bot import TelegramBotHandler
            bot = TelegramBotHandler(config, db)
            bot.send_message_sync(alert_msg)
            logger.warning(f"[healthcheck] Telegram alert sent: {alert_msg}")
        except Exception as e:
            logger.error(f"[healthcheck] Telegram alert failed: {e}")

    # Print report
    status = "OK" if not report["alerts"] else "DEGRADED"
    print(f"\n{'='*44}")
    print(f"  HEALTH CHECK -- {date.today()}  [{status}]")
    print(f"{'='*44}")
    print(f"  DB accessible:       {'YES' if report['db_ok'] else 'NO'}")
    print(f"  DB tables present:   {'YES' if report['db_tables_ok'] else 'NO'}")
    print(f"  Collect within 24h:  {'YES' if report['collect_within_24h'] else 'NO'}")
    print(f"  Predict ran today:   {'YES' if report['predict_ran_today'] else 'NO'}")
    print(f"  Bets today:          {report['bets_today']}")
    print(f"  API keys OK:         {'YES' if report['api_keys_ok'] else 'NO'}")
    if report["missing_api_keys"]:
        print(f"  Missing keys:        {', '.join(report['missing_api_keys'])}")
    if report["alerts"]:
        print(f"\n  Alerts:")
        for alert in report["alerts"]:
            print(f"    - {alert}")
    print(f"{'='*44}\n")

    logger.info(f"[healthcheck] status={status}, alerts={report['alerts']}")
    return report


def cmd_goalie_check(config: dict):
    """Check starting goalie status 90 min before hockey games.

    Fetches today's placed hockey bets and compares the current confirmed
    starting goalie against the one recorded at bet time.  Sends a Telegram
    alert if a goalie change is detected so the user can decide to void.
    """
    from tracking.database import DatabaseHandler
    from notifications.telegram_bot import TelegramBotHandler

    db = DatabaseHandler()

    # Fetch today's hockey bets that are still placed/pending
    hockey_bets = []
    try:
        with db._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM bets
                   WHERE match_date = date('now')
                   AND status IN ('placed', 'pending')
                   AND (LOWER(league) LIKE '%hockey%'
                        OR LOWER(league) LIKE '%ahl%'
                        OR LOWER(league) LIKE '%echl%')""",
            ).fetchall()
            hockey_bets = [dict(r) for r in rows]
    except Exception as exc:
        logger.error(f"goalie_check: DB query failed: {exc}")
        return

    if not hockey_bets:
        logger.info("goalie_check: no hockey bets today")
        print("No hockey bets today — goalie check skipped")
        return

    from data.sources.hockey_source import HockeySource
    source = HockeySource(config)
    alerts = []

    for bet in hockey_bets:
        match_id = bet.get("match_id", "")
        if not match_id:
            continue
        try:
            status = source.get_goalie_status(match_id)
            if not status:
                logger.debug(f"goalie_check: no status for match {match_id}")
                continue
            current = status.get("home_goalie") or status.get("starting_goalie", "")
            recorded = bet.get("notes", "")  # goalie stored in notes if available
            if current and recorded and current.lower() != recorded.lower():
                msg = (
                    f"🏒 Goalie-Wechsel: {bet['home_team']} vs {bet['away_team']}\n"
                    f"Ursprünglich: {recorded}\nJetzt: {current}\nPrüfe Bet {bet['id']}!"
                )
                alerts.append(msg)
                logger.warning(f"Goalie change detected: {msg}")
        except Exception as exc:
            logger.warning(f"goalie_check: could not fetch status for {match_id}: {exc}")

    if alerts:
        try:
            bot = TelegramBotHandler(config, db)
            for msg in alerts:
                bot.send_message_sync(msg)
        except Exception as exc:
            logger.error(f"goalie_check: Telegram send failed: {exc}")
        print(f"⚠️  {len(alerts)} goalie change alert(s) sent")
    else:
        print(f"✅ Goalie check OK — {len(hockey_bets)} hockey bet(s) checked, no changes")


def cmd_test():
    """Run all tests."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-v", "--tb=short"],
        cwd=str(ROOT),
    )
    return result.returncode == 0


def cmd_paper_status(config: dict):
    """Print current paper mode statistics."""
    from tracking.database import DatabaseHandler
    from tracking.performance import PerformanceTracker

    db = DatabaseHandler()
    tracker = PerformanceTracker(db, config)
    summary = tracker.get_full_summary()

    roi_emoji = "🟢" if summary["roi"] > 3 else ("🟡" if summary["roi"] >= 0 else "🔴")
    print(f"\n{'='*40}")
    print(f"  PAPER MODE STATUS")
    print(f"{'='*40}")
    print(f"  Bankroll:    €{summary['bankroll']:.2f}")
    print(f"  ROI:         {summary['roi']:+.2f}% {roi_emoji}")
    print(f"  Total P&L:   €{summary['total_pnl']:+.2f}")
    print(f"  Settled:     {summary['settled_bets']}/200 bets")
    print(f"  Win Rate:    {summary['win_rate']:.1f}%")
    print(f"  Max DD:      €{summary['max_drawdown']:.2f}")
    print(f"  Ready Live:  {'YES ✅' if summary['ready_for_live'] else 'NO ❌'}")
    print(f"{'='*40}\n")
    return summary


def main():
    # Ensure log dir exists before logging starts
    log_dir = ROOT / "data"
    log_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Sports Betting Bot")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--odds_snapshot", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--weekly_report", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--goalie_check", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--paper_status", action="store_true")
    args = parser.parse_args()

    if args.test:
        success = cmd_test()
        sys.exit(0 if success else 1)

    config = load_config()

    if args.setup:
        cmd_setup(config)
    elif args.collect:
        cmd_collect(config)
    elif args.predict:
        cmd_predict(config)
    elif args.brief:
        cmd_brief(config)
    elif args.odds_snapshot:
        cmd_odds_snapshot(config)
    elif args.summarize:
        cmd_summarize(config)
    elif args.retrain:
        cmd_retrain(config)
    elif args.weekly_report:
        cmd_weekly_report(config)
    elif args.results:
        cmd_results(config)
    elif args.healthcheck:
        cmd_healthcheck(config)
    elif args.goalie_check:
        cmd_goalie_check(config)
    elif args.paper_status:
        cmd_paper_status(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
