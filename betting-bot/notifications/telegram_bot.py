import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramBotHandler:
    def __init__(self, config: dict, db, sheets_handler=None):
        self.config = config
        self.db = db
        self.sheets = sheets_handler
        tg_cfg = config.get("telegram", {})
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", tg_cfg.get("bot_token", ""))
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", tg_cfg.get("chat_id", ""))
        self.application = None

    def start(self):
        """Start the bot — call from scheduler."""
        try:
            from telegram.ext import Application, CommandHandler
            self.application = Application.builder().token(self.bot_token).build()
            self.application.add_handler(CommandHandler("placed", self._cmd_placed))
            self.application.add_handler(CommandHandler("skip", self._cmd_skip))
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            self.application.add_handler(CommandHandler("bets", self._cmd_bets))
            self.application.add_handler(CommandHandler("week", self._cmd_week))
            self.application.add_handler(CommandHandler("help", self._cmd_help))
            logger.info("Telegram bot started")
        except Exception as e:
            logger.error(f"Telegram bot start failed: {e}")

    async def send_message(self, text: str) -> bool:
        try:
            from telegram import Bot
            bot = Bot(token=self.bot_token)
            await bot.send_message(chat_id=self.chat_id, text=text, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error(f"send_message failed: {e}")
            return False

    def send_message_sync(self, text: str) -> bool:
        """Synchronous wrapper for send_message."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.send_message(text))
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.send_message(text))
        except Exception as e:
            logger.error(f"send_message_sync failed: {e}")
            return False

    async def _cmd_placed(self, update, context):
        """Mark bet N as placed in paper mode."""
        try:
            args = context.args
            if not args:
                await update.message.reply_text("Usage: /placed N (Bet-Nummer)")
                return
            bet_num = int(args[0])
            today_bets = self.db.get_pending_bets(date.today())
            if bet_num < 1 or bet_num > len(today_bets):
                await update.message.reply_text(f"❌ Bet {bet_num} nicht gefunden. Heute: {len(today_bets)} Bets.")
                return
            bet = today_bets[bet_num - 1]
            self.db.update_bet_status(bet["id"], "placed")
            with self.db._get_conn() as conn:
                conn.execute(
                    "INSERT INTO user_interactions (bet_id, action) VALUES (?, 'placed')",
                    (bet["id"],),
                )
                conn.commit()
            if self.sheets:
                self.sheets.append_bet(bet)
            await update.message.reply_text(
                f"✅ Bet {bet_num} als 'placed' markiert:\n"
                f"{bet['home_team']} vs {bet['away_team']} | {bet['market']} @ {bet['bookmaker_odds']}"
            )
        except ValueError:
            await update.message.reply_text("❌ Ungültige Bet-Nummer. Beispiel: /placed 3")
        except Exception as e:
            logger.error(f"_cmd_placed failed: {e}")
            await update.message.reply_text(f"❌ Fehler: {e}")

    async def _cmd_skip(self, update, context):
        """Mark bet N as skipped."""
        try:
            args = context.args
            if not args:
                await update.message.reply_text("Usage: /skip N (Bet-Nummer)")
                return
            bet_num = int(args[0])
            today_bets = self.db.get_pending_bets(date.today())
            if bet_num < 1 or bet_num > len(today_bets):
                await update.message.reply_text(f"❌ Bet {bet_num} nicht gefunden.")
                return
            bet = today_bets[bet_num - 1]
            self.db.update_bet_status(bet["id"], "skipped")
            with self.db._get_conn() as conn:
                conn.execute(
                    "INSERT INTO user_interactions (bet_id, action) VALUES (?, 'skipped')",
                    (bet["id"],),
                )
                conn.commit()
            await update.message.reply_text(f"⏭️ Bet {bet_num} übersprungen.")
        except ValueError:
            await update.message.reply_text("❌ Ungültige Bet-Nummer.")
        except Exception as e:
            logger.error(f"_cmd_skip failed: {e}")
            await update.message.reply_text(f"❌ Fehler: {e}")

    async def _cmd_status(self, update, context):
        try:
            from tracking.performance import PerformanceTracker
            tracker = PerformanceTracker(self.db, self.config)
            summary = tracker.get_full_summary()
            text = (
                f"📊 <b>Status</b>\n"
                f"💰 Bankroll: €{summary['bankroll']:.2f}\n"
                f"📈 ROI: {summary['roi']:+.1f}%\n"
                f"📋 Settled Bets: {summary['settled_bets']}/200\n"
                f"✅ Won: {summary['won']} | ❌ Lost: {summary['lost']}\n"
                f"🎯 Ready for Live: {'YES ✅' if summary['ready_for_live'] else 'NO'}"
            )
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            logger.error(f"_cmd_status failed: {e}")
            await update.message.reply_text(f"❌ Fehler: {e}")

    async def _cmd_bets(self, update, context):
        try:
            from notifications.morning_briefing import format_morning_briefing
            from tracking.performance import PerformanceTracker
            tracker = PerformanceTracker(self.db, self.config)
            bets = self.db.get_pending_bets(date.today())
            perf = tracker.get_full_summary()
            msg = format_morning_briefing(bets, perf)
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"_cmd_bets failed: {e}")
            await update.message.reply_text(f"❌ Fehler: {e}")

    async def _cmd_week(self, update, context):
        try:
            from notifications.weekly_report import format_weekly_report
            from tracking.performance import PerformanceTracker
            tracker = PerformanceTracker(self.db, self.config)
            perf = tracker.get_full_summary()
            msg = format_weekly_report(perf)
            await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"_cmd_week failed: {e}")
            await update.message.reply_text(f"❌ Fehler: {e}")

    async def _cmd_help(self, update, context):
        help_text = (
            "🤖 <b>Betting Bot Commands</b>\n\n"
            "/placed N — Bet N als gesetzt markieren\n"
            "/skip N — Bet N überspringen\n"
            "/status — Bankroll, ROI, Paper Mode Status\n"
            "/bets — Heutige Bets anzeigen\n"
            "/week — Wochenperformance\n"
            "/help — Diese Hilfe"
        )
        await update.message.reply_text(help_text, parse_mode="HTML")
