import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

STATUS_EMOJI = {
    "won": "🟢",
    "lost": "🔴",
    "pending": "⏳",
    "skipped": "⏭️",
    "void": "⬜",
    "placed": "⏳",
}


class SheetsHandler:
    def __init__(self, config: dict):
        self.config = config
        self.sheets_cfg = config.get("google_sheets", {})
        self.spreadsheet_id = self.sheets_cfg.get("spreadsheet_id", "")
        self.tabs = self.sheets_cfg.get("tabs", {
            "bets": "Bets",
            "analytics": "Analytics",
            "model_health": "Model Health",
        })
        self.client = None
        self.spreadsheet = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            creds_file = Path(self.sheets_cfg.get("credentials_file", "google_credentials.json"))
            if not creds_file.exists():
                logger.error(f"Google credentials file not found: {creds_file}")
                return False

            scopes = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_file(str(creds_file), scopes=scopes)
            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            self._connected = True
            self._ensure_tabs()
            logger.info("Connected to Google Sheets")
            return True
        except Exception as e:
            logger.error(f"Google Sheets connection failed: {e}")
            return False

    def _ensure_tabs(self):
        if not self._connected:
            return
        existing = [ws.title for ws in self.spreadsheet.worksheets()]
        for tab_name in self.tabs.values():
            if tab_name not in existing:
                self.spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=20)
                logger.info(f"Created tab: {tab_name}")

        self._write_bets_header()
        self._write_analytics_header()
        self._write_model_health_header()

    def _write_bets_header(self):
        try:
            ws = self.spreadsheet.worksheet(self.tabs["bets"])
            if ws.row_count == 0 or ws.cell(1, 1).value != "Date":
                headers = [
                    "Date", "Match", "League", "Market", "Our P%",
                    "Bookie Odds", "Bookmaker", "EV%", "Confidence",
                    "Stake €", "Placed?", "Result", "P&L", "Cum P&L",
                    "CLV", "Status"
                ]
                ws.insert_row(headers, 1)
        except Exception as e:
            logger.error(f"Failed to write bets header: {e}")

    def _write_analytics_header(self):
        try:
            ws = self.spreadsheet.worksheet(self.tabs["analytics"])
            if not ws.cell(1, 1).value:
                ws.update("A1", [["Analytics — auto-refreshed weekly"]])
        except Exception as e:
            logger.error(f"Failed to write analytics header: {e}")

    def _write_model_health_header(self):
        try:
            ws = self.spreadsheet.worksheet(self.tabs["model_health"])
            if not ws.cell(1, 1).value:
                ws.update("A1", [["Model Health Dashboard"]])
        except Exception as e:
            logger.error(f"Failed to write model health header: {e}")

    def append_bet(self, bet: dict, cum_pnl: float = 0.0) -> bool:
        if not self._connected:
            logger.warning("Sheets not connected — skipping bet append")
            return False
        try:
            ws = self.spreadsheet.worksheet(self.tabs["bets"])
            status_emoji = STATUS_EMOJI.get(bet.get("status", "pending"), "⏳")
            match_str = f"{bet.get('home_team')} vs {bet.get('away_team')}"
            row = [
                str(bet.get("match_date", "")),
                match_str,
                bet.get("league", ""),
                bet.get("market", ""),
                f"{bet.get('our_probability', 0) * 100:.1f}%",
                bet.get("bookmaker_odds", ""),
                bet.get("bookmaker_name", ""),
                f"+{bet.get('ev_score', 0) * 100:.2f}%",
                bet.get("confidence_score", ""),
                bet.get("stake_recommended", ""),
                "No",
                "",
                "",
                f"{cum_pnl:.2f}",
                "",
                status_emoji,
            ]
            ws.append_row(row)
            return True
        except Exception as e:
            logger.error(f"Sheets append_bet failed: {e}")
            return False

    def update_bet_result(self, bet_id: int, result: dict, cum_pnl: float) -> bool:
        if not self._connected:
            return False
        try:
            ws = self.spreadsheet.worksheet(self.tabs["bets"])
            all_rows = ws.get_all_records()
            for i, row in enumerate(all_rows, start=2):
                if str(row.get("bet_id", "")) == str(bet_id):
                    won = result.get("won", False)
                    pnl = result.get("pnl_simulated", 0)
                    clv = result.get("clv_score", "")
                    status_emoji = STATUS_EMOJI.get("won" if won else "lost", "⏳")
                    ws.update(f"K{i}", [["Yes"]])
                    ws.update(f"L{i}", [[result.get("actual_result", "")]])
                    ws.update(f"M{i}", [[f"{pnl:+.2f}"]])
                    ws.update(f"N{i}", [[f"{cum_pnl:.2f}"]])
                    ws.update(f"O{i}", [[f"{clv:+.4f}" if clv else ""]])
                    ws.update(f"P{i}", [[status_emoji]])
                    return True
            return False
        except Exception as e:
            logger.error(f"Sheets update_bet_result failed: {e}")
            return False

    def refresh_analytics(self, performance_data: dict) -> bool:
        if not self._connected:
            return False
        try:
            ws = self.spreadsheet.worksheet(self.tabs["analytics"])
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            data = [
                ["Last Updated", now],
                [],
                ["Overall Performance"],
                ["ROI Overall", f"{performance_data.get('roi', 0):+.2f}%"],
                ["Total Bets", performance_data.get("total_bets", 0)],
                ["Won", performance_data.get("won", 0)],
                ["Lost", performance_data.get("lost", 0)],
                ["Win Rate", f"{performance_data.get('win_rate', 0):.1f}%"],
                ["Total P&L", f"{performance_data.get('total_pnl', 0):+.2f}€"],
                ["Avg EV", f"{performance_data.get('avg_ev', 0) * 100:+.2f}%"],
                ["Avg Confidence", f"{performance_data.get('avg_confidence', 0):.0f}/100"],
                ["CLV Average", f"{performance_data.get('avg_clv', 0):+.4f}"],
            ]
            ws.update("A1", data)
            return True
        except Exception as e:
            logger.error(f"Sheets refresh_analytics failed: {e}")
            return False

    def update_model_health(self, health_data: dict) -> bool:
        if not self._connected:
            return False
        try:
            ws = self.spreadsheet.worksheet(self.tabs["model_health"])
            roi = health_data.get("roi", 0)
            if roi > 3:
                status = "🟢 ROI > 3%"
            elif roi >= 0:
                status = "🟡 ROI 0-3%"
            else:
                status = "🔴 ROI < 0%"

            settled = health_data.get("settled_bets", 0)
            ready = "YES ✅" if settled >= 200 and roi > 3 else f"NO ({settled}/200 bets)"

            data = [
                ["Model Health Dashboard"],
                [],
                ["System Status", status],
                ["Last Retrain", health_data.get("last_retrain", "Never")],
                ["Next Retrain", "Sunday 03:00"],
                ["Paper Mode Progress", f"{settled}/200 bets"],
                ["Ready for Live", ready],
                ["Total Simulated P&L", f"{health_data.get('total_pnl', 0):+.2f}€"],
                ["Days Running", health_data.get("days_running", 0)],
                [],
                ["Retraining Results"],
                ["XGBoost", str(health_data.get("results", {}).get("xgboost", {}).get("success", "N/A"))],
                ["Poisson", str(health_data.get("results", {}).get("poisson", {}).get("success", "N/A"))],
            ]
            ws.update("A1", data)
            return True
        except Exception as e:
            logger.error(f"Sheets update_model_health failed: {e}")
            return False
