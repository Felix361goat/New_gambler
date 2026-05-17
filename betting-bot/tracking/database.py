from pathlib import Path
import sqlite3
import logging
from datetime import date, datetime
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "betting.db"

CREATE_BETS = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    match_date DATE NOT NULL,
    match_id TEXT,
    league TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market TEXT NOT NULL,
    our_probability REAL NOT NULL,
    bookmaker_odds REAL NOT NULL,
    bookmaker_name TEXT NOT NULL,
    ev_score REAL NOT NULL,
    confidence_score INTEGER,
    stake_recommended REAL,
    kelly_fraction REAL,
    is_watchable BOOLEAN DEFAULT 0,
    watchable_reason TEXT,
    is_favorite_club BOOLEAN DEFAULT 0,
    status TEXT DEFAULT 'pending',
    notes TEXT,
    sport TEXT DEFAULT 'soccer',
    surface TEXT,
    confidence_tier TEXT DEFAULT 'Low',
    data_completeness REAL DEFAULT 0.0,
    matchfixing_warning TEXT,
    bookie_implied_prob REAL,
    our_edge REAL
)
"""

CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id INTEGER REFERENCES bets(id),
    settled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_result TEXT,
    won BOOLEAN,
    pnl_simulated REAL,
    closing_line_odds REAL,
    clv_score REAL
)
"""

CREATE_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE,
    total_bets INTEGER DEFAULT 0,
    won INTEGER DEFAULT 0,
    lost INTEGER DEFAULT 0,
    void INTEGER DEFAULT 0,
    roi_daily REAL,
    roi_cumulative REAL,
    pnl_daily REAL,
    pnl_cumulative REAL,
    max_drawdown REAL,
    avg_confidence REAL,
    avg_ev REAL,
    bankroll_end REAL
)
"""

CREATE_REFEREE_STATS = """
CREATE TABLE IF NOT EXISTS referee_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    referee_name TEXT NOT NULL,
    league TEXT NOT NULL,
    avg_goals_per_game REAL,
    avg_cards_per_game REAL,
    avg_penalties_per_game REAL,
    home_bias_score REAL,
    sample_size INTEGER DEFAULT 0,
    last_updated DATE,
    UNIQUE(referee_name, league)
)
"""

CREATE_ODDS_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    match_id TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,
    odds_home REAL,
    odds_draw REAL,
    odds_away REAL,
    odds_over REAL,
    odds_under REAL
)
"""

CREATE_USER_INTERACTIONS = """
CREATE TABLE IF NOT EXISTS user_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id INTEGER REFERENCES bets(id),
    action TEXT NOT NULL,
    actioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT
)
"""


class DatabaseHandler:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute(CREATE_BETS)
                conn.execute(CREATE_RESULTS)
                conn.execute(CREATE_PERFORMANCE)
                conn.execute(CREATE_REFEREE_STATS)
                conn.execute(CREATE_ODDS_SNAPSHOTS)
                conn.execute(CREATE_USER_INTERACTIONS)
                conn.commit()
        except Exception as e:
            logger.error(f"DB init failed: {e}")
            raise
        self.migrate_schema()

    def migrate_schema(self):
        """Add new columns if they don't exist (safe to run multiple times)."""
        new_columns = [
            ("sport", "TEXT DEFAULT 'soccer'"),
            ("surface", "TEXT"),
            ("confidence_tier", "TEXT DEFAULT 'Low'"),
            ("data_completeness", "REAL DEFAULT 0.0"),
            ("matchfixing_warning", "TEXT"),
            ("bookie_implied_prob", "REAL"),
            ("our_edge", "REAL"),
        ]
        try:
            with self._get_conn() as conn:
                existing = {row[1] for row in conn.execute("PRAGMA table_info(bets)").fetchall()}
                for col_name, col_def in new_columns:
                    if col_name not in existing:
                        conn.execute(f"ALTER TABLE bets ADD COLUMN {col_name} {col_def}")
                        logger.info(f"Added column: bets.{col_name}")
                conn.commit()
        except Exception as e:
            logger.error(f"migrate_schema failed: {e}")

    def insert_bet(self, bet_dict: dict) -> Optional[int]:
        cols = ", ".join(bet_dict.keys())
        placeholders = ", ".join(["?"] * len(bet_dict))
        sql = f"INSERT INTO bets ({cols}) VALUES ({placeholders})"
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(sql, list(bet_dict.values()))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"insert_bet failed: {e}")
            return None

    def update_bet_status(self, bet_id: int, status: str) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE bets SET status = ? WHERE id = ?", (status, bet_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"update_bet_status failed: {e}")
            return False

    def insert_result(self, result_dict: dict) -> Optional[int]:
        cols = ", ".join(result_dict.keys())
        placeholders = ", ".join(["?"] * len(result_dict))
        sql = f"INSERT INTO results ({cols}) VALUES ({placeholders})"
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(sql, list(result_dict.values()))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"insert_result failed: {e}")
            return None

    def insert_odds_snapshot(self, snapshot_dict: dict) -> Optional[int]:
        cols = ", ".join(snapshot_dict.keys())
        placeholders = ", ".join(["?"] * len(snapshot_dict))
        sql = f"INSERT INTO odds_snapshots ({cols}) VALUES ({placeholders})"
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(sql, list(snapshot_dict.values()))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"insert_odds_snapshot failed: {e}")
            return None

    def get_pending_bets(self, target_date: date) -> list:
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM bets WHERE match_date = ? AND status = 'pending'",
                    (str(target_date),),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"get_pending_bets failed: {e}")
            return []

    def get_performance_summary(self, days: int = 30) -> dict:
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_bets,
                        SUM(CASE WHEN r.won = 1 THEN 1 ELSE 0 END) as won,
                        SUM(CASE WHEN r.won = 0 THEN 1 ELSE 0 END) as lost,
                        SUM(r.pnl_simulated) as total_pnl,
                        AVG(b.ev_score) as avg_ev,
                        AVG(b.confidence_score) as avg_confidence
                    FROM bets b
                    LEFT JOIN results r ON r.bet_id = b.id
                    WHERE b.match_date >= date('now', ?)
                    """,
                    (f"-{days} days",),
                ).fetchone()
                if row:
                    d = dict(row)
                    total_staked = conn.execute(
                        "SELECT SUM(stake_recommended) FROM bets WHERE match_date >= date('now', ?)",
                        (f"-{days} days",),
                    ).fetchone()[0] or 0
                    d["roi"] = (
                        (d["total_pnl"] / total_staked * 100)
                        if total_staked > 0
                        else 0.0
                    )
                    return d
                return {}
        except Exception as e:
            logger.error(f"get_performance_summary failed: {e}")
            return {}

    def get_bets_for_retraining(self) -> pd.DataFrame:
        try:
            with self._get_conn() as conn:
                df = pd.read_sql_query(
                    """
                    SELECT b.*, r.won, r.pnl_simulated, r.clv_score
                    FROM bets b
                    JOIN results r ON r.bet_id = b.id
                    ORDER BY b.match_date
                    """,
                    conn,
                )
                return df
        except Exception as e:
            logger.error(f"get_bets_for_retraining failed: {e}")
            return pd.DataFrame()

    def update_daily_performance(self, target_date: date) -> bool:
        try:
            with self._get_conn() as conn:
                stats = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_bets,
                        SUM(CASE WHEN r.won = 1 THEN 1 ELSE 0 END) as won,
                        SUM(CASE WHEN r.won = 0 THEN 1 ELSE 0 END) as lost,
                        SUM(CASE WHEN b.status = 'void' THEN 1 ELSE 0 END) as void,
                        SUM(r.pnl_simulated) as pnl_daily,
                        AVG(b.confidence_score) as avg_confidence,
                        AVG(b.ev_score) as avg_ev,
                        SUM(b.stake_recommended) as total_staked
                    FROM bets b
                    LEFT JOIN results r ON r.bet_id = b.id
                    WHERE b.match_date = ?
                    """,
                    (str(target_date),),
                ).fetchone()

                pnl_daily = stats["pnl_daily"] or 0.0
                total_staked = stats["total_staked"] or 0.0
                roi_daily = (pnl_daily / total_staked * 100) if total_staked > 0 else 0.0

                cum = conn.execute(
                    "SELECT SUM(pnl_daily) FROM performance WHERE date < ?",
                    (str(target_date),),
                ).fetchone()[0] or 0.0
                pnl_cumulative = cum + pnl_daily

                # Cumulative staked: sum every bet ever placed up to and including today
                cum_staked_prev = conn.execute(
                    "SELECT COALESCE(SUM(stake_recommended), 0) FROM bets WHERE match_date < ?",
                    (str(target_date),),
                ).fetchone()[0] or 0.0
                total_staked_cum = cum_staked_prev + total_staked
                roi_cumulative = (pnl_cumulative / total_staked_cum * 100) if total_staked_cum > 0 else 0.0

                conn.execute(
                    """
                    INSERT INTO performance
                        (date, total_bets, won, lost, void, roi_daily,
                         roi_cumulative, pnl_daily, pnl_cumulative, avg_confidence, avg_ev)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        total_bets=excluded.total_bets, won=excluded.won,
                        lost=excluded.lost, void=excluded.void,
                        roi_daily=excluded.roi_daily, roi_cumulative=excluded.roi_cumulative,
                        pnl_daily=excluded.pnl_daily, pnl_cumulative=excluded.pnl_cumulative,
                        avg_confidence=excluded.avg_confidence, avg_ev=excluded.avg_ev
                    """,
                    (
                        str(target_date),
                        stats["total_bets"],
                        stats["won"] or 0,
                        stats["lost"] or 0,
                        stats["void"] or 0,
                        roi_daily,
                        roi_cumulative,
                        pnl_daily,
                        pnl_cumulative,
                        stats["avg_confidence"],
                        stats["avg_ev"],
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"update_daily_performance failed: {e}")
            return False
