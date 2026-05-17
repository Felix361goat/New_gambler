"""
steering/ampel.py — Lenksystem Ampel-Controller

Berechnet den aktuellen Ampelstatus (GRUEN / GELB / ROT) und liefert
die daraus abgeleiteten dynamischen System-Parameter.  Alle Schwellwerte
orientieren sich an den bestehenden config.yaml-Werten; dieses Modul
überschreibt sie situationsbedingt nach unten, NIE nach oben.

Ampel-Logik (unveränderlich):
  GRUEN : CLV positiv  UND  Drawdown < 10 %  UND  ROI letzte 20 > -2 %
  GELB  : Drawdown 10–20 %  ODER  CLV negativ über ≥ 30 Wetten
  ROT   : Drawdown > 20 %   ODER  CLV negativ über ≥ 50 Wetten

Priorität: ROT schlägt GELB schlägt GRUEN.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typen
# ---------------------------------------------------------------------------

class Ampelstatus(str, Enum):
    GRUEN = "GRUEN"
    GELB  = "GELB"
    ROT   = "ROT"


@dataclass
class AmpelParameter:
    """Parameter-Satz der vom Lenksystem an den Rest des Systems übergeben wird."""
    status: Ampelstatus

    # Kelly / Stake
    kelly_fraction: float         # Quarter-Kelly Basis = 0.25
    max_stake_percent: float      # Max % der Bankroll pro Bet

    # EV-Schwellen
    min_ev_threshold: float       # Globale Mindestschwelle
    min_edge_over_implied: float  # Edge über Bookie-Implied

    # Bet-Volumen
    max_daily_bets: int           # Tageslimit

    # Bet365-spezifisch
    bet365_max_stake_eur: float   # Hartes Bet365-Limit in €
    bet365_account_limited: bool  # True = Account limitiert erkannt

    # Meta
    reason: str = ""              # Menschliche Begründung für Status-Wechsel
    auto_pause: bool = False      # True = System pausiert automatisch


@dataclass
class AmpelMetrics:
    """Eingabe-Metriken für die Ampel-Berechnung."""
    clv_avg_last_30: Optional[float]   # None = noch keine 30 Bets
    clv_avg_last_50: Optional[float]   # None = noch keine 50 Bets
    drawdown_pct: float                 # 0–100 (Prozent der Bankroll)
    roi_last_20: Optional[float]        # None = noch keine 20 Bets
    consecutive_neg_clv_bets: int       # Aufeinanderfolgende Bets mit CLV < 0
    bet365_limited: bool = False        # True wenn Limit-Erkennung ausgelöst hat


# ---------------------------------------------------------------------------
# Kernlogik
# ---------------------------------------------------------------------------

def _base_params_from_config(config: dict) -> dict:
    """Liest Basis-Parameter aus config.yaml."""
    b = config.get("betting", {})
    return {
        "kelly_fraction":        b.get("kelly_fraction", 0.25),
        "max_stake_percent":     b.get("max_stake_percent", 0.05),
        "min_ev_threshold":      b.get("min_ev_threshold", 0.05),
        "min_edge_over_implied": b.get("min_edge_over_implied", 0.05),
        "max_daily_bets":        b.get("max_daily_bets", 10),
    }


def calculate_ampel(metrics: AmpelMetrics, config: dict) -> AmpelParameter:
    """
    Hauptfunktion: Berechnet Ampelstatus und gibt angepasste Parameter zurück.

    Aufrufstelle: selection/filter.py — vor select_daily_bets().
    """
    base = _base_params_from_config(config)

    # ---- Status ermitteln (ROT > GELB > GRUEN) ---------------------------

    is_rot = (
        metrics.drawdown_pct > 20.0
        or metrics.consecutive_neg_clv_bets >= 50
    )
    is_gelb = not is_rot and (
        metrics.drawdown_pct >= 10.0
        or metrics.consecutive_neg_clv_bets >= 30
        or (metrics.clv_avg_last_30 is not None and metrics.clv_avg_last_30 < 0)
    )

    clv_positive = (
        metrics.clv_avg_last_30 is None   # noch kein Urteil möglich
        or metrics.clv_avg_last_30 >= 0
    )
    roi_ok = (
        metrics.roi_last_20 is None        # noch kein Urteil möglich
        or metrics.roi_last_20 > -2.0
    )
    is_gruen = not is_rot and not is_gelb and clv_positive and roi_ok

    if is_rot:
        status = Ampelstatus.ROT
    elif is_gelb:
        status = Ampelstatus.GELB
    else:
        status = Ampelstatus.GRUEN

    # ---- Parameter-Satz je Status ----------------------------------------

    if status == Ampelstatus.GRUEN:
        params = AmpelParameter(
            status=status,
            kelly_fraction        = base["kelly_fraction"],           # 0.25 (unverändert)
            max_stake_percent     = base["max_stake_percent"],        # 5 % (unverändert)
            min_ev_threshold      = base["min_ev_threshold"],         # 5 % (unverändert)
            min_edge_over_implied = base["min_edge_over_implied"],    # 5 % (unverändert)
            max_daily_bets        = base["max_daily_bets"],           # 10 (unverändert)
            bet365_max_stake_eur  = 200.0,
            bet365_account_limited= metrics.bet365_limited,
            reason="Alle Indikatoren positiv — Normalbetrieb",
        )

    elif status == Ampelstatus.GELB:
        params = AmpelParameter(
            status=status,
            kelly_fraction        = base["kelly_fraction"] * 0.60,   # → 0.15 (Sixth-Kelly)
            max_stake_percent     = base["max_stake_percent"] * 0.60, # → 3 %
            min_ev_threshold      = max(base["min_ev_threshold"] + 0.02, 0.07),  # → 7 %
            min_edge_over_implied = base["min_edge_over_implied"] + 0.02,         # → 7 %
            max_daily_bets        = max(base["max_daily_bets"] // 2, 3),          # → 5
            bet365_max_stake_eur  = 100.0,
            bet365_account_limited= metrics.bet365_limited,
            reason=(
                f"Drawdown {metrics.drawdown_pct:.1f}% oder "
                f"{metrics.consecutive_neg_clv_bets} neg. CLV-Bets — "
                "Volumen reduziert, EV-Schwelle erhöht"
            ),
        )

    else:  # ROT
        params = AmpelParameter(
            status=status,
            kelly_fraction        = base["kelly_fraction"] * 0.25,   # → 0.0625 (Sixteenth-Kelly)
            max_stake_percent     = base["max_stake_percent"] * 0.40, # → 2 %
            min_ev_threshold      = max(base["min_ev_threshold"] + 0.04, 0.09),  # → 9 %
            min_edge_over_implied = base["min_edge_over_implied"] + 0.04,         # → 9 %
            max_daily_bets        = 2,
            bet365_max_stake_eur  = 50.0,
            bet365_account_limited= metrics.bet365_limited,
            reason=(
                f"KRITISCH: Drawdown {metrics.drawdown_pct:.1f}% oder "
                f"{metrics.consecutive_neg_clv_bets} neg. CLV-Bets — "
                "Notfallbremse aktiv"
            ),
            auto_pause=(metrics.drawdown_pct > 25.0),   # > 25% = vollständige Pause
        )

    # Bet365-Limit überschreibt stake-Parameter zusätzlich
    if metrics.bet365_limited:
        params.bet365_max_stake_eur = min(params.bet365_max_stake_eur, 25.0)
        params.max_stake_percent    = min(params.max_stake_percent, 0.015)  # max 1.5 %
        params.reason += " | Bet365-Account limitiert — Soft-Book-Fallback aktiv"

    logger.info(
        "Ampel: %s | Drawdown=%.1f%% | neg_CLV_streak=%d | "
        "kelly=%.4f max_stake=%.2f%% min_ev=%.1f%% max_bets=%d",
        status.value,
        metrics.drawdown_pct,
        metrics.consecutive_neg_clv_bets,
        params.kelly_fraction,
        params.max_stake_percent * 100,
        params.min_ev_threshold * 100,
        params.max_daily_bets,
    )

    return params


# ---------------------------------------------------------------------------
# Bet365-Limit-Erkennung
# ---------------------------------------------------------------------------

def detect_bet365_limit(recent_accepted_stakes: list[float], requested_stake: float) -> bool:
    """
    Heuristik: Bet365 limitiert accounts ohne Warnung.
    Erkennung: wenn die letzten 3 akzeptierten Stakes deutlich unter
    den empfohlenen Werten liegen, signalisiert das ein aktives Limit.

    recent_accepted_stakes: tatsächlich akzeptierte Einsätze (letzte 3)
    requested_stake        : vom System empfohlener Einsatz

    Gibt True zurück wenn >50 % Reduktion über 3 Bets.
    """
    if len(recent_accepted_stakes) < 3:
        return False
    avg_accepted = sum(recent_accepted_stakes) / len(recent_accepted_stakes)
    if requested_stake <= 0:
        return False
    reduction_ratio = 1.0 - (avg_accepted / requested_stake)
    limited = reduction_ratio > 0.50
    if limited:
        logger.warning(
            "Bet365-Limit erkannt: avg_accepted=%.2f requested=%.2f reduction=%.0f%%",
            avg_accepted, requested_stake, reduction_ratio * 100,
        )
    return limited
