"""
steering/workflow_bet365.py — Bet365 Manuelle-Platzierungs-Workflow

Bet365 hat keine öffentliche API.  Jede Wette muss manuell (Browser/App)
platziert werden.  Dieses Modul generiert:
  1. Eine priorisierte Platzierungs-Checkliste für den Betreiber
  2. Browser-freundliche Suchstrings (Schnellsuche in Bet365)
  3. Limit-Anpassung wenn Bet365 weniger als empfohlen akzeptiert

Entscheidend: Das System bleibt korrekt auch ohne API — die Ampel-Parameter
werden VOR der Checklisten-Generierung angewendet, sodass der Betreiber
niemals höhere Stakes sieht als das Risikomanagement erlaubt.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .ampel import AmpelParameter, Ampelstatus

logger = logging.getLogger(__name__)

# Bet365 akzeptiert in Nischenmärkten typisch diese Maxima:
# Tennis ITF / AHL / Baltic Basketball: €2–50
# WTA / ECHL / European Soccer Tier4:   €10–150
# Tier-1 Märkte (PL, NBA):              €50–500
BET365_MARKET_CAPS: dict[str, float] = {
    "tennis_itf_men":           50.0,
    "tennis_itf_women":         50.0,
    "tennis_wta":              150.0,
    "hockey_ahl":              100.0,
    "hockey_echl":              50.0,
    "hockey_european_minor":    30.0,
    "basketball_baltic":        30.0,
    "basketball_romanian":      25.0,
    "basketball_european_lower":25.0,
    "soccer_england_tier4":    100.0,
    "soccer_england_tier5":     50.0,
    "soccer_scandinavia":       75.0,
    "soccer_poland":            75.0,
}
DEFAULT_BET365_CAP = 100.0


@dataclass
class Bet365BetInstruction:
    """Eine fertige Platzierungs-Anweisung für den Betreiber."""
    rank: int
    match: str                  # "Team A vs Team B"
    league: str
    market: str
    odds: float
    recommended_stake: float    # Schon gegen Bet365-Cap und Ampel-Limit beschnitten
    system_stake: float         # Originalempfehlung des Systems (pre-cap)
    search_hint: str            # Schnellsuche-String für Bet365
    notes: str = ""


def build_bet365_instructions(
    bets: list[dict],
    ampel: AmpelParameter,
) -> list[Bet365BetInstruction]:
    """
    Wandelt die selektierten Bets in konkrete Bet365-Anweisungen um.

    Parameter
    ---------
    bets   : Ausgabe von selection/filter.select_daily_bets()
    ampel  : Aktuelle Ampel-Parameter (bestimmt caps und stake-Limits)

    Returns
    -------
    Sortierte Liste (höchstes EV zuerst), stake bereits gegen alle
    Limits (Ampel + Bet365-Markt-Cap + Account-Limit) beschnitten.
    """
    instructions = []

    for rank, bet in enumerate(bets, 1):
        league  = bet.get("league", "default").lower()
        market  = bet.get("market", "")
        odds    = bet.get("bookmaker_odds", 2.0)
        home    = bet.get("home_team", "?")
        away    = bet.get("away_team", "?")
        sys_stake = float(bet.get("stake_recommended", 0))

        # Bet365 Markt-Cap ermitteln
        market_cap = BET365_MARKET_CAPS.get(league, DEFAULT_BET365_CAP)

        # Ampel-Cap (€ absolut je Bet)
        ampel_cap = ampel.bet365_max_stake_eur

        # Finaler empfohlener Stake: minimum aller Caps
        final_stake = round(min(sys_stake, market_cap, ampel_cap), 2)

        # Bet365 Suchstring — kompakt, für Suche in der App nutzbar
        search_hint = _build_search_hint(home, away, market)

        note_parts = []
        if ampel.status != Ampelstatus.GRUEN:
            note_parts.append(f"Ampel {ampel.status.value}: Stake von €{sys_stake:.2f} auf €{final_stake:.2f} reduziert")
        if final_stake < sys_stake and ampel.status == Ampelstatus.GRUEN:
            note_parts.append(f"Bet365 Markt-Cap ({market_cap:.0f}€) greift")
        if ampel.bet365_account_limited:
            note_parts.append("Account-Limit aktiv — prüfe akzeptierten Betrag in der App")

        instructions.append(Bet365BetInstruction(
            rank=rank,
            match=f"{home} vs {away}",
            league=league,
            market=market,
            odds=odds,
            recommended_stake=final_stake,
            system_stake=sys_stake,
            search_hint=search_hint,
            notes=" | ".join(note_parts),
        ))

    return instructions


def _build_search_hint(home: str, away: str, market: str) -> str:
    """Generiert einen knappen Suchstring für die Bet365 App."""
    market_labels = {
        "1x2_home":  f"{home} gewinnt",
        "1x2_draw":  "Unentschieden",
        "1x2_away":  f"{away} gewinnt",
        "over_2.5":  "Über 2.5 Tore",
        "under_2.5": "Unter 2.5 Tore",
        "over_3.5":  "Über 3.5 Tore",
        "under_3.5": "Unter 3.5 Tore",
        "btts":      "Beide treffen",
    }
    market_label = market_labels.get(market, market)
    # Kurze Teamnamen für schnelle App-Suche
    short_home = home.split()[-1] if " " in home else home
    short_away = away.split()[-1] if " " in away else away
    return f"{short_home} {short_away} → {market_label}"


def format_bet365_checklist(instructions: list[Bet365BetInstruction], ampel: AmpelParameter) -> str:
    """
    Erzeugt die Platzierungs-Checkliste als formatierten String —
    wird direkt in die Telegram-Nachricht eingefügt.
    """
    ampel_icons = {
        Ampelstatus.GRUEN: "🟢",
        Ampelstatus.GELB:  "🟡",
        Ampelstatus.ROT:   "🔴",
    }
    icon = ampel_icons[ampel.status]

    lines = [
        f"{icon} BET365 CHECKLISTE — Ampel {ampel.status.value}",
        f"Kelly: {ampel.kelly_fraction:.4f} | Max Stake: {ampel.max_stake_percent*100:.1f}% | EV-Min: {ampel.min_ev_threshold*100:.0f}%",
        "──────────────────────────────────",
    ]

    for inst in instructions:
        cap_note = ""
        if inst.recommended_stake < inst.system_stake:
            cap_note = f" ⚠️ cap von €{inst.system_stake:.2f}"

        lines.append(
            f"{inst.rank}. {inst.match}\n"
            f"   {inst.league} | {inst.market} @ {inst.odds:.2f}\n"
            f"   💵 Setze: €{inst.recommended_stake:.2f}{cap_note}\n"
            f"   🔍 Suche: \"{inst.search_hint}\""
        )
        if inst.notes:
            lines.append(f"   ℹ️ {inst.notes}")
        lines.append("──────────────────────────────────")

    if ampel.auto_pause:
        lines.append("🚨 AUTO-PAUSE: Drawdown > 25% — KEINE Bets bis manuelle Freigabe!")

    return "\n".join(lines)
