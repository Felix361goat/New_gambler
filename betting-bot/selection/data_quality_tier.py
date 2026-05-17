"""
Data Quality Tier System
========================
Jeder Bet bekommt ein data_quality_tier (1, 2 oder 3).
Dieses Feld wird in der DB gespeichert und ermöglicht später die Analyse,
welche Märkte wirklich profitabel sind.

Tier 1 — Premium Data (Live + Paper)
    - 5+ Datenquellen, 3+ Jahre Historik, 10+ Bookmaker-Odds
    - Beispiele: AHL, WTA, Baltic Basketball, Soccer England Tier 4

Tier 2 — Standard Data (Live + Paper)
    - 3+ Quellen, 1.5+ Jahre Historik, 4+ Bookmaker-Odds
    - Beispiele: ECHL, ITF Damen, Rumänische Liga, Soccer Scandinavia

Tier 3 — Low Data (nur Paper Mode — im Live Mode ignoriert)
    - <3 Quellen ODER <1 Jahr Historik ODER <3 Bookies
    - Beispiele: ITF Herren lower, Europäische Nischenligen (EBEL etc.),
      Basketball Unknown Region, Soccer England Tier 5
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TIER-DEFINITIONEN
# ---------------------------------------------------------------------------

# Liga/Sport → Tier-Zuweisung (explizite Liste)
# Nicht gelistete Märkte fallen automatisch in Tier 3.
LEAGUE_TIER_MAP: dict[str, int] = {
    # ── TIER 1 ── Premium Data ───────────────────────────────────────────────
    # Hockey
    "hockey_ahl": 1,           # AHL: EliteProspects + 3 Stats-Portale, 5+ Jahre
    # Tennis
    "tennis_wta": 1,           # WTA: Tennis Abstract (Sackmann), 10+ Jahre, 15+ Bookies
    "tennis_atp": 1,           # ATP Top 50: idem
    # Basketball
    "basketball_baltic": 1,    # Baltic (Estl/Lettl/Lith): Realscore, Flashscore, 3 Bookies
    # Soccer
    "soccer_england_tier4": 1, # National League EN: Opta, Understat, 8+ Bookies
    "soccer_scandinavia": 1,   # Allsvenskan/Eliteserien: Solidsport-Daten, 5+ Bookies

    # ── TIER 2 ── Standard Data ──────────────────────────────────────────────
    # Hockey
    "hockey_echl": 2,          # ECHL: EliteProspects, 3 Quellen, ~2 Jahre
    "hockey_european_minor": 2, # EBEL, DEL2, NLA: 3 Quellen, 2 Jahre
    # Tennis
    "tennis_itf_women": 2,     # ITF Damen: Tennis Abstract, 3 Quellen, ~2 Jahre
    # Basketball
    "basketball_romanian": 2,  # Romania Liga: SofaScore + 2 Quellen, 2 Jahre
    "basketball_european_lower": 2,  # BSL, Rumänien 2 etc.: 3 Quellen, 1.5 Jahre
    # Soccer
    "soccer_poland": 2,        # Ekstraklasa: Opta light, 4 Bookies, 2 Jahre

    # ── TIER 3 ── Low Data (nur Paper Mode) ──────────────────────────────────
    "tennis_itf_men": 3,       # ITF Herren lower: <3 Bookies, viele unbekannte Spieler
    "soccer_england_tier5": 3, # National League South/North: <4 Bookies, Match-Fixing-Flag
    "basketball_unknown_region": 3,  # Catch-all
    "hockey_tier3_plus": 3,    # Jede Liga unterhalb ECHL/DEL2
    # Alles andere → Tier 3 (siehe assign_tier())
}

# Mindestanforderungen pro Tier — für Laufzeit-Prüfung
TIER_REQUIREMENTS: dict[int, dict] = {
    1: {
        "min_sources": 5,        # Anzahl unabhängiger Datenquellen
        "min_history_years": 3,  # Jahre Historik vorhanden
        "min_bookmakers": 8,     # Bookies mit Odds für diesen Markt
        "min_sample_size": 20,   # Historische Spiele im System
    },
    2: {
        "min_sources": 3,
        "min_history_years": 1.5,
        "min_bookmakers": 4,
        "min_sample_size": 12,
    },
    3: {
        # Tier 3 hat keine harten Anforderungen — alles darunter
        "min_sources": 1,
        "min_history_years": 0,
        "min_bookmakers": 1,
        "min_sample_size": 5,
    },
}

# Im Live Mode werden Tier-3-Bets übersprungen
LIVE_MODE_MIN_TIER = 2


# ---------------------------------------------------------------------------
# KERN-FUNKTION
# ---------------------------------------------------------------------------

def assign_tier(
    league: str,
    sport: str | None = None,
    num_bookmakers: int = 0,
    num_sources: int = 0,
    history_years: float = 0.0,
    sample_size: int = 0,
    data_completeness: float = 0.0,
) -> int:
    """
    Weist einem Bet ein data_quality_tier (1, 2 oder 3) zu.

    Priorität:
    1. Explizite Liga-Zuordnung aus LEAGUE_TIER_MAP
    2. Laufzeit-Prüfung der Datenpunkte (kann downgraden, nie upgraden)
    3. Default: Tier 3

    Parameters
    ----------
    league          : Interne Liga-ID (z.B. "hockey_ahl", "tennis_wta")
    sport           : Sport-Typ als Fallback (z.B. "tennis", "hockey")
    num_bookmakers  : Wieviele Bookies haben Odds für diesen Markt?
    num_sources     : Wieviele Datenquellen wurden erfolgreich abgefragt?
    history_years   : Wieviele Jahre Historik sind im System?
    sample_size     : Anzahl historischer Spiele in der DB für diese Liga
    data_completeness: Feature-Completeness aus filter.py (0.0–1.0)

    Returns
    -------
    int: 1, 2 oder 3
    """
    # Schritt 1: Explizite Zuordnung
    base_tier = LEAGUE_TIER_MAP.get(league, 3)

    # Schritt 2: Laufzeit-Downgrade (nie Upgrade)
    # Wenn echte Datenpunkte verfügbar sind, können sie das Tier nur senken.
    if num_bookmakers > 0 or num_sources > 0 or sample_size > 0:
        runtime_tier = _runtime_tier(
            num_bookmakers=num_bookmakers,
            num_sources=num_sources,
            history_years=history_years,
            sample_size=sample_size,
        )
        base_tier = max(base_tier, runtime_tier)  # max = schlechterer Tier gewinnt

    # Schritt 3: Data-Completeness-Downgrade
    # Wenn <50% Features vorhanden → Tier 3 erzwingen
    if data_completeness > 0.0 and data_completeness < 0.50:
        base_tier = 3

    return base_tier


def _runtime_tier(
    num_bookmakers: int,
    num_sources: int,
    history_years: float,
    sample_size: int,
) -> int:
    """Berechnet Tier nur aus tatsächlich verfügbaren Datenpunkten."""
    req1 = TIER_REQUIREMENTS[1]
    req2 = TIER_REQUIREMENTS[2]

    if (
        num_sources >= req1["min_sources"]
        and num_bookmakers >= req1["min_bookmakers"]
        and history_years >= req1["min_history_years"]
        and sample_size >= req1["min_sample_size"]
    ):
        return 1

    if (
        num_sources >= req2["min_sources"]
        and num_bookmakers >= req2["min_bookmakers"]
        and history_years >= req2["min_history_years"]
        and sample_size >= req2["min_sample_size"]
    ):
        return 2

    return 3


# ---------------------------------------------------------------------------
# LIVE-MODE-GATE
# ---------------------------------------------------------------------------

def is_live_eligible(tier: int) -> bool:
    """
    True wenn der Bet im Live Mode platziert werden darf.
    Tier 3 → nur Paper Mode.
    """
    return tier <= LIVE_MODE_MIN_TIER


def tier_label(tier: int) -> str:
    """Menschenlesbare Bezeichnung für Telegram/Sheets."""
    return {1: "T1-Premium", 2: "T2-Standard", 3: "T3-PaperOnly"}.get(tier, "T3-PaperOnly")


# ---------------------------------------------------------------------------
# BATCH-ZUWEISUNG (wird von filter.py aufgerufen)
# ---------------------------------------------------------------------------

def annotate_bets_with_tier(
    bets: list[dict],
    live_mode: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Annotiert alle Bets mit data_quality_tier und tier_label.
    Im Live Mode werden Tier-3-Bets in die paper_only-Liste verschoben.

    Parameters
    ----------
    bets       : Liste von Bet-Dicts (Output von select_daily_bets)
    live_mode  : Wenn True, werden Tier-3-Bets aus active_bets entfernt

    Returns
    -------
    (active_bets, paper_only_bets)
    - active_bets: werden live platziert (oder paper wenn paper_mode)
    - paper_only_bets: immer nur Paper — auch wenn live_mode=True
    """
    active_bets: list[dict] = []
    paper_only_bets: list[dict] = []

    for bet in bets:
        tier = assign_tier(
            league=bet.get("league", ""),
            sport=bet.get("sport"),
            num_bookmakers=bet.get("_num_bookmakers", 0),
            num_sources=bet.get("_num_sources", 0),
            history_years=bet.get("_history_years", 0.0),
            sample_size=bet.get("_sample_size", 0),
            data_completeness=bet.get("data_completeness", 0.0),
        )
        bet["data_quality_tier"] = tier
        bet["data_quality_tier_label"] = tier_label(tier)
        bet["live_eligible"] = is_live_eligible(tier)

        if live_mode and tier >= 3:
            bet["skipped_reason"] = "Tier 3 — nur Paper Mode"
            paper_only_bets.append(bet)
        else:
            active_bets.append(bet)

    return active_bets, paper_only_bets
