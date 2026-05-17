import logging
from datetime import date
from selection.ev_calculator import scenario_pnl

logger = logging.getLogger(__name__)


def format_morning_briefing(bets: list, performance: dict) -> str:
    if not bets:
        return (
            "🎯 BETTING BRIEFING — Heute keine qualifizierenden Bets\n\n"
            "Kein Bet hat heute die EV-Schwelle (3%) erreicht.\n"
            f"📋 Paper Mode: {performance.get('settled_bets', 0)}/200"
        )

    today = date.today()
    weekday_de = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekday = weekday_de[today.weekday()]
    date_str = today.strftime("%d.%m.%Y")

    lines = [f"🎯 BETTING BRIEFING — {weekday}, {date_str}\n"]
    lines.append("━━━━━━━━━━━━━━━━━━━━━")

    total_ev = 0.0
    for i, bet in enumerate(bets, 1):
        home = bet.get("home_team", "?")
        away = bet.get("away_team", "?")
        league = bet.get("league", "?")
        market = bet.get("market", "?")
        odds = bet.get("bookmaker_odds", 0)
        bookmaker = bet.get("bookmaker_name", "?")
        ev = bet.get("ev_score", 0) * 100
        confidence = bet.get("confidence_score", 0)
        stake = bet.get("stake_recommended", 0)
        bankroll = performance.get("bankroll", 1000)
        stake_pct = (stake / bankroll * 100) if bankroll > 0 else 0
        total_ev += ev

        special = ""
        if bet.get("is_favorite_club"):
            special = "⭐ Dein Klub"
        elif bet.get("is_watchable"):
            special = "🎯 Watchable"

        lines.append(
            f"{i}. {home} vs {away}\n"
            f"   📊 {league} | {market}\n"
            f"   💰 Quote: {odds:.2f} @ {bookmaker}\n"
            f"   📈 EV: +{ev:.1f}% | Confidence: {confidence}/100\n"
            f"   💵 Stake: €{stake:.2f} ({stake_pct:.1f}% Bankroll)"
            + (f"\n   {special}" if special else "")
        )

        # Confidence and data completeness
        confidence_tier = bet.get("confidence_tier", "Low")
        completeness = bet.get("data_completeness", 0)
        tier_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence_tier, "🔴")
        lines.append(f"   {tier_emoji} Konfidenz: {confidence_tier} | Daten: {completeness*100:.0f}%")

        # Breakeven win rate
        from selection.ev_calculator import breakeven_win_rate
        be_rate = breakeven_win_rate(bet.get("bookmaker_odds", 2.0))
        lines.append(f"   🎯 Break-Even: {be_rate*100:.1f}% Win Rate")

        # Match-fixing warning if present
        if bet.get("matchfixing_warning"):
            lines.append(f"   ⚠️ {bet['matchfixing_warning']}")

        lines.append("━━━━━━━━━━━━━━━━━━━━━")

    avg_ev = total_ev / len(bets) if bets else 0
    bankroll = performance.get("bankroll", 1000)
    starting = performance.get("starting_bankroll", 1000)
    pct_change = ((bankroll - starting) / starting * 100) if starting > 0 else 0
    settled = performance.get("settled_bets", 0)

    lines.append(
        f"\n📊 Heute: {len(bets)} Bets | Ø EV: +{avg_ev:.1f}%\n"
        f"💰 Sim. Bankroll: €{bankroll:.2f} ({pct_change:+.1f}%)\n"
        f"📋 Paper Mode: {settled}/200\n\n"
        f"Tippe /placed N oder /skip N"
    )

    # Szenario-Analyse
    if bets:
        scenarios = scenario_pnl(bets)
        lines.append("📊 Szenarien (erw. P&L):")
        lines.append(f"   60% Trefferquote → {scenarios.get(60, 0):+.2f}€")
        lines.append(f"   70% Trefferquote → {scenarios.get(70, 0):+.2f}€")
        lines.append(f"   80% Trefferquote → {scenarios.get(80, 0):+.2f}€")
        lines.append("━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)
