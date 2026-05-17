import logging
from datetime import date

logger = logging.getLogger(__name__)


def format_evening_summary(bets: list, performance_today: dict, performance_total: dict) -> str:
    today = date.today()
    weekday_de = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekday = weekday_de[today.weekday()]
    date_str = today.strftime("%d.%m.%Y")

    lines = [f"📋 TAGES-ABSCHLUSS — {weekday}, {date_str}\n"]

    won = lost = pending = 0
    for bet in bets:
        status = bet.get("status", "pending")
        result = bet.get("result", {})
        match_str = f"{bet.get('home_team', '?')} vs {bet.get('away_team', '?')}"
        market = bet.get("market", "?")

        if status in ("placed",) and result.get("won") is not None:
            is_won = result.get("won", False)
            pnl = result.get("pnl_simulated", 0)
            icon = "✅" if is_won else "❌"
            check = "✓" if is_won else "✗"
            if is_won:
                won += 1
            else:
                lost += 1
            lines.append(f"{icon} Bet {bet.get('id', '?')}: {match_str} {market} {check} {pnl:+.2f}€")
        else:
            pending += 1
            lines.append(f"⏳ Bet {bet.get('id', '?')}: {match_str} — läuft noch")

    total_today = won + lost
    win_rate = (won / total_today * 100) if total_today > 0 else 0
    daily_pnl = performance_today.get("pnl_daily", 0)
    cum_pnl = performance_total.get("total_pnl", 0)
    roi = performance_total.get("roi", 0)
    clv = performance_today.get("clv_today", None)

    lines.append(f"\n📊 Heute: {won}W {lost}L | Win Rate: {win_rate:.0f}%")
    lines.append(f"💵 Tages-P&L: {daily_pnl:+.2f}€")
    lines.append(f"📈 Gesamt: {cum_pnl:+.2f}€ ({roi:+.1f}% ROI)")

    if clv is not None:
        clv_check = "✓" if clv > 0 else ""
        lines.append(f"🎯 CLV heute: {clv:+.2f}% {clv_check}")

    return "\n".join(lines)
