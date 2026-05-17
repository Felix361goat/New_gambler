import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def format_weekly_report(performance: dict, league_stats: dict = None) -> str:
    today = date.today()
    week_num = today.isocalendar()[1]
    settled = performance.get("settled_bets", 0)
    min_paper = 200
    remaining = max(0, min_paper - settled)

    lines = [f"📊 WOCHEN-REPORT KW {week_num}\n"]
    lines.append(
        f"Bets: {performance.get('total_bets', 0)} | "
        f"Won: {performance.get('won', 0)} ({performance.get('win_rate', 0):.1f}%)"
    )
    lines.append(f"ROI: {performance.get('roi', 0):+.1f}% | P&L: {performance.get('total_pnl', 0):+.2f}€")
    lines.append(f"Max Drawdown: {performance.get('max_drawdown', 0):.2f}€")
    lines.append(f"Ø Confidence: {performance.get('avg_confidence', 0):.0f}/100")
    lines.append("")

    if league_stats:
        sorted_leagues = sorted(league_stats.items(), key=lambda x: x[1].get("roi", 0), reverse=True)
        if sorted_leagues:
            best_l, best_s = sorted_leagues[0]
            worst_l, worst_s = sorted_leagues[-1]
            lines.append(f"🏆 Beste Liga: {best_l} ({best_s.get('roi', 0):+.1f}% ROI)")
            lines.append(f"⚠️ Schwächste: {worst_l} ({worst_s.get('roi', 0):+.1f}% ROI)")
            lines.append("")

    insight = _generate_insight(performance)
    lines.append(f"Modell-Empfehlung: {insight}")
    lines.append("")
    lines.append("Retraining: ✅ Sonntag 03:00 Uhr")
    lines.append(f"Paper Mode: {settled}/200 | {remaining} bis Live möglich")

    return "\n".join(lines)


def _generate_insight(performance: dict) -> str:
    roi = performance.get("roi", 0)
    win_rate = performance.get("win_rate", 0)
    avg_ev = performance.get("avg_ev", 0) * 100
    settled = performance.get("settled_bets", 0)

    if settled < 20:
        return "Noch zu wenig Daten für verlässliche Einschätzung."

    if roi > 10:
        return f"Ausgezeichnete Performance mit {roi:.1f}% ROI. Strategie beibehalten."
    if roi > 3:
        return f"Solide {roi:.1f}% ROI. Modell im Zielbereich. Weiter beobachten."
    if roi > 0:
        return f"Positiv aber knapp ({roi:.1f}%). EV-Schwelle könnte auf 3.5% erhöht werden."
    if win_rate > 60:
        return f"Hohe Win Rate ({win_rate:.0f}%) aber niedriger ROI — mögliche Stake-Optimierung nötig."
    return f"ROI bei {roi:.1f}%. Prüfe ob Datenqualität oder Modell-Rekalibrierung nötig ist."
