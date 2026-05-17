# Autonomous Sports Betting Bot

Autonomous sports betting prediction bot. Generates up to 10 daily bet recommendations based strictly on positive Expected Value (EV). No bet without EV justification.

## Mode

**Paper mode first.** System simulates bets and tracks performance. Live betting only after 200+ simulated bets with ROI > 3%.

## Stack

- Python 3.11+
- SQLite (local database)
- Google Sheets API (tracking dashboard)
- Telegram Bot API (user interface)
- No LLM in runtime — pure ML/statistics, zero variable API costs

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
- `FOOTBALL_API_KEY` — free at [football-data.org](https://football-data.org)
- `ODDS_API_KEY` — free at [the-odds-api.com](https://the-odds-api.com)
- `WEATHER_API_KEY` — free at [openweathermap.org](https://openweathermap.org)
- `TELEGRAM_BOT_TOKEN` — create bot via @BotFather on Telegram
- `TELEGRAM_CHAT_ID` — message @userinfobot on Telegram
- `GOOGLE_SHEET_ID` — create empty Sheet, copy ID from URL
- Place `google_credentials.json` in project root (Google Cloud Console)

### 3. Run setup

```bash
python main.py --setup
```

This initializes the database, verifies API keys, connects to Google Sheets, and installs cron jobs automatically.

Paper mode starts next morning at **08:30**.

## CLI Commands

| Command | Description |
|---------|-------------|
| `--setup` | First-time setup: DB, Sheets, verify API keys, install cron |
| `--collect` | Fetch all data sources |
| `--predict` | Run prediction engine, generate daily bets |
| `--brief` | Send morning briefing via Telegram |
| `--odds_snapshot` | Save current odds snapshot to DB |
| `--summarize` | Send evening summary via Telegram |
| `--retrain` | Retrain all models |
| `--weekly_report` | Send weekly report |
| `--test` | Run all tests |
| `--paper_status` | Print current paper mode statistics |

## Cron Schedule (auto-installed by --setup)

```
06:00  --collect        Fetch all data
07:30  --predict        Generate daily bets
08:30  --brief          Send morning briefing
*/6h   --odds_snapshot  Save odds snapshots
23:00  --summarize      Send evening summary
Sun 03:00  --retrain    Weekly model retraining
Sun 20:00  --weekly_report  Weekly performance report
```

## Telegram Commands

| Command | Action |
|---------|--------|
| `/placed N` | Mark bet N as placed |
| `/skip N` | Skip bet N |
| `/status` | Current bankroll, ROI, paper mode progress |
| `/bets` | Show today's bets |
| `/week` | This week's performance |
| `/help` | List all commands |

## Bet Selection Logic (5 Layers)

1. **Hard filter**: EV ≥ 3% (never below, no exceptions)
2. **Ranking**: Sort by EV descending, take top 10
3. **Tiebreaker**: League priority order (only within 0.5% EV)
4. **Watchable flag**: Preferred leagues (PL/Bundesliga/NBA) between 17:00–23:00 Austrian time
5. **Favorite clubs**: Arsenal/Bristol Rovers included at 80% threshold (≥ 2.4% EV)

## Models

- **Poisson** (35% weight): Dixon-Coles model with time-decay
- **XGBoost** (40% weight): TimeSeriesSplit — no data leakage
- **ELO** (25% weight): K=32, home advantage 100 points

All 3 models must agree (≥ 2/3) for a bet to be generated.

## Google Sheets Dashboard

Three auto-created tabs:
- **Bets**: All bets with EV, odds, P&L, CLV, status emoji
- **Analytics**: ROI by league/market, win rates, model calibration
- **Model Health**: System status, retrain dates, paper mode progress

## Go Live Conditions

- ≥ 200 simulated bets settled
- ROI > 3% overall
- System reports "Ready for Live: YES ✅" in `/status`

## Running Tests

```bash
python main.py --test
# or directly:
pytest tests/ -v
```
