# CLAUDE.md — Autonomous Sports Betting Bot

## Your Role as Claude Code

You are the **Orchestrator**. You manage two subagents simultaneously:

- **BUILDER**: Writes all code, creates all files, implements all modules
- **QA**: Reviews every completed batch, identifies bugs, validates logic, instructs BUILDER on exact fixes

### Rules
- Run BUILDER and QA in parallel wherever possible
- QA reviews every batch before BUILDER moves to the next
- QA fixes trivial issues itself — only escalates real logic errors to BUILDER
- No module is "done" without QA sign-off
- Never read the same file twice unnecessarily
- Prefer writing complete files over incremental edits to save tokens

---

## Project Overview

**Goal:** Autonomous sports betting prediction bot. Generates up to 10 daily bet recommendations based strictly on positive Expected Value (EV). No bet without EV justification.

**Mode:** Paper mode first. System simulates bets, tracks performance. Live betting only after 200+ simulated bets with ROI > 3%.

**Stack:**
- Python 3.11+
- SQLite (local database)
- Google Sheets API (tracking dashboard)
- Telegram Bot API (user interface)
- Deployable on Hetzner VPS (€3.29/mo) or Oracle Cloud Free Tier
- No LLM in runtime — pure ML/statistics, zero variable API costs

**Total variable costs: €0**

---

## Project Structure

Create this exact structure first before writing any logic:

```
betting-bot/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── main.py
│
├── data/
│   ├── collector.py
│   ├── quality_check.py
│   └── sources/
│       ├── base_source.py
│       ├── football_api.py
│       ├── odds_api.py
│       ├── understat.py
│       ├── transfermarkt.py
│       ├── weather.py
│       ├── nba_api.py
│       └── news_rss.py
│
├── features/
│   ├── builder.py
│   ├── form.py
│   ├── xg_features.py
│   ├── context.py
│   ├── squad.py
│   ├── referee.py
│   ├── h2h.py
│   └── odds_movement.py
│
├── models/
│   ├── ensemble.py
│   ├── poisson_model.py
│   ├── xgboost_model.py
│   ├── elo_model.py
│   └── retrainer.py
│
├── selection/
│   ├── ev_calculator.py
│   ├── kelly.py
│   ├── filter.py
│   └── preference.py
│
├── tracking/
│   ├── database.py
│   ├── sheets.py
│   └── performance.py
│
├── notifications/
│   ├── telegram_bot.py
│   ├── morning_briefing.py
│   ├── evening_summary.py
│   └── weekly_report.py
│
├── scheduler/
│   └── cron_setup.py
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    ├── test_ev.py
    └── test_telegram.py
```

---

## config.yaml — Write This Exactly

```yaml
betting:
  max_daily_bets: 10
  min_ev_threshold: 0.03          # 3% minimum EV — never go below
  watchable_ev_threshold: 0.024   # 80% of min — only for preferred leagues
  min_paper_bets_before_live: 200
  bankroll_paper: 1000.0          # Simulated starting bankroll in €
  kelly_fraction: 0.25            # Quarter-Kelly — conservative
  max_stake_percent: 0.05         # Max 5% of bankroll per bet

leagues:
  # Tiebreaker order — only used when EV is equal (within 0.5%)
  # PROFIT is always Layer 1. This list only breaks ties.
  priority:
    - premier_league
    - bundesliga
    - la_liga
    - serie_a
    - ligue_1
    - austrian_bundesliga
    - nba
  # At least 1 watchable bet per week from preferred leagues
  # ONLY if EV >= watchable_ev_threshold. Never forced.
  watchable_min_per_week: 1
  watchable_max_per_week: 3
  watchable_preferred:
    - premier_league
    - bundesliga
    - nba
  watchable_kickoff_window:
    earliest_hour: 17             # Austrian time
    latest_hour: 23

favorite_clubs:
  - "Arsenal"
  - "Bristol Rovers"
  # These clubs get ev_override_factor applied to threshold
  # Still needs at least 80% of normal EV — not a free pass
  ev_override_factor: 0.80

data_sources:
  football_api_key: ${FOOTBALL_API_KEY}
  odds_api_key: ${ODDS_API_KEY}
  weather_api_key: ${WEATHER_API_KEY}
  request_delay_seconds: 2        # Respectful scraping
  max_retries: 3
  timeout_seconds: 10

telegram:
  bot_token: ${TELEGRAM_BOT_TOKEN}
  chat_id: ${TELEGRAM_CHAT_ID}
  morning_briefing_hour: 8
  morning_briefing_minute: 30
  evening_summary_hour: 23
  evening_summary_minute: 0
  weekly_report_day: "sunday"
  weekly_report_hour: 20

google_sheets:
  credentials_file: "google_credentials.json"
  spreadsheet_id: ${GOOGLE_SHEET_ID}
  tabs:
    bets: "Bets"
    analytics: "Analytics"
    model_health: "Model Health"

model:
  retrain_day: "sunday"
  retrain_hour: 3
  min_samples_to_train: 50        # Don't retrain with less data
  form_weights:
    last_5_games: 0.35
    last_10_to_20_games: 0.25
    full_current_season: 0.15
    last_2_to_3_seasons: 0.15
    head_to_head: 0.10
  ensemble_weights:
    poisson: 0.35
    xgboost: 0.40
    elo: 0.25
  min_models_agreeing: 2          # At least 2/3 models must agree
```

---

## .env.example — Write This Exactly

```
FOOTBALL_API_KEY=your_key_here         # football-data.org (free)
ODDS_API_KEY=your_key_here             # the-odds-api.com (free tier)
WEATHER_API_KEY=your_key_here          # openweathermap.org (free)
TELEGRAM_BOT_TOKEN=your_token_here     # from @BotFather on Telegram
TELEGRAM_CHAT_ID=your_chat_id_here     # your personal chat ID
GOOGLE_SHEET_ID=your_sheet_id_here     # from Google Sheets URL
```

---

## BATCH 1 — Database & Foundation

**BUILDER implements. QA reviews before Batch 2 starts.**

### tracking/database.py

Create SQLite database with these exact tables:

```sql
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    match_date DATE NOT NULL,
    match_id TEXT,
    league TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market TEXT NOT NULL,           -- e.g. "over_2.5", "1x2_home"
    our_probability REAL NOT NULL,  -- 0.0 to 1.0
    bookmaker_odds REAL NOT NULL,   -- decimal odds
    bookmaker_name TEXT NOT NULL,
    ev_score REAL NOT NULL,         -- e.g. 0.073 = +7.3%
    confidence_score INTEGER,       -- 0 to 100
    stake_recommended REAL,         -- in €
    kelly_fraction REAL,
    is_watchable BOOLEAN DEFAULT 0,
    watchable_reason TEXT,
    is_favorite_club BOOLEAN DEFAULT 0,
    status TEXT DEFAULT 'pending',  -- pending/placed/skipped/void
    notes TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id INTEGER REFERENCES bets(id),
    settled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_result TEXT,             -- e.g. "2-1", "over"
    won BOOLEAN,
    pnl_simulated REAL,             -- +/- in €
    closing_line_odds REAL,         -- odds at kickoff
    clv_score REAL                  -- closing line value
);

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
);

CREATE TABLE IF NOT EXISTS referee_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    referee_name TEXT NOT NULL,
    league TEXT NOT NULL,
    avg_goals_per_game REAL,
    avg_cards_per_game REAL,
    avg_penalties_per_game REAL,
    home_bias_score REAL,           -- positive = favors home
    sample_size INTEGER DEFAULT 0,
    last_updated DATE,
    UNIQUE(referee_name, league)
);

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
);

CREATE TABLE IF NOT EXISTS user_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id INTEGER REFERENCES bets(id),
    action TEXT NOT NULL,           -- placed/skipped
    actioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT
);
```

Also implement these methods in the DatabaseHandler class:
- `insert_bet(bet_dict)` → returns bet_id
- `update_bet_status(bet_id, status)`
- `insert_result(result_dict)`
- `insert_odds_snapshot(snapshot_dict)`
- `get_pending_bets(date)` → list
- `get_performance_summary(days=30)` → dict
- `get_bets_for_retraining()` → DataFrame
- `update_daily_performance(date)`

### data/quality_check.py

Implement `QualityChecker` class with:

```python
def check_match(self, match_data) -> tuple[bool, list[str]]:
    """
    Returns (is_valid, list_of_warnings).
    Fails hard if:
    - Fewer than 5 historical games for either team
    - No odds data available
    - Match date in the past
    Warns (but continues) if:
    - Injury data older than 24h → sets injury_data_stale=True flag
    - Weather API failed → sets weather_available=False flag
    - xG data unavailable → model runs without xG features
    """

def check_data_freshness(self, source_name, last_updated) -> bool:
    """Returns False if data is too stale to use."""

def validate_odds(self, odds_value) -> bool:
    """Rejects odds < 1.01 or > 50.0 as likely errors."""
```

**QA checks Batch 1:**
- [ ] All SQL tables have correct foreign keys and constraints?
- [ ] DatabaseHandler methods handle exceptions and log errors?
- [ ] QualityChecker covers all edge cases listed?
- [ ] No hardcoded file paths — use pathlib throughout?

---

## BATCH 2 — Data Sources

**BUILDER implements all sources in parallel. QA reviews before Batch 3.**

### data/sources/base_source.py

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseSource(ABC):
    @abstractmethod
    def fetch(self) -> dict:
        """Fetch raw data from source."""

    @abstractmethod
    def validate(self, data: dict) -> bool:
        """Validate raw data quality."""

    @abstractmethod
    def normalize(self, data: dict) -> pd.DataFrame:
        """Transform to standard internal format."""

    def fetch_and_normalize(self) -> pd.DataFrame | None:
        """Full pipeline with error handling."""
        try:
            raw = self.fetch()
            if not self.validate(raw):
                return None
            return self.normalize(raw)
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__} failed: {e}")
            return None
```

### data/sources/football_api.py

- Base URL: `https://api.football-data.org/v4`
- Fetch: standings, matches (results + upcoming), head-to-head
- Leagues to fetch: PL, BL1, PD, SA, FL1, OSL (Austrian)
- Rate limit: max 10 requests/minute on free tier — implement rate limiter
- Normalize to standard match format:
  ```
  {match_id, date, league, home_team, away_team, 
   home_goals, away_goals, status, matchday, season}
  ```

### data/sources/odds_api.py

- Base URL: `https://api.the-odds-api.com/v4`
- Fetch: all available bookmakers, markets: h2h + totals
- Store full snapshot to odds_snapshots table every 6 hours
- Also return: best available odds per market (line shopping)
- Track which bookmaker has best odds per market

### data/sources/understat.py

- Scrape: `https://understat.com/league/{league}/{season}`
- Extract per match: xG home, xG away, npxG home, npxG away
- Leagues: EPL, Bundesliga, La Liga, Serie A, Ligue 1
- Use requests + BeautifulSoup, parse embedded JSON in HTML
- Delay 2 seconds between requests, rotate User-Agent

### data/sources/transfermarkt.py

- Scrape injury/suspension lists per team
- Extract: player name, position, injury type, expected return date
- Extract: squad market values (top 11 average as depth proxy)
- Respectful scraping: delay 3s, proper headers, robots.txt check

### data/sources/weather.py

- API: `https://api.openweathermap.org/data/2.5/forecast`
- For each upcoming match: get weather at stadium city at kickoff time
- Extract: wind_speed_kmh, precipitation_mm, temperature_c
- Cache results — don't re-fetch same city/time twice
- If API fails: return None, downstream code handles gracefully

### data/sources/nba_api.py

- Base URL: `https://api.balldontlie.io/v1`
- Fetch: team stats, game schedule, player injury report
- Calculate per team: 
  - Days rest since last game
  - Back-to-back flag (game yesterday?)
  - Home/away record last 10
  - Offensive rating, defensive rating, pace (rolling 10 games)

### data/sources/news_rss.py

- Fetch RSS feeds:
  - `https://feeds.bbci.co.uk/sport/football/rss.xml`
  - `https://www.skysports.com/rss/12040`
  - `https://www.espn.com/espn/rss/nba/news`
  - `https://www.kicker.de/news/fussball/bundesliga/rss.xml`
- Keywords that LOWER confidence score (injury/doubt signals):
  ```python
  NEGATIVE_KEYWORDS = [
      "verletzt", "fehlt", "fällt aus", "fraglich",
      "injured", "out", "doubt", "suspended", "ban",
      "injury concern", "rotation expected", "rested",
      "geschont", "gesperrt"
  ]
  ```
- Keywords that RAISE confidence score:
  ```python
  POSITIVE_KEYWORDS = [
      "fit again", "returns", "available", "full squad",
      "zurück", "wieder fit", "vollständig"
  ]
  ```
- Output per match: list of relevant snippets + net_sentiment_modifier (-0.10 to +0.05)

**QA checks Batch 2:**
- [ ] All sources return None gracefully on failure (no crashes)?
- [ ] Rate limiting actually works — test with mock that counts calls?
- [ ] Normalized DataFrames have consistent column names across sources?
- [ ] Scraping includes delays and proper headers?
- [ ] News sentiment modifier stays within bounds?

---

## BATCH 3 — Feature Engineering

**BUILDER implements. QA reviews before Batch 4.**

### features/form.py

Weighted form window exactly as configured:

```python
def calculate_team_form(team_id, matches_df, weights_config) -> dict:
    """
    Returns:
    {
        weighted_form_score: float (0-100),
        last5_ppg: float,           # points per game last 5
        last20_ppg: float,
        season_ppg: float,
        goals_scored_weighted: float,
        goals_conceded_weighted: float,
        home_form_score: float,
        away_form_score: float,
        form_trajectory: float      # positive = improving, negative = declining
    }
    """
    # Apply weights from config:
    # last 5 games = 0.35
    # last 10-20 games = 0.25
    # full season = 0.15
    # last 2-3 seasons = 0.15
    # h2h = 0.10
    # Adjust for opponent quality using ELO ratings
```

### features/xg_features.py

```python
def calculate_xg_features(team_id, matches_with_xg) -> dict:
    """
    Returns:
    {
        xg_scored_rolling10: float,
        xg_conceded_rolling10: float,
        xg_difference: float,
        xg_overperformance: float,   # actual_goals - xg (regression signal)
        xg_trend: float,             # improving or declining
        npxg_scored: float           # non-penalty xG
    }
    """
    # Key insight: teams consistently scoring > their xG are 
    # likely to regress. Teams scoring < xG are "due".
    # This is a real predictive signal — implement carefully.
```

### features/context.py

```python
def calculate_context_features(match, team_schedule) -> dict:
    """
    Returns:
    {
        days_since_last_game_home: int,
        days_since_last_game_away: int,
        fixture_congestion_home: float,  # games in last 21 days
        fixture_congestion_away: float,
        travel_distance_km: float,       # away team travel
        match_importance_score: float,   # relegation/title pressure
        intl_players_returning_home: int,
        intl_players_returning_away: int
    }
    """
    # Fixture congestion: measurable 8-12% performance drop
    # Travel distance: use geopy to calculate
    # Match importance: based on table position + points to safety/title
```

### features/squad.py

```python
def calculate_squad_features(team_id, injury_data, market_values) -> dict:
    """
    Returns:
    {
        injured_players_count: int,
        key_players_missing: bool,       # True if top-3 by value missing
        squad_depth_score: float,        # market value distribution
        available_squad_value: float,    # value of available players
        injury_impact_score: float       # 0-1, how much injuries hurt
    }
    """
    # key_players_missing: if any player in top 20% market value is out
    # This is a binary flag that strongly modifies predictions
```

### features/referee.py

```python
def get_referee_modifier(referee_name, league, db) -> dict:
    """
    Looks up referee in referee_stats table.
    Returns modifier dict:
    {
        goals_modifier: float,    # e.g. +0.3 means this ref has 0.3 more goals/game
        cards_modifier: float,
        penalty_modifier: float,
        home_bias: float,
        confidence: float         # based on sample_size
    }
    # If referee unknown: return all zeros, log for future tracking
    # After each game: update referee stats automatically
    """
```

### features/odds_movement.py

```python
def calculate_movement_features(match_id, db, hours_lookback=48) -> dict:
    """
    Reads odds_snapshots from DB for this match.
    Returns:
    {
        opening_odds: float,
        current_odds: float,
        movement_direction: str,     # 'shortening' or 'drifting'
        movement_magnitude: float,   # % change
        bookmaker_consensus: float,  # % of books moving same direction
        sharp_money_signal: float    # -1 to +1
    }
    """
    # Sharp money signal:
    # If > 70% of bookmakers move in same direction = sharp money
    # Sharp money moving ON a selection = positive signal
    # Sharp money moving AGAINST = strong negative signal
```

### features/builder.py

Master coordinator that:
1. Calls all feature modules
2. Handles missing data gracefully (None values become 0 or are flagged)
3. Returns single feature vector per match per team
4. Logs which features were available vs. imputed

**QA checks Batch 3:**
- [ ] All feature functions return consistent dict structure?
- [ ] No feature function crashes if input data is partially missing?
- [ ] xG overperformance calculation is mathematically correct?
- [ ] Referee modifier returns neutral values for unknown referees?
- [ ] Odds movement correctly identifies sharp money direction?

---

## BATCH 4 — Models & Prediction

**BUILDER implements. QA reviews before Batch 5.**

### models/poisson_model.py

Implement Dixon-Coles Poisson model:

```python
class PoissonModel:
    def fit(self, historical_matches: pd.DataFrame):
        """
        Fit attack/defense strengths per team.
        Use Dixon-Coles correction for low-scoring results (0-0, 1-0, 0-1).
        Include home advantage parameter.
        Weight recent matches higher using time-decay factor.
        """

    def predict_match(self, home_team, away_team) -> dict:
        """
        Returns:
        {
            home_win_prob: float,
            draw_prob: float,
            away_win_prob: float,
            over_25_prob: float,
            under_25_prob: float,
            over_35_prob: float,
            under_35_prob: float,
            btts_prob: float,       # both teams to score
            predicted_home_goals: float,
            predicted_away_goals: float
        }
        """
```

### models/elo_model.py

```python
class EloModel:
    K_FACTOR = 32
    HOME_ADVANTAGE = 100  # ELO points

    def update_ratings(self, match_result):
        """Update ELO after each match."""

    def predict_match(self, home_team, away_team) -> dict:
        """Returns win/draw/loss probabilities."""

    def get_rating(self, team) -> float:
        """Returns current ELO, default 1500 for new teams."""
```

### models/xgboost_model.py

```python
class XGBoostModel:
    def train(self, features_df: pd.DataFrame, targets: pd.Series):
        """
        CRITICAL: Use TimeSeriesSplit for cross-validation.
        NEVER random split — this leaks future data.
        Features: all outputs from features/builder.py
        Targets: over_25 (binary), match_outcome (3-class)
        """

    def predict(self, features: dict) -> dict:
        """Returns probabilities for each market."""
```

### models/ensemble.py

```python
class EnsembleModel:
    def predict(self, home_team, away_team, features) -> dict:
        """
        1. Get prediction from each model
        2. Check agreement: need >= 2/3 models to agree direction
        3. Weighted average of probabilities
        4. Calculate confidence_score (0-100):
           - Based on model agreement
           - Based on feature completeness
           - Based on data recency
        5. Apply news sentiment modifier from features
        Returns final probability per market + confidence_score
        """

    def _models_agree(self, predictions: list) -> bool:
        """At least 2 of 3 models predict same outcome."""
```

### models/retrainer.py

```python
class ModelRetrainer:
    def retrain_weekly(self):
        """
        Runs every Sunday at 03:00.
        1. Load all settled bets from DB
        2. Calculate which features were most predictive
        3. Retrain XGBoost on fresh data
        4. Update Poisson team strengths
        5. Log retraining results to Model Health sheet
        6. Only deploy new model if it beats old model on holdout set
        """
```

**QA checks Batch 4:**
- [ ] Poisson probabilities sum to 1.0 (within float tolerance)?
- [ ] XGBoost uses TimeSeriesSplit — verify no data leakage?
- [ ] Ensemble handles case where one model fails/returns None?
- [ ] Retrainer has safeguard — doesn't deploy worse model?
- [ ] confidence_score is always 0-100, never outside range?

---

## BATCH 5 — Selection Engine

**BUILDER implements. QA reviews before Batch 6.**

### selection/ev_calculator.py

```python
def calculate_ev(our_probability: float, decimal_odds: float) -> float:
    """
    EV = (our_probability * decimal_odds) - 1
    Example: P=0.55, Odds=2.10 → (0.55 * 2.10) - 1 = +0.155 = +15.5%
    Returns float, e.g. 0.073 for +7.3%
    """

def find_best_odds(match_id, market, odds_api_data) -> tuple[float, str]:
    """
    Line shopping: scan all bookmakers for this market.
    Returns (best_odds, bookmaker_name).
    This alone improves ROI by 1-2% over using single bookmaker.
    """

def calculate_clv(bet_odds: float, closing_odds: float) -> float:
    """
    Closing Line Value: were our odds better than closing line?
    Positive CLV = we found value. This is the best model health metric.
    CLV = (bet_odds / closing_odds) - 1
    """
```

### selection/kelly.py

```python
def kelly_stake(probability: float, decimal_odds: float, 
                bankroll: float, config: dict) -> float:
    """
    Full Kelly = (bp - q) / b
    b = decimal_odds - 1
    p = our_probability  
    q = 1 - p

    Apply fraction from config (0.25 = Quarter Kelly).
    Cap at max_stake_percent of bankroll.
    Return 0 if Kelly is negative (no edge).
    """
```

### selection/filter.py

```python
def select_daily_bets(all_predictions: list, config: dict, 
                      week_watchable_count: int) -> list:
    """
    LAYER 1 — MANDATORY (profit always comes first):
    Filter: EV >= min_ev_threshold (3%)
    If fewer than 10 qualify: return fewer. Never fill with weak bets.

    LAYER 2 — RANKING:
    Sort by EV descending. Take top 10.

    LAYER 3 — TIEBREAKER (only when EV within 0.5% of each other):
    Prefer leagues in priority order from config.

    LAYER 4 — WATCHABLE FLAG:
    Check if any bet in preferred leagues (PL, Bundesliga, NBA)
    qualifies this week. Conditions ALL must be true:
    - EV >= watchable_ev_threshold (2.4%)
    - Kickoff between 17:00-23:00 Austrian time
    - week_watchable_count < watchable_max_per_week

    LAYER 5 — FAVORITE CLUB OVERRIDE:
    If Arsenal or Bristol Rovers playing:
    - Only include if EV >= min_ev_threshold * ev_override_factor (2.4%)
    - Tag with is_favorite_club=True
    - Never include if EV below 2.4% regardless

    Returns sorted list of up to 10 bet dicts.
    """
```

**QA checks Batch 5:**
- [ ] EV formula is mathematically correct — test with known values?
- [ ] Kelly never returns negative stake?
- [ ] Filter never returns more than 10 bets?
- [ ] Watchable logic correctly checks week_watchable_count?
- [ ] Favorite club override respects the 80% threshold hard limit?

---

## BATCH 6 — Tracking & Notifications

**BUILDER implements. QA reviews before Batch 7.**

### tracking/sheets.py

Auto-create Google Sheet with 3 tabs if it doesn't exist.

**Tab "Bets" columns (auto-written after each prediction run):**
```
Date | Match | League | Market | Our P% | Bookie Odds | Bookmaker 
| EV% | Confidence | Stake € | Placed? | Result | P&L | Cum P&L 
| CLV | Status Emoji
```

Status Emoji logic:
- 🟢 if bet won
- 🔴 if bet lost  
- ⏳ if pending
- ⏭️ if skipped

**Tab "Analytics" (auto-calculated, refresh weekly):**
```
ROI Overall | ROI Last 20 Bets | ROI by League | ROI by Market
Win Rate | Avg Confidence (wins) | Avg Confidence (losses)
Best Day of Week | Worst Day of Week | Max Drawdown
Model Calibration: [confidence bracket → actual win rate table]
CLV Average (target: > 0)
```

**Tab "Model Health":**
```
System Status: 🟢 ROI>3% | 🟡 ROI 0-3% | 🔴 ROI<0%
Last Retrain Date | Retrain Improvement | Next Retrain
Paper Mode Progress: X/200 bets | Ready for Live: YES/NO
Total Simulated P&L | Days Running
```

### notifications/morning_briefing.py

Exact message format:

```
🎯 BETTING BRIEFING — {weekday}, {date}

━━━━━━━━━━━━━━━━━━━━━
{for each bet:}
{N}. {home_team} vs {away_team}
   📊 {league} | {market}
   💰 Quote: {odds} @ {bookmaker}
   📈 EV: +{ev}% | Confidence: {confidence}
   💵 Stake: €{stake} ({stake_pct}% Bankroll)
   {⭐ Dein Klub | 🎯 Watchable | nothing}
━━━━━━━━━━━━━━━━━━━━━

📊 Heute: {n} Bets | Ø EV: +{avg_ev}%
💰 Sim. Bankroll: €{bankroll} ({pct_change:+.1f}%)
📋 Paper Mode: {settled_bets}/200

Tippe /placed N oder /skip N
```

### notifications/evening_summary.py

```
📋 TAGES-ABSCHLUSS — {weekday}, {date}

{for each settled bet:}
{✅|❌} Bet {N}: {match} {market} {✓|✗} {pnl:+.2f}€
{for pending:}
⏳ Bet {N}: {match} — läuft noch

📊 Heute: {won}W {lost}L | Win Rate: {rate}%
💵 Tages-P&L: {daily_pnl:+.2f}€
📈 Gesamt: {cum_pnl:+.2f}€ ({roi:+.1f}% ROI)
🎯 CLV heute: {clv:+.2f}% {✓ if positive}
```

### notifications/weekly_report.py

```
📊 WOCHEN-REPORT KW {week}

Bets: {total} | Won: {won} ({win_rate:.1f}%)
ROI: {roi:+.1f}% | P&L: {pnl:+.2f}€
Max Drawdown: {drawdown}€
Ø Confidence: {avg_conf}

🏆 Beste Liga: {best_league} ({best_roi:+.1f}% ROI)
⚠️ Schwächste: {worst_league} ({worst_roi:+.1f}% ROI)

Modell-Empfehlung: {auto_generated_insight}

Retraining: ✅ Sonntag 03:00 Uhr
Paper Mode: {settled}/200 | {remaining} bis Live möglich
```

### notifications/telegram_bot.py

Handle these commands:
- `/placed N` — mark bet N as placed in paper mode
- `/skip N` — mark bet N as skipped
- `/status` — current bankroll, ROI, bets today
- `/bets` — show today's bets again
- `/week` — show this week's performance
- `/help` — list commands

**QA checks Batch 6:**
- [ ] Sheets API writes don't fail silently — log all errors?
- [ ] Morning briefing handles 0 bets available gracefully?
- [ ] All Telegram commands have error handling?
- [ ] CLV calculation requires closing odds — handled if unavailable?
- [ ] Weekly report generates sensible insight even with little data?

---

## BATCH 7 — Main Orchestrator & Scheduler

**BUILDER implements. QA does final review.**

### main.py

```python
"""
Entry points via CLI arguments:
  --setup          First-time setup: DB, Sheets, verify API keys
  --collect        Fetch all data sources
  --predict        Run prediction engine, generate daily bets
  --brief          Send morning briefing via Telegram
  --odds_snapshot  Save current odds snapshot to DB
  --summarize      Send evening summary via Telegram  
  --retrain        Retrain all models
  --weekly_report  Send weekly report
  --test           Run all tests
  --paper_status   Print current paper mode statistics
"""
```

### scheduler/cron_setup.py

Script that installs cron jobs automatically when `--setup` is run:

```
0 6 * * *    python /path/to/main.py --collect
30 7 * * *   python /path/to/main.py --predict
30 8 * * *   python /path/to/main.py --brief
0 */6 * * *  python /path/to/main.py --odds_snapshot
0 23 * * *   python /path/to/main.py --summarize
0 3 * * 0    python /path/to/main.py --retrain
0 20 * * 0   python /path/to/main.py --weekly_report
```

### requirements.txt

```
requests==2.31.0
beautifulsoup4==4.12.2
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
xgboost==2.0.0
scipy==1.11.0
python-telegram-bot==20.6
google-api-python-client==2.100.0
google-auth==2.23.0
gspread==5.11.0
openmeteo-requests==1.1.0
geopy==2.4.0
feedparser==6.0.10
python-dotenv==1.0.0
pyyaml==6.0.1
schedule==1.2.1
loguru==0.7.2
pytest==7.4.0
```

---

## QA — Final Validation Checklist

Before marking project complete, QA runs these checks:

```
INFRASTRUCTURE
[ ] python main.py --test → all tests pass
[ ] python main.py --setup → no errors, DB created, sheets accessible
[ ] .env.example has all required keys documented

DATA PIPELINE  
[ ] Dry run --collect completes without crashes
[ ] All sources return data or fail gracefully with logged warning
[ ] Quality check correctly flags insufficient data

PREDICTION ENGINE
[ ] Poisson probabilities sum to 1.0 for test match
[ ] XGBoost trains without data leakage (TimeSeriesSplit verified)
[ ] Ensemble returns None when models disagree — no forced bet
[ ] EV calculation correct: P=0.55 odds=2.10 → EV=+0.155

SELECTION LOGIC
[ ] With EV threshold 3%: only bets above 3% pass
[ ] With 15 qualifying bets: only top 10 by EV selected
[ ] With 3 qualifying bets: returns 3, not padded to 10
[ ] Watchable flag only triggers with correct conditions
[ ] Favorite club override respects 80% threshold

NOTIFICATIONS
[ ] Morning briefing sends successfully to Telegram
[ ] /placed and /skip commands update DB correctly
[ ] Evening summary calculates correct P&L
[ ] Google Sheet updates after /placed command

PAPER MODE
[ ] Bankroll starts at configured amount
[ ] P&L tracked correctly across multiple days
[ ] ROI calculation: (total_pnl / total_staked) * 100
[ ] System correctly reports when 200 bets milestone reached
```

---

## What Felix Does (Only This)

1. Open terminal in empty folder
2. Create file `CLAUDE.md` and paste this entire document
3. Run Claude Code and say: **"Execute the CLAUDE.md build plan completely."**
4. When Claude Code finishes, fill in `.env` with API keys:
   - `FOOTBALL_API_KEY` → free at football-data.org
   - `ODDS_API_KEY` → free at the-odds-api.com  
   - `WEATHER_API_KEY` → free at openweathermap.org
   - `TELEGRAM_BOT_TOKEN` → create bot via @BotFather on Telegram
   - `TELEGRAM_CHAT_ID` → message @userinfobot on Telegram
   - `GOOGLE_SHEET_ID` → create empty Sheet, copy ID from URL
   - Place `google_credentials.json` in project root (Google Cloud Console)
5. Run: `python main.py --setup`
6. Paper mode starts automatically next morning at 08:30

**Total setup time after Claude Code builds: ~20 minutes (API keys only)**

---

## If Claude Code Runs Low on Context

Priority order — finish in this sequence, stop cleanly after each:

1. Batch 1 (DB + Config) — absolute minimum to function
2. Batch 5 (EV Calculator + Filter) — core logic
3. Batch 2 (Odds API + Football API only) — minimum data
4. Batch 6 (Telegram morning briefing) — usable output
5. Batch 4 (Models — Poisson first) — predictions
6. Batch 3 (Features — form.py first) — feature engineering
7. Batch 6 remainder (Sheets, evening summary)
8. Batch 7 (Main + Scheduler)
9. Everything else

