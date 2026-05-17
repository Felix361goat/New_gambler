import math
import logging
from typing import Optional
import numpy as np
from scipy.optimize import minimize
import pandas as pd

logger = logging.getLogger(__name__)


class PoissonModel:
    def __init__(self):
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_advantage = 0.25
        self.fitted = False

    def _dc_correction(self, home_goals: int, away_goals: int, mu_h: float, mu_a: float, rho: float) -> float:
        if home_goals == 0 and away_goals == 0:
            return 1 - mu_h * mu_a * rho
        if home_goals == 1 and away_goals == 0:
            return 1 + mu_a * rho
        if home_goals == 0 and away_goals == 1:
            return 1 + mu_h * rho
        if home_goals == 1 and away_goals == 1:
            return 1 - rho
        return 1.0

    def fit(self, historical_matches: pd.DataFrame):
        if historical_matches is None or historical_matches.empty:
            logger.warning("No data to fit PoissonModel")
            return

        finished = historical_matches[
            historical_matches.get("status", pd.Series(["FINISHED"])).isin(
                ["FINISHED", "FT", "finished"]
            )
        ].copy() if "status" in historical_matches.columns else historical_matches.copy()

        finished = finished.dropna(subset=["home_goals", "away_goals"])
        if len(finished) < 10:
            logger.warning(f"Only {len(finished)} finished matches — using defaults")
            return

        # Time decay: more recent = higher weight
        if "date" in finished.columns:
            finished = finished.sort_values("date")
            max_date = finished["date"].max()
            finished["weight"] = finished["date"].apply(
                lambda d: math.exp(-0.003 * max(0, (pd.Timestamp(max_date) - pd.Timestamp(d)).days))
            )
        else:
            finished["weight"] = 1.0

        teams = pd.unique(pd.concat([finished["home_team"], finished["away_team"]]))
        team_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        def params_to_dicts(params):
            atk = {t: params[i] for t, i in team_idx.items()}
            dfc = {t: params[n_teams + i] for t, i in team_idx.items()}
            ha = params[2 * n_teams]
            rho = params[2 * n_teams + 1]
            return atk, dfc, ha, rho

        def neg_log_likelihood(params):
            atk, dfc, ha, rho = params_to_dicts(params)
            ll = 0.0
            for _, row in finished.iterrows():
                ht, at = row["home_team"], row["away_team"]
                hg, ag = int(row["home_goals"]), int(row["away_goals"])
                w = row.get("weight", 1.0)
                mu_h = math.exp(atk.get(ht, 0) - dfc.get(at, 0) + ha)
                mu_a = math.exp(atk.get(at, 0) - dfc.get(ht, 0))
                mu_h = max(mu_h, 1e-10)
                mu_a = max(mu_a, 1e-10)
                dc = self._dc_correction(hg, ag, mu_h, mu_a, rho)
                dc = max(dc, 1e-10)
                ll += w * (
                    hg * math.log(mu_h) - mu_h - sum(math.log(i+1) for i in range(hg))
                    + ag * math.log(mu_a) - mu_a - sum(math.log(i+1) for i in range(ag))
                    + math.log(dc)
                )
            return -ll

        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams] = 0.25  # home advantage
        x0[2 * n_teams + 1] = 0.0  # rho

        try:
            result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                              options={"maxiter": 200, "ftol": 1e-6})
            atk, dfc, ha, rho = params_to_dicts(result.x)
            self.attack = atk
            self.defense = dfc
            self.home_advantage = ha
            self._rho = rho
            self.fitted = True
            logger.info(f"Poisson fitted on {len(finished)} matches, {n_teams} teams")
        except Exception as e:
            logger.error(f"Poisson fit failed: {e}")

    def _predict_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        mu_h = math.exp(
            self.attack.get(home_team, 0) - self.defense.get(away_team, 0) + self.home_advantage
        )
        mu_a = math.exp(
            self.attack.get(away_team, 0) - self.defense.get(home_team, 0)
        )
        return max(mu_h, 0.1), max(mu_a, 0.1)

    def _poisson_pmf(self, k: int, mu: float) -> float:
        return math.exp(-mu) * (mu ** k) / math.factorial(k)

    def predict_match(self, home_team: str, away_team: str) -> dict:
        mu_h, mu_a = self._predict_goals(home_team, away_team)
        max_goals = 10

        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                dc = self._dc_correction(i, j, mu_h, mu_a, getattr(self, "_rho", 0.0))
                dc = max(dc, 0.0)
                prob_matrix[i, j] = self._poisson_pmf(i, mu_h) * self._poisson_pmf(j, mu_a) * dc

        # Normalize
        total = prob_matrix.sum()
        if total > 0:
            prob_matrix /= total

        home_win = float(np.sum(np.tril(prob_matrix, -1)))
        draw = float(np.sum(np.diag(prob_matrix)))
        away_win = float(np.sum(np.triu(prob_matrix, 1)))

        over25 = float(np.sum([prob_matrix[i, j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j > 2.5]))
        under25 = 1.0 - over25
        over35 = float(np.sum([prob_matrix[i, j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j > 3.5]))
        under35 = 1.0 - over35
        btts = float(np.sum([prob_matrix[i, j] for i in range(1, max_goals+1) for j in range(1, max_goals+1)]))

        return {
            "home_win_prob": round(home_win, 4),
            "draw_prob": round(draw, 4),
            "away_win_prob": round(away_win, 4),
            "over_25_prob": round(over25, 4),
            "under_25_prob": round(under25, 4),
            "over_35_prob": round(over35, 4),
            "under_35_prob": round(under35, 4),
            "btts_prob": round(btts, 4),
            "predicted_home_goals": round(mu_h, 3),
            "predicted_away_goals": round(mu_a, 3),
        }
