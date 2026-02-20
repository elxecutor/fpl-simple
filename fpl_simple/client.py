from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict

import requests

from .models import Player, Fixture

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

POSITION_NAMES = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


class FPLClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawPlayer:
    id: int
    web_name: str
    first_name: str
    second_name: str
    team: int
    element_type: int
    total_points: int
    ict_index: float
    now_cost: int
    form: float
    points_per_game: float
    chance_of_playing_next_round: int | None
    minutes: int
    expected_points_next: float = 0.0
    selected_by_percent: float = 0.0
    cost_change_event: int = 0  # Price change this GW (+1, -1, 0)

    @classmethod
    def from_api(cls, payload: dict) -> "RawPlayer":
        ict_raw = payload.get("ict_index") or 0
        try:
            ict_index = float(ict_raw)
        except (TypeError, ValueError):
            ict_index = 0.0
        
        try:
            form = float(payload.get("form") or 0.0)
        except (TypeError, ValueError):
            form = 0.0
            
        try:
            ppg = float(payload.get("points_per_game") or 0.0)
        except (TypeError, ValueError):
            ppg = 0.0

        try:
            ep_next = float(payload.get("ep_next") or 0.0)
        except (TypeError, ValueError):
            ep_next = 0.0

        try:
            selected_by = float(payload.get("selected_by_percent") or 0.0)
        except (TypeError, ValueError):
            selected_by = 0.0

        return cls(
            id=payload["id"],
            web_name=payload.get("web_name", ""),
            first_name=payload.get("first_name", ""),
            second_name=payload.get("second_name", ""),
            team=payload.get("team", 0),
            element_type=payload.get("element_type", 0),
            total_points=int(payload.get("total_points", 0)),
            ict_index=ict_index,
            now_cost=int(payload.get("now_cost", 0)),
            form=form,
            points_per_game=ppg,
            chance_of_playing_next_round=payload.get("chance_of_playing_next_round"),
            minutes=int(payload.get("minutes", 0)),
            expected_points_next=ep_next,
            selected_by_percent=selected_by,
            cost_change_event=int(payload.get("cost_change_event", 0)),
        )


class FPLClient:
    """Small helper around FPL's public bootstrap endpoint."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch_fixtures(self) -> List[Fixture]:
        response = self._session.get(FPL_FIXTURES_URL, timeout=20)
        if response.status_code != 200:
            raise FPLClientError(f"Failed to fetch fixtures (status {response.status_code})")
        
        try:
            data = response.json()
        except ValueError as exc:
            raise FPLClientError("Failed to parse JSON from FPL Fixtures API") from exc
            
        fixtures = []
        for f in data:
            fixtures.append(Fixture(
                id=f["id"],
                event=f.get("event"),
                team_h=f["team_h"],
                team_a=f["team_a"],
                team_h_difficulty=f.get("team_h_difficulty", 3),
                team_a_difficulty=f.get("team_a_difficulty", 3),
                kickoff_time=f.get("kickoff_time", ""),
                finished=f.get("finished", False)
            ))
        return fixtures

    def fetch_players(self) -> List[Player]:
        payload = self._get_bootstrap()
        teams = {team["id"]: team.get("name", "?") for team in payload.get("teams", [])}
        raw_players: Iterable[RawPlayer] = (
            RawPlayer.from_api(player) for player in payload.get("elements", [])
        )
        
        # Fetch fixtures to enrich players
        try:
            all_fixtures = self.fetch_fixtures()
            # Filter for future fixtures
            future_fixtures = [f for f in all_fixtures if not f.finished and f.event is not None]
        except FPLClientError:
            future_fixtures = []
            print("Warning: Could not fetch fixtures, difficulty scoring will be disabled.")

        players: List[Player] = []
        for rp in raw_players:
            position_name = POSITION_NAMES.get(rp.element_type, "?")
            
            # Find upcoming fixtures for this player's team
            player_fixtures = []
            for f in future_fixtures:
                if f.team_h == rp.team or f.team_a == rp.team:
                    player_fixtures.append(f)
            
            # Sort by kickoff time/event
            player_fixtures.sort(key=lambda x: (x.event if x.event else 999, x.kickoff_time))

            players.append(
                Player(
                    id=rp.id,
                    web_name=rp.web_name,
                    first_name=rp.first_name,
                    second_name=rp.second_name,
                    team_id=rp.team,
                    team_name=teams.get(rp.team, "Unknown"),
                    element_type=rp.element_type,
                    position_name=position_name,
                    total_points=rp.total_points,
                    ict_index=rp.ict_index,
                    now_cost=rp.now_cost,
                    form=rp.form,
                    points_per_game=rp.points_per_game,
                    chance_of_playing_next_round=rp.chance_of_playing_next_round,
                    minutes=rp.minutes,
                    expected_points_next=rp.expected_points_next,
                    selected_by_percent=rp.selected_by_percent,
                    cost_change_event=rp.cost_change_event,
                    upcoming_fixtures=player_fixtures
                )
            )
        return players

    def _get_bootstrap(self) -> dict:
        response = self._session.get(FPL_BOOTSTRAP_URL, timeout=20)
        if response.status_code != 200:
            raise FPLClientError(
                f"Failed to fetch bootstrap data (status {response.status_code})"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise FPLClientError("Failed to parse JSON from FPL API") from exc
