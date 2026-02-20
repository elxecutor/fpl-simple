from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


MIN_MINUTES_THRESHOLD = 60  # Minimum average minutes per match to consider a player reliable
MIN_APPEARANCES_THRESHOLD = 5  # Require at least this many appearances before trusting projections


@dataclass(frozen=True)
class Fixture:
    id: int
    event: int | None  # Gameweek number
    team_h: int
    team_a: int
    team_h_difficulty: int
    team_a_difficulty: int
    kickoff_time: str
    finished: bool


@dataclass(frozen=True)
class Player:
    """Represents the subset of player data we care about."""

    id: int
    web_name: str
    first_name: str
    second_name: str
    team_id: int
    team_name: str
    element_type: int
    position_name: str
    total_points: int
    ict_index: float
    now_cost: int
    form: float
    points_per_game: float
    chance_of_playing_next_round: int | None = None
    minutes: int = 0
    expected_points_next: float = 0.0
    selected_by_percent: float = 0.0
    cost_change_event: int = 0

    # Enriched data
    upcoming_fixtures: List[Fixture] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.second_name}".strip()

    @property
    def ownership_str(self) -> str:
        return f"{self.selected_by_percent:.1f}%"

    @property
    def price_trend(self) -> str:
        if self.cost_change_event > 0:
            return "↑"
        elif self.cost_change_event < 0:
            return "↓"
        return ""

    @property
    def is_differential(self) -> bool:
        """Low ownership (<10%) player with decent potential."""
        return self.selected_by_percent < 10.0

    @property
    def price_momentum(self) -> float:
        """Price trend indicator: +1 rising, -1 falling, 0 stable."""
        if self.cost_change_event > 0:
            return 1.0
        elif self.cost_change_event < 0:
            return -1.0
        return 0.0

    @property
    def dgw_fixture_count(self) -> int:
        """Count fixtures in next GW (2 = DGW, 0 = BGW, 1 = normal)."""
        if not self.upcoming_fixtures:
            return 0
        # Check fixtures in the first event
        first_event = self.upcoming_fixtures[0].event if self.upcoming_fixtures else None
        if first_event is None:
            return 0
        return sum(1 for f in self.upcoming_fixtures if f.event == first_event)

    @property
    def composite_score(self) -> float:
        """
        Improved scoring formula:
        - Uses xP (expected points) as anchor when available
        - Falls back to PPG * form blend for reliability
        - ICT used as tiebreaker, not primary driver
        - Position multiplier for attacking potential
        """
        # Base: blend of xP and historical PPG
        xp = self.expected_points_next
        ppg = self.points_per_game
        
        if xp > 0 and ppg > 0:
            # Trust xP more for immediate, PPG for consistency
            base = (xp * 0.6 + ppg * 0.4) * 10  # Scale up for visibility
        elif xp > 0:
            base = xp * 10
        elif ppg > 0:
            base = ppg * 10
        else:
            base = 0.0
        
        # ICT as secondary factor (10% weight) - rewards underlying performance
        ict_bonus = self.ict_index * 0.1
        
        # Position multiplier - attackers score more points
        position_mult = {"GKP": 0.85, "DEF": 1.0, "MID": 1.1, "FWD": 1.15}.get(self.position_name, 1.0)
        
        return (base + ict_bonus) * position_mult

    @property
    def cost_str(self) -> str:
        return f"{self.now_cost / 10:.1f}m"

    @property
    def selling_price(self) -> int:
        """Selling price is typically 0.1m less than current price in FPL."""
        return max(0, self.now_cost - 1)

    @property
    def selling_cost_str(self) -> str:
        return f"{self.selling_price / 10:.1f}m"

    @property
    def appearances(self) -> float:
        """Return estimated appearances using total points and points per game."""
        if self.points_per_game > 0:
            return self.total_points / self.points_per_game
        return 0.0

    @property
    def average_minutes_per_appearance(self) -> float:
        apps = self.appearances
        if apps <= 0:
            return 0.0
        return self.minutes / apps
