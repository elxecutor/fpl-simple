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
    next_global_event: int | None = None  # Global next gameweek for accurate BGW detection

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
        """Count fixtures in next GW (2 = DGW, 0 = BGW, 1 = normal).
        Compares against the global next event to accurately detect blanks.
        """
        # Compare against the global next event to catch teams that skip a gameweek
        if not self.upcoming_fixtures or not self.next_global_event:
            return 0
        return sum(1 for f in self.upcoming_fixtures if f.event == self.next_global_event)

    @property
    def composite_score(self) -> float:
        """
        Pure Double-Digit Ceiling Formula:
        Explicitly targets players with explosive underlying stats and punishes blankers.
        """
        # 1. Per-game Threat and Creativity
        ict_per_game = self.ict_index / max(1.0, self.appearances)
        
        # 2. Exponential Form Multiplier: 
        # If a player is blanking (form < 2), they get crushed. 
        # If they are hauling (form > 6), their score skyrockets.
        form_factor = max(0, self.form) ** 1.5 
        
        # 3. Base calculation favoring xP and Explosiveness over season-long safety
        base = (ict_per_game * 5.0) + (form_factor * 3.0) + (self.expected_points_next * 4.0)
        
        # 4. Positional Bias for Hauls
        # Goalscoring midfielders and forwards are where double digits live. 
        # Goalkeepers and steady defenders are nerfed so they don't clog the recommendations.
        position_mult = {"GKP": 0.5, "DEF": 0.9, "MID": 1.5, "FWD": 1.4}.get(self.position_name, 1.0)
        
        return base * position_mult

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
