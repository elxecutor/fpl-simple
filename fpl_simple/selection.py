from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

from .models import Player, MIN_MINUTES_THRESHOLD, MIN_APPEARANCES_THRESHOLD
from .analysis import calculate_adjusted_score


@dataclass(frozen=True)
class PositionSelection:
    position: str
    element_type: int
    count: int


DEFAULT_REQUIREMENTS: tuple[PositionSelection, ...] = (
    PositionSelection("GKP", 1, 10),
    PositionSelection("DEF", 2, 10),
    PositionSelection("MID", 3, 10),
    PositionSelection("FWD", 4, 10),
)


def select_best_squad(
    players: Iterable[Player],
    requirements: Iterable[PositionSelection] = DEFAULT_REQUIREMENTS,
) -> Mapping[str, List[Player]]:
    """Return a position -> best players mapping prioritizing fixture difficulty."""
    from .analysis import calculate_fixture_difficulty_score

    players_by_position: dict[int, List[Player]] = {}
    for player in players:
        # Filter out unavailable players (Chance < 75%)
        chance = player.chance_of_playing_next_round
        if chance is not None and chance < 75:
            continue
        if player.average_minutes_per_appearance < MIN_MINUTES_THRESHOLD:
            continue
        if player.appearances < MIN_APPEARANCES_THRESHOLD:
            continue
            
        players_by_position.setdefault(player.element_type, []).append(player)

    # Score with doubled fixture difficulty weighting and fixture count consideration
    def get_squad_score(p: Player) -> float:
        from .analysis import calculate_fixture_difficulty_score, calculate_fixture_count_multiplier
        adj_score = calculate_adjusted_score(p)
        fix_mult = calculate_fixture_difficulty_score(p)
        # Triple fixture difficulty impact for better prioritization
        return adj_score * (1.0 + (fix_mult - 1.0) * 3.0)

    selection: dict[str, List[Player]] = {}
    for requirement in requirements:
        candidates = players_by_position.get(requirement.element_type, [])
        sorted_candidates = sorted(
            candidates,
            key=get_squad_score,
            reverse=True,
        )
        selection[requirement.position] = sorted_candidates[: requirement.count]
    return selection
