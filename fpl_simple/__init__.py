"""Simple utilities for selecting Fantasy Premier League squads."""

from .client import FPLClient
from .selection import select_best_squad, PositionSelection
from .analysis import select_best_xi, select_budget_dream_xi

__all__ = [
    "FPLClient",
    "select_best_squad",
    "PositionSelection",
    "select_best_xi",
    "select_budget_dream_xi",
]
