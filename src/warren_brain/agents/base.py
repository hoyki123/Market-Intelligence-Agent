"""Abstract base class for all Warren Brain agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState


class BaseAgent(ABC):
    """
    Each agent receives the full shared state and returns a dict
    that gets merged back into the relevant state field.
    """

    name: str = "BaseAgent"

    @abstractmethod
    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        """
        Run analysis for the given ticker.
        Returns a dict that will be stored in state under this agent's key.
        Should never raise — return {"error": str(e)} on failure.
        """
        ...

    def _score(self, value: float | None, low: float, high: float) -> float:
        """Normalize value to [0, 1] range given expected low/high bounds."""
        if value is None:
            return 0.5
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def _safe_divide(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
