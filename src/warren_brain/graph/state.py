"""LangGraph shared state schema for Warren Brain."""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class WarrenBrainState(TypedDict, total=False):
    """
    Shared state that flows through the agent graph.
    Each agent reads existing state and writes to its designated key.
    """

    # Input
    ticker: str

    # Agent outputs (each agent writes to its own key)
    fundamentals: dict[str, Any]
    technicals: dict[str, Any]
    sentiment: dict[str, Any]
    thirteen_f: dict[str, Any]
    ontology: dict[str, Any]
    risk: dict[str, Any]
    buffett_brain: dict[str, Any]

    # Final output from PortfolioManagerAgent
    recommendation: dict[str, Any]

    # LangGraph message history (optional — for conversational interface)
    messages: Annotated[list[BaseMessage], add_messages]

    # Metadata
    errors: list[str]
