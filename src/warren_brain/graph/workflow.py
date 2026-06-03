"""LangGraph workflow — wires all agents into the Warren Brain pipeline.

Execution order:
  fundamentals ──┐
  technicals   ──┤ (parallel, no data dependencies)
  sentiment    ──┤
  thirteen_f   ──┤
  ontology     ──┘
       ↓
     risk          (depends on price history, independent of above agents)
       ↓
  buffett_brain    (reads all prior agent outputs via state)
       ↓
  portfolio_mgr    (aggregates all signals → final recommendation)
"""

from __future__ import annotations

from rich.console import Console

from langgraph.graph import StateGraph, START, END
from warren_brain.data.database import init_db
from warren_brain.data.repository import save_analysis
from warren_brain.guardrails import check_data_quality, validate_llm_output

from warren_brain.agents.fundamentals import FundamentalsAgent
from warren_brain.agents.technicals import TechnicalsAgent
from warren_brain.agents.sentiment import SentimentAgent
from warren_brain.agents.thirteen_f import ThirteenFAgent
from warren_brain.agents.ontology import OntologyAgent
from warren_brain.agents.risk import RiskAgent
from warren_brain.agents.buffett_brain import BuffettBrainAgent
from warren_brain.agents.portfolio_manager import PortfolioManagerAgent
from warren_brain.graph.state import WarrenBrainState

console = Console()

# Instantiate agents once (they hold API clients)
_fundamentals = FundamentalsAgent()
_technicals = TechnicalsAgent()
_sentiment = SentimentAgent()
_thirteen_f = ThirteenFAgent()
_ontology = OntologyAgent()
_risk = RiskAgent()
_buffett = BuffettBrainAgent()
_portfolio = PortfolioManagerAgent()


def _node(agent, key: str):
    """Wrap an agent into a LangGraph node function."""
    def node_fn(state: WarrenBrainState) -> dict:
        ticker = state["ticker"]
        console.print(f"  [dim]→ {agent.name}[/dim]")
        result = agent.analyze(ticker, state)
        if "error" in result:
            errors = list(state.get("errors") or [])
            errors.append(f"{agent.name}: {result['error']}")
            return {key: result, "errors": errors}
        return {key: result}
    node_fn.__name__ = key
    return node_fn


def build_graph() -> StateGraph:
    graph = StateGraph(WarrenBrainState)

    # Register nodes
    graph.add_node("fundamentals", _node(_fundamentals, "fundamentals"))
    graph.add_node("technicals", _node(_technicals, "technicals"))
    graph.add_node("sentiment", _node(_sentiment, "sentiment"))
    graph.add_node("thirteen_f", _node(_thirteen_f, "thirteen_f"))
    graph.add_node("ontology", _node(_ontology, "ontology"))
    graph.add_node("risk", _node(_risk, "risk"))
    graph.add_node("buffett_brain", _node(_buffett, "buffett_brain"))
    graph.add_node("portfolio_manager", _portfolio_manager_node)

    # Parallel first wave: all data-gathering agents run from START
    for agent_node in ("fundamentals", "technicals", "sentiment", "thirteen_f", "ontology", "risk"):
        graph.add_edge(START, agent_node)
        graph.add_edge(agent_node, "buffett_brain")

    # buffett_brain has a fan-in from 6 nodes — LangGraph handles this automatically
    # when multiple edges point to the same node (it waits for all upstream to complete)
    graph.add_edge("buffett_brain", "portfolio_manager")
    graph.add_edge("portfolio_manager", END)

    return graph.compile()


def _portfolio_manager_node(state: WarrenBrainState) -> dict:
    ticker = state["ticker"]
    console.print(f"  [dim]→ {_portfolio.name}[/dim]")
    result = _portfolio.analyze(ticker, state)
    return {"recommendation": result}


# Compiled graph singleton
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


TTL_ANALYSIS = 60 * 60  # 1 hour — reuse full pipeline result within a session


def run_analysis(ticker: str, persist: bool = True) -> dict:
    """
    Run the full Warren Brain pipeline for a single ticker.
    Automatically saves result to the database unless persist=False.
    Returns the completed state as a dict.
    """
    from warren_brain.data.cache import get_cache
    cache = get_cache()
    cache_key = f"analysis:{ticker.upper()}"
    if cached := cache.get(cache_key):
        console.print(f"  [dim]↩ {ticker.upper()} loaded from cache[/dim]")
        return cached

    init_db()  # no-op if tables already exist

    graph = get_graph()
    initial_state: WarrenBrainState = {
        "ticker": ticker.upper(),
        "errors": [],
        "messages": [],
    }

    final_state = graph.invoke(initial_state)

    rec = final_state.get("recommendation", {})
    current_price = (final_state.get("fundamentals") or {}).get("metrics", {}).get("current_price")
    rec, llm_warnings = validate_llm_output(rec, current_price)
    for w in llm_warnings:
        console.print(f"  [yellow]⚠ LLM output guardrail: {w}[/yellow]")

    result = {
        "ticker": ticker.upper(),
        "recommendation": rec,
        "fundamentals": final_state.get("fundamentals", {}),
        "technicals": final_state.get("technicals", {}),
        "sentiment": final_state.get("sentiment", {}),
        "thirteen_f": final_state.get("thirteen_f", {}),
        "ontology": final_state.get("ontology", {}),
        "risk": final_state.get("risk", {}),
        "buffett_brain": final_state.get("buffett_brain", {}),
        "errors": final_state.get("errors", []),
    }

    quality, dq_warnings = check_data_quality(result)
    for w in dq_warnings:
        console.print(f"  [yellow]⚠ Data quality guardrail: {w}[/yellow]")

    cache.set(cache_key, result, ttl=TTL_ANALYSIS)

    if persist:
        try:
            analysis_id = save_analysis(result)
            result["db_id"] = analysis_id
            console.print(f"  [dim]✓ Saved to DB (id={analysis_id})[/dim]")
        except Exception as e:
            console.print(f"  [yellow]⚠ DB save failed: {e}[/yellow]")

    return result
