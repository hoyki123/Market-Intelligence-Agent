# Warren Brain Analysis Skill

Triggered when: user asks to analyze a stock, add a new agent, modify scoring weights, or debug why a ticker got a specific recommendation.

## Key context to load
- `src/warren_brain/agents/base.py` — BaseAgent contract
- `src/warren_brain/graph/state.py` — WarrenBrainState fields
- `src/warren_brain/graph/workflow.py` — graph topology and run_analysis()
- `src/warren_brain/config.py` — all settings and weight fields

## Score debugging pattern
```python
from warren_brain.graph.workflow import run_analysis
result = run_analysis("TICKER")
print(result["recommendation"]["agent_scores"])
print(result["fundamentals"]["summary"])
print(result["buffett_brain"]["concerns"])
```

## Adding a new agent checklist
1. Create `src/warren_brain/agents/new_agent.py` inheriting `BaseAgent`
2. Add node in `workflow.py`: `graph.add_node("new_agent", _node(_new_agent, "new_agent"))`
3. Add edges: `graph.add_edge(START, "new_agent")` and `graph.add_edge("new_agent", "buffett_brain")`
4. Add field to `WarrenBrainState` in `state.py`
5. Add weight to `config.py` and `.env`
6. Add tool definition to `src/warren_brain/mcp/tools.py` if it should be available in MCP mode
