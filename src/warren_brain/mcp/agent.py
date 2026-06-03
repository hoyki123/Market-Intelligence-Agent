"""Dynamic Claude agent using Anthropic tool use (MCP-style).

Unlike the static LangGraph graph that always runs all 6 agents in parallel,
this agent lets Claude decide:
  - Which tools to call
  - In what order
  - Whether to call the same tool twice with different context
  - When it has enough information to make a recommendation

This makes the analysis adaptive: a bank gets different tools than a semiconductor
company; a distressed stock triggers more risk calls than a stable one.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

from warren_brain.config import settings
from warren_brain.guardrails import validate_tool_input
from warren_brain.llm import _get_anthropic_client
from warren_brain.mcp.tools import TOOLS, execute_tool

console = Console()

SYSTEM_PROMPT = """You are Warren Brain — an AI investment analyst trained in the philosophy
of Warren Buffett and Charlie Munger.

Investment principles you apply:
1. MOAT: Only invest in businesses with durable competitive advantages (brand, network
   effects, switching costs, cost advantages, efficient scale).
2. MANAGEMENT: Prefer shareholder-friendly management with skin in the game.
3. FINANCIALS: Prioritize high ROE (>15%), consistent FCF, low debt, predictable earnings.
4. PRICE: Buy at a significant margin of safety below intrinsic value. Never overpay.
5. CIRCLE OF COMPETENCE: Only opine on businesses predictable 10 years out.
6. PATIENCE: Prefer holding forever. Never buy on short-term noise.
7. RISK: Permanent capital loss is the only real risk. Volatility is opportunity.

You have tools to gather real data about any stock. You decide which tools to call and
in what order based on what you are learning. You do not need to call all tools —
use judgment:
- Always call get_fundamentals and get_risk_metrics
- Call get_sentiment for any stock where market perception matters
- Call get_institutional_holdings to check if Buffett himself owns it
- Call get_company_overview if you need to understand the business or ecosystem
- Call get_technicals if entry timing is relevant to your recommendation
- Call get_realtime_price if you need the most current price

When you have gathered sufficient data, output ONLY valid JSON — no markdown fences,
no explanation text before or after, just the raw JSON object:

{
  "action": "BUY | SELL | HOLD",
  "buy_price": <float — price you would start buying at, or null if never interested>,
  "sell_price": <float — ALWAYS provide this: the price target at which you would exit or trim, even for HOLD>,
  "conviction": "HIGH | MEDIUM | LOW",
  "moat_assessment": "<STRONG|MODERATE|WEAK|UNKNOWN> — one sentence explanation",
  "reasoning": "2-3 sentences in Buffett's voice explaining the decision",
  "concerns": ["risk 1", "risk 2", "risk 3"],
  "composite_score": <float between 0.0 and 1.0>,
  "tools_called": ["list of every tool name you called"]
}"""


TTL_MCP = 60 * 60  # 1 hour


def run_mcp_analysis(ticker: str) -> dict:
    """
    Run a dynamic Claude agent analysis for a ticker.

    Claude iteratively calls tools, reads results, and decides what else
    it needs — rather than running a fixed predetermined pipeline.

    Returns a dict in the same shape as PortfolioManagerAgent's recommendation.
    """
    from warren_brain.data.cache import get_cache
    cache = get_cache()
    cache_key = f"mcp:{ticker.upper()}"
    if cached := cache.get(cache_key):
        console.print(f"  [dim]↩ {ticker.upper()} MCP result from cache[/dim]")
        return cached

    client = _get_anthropic_client()
    ticker = ticker.upper()

    console.print(f"  [bold cyan]⚡ MCP dynamic agent — {ticker}[/bold cyan]")

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Analyze {ticker} as a Buffett-style investment. "
                f"Use the available tools to gather data, then provide your recommendation."
            ),
        }
    ]

    tools_called: list[str] = []
    max_iterations = 15

    for _ in range(max_iterations):
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # Claude finished — no more tool calls
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            text_blocks = [b for b in response.content if b.type == "text"]
            raw_text = text_blocks[0].text if text_blocks else ""
            result = _parse_final(raw_text, ticker, tools_called)
            console.print(
                f"  [green]✓ MCP agent done — called {len(tools_called)} tool(s): "
                f"{', '.join(tools_called)}[/green]"
            )
            cache.set(cache_key, result, ttl=TTL_MCP)
            return result

        # Execute all tool calls Claude requested in this turn — in parallel
        messages.append({"role": "assistant", "content": response.content})

        names = [b.name for b in tool_use_blocks]
        console.print(f"  [cyan]  ⚙ parallel: {', '.join(names)}[/cyan]")
        tools_called.extend(names)

        def _run(block):
            inp = dict(block.input)
            valid, msg = validate_tool_input(block.name, inp, ticker)
            if not valid:
                console.print(f"  [yellow]⚠ Tool guardrail: {msg}[/yellow]")
                inp["ticker"] = ticker
            return block, execute_tool(block.name, inp)

        tool_results_content = []
        with ThreadPoolExecutor(max_workers=len(tool_use_blocks)) as pool:
            futures = {pool.submit(_run, block): block for block in tool_use_blocks}
            for future in as_completed(futures):
                block, result = future.result()
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        messages.append({"role": "user", "content": tool_results_content})

    return {
        "action": "HOLD",
        "conviction": "LOW",
        "reasoning": "Analysis timed out — max tool iterations reached.",
        "tools_called": tools_called,
        "ticker": ticker,
        "error": "max_iterations_exceeded",
    }


def _parse_final(text: str, ticker: str, tools_called: list[str]) -> dict:
    """Extract JSON from Claude's final response text."""
    for candidate in [text.strip(), re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()]:
        try:
            result = json.loads(candidate)
            result.setdefault("tools_called", tools_called)
            result["ticker"] = ticker
            return result
        except (json.JSONDecodeError, ValueError):
            continue

    # Last resort: find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            result.setdefault("tools_called", tools_called)
            result["ticker"] = ticker
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "action": "HOLD",
        "conviction": "LOW",
        "reasoning": text[:500] if text else "No response generated.",
        "tools_called": tools_called,
        "ticker": ticker,
        "error": "json_parse_failed",
    }
