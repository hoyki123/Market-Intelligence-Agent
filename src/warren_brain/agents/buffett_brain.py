"""BuffettBrainAgent — GPT-4o reasoning in Buffett's investment style."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from warren_brain.agents.base import BaseAgent
from warren_brain.config import settings
from warren_brain.llm import complete

if TYPE_CHECKING:
    from warren_brain.graph.state import WarrenBrainState

BUFFETT_SYSTEM_PROMPT = """You are Warren Brain — an AI investment analyst trained in the philosophy
of Warren Buffett and Charlie Munger. You reason with the following principles:

1. MOAT: Only invest in businesses with durable competitive advantages (brand, network effects,
   switching costs, cost advantages, efficient scale).
2. MANAGEMENT: Prefer shareholder-friendly management with skin in the game and long-term focus.
3. FINANCIALS: Prioritize high ROE (>15%), consistent free cash flow, low debt, and predictable earnings.
4. PRICE: Buy at a significant margin of safety below intrinsic value. Avoid over-paying even for great businesses.
5. CIRCLE OF COMPETENCE: Only opine on businesses you can understand and predict 10 years out.
6. PATIENCE: Prefer holding forever. Never buy based on short-term market noise.
7. RISK: Losing money permanently is the only real risk. Volatility is opportunity, not risk.

When analyzing a stock, express your reasoning in Buffett's voice — direct, common-sense, and grounded
in long-term business fundamentals. Be skeptical of high-P/E growth stories. Favor boring businesses
with recurring revenue and pricing power.

Always output valid JSON with keys: action, buy_price, sell_price, conviction, reasoning, concerns, moat_assessment
"""


class BuffettBrainAgent(BaseAgent):
    name = "BuffettBrainAgent"

    pass

    def analyze(self, ticker: str, state: "WarrenBrainState") -> dict:
        try:
            context = self._build_context(ticker, state)
            result = self._llm_analysis(ticker, context)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "action": "HOLD",
                "conviction": "LOW",
                "reasoning": "Analysis failed — defaulting to HOLD.",
            }

    def _build_context(self, ticker: str, state: "WarrenBrainState") -> str:
        parts = [f"=== Investment Analysis Context for {ticker} ===\n"]

        if fund := state.get("fundamentals"):
            parts.append(f"FUNDAMENTALS:\n{fund.get('summary', '')}")
            metrics = fund.get("metrics", {})
            if metrics.get("pe_ratio"):
                parts.append(f"  P/E: {metrics['pe_ratio']:.1f}x")
            if metrics.get("roe"):
                parts.append(f"  ROE: {metrics['roe']:.1%}")
            if iv := fund.get("intrinsic_value_estimate"):
                parts.append(f"  Estimated intrinsic value: ${iv:.2f}")
                if cp := metrics.get("current_price"):
                    parts.append(f"  Current price: ${cp:.2f}")

        if tech := state.get("technicals"):
            parts.append(f"\nTECHNICALS:\n{tech.get('summary', '')}")

        if sent := state.get("sentiment"):
            parts.append(f"\nSENTIMENT:\n{sent.get('summary', '')}")
            parts.append(f"  Label: {sent.get('sentiment_label', 'N/A')}")
            if themes := sent.get("key_themes"):
                parts.append(f"  Themes: {', '.join(themes)}")

        if f13 := state.get("thirteen_f"):
            parts.append(f"\nINSTITUTIONAL (13F):\n{f13.get('summary', '')}")

        if ont := state.get("ontology"):
            parts.append(f"\nSUPPLY CHAIN / ONTOLOGY:\n{ont.get('summary', '')}")

        if risk := state.get("risk"):
            parts.append(f"\nRISK:\n{risk.get('summary', '')}")

        return "\n".join(parts)

    def _llm_analysis(self, ticker: str, context: str) -> dict:
        content = complete(
            messages=[
                {"role": "system", "content": BUFFETT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"Based on all the above data, provide your Buffett-style investment analysis "
                        f"for {ticker}. Output JSON with:\n"
                        f'  "action": BUY | SELL | HOLD\n'
                        f'  "buy_price": float (price to start buying, or null if not interested)\n'
                        f'  "sell_price": float (target sell price, or null)\n'
                        f'  "conviction": HIGH | MEDIUM | LOW\n'
                        f'  "moat_assessment": brief string (STRONG | MODERATE | WEAK | UNKNOWN with explanation)\n'
                        f'  "reasoning": 2-3 sentence Buffett-style rationale\n'
                        f'  "concerns": list of 1-3 key risks or reasons to be cautious\n'
                    ),
                },
            ],
            json_mode=True,
        )
        return json.loads(content)
