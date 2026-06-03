"""Streamlit dashboard — run with: warren ui  OR  streamlit run src/warren_brain/ui/dashboard.py"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from warren_brain.guardrails import (
    STALE_ANALYSIS_HOURS,
    apply_chat_guardrails,
    cache_age_seconds,
    check_action_vs_composite,
    check_data_quality,
    check_high_risk,
    is_leveraged_etf,
    sanitize_chat_input,
    validate_batch,
    validate_ticker,
)

st.set_page_config(
    page_title="Warren Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _check_secrets():
    """Show a clear error and stop if required API keys are missing."""
    import os
    def _get(key: str) -> str:
        if val := os.environ.get(key, ""):
            return val
        try:
            return st.secrets.get(key, "") or ""
        except Exception:
            return ""

    llm_provider = _get("LLM_PROVIDER") or "anthropic"
    if llm_provider == "anthropic":
        if not _get("ANTHROPIC_API_KEY"):
            st.error(
                "**Missing ANTHROPIC_API_KEY.** "
                "Go to **Manage app → Settings → Secrets** and add:\n\n"
                "```toml\n"
                'LLM_PROVIDER = "anthropic"\n'
                'ANTHROPIC_API_KEY = "sk-ant-..."\n'
                'ANTHROPIC_BASE_URL = "https://www.dataexpert.io/api/v1/anthropic"\n'
                'ANTHROPIC_MODEL = "claude-sonnet-4-6"\n'
                "```"
            )
            st.stop()
    else:
        if not _get("OPENAI_API_KEY"):
            st.error(
                "**Missing OPENAI_API_KEY.** "
                "Go to **Manage app → Settings → Secrets** and add your OpenAI key."
            )
            st.stop()


def main():
    _check_secrets()
    st.title("🧠 Warren Brain — Agentic Investment Intelligence")
    st.caption("Buffett-style multi-agent stock analysis powered by Claude + LangGraph")

    # Initialise session state
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "use_mcp" not in st.session_state:
        st.session_state.use_mcp = False

    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Settings")
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value="NVDA, AAPL, TSM",
            help="Enter one or more stock tickers",
        )
        mode = st.radio(
            "Analysis Mode",
            options=["Static Graph (LangGraph)", "Dynamic MCP Agent (Claude)"],
            index=0,
            help=(
                "Static: all 6 agents always run in parallel.\n"
                "Dynamic: Claude decides which tools to call and in what order."
            ),
        )
        use_mcp = mode.startswith("Dynamic")
        st.session_state.use_mcp = use_mcp
        run_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        st.divider()
        if use_mcp:
            st.caption("Mode: Dynamic MCP Agent\nClaude decides which tools to call")
        else:
            st.caption("Mode: Static LangGraph\nAll agents run in parallel")

    if run_btn and tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        _run_analysis_ui(tickers, use_mcp=use_mcp)
    elif st.session_state.results:
        _show_results()
    else:
        _show_landing()


def _run_analysis_ui(tickers: list[str], use_mcp: bool = False):
    tickers, batch_warning = validate_batch(tickers)
    if batch_warning:
        st.warning(batch_warning)

    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(tickers):
        if is_leveraged_etf(ticker):
            st.warning(
                f"⚠ **{ticker}** is a leveraged/inverse ETF. These instruments decay over time "
                "and are not suitable for long-term holding. Warren Brain's analysis may not apply."
            )

        valid, err = validate_ticker(ticker)
        if not valid:
            st.error(f"❌ {err}")
            progress.progress((i + 1) / len(tickers))
            continue

        mode_label = "MCP agent" if use_mcp else "pipeline"
        with st.spinner(f"Running {mode_label} for {ticker}… (first run ~20s, cached runs instant)"):
            if use_mcp:
                st.session_state.results[ticker] = _run_mcp(ticker)
            else:
                from warren_brain.graph.workflow import run_analysis
                st.session_state.results[ticker] = run_analysis(ticker)
            st.session_state.chat_history[ticker] = []
        progress.progress((i + 1) / len(tickers))
        status.text(f"✓ {ticker} done")

    status.empty()
    progress.empty()
    _show_results()


def _show_results():
    use_mcp = st.session_state.use_mcp
    tickers = list(st.session_state.results.keys())
    if not tickers:
        return

    tabs = st.tabs(tickers)
    for ticker, tab in zip(tickers, tabs):
        with tab:
            result = st.session_state.results[ticker]
            _render_ticker(ticker, result, is_mcp=use_mcp)
            st.divider()
            _render_chat(ticker, result, is_mcp=use_mcp)


def _run_mcp(ticker: str) -> dict:
    from warren_brain.mcp.agent import run_mcp_analysis
    rec = run_mcp_analysis(ticker)
    return {"recommendation": rec, "mcp_mode": True}


SYSTEM_PROMPTS = {
    "Warren (Long-term)": (
        "You are Warren Brain, an AI investment analyst trained in the philosophy of Warren Buffett. "
        "You have just completed an analysis of {ticker}. "
        "Answer the user's question concisely using the analysis data provided. "
        "Speak in Buffett's voice — direct, plain-English, long-term focused. "
        "If the data doesn't contain the answer, say so clearly."
    ),
    "Trader (Tactical Entry)": (
        "You are a tactical trading analyst specialising in short-to-medium term entries (days to months, NOT buy-and-hold). "
        "You have just completed a technical and fundamental analysis of {ticker}. "
        "The user wants practical, actionable entry timing advice — not long-term holding philosophy. "
        "Focus on: RSI levels (oversold <30, cheap entry 30-40), MACD crossovers and histogram direction, "
        "price vs SMA50/SMA200 (support levels), % from 52-week low/high, ATR-based stop-loss zones, "
        "and recent momentum. "
        "Give specific price levels, conditions to watch for, and risk/reward context. "
        "Be direct — say 'buy when X happens' not 'consider evaluating whether'. "
        "If the data doesn't contain the answer, say so."
    ),
}


# ── Chat ──────────────────────────────────────────────────────────────────────

def _render_chat(ticker: str, result: dict, is_mcp: bool):
    st.subheader(f"💬 Ask about {ticker}")

    perspective = st.radio(
        "Perspective",
        options=list(SYSTEM_PROMPTS.keys()),
        horizontal=True,
        key=f"perspective_{ticker}",
        help="Warren: long-term value investing. Trader: tactical entry timing for shorter holds.",
    )

    if ticker not in st.session_state.chat_history:
        st.session_state.chat_history[ticker] = []

    # Render existing messages
    for msg in st.session_state.chat_history[ticker]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    mode_hint = "MCP: Claude can call tools for fresh data" if is_mcp else "answers from analysis data"
    question = st.chat_input(f"Ask anything about {ticker}… ({mode_hint})")

    if question:
        valid, err = sanitize_chat_input(question)
        if not valid:
            st.error(err)
        else:
            st.session_state.chat_history[ticker].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = _answer_question(question, ticker, result, is_mcp, perspective)
                st.write(answer)

            st.session_state.chat_history[ticker].append({"role": "assistant", "content": answer})


def _answer_question(question: str, ticker: str, result: dict, is_mcp: bool, perspective: str) -> str:
    from warren_brain.llm import _get_anthropic_client
    from warren_brain.config import settings

    client = _get_anthropic_client()

    context_data = {k: v for k, v in result.items() if k not in ("raw_output",)}
    context = json.dumps(context_data, indent=2, default=str)[:6000]

    system = apply_chat_guardrails(SYSTEM_PROMPTS[perspective].format(ticker=ticker))

    base_messages = [
        {
            "role": "user",
            "content": f"Analysis data for {ticker}:\n{context}\n\nQuestion: {question}",
        }
    ]

    if is_mcp:
        # In MCP mode: give Claude tools so it can fetch fresh data if needed
        from warren_brain.mcp.tools import TOOLS, execute_tool

        messages = base_messages.copy()
        for _ in range(6):
            with client.messages.stream(
                model=settings.anthropic_model,
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                response = stream.get_final_message()

            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_blocks or response.stop_reason == "end_turn":
                text_blocks = [b for b in response.content if b.type == "text"]
                return text_blocks[0].text if text_blocks else "No answer generated."

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_blocks:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(execute_tool(block.name, block.input), default=str),
                })
            messages.append({"role": "user", "content": tool_results})

        return "Could not generate an answer within the allowed steps."

    else:
        # Static mode: answer purely from pre-fetched data
        content = ""
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=system,
            messages=base_messages,
        ) as stream:
            for text in stream.text_stream:
                content += text
        return content


# ── Analysis render ───────────────────────────────────────────────────────────

def _render_ticker(ticker: str, result: dict, is_mcp: bool = False):
    rec = result.get("recommendation", {})

    st.caption("⚠ For informational purposes only — not financial advice.")

    age = cache_age_seconds(ticker, is_mcp=is_mcp)
    if age is not None and age > STALE_ANALYSIS_HOURS * 3600:
        st.info(f"⏱ This analysis is {age // 3600}h old. Re-run to get fresher data.")

    quality, dq_warnings = check_data_quality(result)
    if quality == "INSUFFICIENT":
        st.error("🚨 Data quality INSUFFICIENT — multiple data sources failed. Treat this recommendation with caution.")
    elif quality == "DEGRADED":
        for w in dq_warnings:
            st.warning(f"⚠ {w}")

    if is_mcp:
        st.info(
            "**Dynamic MCP Agent** — Claude called these tools: "
            + ", ".join(f"`{t}`" for t in rec.get("tools_called", []))
        )

    col1, col2, col3, col4 = st.columns(4)
    action = rec.get("action", "HOLD")

    def _fmt_price(val):
        return f"${val:,.2f}" if val is not None else "N/A"

    col1.metric("Action", action)
    col2.metric("Buy Price", _fmt_price(rec.get("buy_price")))
    col3.metric("Sell Price", _fmt_price(rec.get("sell_price")))
    col4.metric("Confidence", rec.get("confidence") or rec.get("conviction", "N/A"))

    st.divider()

    composite = rec.get("composite_score", 0.5)
    left, right = st.columns([1, 2])
    with left:
        fig = _gauge_chart(composite, ticker)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Buffett Brain Rationale")
        rationale = rec.get("rationale") or rec.get("reasoning") or "No rationale generated."
        st.write(rationale)

        buffett = result.get("buffett_brain", {})
        concerns = rec.get("concerns") or buffett.get("concerns", [])
        if concerns:
            st.subheader("Key Risks")
            for c in concerns:
                st.warning(f"⚠ {c}")

        moat = rec.get("moat_assessment") or buffett.get("moat_assessment", "")
        if moat:
            st.subheader("Moat Assessment")
            st.info(moat)

        mismatch = check_action_vs_composite(action, composite)
        if mismatch:
            st.warning(f"⚠ {mismatch}")

        risk_warnings = check_high_risk(result)
        if risk_warnings:
            st.subheader("High-Risk Flags")
            for w in risk_warnings:
                st.error(f"🚨 {w}")

    st.divider()

    st.subheader("Agent Signal Scores")
    scores = rec.get("agent_scores", {})
    if scores:
        score_cols = st.columns(len(scores))
        for col, (key, val) in zip(score_cols, scores.items()):
            label = key.replace("_", " ").title()
            if val is not None:
                delta_color = "normal" if val >= 0.5 else "inverse"
                col.metric(label, f"{val:.2f}", delta=f"{(val - 0.5):.2f}", delta_color=delta_color)
            else:
                col.metric(label, "N/A")

    st.divider()

    with st.expander("📊 Fundamentals Detail"):
        fund = result.get("fundamentals", {})
        if m := fund.get("metrics"):
            _render_metrics_table(m)
        if s := fund.get("summary"):
            st.write(s)

    with st.expander("📈 Technicals Detail"):
        tech = result.get("technicals", {})
        if s := tech.get("summary"):
            st.write(s)
        if ind := tech.get("indicators"):
            _render_metrics_table(ind)

    with st.expander("🗞 Sentiment Detail"):
        sent = result.get("sentiment", {})
        st.write(sent.get("summary", ""))
        if themes := sent.get("key_themes"):
            st.write("**Key themes:**", ", ".join(themes))

    with st.expander("🏛 13F / Institutional"):
        f13 = result.get("thirteen_f", {})
        st.write(f13.get("summary", ""))
        if berk := f13.get("berkshire_position"):
            st.json(berk)

    with st.expander("⚠ Risk Metrics"):
        risk = result.get("risk", {})
        if m := risk.get("metrics"):
            _render_metrics_table(m)

    with st.expander("🔧 Raw JSON"):
        st.json(result)


def _render_metrics_table(data: dict):
    items = {k: v for k, v in data.items() if v is not None}
    if not items:
        return
    rows = [(k, str(v)) for k, v in items.items()]
    import pandas as pd
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df, hide_index=True, use_container_width=True)


def _gauge_chart(score: float, ticker: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "valueformat": ".2f"},
        title={"text": f"{ticker} Composite Score"},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 0.35], "color": "#ff4b4b"},
                {"range": [0.35, 0.55], "color": "#ffa500"},
                {"range": [0.55, 0.70], "color": "#90EE90"},
                {"range": [0.70, 1.0], "color": "#2ecc71"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def _show_landing():
    st.info(
        "Enter one or more tickers in the sidebar and click **Analyze** to run Warren Brain.\n\n"
        "**Example tickers:** NVDA, AAPL, TSM, BRK.B, OXY, KO"
    )
    st.markdown("""
### How it works
1. **FundamentalsAgent** — evaluates P/E, ROE, FCF, and intrinsic value
2. **TechnicalsAgent** — RSI, MACD, SMA trends, entry zones
3. **SentimentAgent** — analyzes recent news sentiment via Massive.com
4. **13FAgent** — checks Berkshire Hathaway and institutional holdings from SEC EDGAR
5. **OntologyAgent** — maps supply chain, competitors, and ETF exposure
6. **RiskAgent** — beta, VaR(95%), max drawdown, Sharpe ratio
7. **BuffettBrainAgent** — Claude reasons in Warren Buffett's voice
8. **PortfolioManagerAgent** — aggregates all signals into buy/sell recommendation

After analysis, a **chat bar** appears under each ticker so you can ask follow-up questions.
In **MCP mode**, Claude can call tools to fetch fresh data to answer your question.
""")


if __name__ == "__main__":
    main()
