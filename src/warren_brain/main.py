"""CLI entry point — `warren analyze NVDA AAPL TSM`"""

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from warren_brain.graph.workflow import run_analysis

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="warren",
        description="Warren Brain — Agentic Investment Intelligence",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one or more tickers")
    analyze.add_argument("tickers", nargs="+", help="Stock tickers (e.g. NVDA AAPL TSM)")
    analyze.add_argument("--json", action="store_true", help="Output raw JSON")

    sub.add_parser("ui", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "analyze":
        _run_analyze(args.tickers, raw_json=args.json)
    elif args.command == "ui":
        _launch_ui()


def _run_analyze(tickers: list[str], raw_json: bool) -> None:
    results = []
    for ticker in tickers:
        console.print(f"\n[bold cyan]Analyzing {ticker}...[/bold cyan]")
        result = run_analysis(ticker)
        results.append(result)

        if raw_json:
            console.print_json(json.dumps(result))
        else:
            _render_result(result)

    if raw_json and len(results) > 1:
        print(json.dumps(results, indent=2))


def _render_result(result: dict) -> None:
    rec = result.get("recommendation", {})
    ticker = result.get("ticker", "?")

    table = Table(title=f"Warren Brain — {ticker}", show_header=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    action = rec.get("action", "HOLD")
    action_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")

    table.add_row("Action", f"[{action_color}]{action}[/{action_color}]")
    table.add_row("Buy Price", f"${rec.get('buy_price', 'N/A')}")
    table.add_row("Sell Price", f"${rec.get('sell_price', 'N/A')}")
    table.add_row("Confidence", rec.get("confidence", "N/A"))
    table.add_row("Composite Score", str(rec.get("composite_score", "N/A")))

    console.print(table)

    rationale = rec.get("rationale", "")
    if rationale:
        console.print(f"\n[bold]Rationale:[/bold] {rationale}\n")

    backtest = rec.get("backtest", {})
    if backtest:
        bt_table = Table(title="Backtest Metrics")
        bt_table.add_column("Metric")
        bt_table.add_column("Value")
        for k, v in backtest.items():
            bt_table.add_row(k, str(v))
        console.print(bt_table)


def _launch_ui() -> None:
    import subprocess

    ui_path = __file__.replace("main.py", "ui/dashboard.py")
    console.print("[bold]Launching Streamlit dashboard...[/bold]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path], check=True)


if __name__ == "__main__":
    main()
