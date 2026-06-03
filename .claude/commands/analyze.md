# /project:analyze

Run a full Warren Brain analysis for one or more tickers and print the recommendation.

```bash
.venv/bin/python -c "
import sys, json
sys.path.insert(0, 'src')
from warren_brain.graph.workflow import run_analysis

tickers = '$ARGUMENTS'.upper().split(',')
for ticker in tickers:
    ticker = ticker.strip()
    result = run_analysis(ticker)
    rec = result['recommendation']
    print(f'\n=== {ticker} ===')
    print(f'Action:    {rec.get(\"action\")}')
    print(f'Score:     {rec.get(\"composite_score\")}')
    print(f'Buy at:    \${rec.get(\"buy_price\")}')
    print(f'Sell at:   \${rec.get(\"sell_price\")}')
    print(f'Confidence:{rec.get(\"confidence\")}')
    print(f'Rationale: {rec.get(\"rationale\", \"\")[:300]}')
"
```

Usage: `/project:analyze AAPL` or `/project:analyze AAPL,NVDA,TSM`
