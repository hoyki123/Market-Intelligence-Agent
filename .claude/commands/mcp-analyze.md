# /project:mcp-analyze

Run the dynamic MCP agent (Claude decides which tools to call) for a ticker.

```bash
.venv/bin/python -c "
import sys, json
sys.path.insert(0, 'src')
from warren_brain.mcp.agent import run_mcp_analysis

ticker = '$ARGUMENTS'.upper().strip()
result = run_mcp_analysis(ticker)
print(json.dumps(result, indent=2, default=str))
"
```

Usage: `/project:mcp-analyze NVDA`

Shows which tools Claude called, in what order, and the full Buffett-style reasoning.
