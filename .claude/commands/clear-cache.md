# /project:clear-cache

Clear cached analysis results so the next run fetches fresh data.

```bash
.venv/bin/python -c "
import sqlite3
with sqlite3.connect('warren_brain.db') as conn:
    n1 = conn.execute(\"DELETE FROM cache WHERE key LIKE 'analysis:%'\").rowcount
    n2 = conn.execute(\"DELETE FROM cache WHERE key LIKE 'mcp:%'\").rowcount
    conn.commit()
    print(f'Cleared {n1} static analysis + {n2} MCP analysis cache entries.')
    print('Data caches (prices, news, fundamentals) left intact.')
"
```

To also clear all data caches (forces fresh API calls):
```bash
.venv/bin/python -c "
import sqlite3
with sqlite3.connect('warren_brain.db') as conn:
    n = conn.execute('DELETE FROM cache').rowcount
    conn.commit()
    print(f'Cleared all {n} cache entries.')
"
```
