# API & Data Conventions

## Massive.com
- Base URL: `https://api.massive.com`
- Auth: `?apiKey=settings.massive_api_key` query param (Polygon-compatible)
- Always check `_is_available()` before calling — returns `{}` or `[]` if no key set
- Free plan endpoints: `/v2/reference/news`, `/v1/related-companies/{ticker}`, `/v3/reference/tickers/{ticker}`, `/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}`
- NOT on free plan (returns NOT_AUTHORIZED): financial ratios, income statements

## yfinance
- Use `fetch_key_metrics(ticker)` for fundamental ratios — never call `yf.Ticker()` directly in agents
- Price history: `fetch_price_history(ticker, period_years=N)` — returns DataFrame with DatetimeIndex reset to string
- All yfinance calls are cached 1 hr by default

## SEC EDGAR
- No API key required. Use `User-Agent` header: `WarrenBrain/0.1 (research project; ...)`
- CIK lookup: `get_cik(ticker)` from `data/edgar.py`
- 13F filings: `fetch_13f_holdings(cik, max_filings=N)` — parses XML infotable

## OpenAI / Anthropic Proxy (dataexpert.io)
- Both proxies return SSE streaming format regardless of `stream` parameter
- OpenAI: must use `stream=True` + iterate chunks
- Anthropic: must use `.messages.stream()` + `.get_final_message()`
- Both require `x-session-id: warren-brain` header
- Never use the raw SDK clients directly — always use `complete()` from `llm.py`

## Database
- Never write raw SQL — use SQLAlchemy ORM models from `data/models.py`
- Always call `init_db()` before first write — it's a no-op if tables exist
- Password encoding for PostgreSQL: `quote_plus(settings.db_password)` — never manually encode in `.env`
- SQLite needs `StaticPool` + `check_same_thread=False` for Streamlit multi-thread use

## Cache Keys Convention
```
price:{period}:{TICKER}          # yfinance price history
info:{TICKER}                    # yfinance info
massive:news:{TICKER}:{days}     # Massive news
massive:snapshot:{TICKER}        # Massive real-time price
massive:overview:{TICKER}        # Massive company overview
massive:related:{TICKER}         # Massive related companies
13f:{cik}:{max_filings}          # SEC EDGAR 13F
analysis:{TICKER}                # full static graph result
mcp:{TICKER}                     # full MCP agent result
```
