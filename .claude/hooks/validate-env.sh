#!/bin/bash
# Runs before any Bash tool execution that touches .env
# Warns if critical keys are missing

if [ -f ".env" ]; then
  if ! grep -q "ANTHROPIC_API_KEY=sk-" .env && ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "⚠ WARNING: No LLM API key found in .env — BuffettBrainAgent will fail"
  fi
  if ! grep -q "MASSIVE_API_KEY=" .env || grep -q "MASSIVE_API_KEY=$" .env; then
    echo "⚠ WARNING: MASSIVE_API_KEY not set — falling back to yfinance/Finnhub for all data"
  fi
fi
