"""Shared LLM client — routes to OpenAI or Anthropic based on LLM_PROVIDER setting.

OpenAI proxy (dataexpert.io) always returns SSE streaming format → stream=True required.
Anthropic proxy uses the standard Anthropic SDK messages API.
"""

from __future__ import annotations

import os

from warren_brain.config import settings


def _secret(env_key: str, settings_value: str) -> str:
    """Return settings value, then OS env, then st.secrets — handles Streamlit Cloud."""
    if settings_value:
        return settings_value
    if val := os.environ.get(env_key, ""):
        return val
    try:
        import streamlit as st
        return st.secrets.get(env_key, "") or ""
    except Exception:
        return ""


# ── OpenAI ───────────────────────────────────────────────────────────────────

def _get_openai_client():
    from openai import OpenAI
    api_key = _secret("OPENAI_API_KEY", settings.openai_api_key)
    base_url = _secret("OPENAI_BASE_URL", settings.openai_base_url)
    kwargs: dict = {"api_key": api_key or None}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["default_headers"] = {"x-session-id": "warren-brain"}
    return OpenAI(**kwargs)


def _complete_openai(
    messages: list[dict],
    model: str | None,
    temperature: float | None,
    json_mode: bool,
) -> str:
    client = _get_openai_client()
    kwargs: dict = {
        "model": model or settings.openai_model,
        "temperature": temperature if temperature is not None else settings.openai_temperature,
        "messages": messages,
        "stream": True,  # proxy always returns SSE — must stream
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    content = ""
    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return content


# ── Anthropic ────────────────────────────────────────────────────────────────

def _get_anthropic_client():
    from anthropic import Anthropic
    api_key = _secret("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    base_url = _secret("ANTHROPIC_BASE_URL", settings.anthropic_base_url)
    kwargs: dict = {
        "api_key": api_key or None,
        "default_headers": {"x-session-id": "warren-brain"},
    }
    if base_url:
        kwargs["base_url"] = base_url
    return Anthropic(**kwargs)


def _complete_anthropic(
    messages: list[dict],
    model: str | None,
    temperature: float | None,
    json_mode: bool,
) -> str:
    client = _get_anthropic_client()

    # Anthropic separates system prompt from the messages list
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_messages = [m for m in messages if m["role"] != "system"]
    system = "\n\n".join(system_parts)

    if json_mode:
        suffix = "IMPORTANT: Respond with valid JSON only. No markdown fences, no explanation — pure JSON."
        system = f"{system}\n\n{suffix}" if system else suffix

    kwargs: dict = {
        "model": model or settings.anthropic_model,
        "max_tokens": 2048,
        "messages": user_messages,
    }
    if system:
        kwargs["system"] = system
    kwargs["temperature"] = temperature if temperature is not None else settings.openai_temperature

    content = ""
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            content += text
    return content


# ── Public interface ──────────────────────────────────────────────────────────

def complete(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """
    Make a chat completion and return the response text.
    Routes to OpenAI or Anthropic based on LLM_PROVIDER env var.
    All agents pass OpenAI-style messages (role: system/user/assistant).
    """
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return _complete_anthropic(messages, model, temperature, json_mode)
    return _complete_openai(messages, model, temperature, json_mode)
