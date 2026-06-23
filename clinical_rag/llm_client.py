"""LLM client (MiniMax Anthropic-compatible API from api_key.md / config.json)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import httpx


def get_llm_settings(config: dict) -> dict[str, str]:
    llm = config.get("llm", {})
    api_keys = config.get("api_keys", {})
    return {
        "base_url": (
            llm.get("base_url")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or "https://api.minimaxi.com/anthropic"
        ).rstrip("/"),
        "api_key": (
            llm.get("api_key")
            or api_keys.get("llm_api_key")
            or os.environ.get("LLM_API_KEY")
            or ""
        ).strip(),
        "model": (
            llm.get("model")
            or os.environ.get("LLM_MODEL")
            or "MiniMax-M2.7"
        ).strip(),
    }


def _strip_markdown_json(text: str) -> str:
    text = (text or "").strip()
    markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if markdown_match:
        return markdown_match.group(1).strip()
    return text


def chat_completion(
    messages: list[dict[str, str]],
    config: dict,
    *,
    response_format: Optional[str] = "json_object",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Optional[str]:
    """Call MiniMax via Anthropic-compatible Messages API."""
    settings = get_llm_settings(config)
    if not settings["api_key"]:
        raise ValueError("LLM API key missing. Set llm.api_key in config or api_key.md")

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_content = "\n\n".join(
        m["content"] for m in messages if m.get("role") == "user"
    )
    if not user_content:
        raise ValueError("At least one user message is required")

    system = system_parts[0] if system_parts else ""
    if response_format == "json_object":
        system = (system + "\nRespond with valid JSON only, no markdown.").strip()

    payload: dict[str, Any] = {
        "model": settings["model"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_content}],
    }
    if system:
        payload["system"] = system

    url = f"{settings['base_url']}/v1/messages"
    headers = {
        "x-api-key": settings["api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code >= 400:
                print(f"--- [ERROR] LLM API HTTP {response.status_code}: {response.text[:500]} ---")
                return None
            data = response.json()

        # Check for truncation — Anthropic returns stop_reason: "max_tokens"
        # when the response is cut off mid-generation.
        stop_reason = data.get("stop_reason", "")
        if stop_reason == "max_tokens":
            print(f"--- [ERROR] LLM response truncated (stop_reason=max_tokens, max_tokens={max_tokens}). Increase max_tokens. ---")
            return None

        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text and blocks:
            text = "".join(
                str(block.get("text", block))
                for block in blocks
                if isinstance(block, dict)
            )
        return _strip_markdown_json(text) if text else None
    except Exception as exc:
        print(f"--- [ERROR] LLM API request failed: {exc} ---")
        return None
