"""Helpers for talking to OpenAI-compatible HistoGraph chat endpoints."""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_HISTOGRAPH_ENDPOINT = "http://localhost:8000/v1/chat/completions"


def normalize_endpoint(endpoint: str | None) -> str:
    """Normalize a chat-completions endpoint URL."""
    cleaned = (endpoint or DEFAULT_HISTOGRAPH_ENDPOINT).strip()
    if cleaned == "":
        return DEFAULT_HISTOGRAPH_ENDPOINT
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    return cleaned


def _extract_content(content: Any) -> str:
    """Extract text content from OpenAI-style message content payloads."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def request_chat_completion(
    endpoint: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    user: str,
    conversation_id: str,
    timeout: tuple[float, float] = (5.0, 120.0),
) -> str:
    """Call an OpenAI-compatible chat completion endpoint and return text."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "user": user,
        "conversation_id": conversation_id,
    }

    session = requests.Session()
    session.trust_env = False  # bypass system proxies for local requests
    session.proxies.update({"http": None, "https": None})

    response = session.post(
        normalize_endpoint(endpoint),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        msg = (
            "Streaming response returned from endpoint. "
            "Disable stream in the UI for now."
        )
        raise RuntimeError(msg)

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("No choices in chat completion response.")

    first_choice = choices[0]
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    if not isinstance(message, dict):
        raise RuntimeError("Unexpected response format from chat endpoint.")

    content = _extract_content(message.get("content", ""))
    if content.strip() == "":
        raise RuntimeError("Chat completion returned empty content.")
    return content
