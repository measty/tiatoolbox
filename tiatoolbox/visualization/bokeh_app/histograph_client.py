"""Helpers for talking to OpenAI-compatible HistoGraph chat endpoints."""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

DEFAULT_HISTOGRAPH_ENDPOINT = "http://localhost:8000/v1/chat/completions"

_WORKFLOW_ARTIFACT_RE = re.compile(
    r"(https?://[^\s\]\[\)\(\"'<>]+/comfy_workflow\.json(?:\?[^\s\]\[\)\(\"'<>]+)?)",
    flags=re.IGNORECASE,
)
_WORKFLOW_RUN_ID_RE = re.compile(
    r"/v1/artifacts/([^/\s]+)/comfy_workflow\.json(?:\?|$)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class WorkflowArtifact:
    """Describes a detected workflow artifact link in assistant markdown."""

    url: str
    run_id: str | None


def normalize_endpoint(endpoint: str | None) -> str:
    """Normalize a chat-completions endpoint URL."""
    cleaned = (endpoint or DEFAULT_HISTOGRAPH_ENDPOINT).strip()
    if cleaned == "":
        return DEFAULT_HISTOGRAPH_ENDPOINT
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    return cleaned


def normalize_base_url(url: str | None, *, default: str) -> str:
    """Normalize a base URL (scheme + host + optional path)."""
    cleaned = (url or default).strip()
    if cleaned == "":
        cleaned = default
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    return cleaned.rstrip("/")


def build_session() -> requests.Session:
    """Build a local-safe requests session (no proxy inheritance)."""
    session = requests.Session()
    session.trust_env = False  # bypass system proxies for local requests
    session.proxies.update({"http": None, "https": None})
    return session


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


def extract_workflow_artifact(markdown_text: str) -> WorkflowArtifact | None:
    """Extract first Comfy workflow artifact URL from assistant markdown."""
    match = _WORKFLOW_ARTIFACT_RE.search(markdown_text)
    if not match:
        return None

    url = match.group(1)
    run_match = _WORKFLOW_RUN_ID_RE.search(url)
    run_id = run_match.group(1) if run_match else None
    return WorkflowArtifact(url=url, run_id=run_id)


def _safe_workflow_name(run_id: str | None) -> str:
    """Build a safe workflow filename stem."""
    if run_id:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
        if safe:
            return f"histograph_{safe}"
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"histograph_{ts}"


def upload_workflow_artifact_to_comfyui(
    *,
    artifact_url: str,
    run_id: str | None,
    comfyui_url: str,
    timeout: tuple[float, float] = (5.0, 60.0),
) -> tuple[str, str]:
    """Download workflow JSON artifact and upload to ComfyUI userdata/workflows.

    Returns:
        Tuple of uploaded workflow filename and the ComfyUI endpoint used.

    """
    base_url = normalize_base_url(comfyui_url, default="http://127.0.0.1:8188")
    session = build_session()

    download_resp = session.get(artifact_url, timeout=timeout)
    download_resp.raise_for_status()
    raw_bytes = download_resp.content

    # Validate payload is JSON so ComfyUI load dialog sees a valid workflow file.
    parsed = json.loads(raw_bytes.decode("utf-8"))
    payload_bytes = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8",
    )

    workflow_name = f"{_safe_workflow_name(run_id)}.json"
    rel_path = f"workflows/{workflow_name}"
    encoded_rel_path = urllib.parse.quote(rel_path, safe="")

    candidate_urls = [
        f"{base_url}/userdata/{encoded_rel_path}",
        f"{base_url}/api/userdata/{encoded_rel_path}",
    ]

    headers = {"Content-Type": "application/json"}
    errors: list[str] = []
    for candidate in candidate_urls:
        try:
            upload_resp = session.post(
                candidate,
                params={"overwrite": "true"},
                data=payload_bytes,
                headers=headers,
                timeout=timeout,
            )
            if upload_resp.status_code in {200, 201, 204}:
                return workflow_name, candidate
            errors.append(f"{candidate} -> {upload_resp.status_code}: {upload_resp.text[:180]}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate} -> {exc}")

    msg = "Failed to upload workflow to ComfyUI userdata endpoint. " + " | ".join(errors)
    raise RuntimeError(msg)


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

    session = build_session()

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
