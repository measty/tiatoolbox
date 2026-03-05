"""Helpers for talking to OpenAI-compatible HistoGraph chat endpoints."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

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


def _workflow_json_from_artifact(
    artifact_url: str,
    *,
    timeout: tuple[float, float] = (5.0, 60.0),
) -> dict[str, Any]:
    """Download and parse a workflow artifact JSON.

    Note: HistoGraph currently emits *ComfyUI UI workflow* JSON (graph with nodes/links/etc).
    ComfyUI's headless `/prompt` endpoint expects a different JSON shape (the API "prompt"),
    so we convert UI workflow -> API prompt before queueing.
    """
    session = build_session()
    response = session.get(artifact_url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        msg = "Workflow artifact is not a JSON object."
        raise RuntimeError(msg)
    return payload


def _looks_like_comfyui_prompt(prompt: dict[str, Any]) -> bool:
    """Heuristic check for ComfyUI API prompt format.

    Expected shape:
      {"<node_id>": {"class_type": "SomeNode", "inputs": {...}}, ...}
    """
    if not prompt:
        return False
    if "nodes" in prompt and "links" in prompt:
        return False
    for key, value in list(prompt.items())[:5]:
        if not isinstance(key, str):
            return False
        if not isinstance(value, dict):
            return False
        if "class_type" not in value or "inputs" not in value:
            return False
    return True


def _workflow_to_comfyui_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    """Convert ComfyUI *UI workflow* JSON into ComfyUI API prompt JSON.

    This works for our HistoGraph-emitted workflows because each node carries a
    `properties.pathology_params` dict which maps parameter names to values.

    References:
    - UI workflow has top-level keys like: nodes, links, extra, last_node_id...
    - API prompt expects node-id-keyed objects with class_type + inputs.
    """
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        msg = "Workflow JSON missing required keys: nodes/links."
        raise RuntimeError(msg)

    link_map: dict[int, tuple[int, int]] = {}
    for link in links:
        if not (isinstance(link, list) and len(link) >= 3):
            continue
        link_id, src_id, src_slot = link[0], link[1], link[2]
        if isinstance(link_id, int) and isinstance(src_id, int) and isinstance(src_slot, int):
            link_map[link_id] = (src_id, src_slot)

    prompt: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        class_type = node.get("type")
        if not isinstance(node_id, int) or not isinstance(class_type, str) or class_type.strip() == "":
            continue

        inputs: dict[str, Any] = {}

        for inp in node.get("inputs", []) or []:
            if not isinstance(inp, dict):
                continue
            link_id = inp.get("link")
            if link_id is None:
                continue
            if not isinstance(link_id, int) or link_id not in link_map:
                continue
            src_id, src_slot = link_map[link_id]
            name = str(inp.get("name", "")).strip()
            if name:
                inputs[name] = [str(src_id), src_slot]

        props = node.get("properties") or {}
        if isinstance(props, dict):
            params = props.get("pathology_params")
            if isinstance(params, dict):
                for key, value in params.items():
                    if value is not None:
                        inputs[str(key)] = value

        prompt[str(node_id)] = {"class_type": class_type, "inputs": inputs}

    if not prompt:
        msg = "Converted prompt is empty; cannot queue workflow."
        raise RuntimeError(msg)
    return prompt


def prompt_json_from_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize an artifact payload into a valid ComfyUI API prompt JSON."""
    if _looks_like_comfyui_prompt(payload):
        return payload
    if "nodes" in payload and "links" in payload:
        return _workflow_to_comfyui_prompt(payload)
    msg = "Artifact JSON is neither a ComfyUI prompt nor a ComfyUI workflow."
    raise RuntimeError(msg)


def queue_comfyui_prompt(
    *,
    comfyui_url: str,
    prompt_json: dict[str, Any],
    timeout: tuple[float, float] = (5.0, 60.0),
) -> dict[str, Any]:
    """Queue a ComfyUI API prompt and return response payload."""
    base_url = normalize_base_url(comfyui_url, default="http://127.0.0.1:8188")
    session = build_session()

    candidate_urls = [
        f"{base_url}/prompt",
        f"{base_url}/api/prompt",
    ]

    payload = {"prompt": prompt_json}
    errors: list[str] = []
    for candidate in candidate_urls:
        try:
            response = session.post(candidate, json=payload, timeout=timeout)
            if response.status_code in {200, 201, 202}:
                data = response.json()
                if isinstance(data, dict):
                    return data
                msg = f"Unexpected queue response type from {candidate}."
                raise RuntimeError(msg)
            errors.append(f"{candidate} -> {response.status_code}: {response.text[:180]}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate} -> {exc}")

    msg = "Failed to queue ComfyUI prompt. " + " | ".join(errors)
    raise RuntimeError(msg)


def fetch_comfyui_history(
    *,
    comfyui_url: str,
    prompt_id: str,
    timeout: tuple[float, float] = (5.0, 60.0),
) -> dict[str, Any]:
    """Fetch ComfyUI history for a given prompt id."""
    base_url = normalize_base_url(comfyui_url, default="http://127.0.0.1:8188")
    session = build_session()

    encoded_prompt = urllib.parse.quote(prompt_id, safe="")
    candidate_urls = [
        f"{base_url}/history/{encoded_prompt}",
        f"{base_url}/api/history/{encoded_prompt}",
    ]

    errors: list[str] = []
    for candidate in candidate_urls:
        try:
            response = session.get(candidate, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    return data
                msg = f"Unexpected history payload type from {candidate}."
                raise RuntimeError(msg)
            errors.append(f"{candidate} -> {response.status_code}: {response.text[:180]}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate} -> {exc}")

    msg = "Failed to fetch ComfyUI history. " + " | ".join(errors)
    raise RuntimeError(msg)


def wait_for_comfyui_prompt_completion(
    *,
    comfyui_url: str,
    prompt_id: str,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.5,
) -> dict[str, Any]:
    """Poll ComfyUI history until a prompt is completed or fails."""
    deadline = time.monotonic() + max(timeout_s, 1.0)

    while time.monotonic() < deadline:
        history = fetch_comfyui_history(comfyui_url=comfyui_url, prompt_id=prompt_id)
        prompt_data = history.get(prompt_id)
        if isinstance(prompt_data, dict):
            status = prompt_data.get("status", {})
            print(f"Prompt finished; Status: {status}")
            if isinstance(status, dict):
                status_str = str(status.get("status_str", "")).lower()
                completed = bool(status.get("completed", False))
                if status_str in {"error", "failed"}:
                    msg = f"ComfyUI run failed ({status_str})."
                    raise RuntimeError(msg)
                if completed or status_str in {"success", "succeeded", "completed"}:
                    return prompt_data

            # Some ComfyUI versions omit explicit status but include outputs on completion.
            if isinstance(prompt_data.get("outputs"), dict) and prompt_data.get("outputs"):
                return prompt_data

        time.sleep(max(poll_interval_s, 0.25))

    msg = f"Timed out waiting for ComfyUI prompt {prompt_id}."
    raise TimeoutError(msg)


def _maybe_append_extension(path_str: str, format_hint: str | None = None) -> str:
    """Append format extension when path has no suffix."""
    candidate = Path(path_str)
    if candidate.suffix != "":
        return str(candidate)

    ext = str(format_hint or "").strip().lstrip(".")
    if ext == "":
        return str(candidate)
    return str(candidate.with_suffix(f".{ext}"))


def _path_exists(path_str: str) -> bool:
    """Best-effort filesystem existence check."""
    try:
        return Path(path_str).expanduser().exists()
    except Exception:  # noqa: BLE001
        return False


def _path_dedupe_key(path_str: str) -> str:
    """Create a stable path key for deduplication."""
    path_obj = Path(path_str).expanduser()
    try:
        return str(path_obj.resolve(strict=False))
    except Exception:  # noqa: BLE001
        return str(path_obj)


def extract_export_outputs_from_workflow(
    workflow_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract export-path outputs directly from a ComfyUI UI workflow JSON."""
    nodes = workflow_json.get("nodes")
    if not isinstance(nodes, list):
        return []

    items: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue

        props = node.get("properties")
        if not isinstance(props, dict):
            props = {}

        pathology_op = str(props.get("pathology_op", "")).strip()
        node_type = str(node.get("type", "")).strip()
        is_export = pathology_op.startswith("Export") or node_type.startswith(
            "Pathology.Export",
        )
        if not is_export:
            continue

        params = props.get("pathology_params")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:  # noqa: BLE001
                params = {}
        if not isinstance(params, dict):
            params = {}

        raw_path = str(params.get("path", "")).strip()
        if raw_path == "":
            continue

        format_hint = str(params.get("format", "")).strip()
        resolved_path = _maybe_append_extension(raw_path, format_hint)
        path_obj = Path(resolved_path).expanduser()
        op_name = pathology_op or node_type

        items.append(
            {
                "node_id": str(node.get("id", "")),
                "op": op_name,
                "source": "workflow_export",
                "path": raw_path,
                "format": format_hint,
                "filename": path_obj.name,
                "display_name": f"{op_name}: {path_obj.name}",
                "resolved_path": str(path_obj),
                "exists": _path_exists(str(path_obj)),
            },
        )

    return items


def extract_comfyui_output_images(
    prompt_history: dict[str, Any],
    *,
    comfyui_output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Extract image outputs from a ComfyUI prompt history object."""
    outputs = prompt_history.get("outputs")
    if not isinstance(outputs, dict):
        return []

    base_dir = Path(comfyui_output_dir).expanduser() if comfyui_output_dir else None
    items: list[dict[str, Any]] = []

    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if not isinstance(images, list):
            continue

        for idx, image_entry in enumerate(images):
            if not isinstance(image_entry, dict):
                continue

            filename = str(image_entry.get("filename", "")).strip()
            if filename == "":
                continue
            subfolder = str(image_entry.get("subfolder", "")).strip()
            image_type = str(image_entry.get("type", "output")).strip() or "output"

            rel_parts = [part for part in [subfolder, filename] if part]
            rel_path = Path(*rel_parts) if rel_parts else Path(filename)

            resolved_path = rel_path
            if not rel_path.is_absolute() and base_dir is not None:
                resolved_path = base_dir / rel_path

            display_name = filename if subfolder == "" else f"{subfolder}/{filename}"
            resolved_path_str = str(resolved_path.expanduser())
            items.append(
                {
                    "node_id": str(node_id),
                    "index": str(idx),
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": image_type,
                    "source": "history_images",
                    "display_name": display_name,
                    "resolved_path": resolved_path_str,
                    "exists": _path_exists(resolved_path_str),
                },
            )

    return items


def _merge_existing_outputs(
    export_outputs: list[dict[str, Any]],
    history_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge export-derived and history-derived outputs, keeping existing files only."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in [*export_outputs, *history_outputs]:
        if not isinstance(item, dict):
            continue

        resolved_path = str(item.get("resolved_path", "")).strip()
        if resolved_path == "":
            continue

        exists = bool(item.get("exists", False))
        if not exists:
            continue

        key = _path_dedupe_key(resolved_path)
        if key in seen:
            continue

        seen.add(key)
        merged.append(item)

    return merged


def run_comfyui_workflow_from_artifact(
    *,
    artifact_url: str,
    comfyui_url: str,
    comfyui_output_dir: str | None = None,
    queue_timeout: tuple[float, float] = (5.0, 60.0),
    wait_timeout_s: float = 300.0,
    poll_interval_s: float = 1.5,
    on_prompt_queued: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Queue workflow artifact on ComfyUI, wait, and collect image outputs."""
    workflow_json = _workflow_json_from_artifact(artifact_url, timeout=queue_timeout)
    prompt_json = prompt_json_from_artifact_payload(workflow_json)
    queue_result = queue_comfyui_prompt(
        comfyui_url=comfyui_url,
        prompt_json=prompt_json,
        timeout=queue_timeout,
    )

    prompt_id = str(queue_result.get("prompt_id", "")).strip()
    if prompt_id == "":
        msg = "ComfyUI queue response did not include prompt_id."
        raise RuntimeError(msg)

    if on_prompt_queued is not None:
        on_prompt_queued(prompt_id)

    history = wait_for_comfyui_prompt_completion(
        comfyui_url=comfyui_url,
        prompt_id=prompt_id,
        timeout_s=wait_timeout_s,
        poll_interval_s=poll_interval_s,
    )

    export_outputs = extract_export_outputs_from_workflow(workflow_json)
    history_outputs = extract_comfyui_output_images(
        history,
        comfyui_output_dir=comfyui_output_dir,
    )
    outputs = _merge_existing_outputs(export_outputs, history_outputs)

    return {
        "prompt_id": prompt_id,
        "queue_response": queue_result,
        "history": history,
        "outputs": outputs,
        "output_summary": {
            "export_found": sum(1 for x in export_outputs if bool(x.get("exists", False))),
            "history_found": sum(1 for x in history_outputs if bool(x.get("exists", False))),
            "total_found": len(outputs),
        },
    }


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
