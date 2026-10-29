import time
from types import SimpleNamespace
from typing import Any

import litellm

from .config import Deployment


def infer(
    deployment: Deployment,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    request_timeout: float,
    activity: Any | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": deployment.model,
        "messages": messages,
        "timeout": request_timeout,
        "num_retries": 0,
        "stream": True,
    }
    if deployment.api_base:
        kwargs["api_base"] = deployment.api_base
    if deployment.api_key:
        kwargs["api_key"] = deployment.api_key
    if response_format is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": response_format}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    content_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, Any]] = {}

    for chunk in litellm.completion(**kwargs):
        delta = chunk.choices[0].delta
        delta_content = getattr(delta, "content", None)
        delta_tool_calls = getattr(delta, "tool_calls", None)

        if activity is not None and (delta_content or delta_tool_calls):
            now = time.monotonic()
            if activity.first_token_at is None:
                activity.first_token_at = now
            activity.state = "streaming"
            activity.last_chunk_at = now

        if delta_content:
            content_parts.append(delta_content)
            if activity is not None:
                activity.tokens_emitted += max(1, len(delta_content) // 4)

        if delta_tool_calls:
            for tc in delta_tool_calls:
                entry = tool_call_parts.setdefault(
                    tc.index, {"id": None, "type": None, "name": None, "arguments": ""}
                )
                if tc.id:
                    entry["id"] = tc.id
                if tc.type:
                    entry["type"] = tc.type
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        entry["name"] = fn.name
                    if fn.arguments:
                        entry["arguments"] += fn.arguments
            if activity is not None:
                activity.tokens_emitted += 1

    tool_calls = None
    if tool_call_parts:
        tool_calls = [
            SimpleNamespace(
                id=entry["id"],
                type=entry["type"] or "function",
                function=SimpleNamespace(name=entry["name"], arguments=entry["arguments"]),
            )
            for _, entry in sorted(tool_call_parts.items())
        ]

    return SimpleNamespace(content="".join(content_parts) or None, tool_calls=tool_calls)
