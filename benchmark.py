"""Throughput calibration and ad-hoc prompt benchmarking."""
from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from statistics import mean
from typing import Any

import litellm

from .config import Deployment
from .router import Router, estimate_tokens

_FILLER_WORDS = (
    "the quick brown fox jumps over lazy dog river mountain valley cloud storm "
    "forest ocean desert circuit engine lantern harbor compass glacier meadow "
    "ember horizon lattice prism velvet granite willow canyon"
).split()


def _filler_prompt(word_count: int = 150) -> str:
    """Randomly shuffled filler text -- not a fixed string, so repeated
    calibration pings don't just measure a cached shared-prefix hit."""
    words = random.choices(_FILLER_WORDS, k=word_count)
    random.shuffle(words)
    return " ".join(words)


def _stream_ping(deployment: Deployment, prompt_text: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": deployment.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": True,
    }
    if deployment.api_base:
        kwargs["api_base"] = deployment.api_base
    if deployment.api_key:
        kwargs["api_key"] = deployment.api_key

    start = time.monotonic()
    first_reasoning_at: float | None = None
    first_content_at: float | None = None
    last_content_at: float | None = None
    content_chars = 0

    for chunk in litellm.completion(**kwargs):
        delta = chunk.choices[0].delta
        now = time.monotonic()
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if reasoning and first_reasoning_at is None:
            first_reasoning_at = now
        if content:
            if first_content_at is None:
                first_content_at = now
            last_content_at = now
            content_chars += len(content)

    return {
        "start": start,
        "first_reasoning_at": first_reasoning_at,
        "first_content_at": first_content_at,
        "last_content_at": last_content_at,
        "content_tokens": max(1, content_chars // 4) if content_chars else 0,
    }


async def calibrate(
    router: Router,
    pool_name: str,
    model_id: str | None = None,
    pings: int = 3,
) -> dict[str, dict[str, Any]]:
    states = router.pools[pool_name]
    if model_id:
        states = [s for s in states if s.deployment.id == model_id]

    results: dict[str, dict[str, Any]] = {}
    for state in states:
        load_time_seconds: float | None = None
        prefill_rates: list[float] = []
        decode_rates: list[float] = []
        completion_tokens: list[int] = []

        for i in range(pings):
            prompt_text = _filler_prompt()
            prompt_tokens = estimate_tokens([{"role": "user", "content": prompt_text}])
            timestamps = await asyncio.to_thread(_stream_ping, state.deployment, prompt_text)
            end = timestamps["last_content_at"] or timestamps["start"]

            if i == 0:
                # The first ping pays for a cold model load -- report it
                # separately, never blend it into the throughput rates.
                load_time_seconds = end - timestamps["start"]
                continue

            prefill_anchor = (
                timestamps["first_reasoning_at"] or timestamps["first_content_at"] or end
            )
            prefill_time = max(prefill_anchor - timestamps["start"], 1e-6)
            decode_time = max(end - prefill_anchor, 1e-6)
            prefill_rates.append(prompt_tokens / prefill_time)
            if timestamps["content_tokens"]:
                decode_rates.append(timestamps["content_tokens"] / decode_time)
                completion_tokens.append(timestamps["content_tokens"])

        metrics = {
            "load_time_seconds": load_time_seconds,
            "prefill_tokens_per_second": mean(prefill_rates) if prefill_rates else None,
            "decode_tokens_per_second": mean(decode_rates) if decode_rates else None,
            "avg_completion_tokens": mean(completion_tokens) if completion_tokens else None,
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }
        router.save_metrics_entry(pool_name, state.deployment.id, metrics)
        results[state.deployment.id] = metrics

    return results


async def run_adhoc(
    router: Router,
    pool_name: str,
    prompts: list[dict[str, Any]],
    repeat: int = 1,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    states = router.pools[pool_name]
    if model_id:
        states = [s for s in states if s.deployment.id == model_id]

    outputs: list[dict[str, Any]] = []
    for state in states:
        for item in prompts:
            messages = [{"role": "user", "content": item["prompt"]}]
            response_format = item.get("response_format")
            for rep in range(repeat):
                result = await router.complete(
                    pool=pool_name,
                    messages=messages,
                    response_format=response_format,
                    deployment_id=state.deployment.id,
                )
                outputs.append(
                    {
                        "deployment_id": state.deployment.id,
                        "prompt": item["prompt"],
                        "repeat_index": rep,
                        "success": result.success,
                        "content": result.content,
                        "attempts": result.attempts,
                        "latency_seconds": result.latency_seconds,
                        "blocked_reason": result.blocked_reason,
                    }
                )
    return outputs
