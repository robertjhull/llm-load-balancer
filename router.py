"""Model selection, per-deployment capacity, the call path, and health checks."""
from __future__ import annotations

import asyncio
import contextlib
import heapq
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import litellm

from .config import Deployment, PoolsConfig
from . import schema_check

DEFAULT_DURATION_SECONDS = 5.0

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


class NoHealthyDeploymentError(Exception):
    """Raised when every deployment in a pool is currently unhealthy."""


@dataclass
class CompleteResult:
    success: bool
    content: str | None
    model_id: str | None
    attempts: int
    blocked_reason: str | None
    latency_seconds: float


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Character-count heuristic -- good enough for routing/preflight, not exact."""
    text = " ".join(str(m.get("content") or "") for m in messages)
    return max(1, len(text) // 4)


async def _ping_health(deployment: Deployment) -> bool:
    if not deployment.api_base:
        return True

    parts = urlsplit(deployment.api_base)
    url = f"{parts.scheme}://{parts.netloc}/health"

    def _ping() -> bool:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    return await asyncio.to_thread(_ping)


class DeploymentState:
    def __init__(self, deployment: Deployment):
        self.deployment = deployment
        self.lanes: list[float] = [0.0] * max(1, deployment.capacity)
        heapq.heapify(self.lanes)
        self.semaphore = asyncio.Semaphore(max(1, deployment.capacity))
        self.healthy = True
        self.prefill_tokens_per_second: float | None = None
        self.decode_tokens_per_second: float | None = None
        self.avg_completion_tokens: float | None = None
        self.load_time_seconds: float | None = None


class Router:
    def __init__(
        self,
        config: PoolsConfig,
        metrics_path: str | Path,
        log_dir: str | Path,
        health_interval: int = 60,
    ):
        self.pools: dict[str, list[DeploymentState]] = {
            name: [DeploymentState(d) for d in deployments]
            for name, deployments in config.pools.items()
        }
        self.default_max_retries = config.max_retries
        self.metrics_path = Path(metrics_path)
        self.log_dir = Path(log_dir)
        self.health_interval = health_interval
        self._health_task: asyncio.Task | None = None

        self._apply_metrics(self._load_metrics())

    # -- lifecycle ---------------------------------------------------

    async def start(self) -> None:
        await self.check_health()
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.health_interval)
            await self.check_health()

    # -- health --------------------------------------------------------

    async def check_health(self, pool: str | None = None) -> dict[str, dict[str, bool]]:
        targets = [
            (name, state)
            for name, states in self.pools.items()
            for state in states
            if pool is None or name == pool
        ]

        async def _one(state: DeploymentState) -> None:
            state.healthy = await _ping_health(state.deployment)

        await asyncio.gather(*(_one(state) for _, state in targets))

        report: dict[str, dict[str, bool]] = {}
        for name, state in targets:
            report.setdefault(name, {})[state.deployment.id] = state.healthy
        return report

    # -- introspection ---------------------------------------------------

    def describe_pools(self) -> dict[str, list[dict[str, Any]]]:
        return {
            pool: [
                {
                    "id": state.deployment.id,
                    "model": state.deployment.model,
                    "api_base": state.deployment.api_base,
                    "capacity": state.deployment.capacity,
                    "max_context_tokens": state.deployment.max_context_tokens,
                    "healthy": state.healthy,
                    "calibration": {
                        "prefill_tokens_per_second": state.prefill_tokens_per_second,
                        "decode_tokens_per_second": state.decode_tokens_per_second,
                        "avg_completion_tokens": state.avg_completion_tokens,
                        "load_time_seconds": state.load_time_seconds,
                    },
                }
                for state in states
            ]
            for pool, states in self.pools.items()
        }

    # -- routing -----------------------------------------------------

    def _estimate_duration(self, state: DeploymentState, prompt_tokens: int) -> float:
        if (
            state.prefill_tokens_per_second
            and state.decode_tokens_per_second
            and state.avg_completion_tokens
        ):
            return (
                prompt_tokens / state.prefill_tokens_per_second
                + state.avg_completion_tokens / state.decode_tokens_per_second
            )
        return DEFAULT_DURATION_SECONDS

    def _select(self, pool: str, prompt_tokens: int) -> DeploymentState:
        candidates = [s for s in self.pools[pool] if s.healthy]
        if not candidates:
            raise NoHealthyDeploymentError(f"no healthy deployment in pool {pool!r}")

        best = min(candidates, key=lambda s: s.lanes[0])
        duration = self._estimate_duration(best, prompt_tokens)

        now = time.monotonic()
        free_at = heapq.heappop(best.lanes)
        start = max(now, free_at)
        heapq.heappush(best.lanes, start + duration)
        return best

    def _get_state(self, pool: str, deployment_id: str) -> DeploymentState:
        for state in self.pools[pool]:
            if state.deployment.id == deployment_id:
                return state
        raise KeyError(f"no deployment {deployment_id!r} in pool {pool!r}")

    # -- the call path -------------------------------------------------

    def _call_litellm(
        self,
        deployment: Deployment,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> str:
        kwargs: dict[str, Any] = {"model": deployment.model, "messages": messages}
        if deployment.api_base:
            kwargs["api_base"] = deployment.api_base
        if deployment.api_key:
            kwargs["api_key"] = deployment.api_key
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    def _log_attempt(
        self,
        trace_id: str | None,
        deployment_id: str,
        attempt: int,
        raw: str | None,
        error: Exception | None,
    ) -> None:
        directory = self.log_dir / (trace_id or "untraced")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{deployment_id}_attempt{attempt}.json"
        payload = {"raw_response": raw, "error": str(error) if error else None}
        path.write_text(json.dumps(payload, indent=2))

    async def complete(
        self,
        pool: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        max_retries: int | None = None,
        trace_id: str | None = None,
        deployment_id: str | None = None,
    ) -> CompleteResult:
        if pool not in self.pools:
            raise KeyError(f"unknown pool: {pool!r}")

        prompt_tokens = estimate_tokens(messages)
        state = (
            self._get_state(pool, deployment_id)
            if deployment_id is not None
            else self._select(pool, prompt_tokens)
        )
        retries = max_retries if max_retries is not None else self.default_max_retries

        start = time.monotonic()
        async with state.semaphore:
            mct = state.deployment.max_context_tokens
            if mct and prompt_tokens > mct:
                return CompleteResult(
                    success=False,
                    content=None,
                    model_id=state.deployment.id,
                    attempts=0,
                    blocked_reason=(
                        f"estimated prompt tokens ({prompt_tokens}) exceeds "
                        f"max_context_tokens ({mct})"
                    ),
                    latency_seconds=time.monotonic() - start,
                )

            last_error: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    raw = await asyncio.to_thread(
                        self._call_litellm, state.deployment, messages, response_format
                    )
                except Exception as exc:  # noqa: BLE001 -- surfaced via blocked_reason
                    last_error = exc
                    self._log_attempt(trace_id, state.deployment.id, attempt, None, exc)
                    continue

                text = strip_code_fence(raw or "")

                if response_format is None:
                    return CompleteResult(
                        success=True,
                        content=text,
                        model_id=state.deployment.id,
                        attempts=attempt,
                        blocked_reason=None,
                        latency_seconds=time.monotonic() - start,
                    )

                try:
                    schema_check.validate(text, response_format["schema"])
                except schema_check.SchemaValidationError as exc:
                    last_error = exc
                    self._log_attempt(trace_id, state.deployment.id, attempt, text, exc)
                    continue

                return CompleteResult(
                    success=True,
                    content=text,
                    model_id=state.deployment.id,
                    attempts=attempt,
                    blocked_reason=None,
                    latency_seconds=time.monotonic() - start,
                )

            return CompleteResult(
                success=False,
                content=None,
                model_id=state.deployment.id,
                attempts=retries,
                blocked_reason=f"all attempts failed: {last_error}",
                latency_seconds=time.monotonic() - start,
            )

    # -- metrics store -------------------------------------------------

    @staticmethod
    def _metrics_key(pool: str, deployment_id: str) -> str:
        return f"{pool}/{deployment_id}"

    def _load_metrics(self) -> dict[str, Any]:
        if self.metrics_path.exists():
            return json.loads(self.metrics_path.read_text())
        return {}

    def _apply_metrics(self, metrics: dict[str, Any]) -> None:
        for pool, states in self.pools.items():
            for state in states:
                entry = metrics.get(self._metrics_key(pool, state.deployment.id))
                if entry:
                    state.prefill_tokens_per_second = entry.get("prefill_tokens_per_second")
                    state.decode_tokens_per_second = entry.get("decode_tokens_per_second")
                    state.avg_completion_tokens = entry.get("avg_completion_tokens")
                    state.load_time_seconds = entry.get("load_time_seconds")

    def save_metrics_entry(self, pool: str, deployment_id: str, metrics_entry: dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        store = self._load_metrics()
        store[self._metrics_key(pool, deployment_id)] = metrics_entry
        self.metrics_path.write_text(json.dumps(store, indent=2))
        self._apply_metrics(store)
