from __future__ import annotations

import asyncio
import heapq
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

DEFAULT_DURATION_SECONDS = 5.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 120.0


class Deployment(BaseModel):
    id: str | None = None
    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    capacity: int = 1
    max_context_tokens: int | None = None
    max_retries: int | None = None
    request_timeout: float | None = None


@dataclass
class PoolsConfig:
    pools: dict[str, list[Deployment]]


@dataclass
class CallActivity:
    label: dict[str, Any] | None = None
    state: str = "queued"  # queued | inferring | streaming | complete
    started_at: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    tokens_emitted: int = 0
    last_chunk_at: float | None = None


class DeploymentState:
    def __init__(self, deployment: Deployment):
        self.deployment = deployment
        self.lanes: list[float] = [0.0] * max(1, deployment.capacity)
        heapq.heapify(self.lanes)
        self.semaphore = asyncio.Semaphore(max(1, deployment.capacity))
        self.healthy = False
        self.prefill_tokens_per_second: float | None = None
        self.decode_tokens_per_second: float | None = None
        self.avg_completion_tokens: float | None = None
        self.active_calls: dict[str, CallActivity] = {}

    def get_max_retries(self, override: int | None) -> int:
        if override is not None:
            return override
        if self.deployment.max_retries is not None:
            return self.deployment.max_retries
        return DEFAULT_MAX_RETRIES

    def get_timeout(self) -> float:
        return self.deployment.request_timeout or DEFAULT_REQUEST_TIMEOUT

    def estimate_duration(self, prompt_tokens: int) -> float:
        if self.prefill_tokens_per_second and self.decode_tokens_per_second and self.avg_completion_tokens:
            return (
                prompt_tokens / self.prefill_tokens_per_second
                + self.avg_completion_tokens / self.decode_tokens_per_second
            )
        return DEFAULT_DURATION_SECONDS

    async def is_healthy(self) -> bool:
        if not self.deployment.api_base:
            return True

        parts = urlsplit(self.deployment.api_base)
        url = f"{parts.scheme}://{parts.netloc}/health"

        def _ping() -> bool:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    return 200 <= resp.status < 300
            except (urllib.error.URLError, OSError, ValueError):
                return False

        self.healthy = await asyncio.to_thread(_ping)
        return self.healthy


def load_pools(path: str | Path) -> PoolsConfig:
    raw = json.loads(Path(path).read_text())

    pools: dict[str, list[Deployment]] = {}
    id: int = 0
    for pool_name, deployments in raw.items():
        models = [Deployment(**d) for d in deployments]
        names = [m.api_base + m.model for m in models]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate model url in pool {pool_name!r}")
        for model in models:
            id += 1
            model.id = f"m{id}"
        pools[pool_name] = models

    return PoolsConfig(pools)
