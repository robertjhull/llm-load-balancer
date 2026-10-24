"""Load and validate pools.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

DEFAULT_MAX_RETRIES = 2


class Deployment(BaseModel):
    id: str
    model: str
    api_base: str | None = None
    api_key: str | None = None
    capacity: int = 1
    max_context_tokens: int | None = None


@dataclass
class PoolsConfig:
    pools: dict[str, list[Deployment]]
    max_retries: int = DEFAULT_MAX_RETRIES


def load_pools(path: str | Path) -> PoolsConfig:
    raw = json.loads(Path(path).read_text())
    defaults = raw.pop("_defaults", {})

    pools: dict[str, list[Deployment]] = {}
    for pool_name, deployments in raw.items():
        parsed = [Deployment(**d) for d in deployments]
        ids = [d.id for d in parsed]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate deployment id in pool {pool_name!r}")
        pools[pool_name] = parsed

    return PoolsConfig(
        pools=pools,
        max_retries=defaults.get("max_retries", DEFAULT_MAX_RETRIES),
    )
