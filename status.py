from __future__ import annotations

import asyncio
import html
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import DeploymentState
    from .router import Router

_STYLE = """
body { font-family: Arial, sans-serif; background-color: #1e1e1e; color: #ddd; padding: 20px; }
h2 { color: #eee; }
.pool { background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 6px; padding: 15px;
  margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
table { width: 100%; border-collapse: collapse; table-layout: fixed; }
th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #3a3a3a;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
th { color: #999; }
.healthy, .unhealthy { font-size: 1.2em; }
.healthy { color: #3ddc84; }
.unhealthy { color: #e05d5d; }
"""


# -- health ------------------------------------------------------------

async def check_health(pools: dict[str, list["DeploymentState"]], pool: str | None = None) -> None:
    targets = [
        state
        for name, states in pools.items()
        for state in states
        if pool is None or name == pool
    ]

    await asyncio.gather(*(state.is_healthy() for state in targets))


async def run_health_loop(pools: dict[str, list["DeploymentState"]], interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        await check_health(pools)


# -- rendering -----------------------------------------------------------

def _rate(value: float | None, suffix: str) -> str:
    return f"{value:.1f}{suffix}" if value else "-"


def _queued_cell(state: "DeploymentState") -> str:
    return str(sum(1 for a in state.active_calls.values() if a.state == "queued"))


def _status_cell(state: "DeploymentState") -> str:
    now = time.monotonic()
    active = [a for a in state.active_calls.values() if a.state != "queued"]
    if not active:
        return "idle"
    return "<br>".join(
        f"{html.escape(a.state)} ({now - a.started_at:.1f}s)" for a in active
    )


_COLUMNS: list[tuple[str, str, Any]] = [
    ("", "3%", lambda s: f'<span class="{"healthy" if s.healthy else "unhealthy"}">●</span>'),
    ("id", "7%", lambda s: html.escape(s.deployment.id or "-")),
    ("model", "30%", lambda s: html.escape(s.deployment.model or "-")),
    ("api_base", "18%", lambda s: html.escape(s.deployment.api_base or "-")),
    ("capacity", "6%", lambda s: str(s.deployment.capacity)),
    ("max_context", "9%", lambda s: str(s.deployment.max_context_tokens or "-")),
    ("tok/s", "6%", lambda s: _rate(s.decode_tokens_per_second, "")),
    ("queued", "9%", _queued_cell),
    ("status", "12%", _status_cell),
]


def _pool_table(states: list["DeploymentState"]) -> str:
    colgroup = "".join(f'<col style="width: {width}">' for _, width, _ in _COLUMNS)
    header = "".join(f"<th>{html.escape(name)}</th>" for name, _, _ in _COLUMNS)
    rows = "".join(
        "<tr>" + "".join(f"<td>{getter(state)}</td>" for _, _, getter in _COLUMNS) + "</tr>"
        for state in states
    )
    return f"<table><colgroup>{colgroup}</colgroup><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


def render(router: "Router") -> str:
    pools_html = "".join(
        f'<div class="pool"><h2>{html.escape(pool)}</h2>{_pool_table(states)}</div>'
        for pool, states in router.pools.items()
    )

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f"<title>Status</title><style>{_STYLE}</style></head><body>"
        f"{pools_html}"
        "</body></html>"
    )
