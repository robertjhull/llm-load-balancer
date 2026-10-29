from __future__ import annotations

import asyncio
import html
import urllib.error
import urllib.request
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .config import DeploymentState
    from .router import Job, Router

_STYLE = """
body { font-family: Arial, sans-serif; background-color: #f9f9f9; color: #333; padding: 20px; }
h1 { text-align: center; color: #2c3e50; }
h2 { color: #2c3e50; margin-top: 30px; }
.pool, .job { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 15px;
  margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.pool-name { font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
.entry { margin-bottom: 10px; }
.key { display: inline-block; width: 220px; font-weight: bold; color: #34495e; }
.value { display: inline-block; color: #2c3e50; }
.nested { margin-left: 20px; padding-left: 10px; border-left: 2px dashed #ccc; }
.healthy { color: #27ae60; }
.unhealthy { color: #c0392b; }
"""


# -- health ------------------------------------------------------------

async def _ping(api_base: str | None) -> bool:
    if not api_base:
        return True

    parts = urlsplit(api_base)
    url = f"{parts.scheme}://{parts.netloc}/health"

    def _ping_sync() -> bool:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    return await asyncio.to_thread(_ping_sync)


async def check_health(pools: dict[str, list["DeploymentState"]], pool: str | None = None) -> None:
    targets = [
        state
        for name, states in pools.items()
        for state in states
        if pool is None or name == pool
    ]
    unique_bases = {state.deployment.api_base for state in targets}
    results: dict[str | None, bool] = {}

    async def _ping_one(base: str | None) -> None:
        results[base] = await _ping(base)

    await asyncio.gather(*(_ping_one(base) for base in unique_bases))
    for state in targets:
        state.healthy = results[state.deployment.api_base]


async def run_health_loop(pools: dict[str, list["DeploymentState"]], interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        await check_health(pools)


# -- rendering -----------------------------------------------------------

def _entry(key: str, value: str) -> str:
    return f'<div class="entry"><div class="key">{html.escape(key)}:</div><div class="value">{value}</div></div>'


def _rate(value: float | None, suffix: str) -> str:
    return f"{value:.1f}{suffix}" if value else "-"


def _deployment_html(state: "DeploymentState") -> str:
    d = state.deployment
    health_class = "healthy" if state.healthy else "unhealthy"

    calls = list(state.active_calls.values())
    calls_html = "".join(
        f'<div class="entry nested">{html.escape(str(a.label))} -- {html.escape(a.state)}, '
        f"{a.tokens_emitted} tokens</div>"
        for a in calls
    ) or '<div class="entry nested">none</div>'

    return (
        '<div class="pool">'
        f'<div class="pool-name">{html.escape(d.name)}</div>'
        + _entry("model", html.escape(d.model))
        + _entry("api_base", html.escape(d.api_base or "-"))
        + _entry("capacity", str(d.capacity))
        + _entry("max_context_tokens", str(d.max_context_tokens or "-"))
        + _entry("healthy", f'<span class="{health_class}">{state.healthy}</span>')
        + _entry("prefill_tokens_per_second", _rate(state.prefill_tokens_per_second, " tok/s"))
        + _entry("decode_tokens_per_second", _rate(state.decode_tokens_per_second, " tok/s"))
        + _entry("avg_completion_tokens", _rate(state.avg_completion_tokens, " tok"))
        + _entry("active_calls", calls_html)
        + "</div>"
    )


def _job_html(job: "Job") -> str:
    return (
        '<div class="job">'
        + _entry("job_id", html.escape(job.job_id))
        + _entry("pool", html.escape(job.pool))
        + _entry("status", html.escape(job.status))
        + _entry("model_id", html.escape(job.deployment_id or "-"))
        + "</div>"
    )


def render(router: "Router") -> str:
    pools_html = "".join(
        f"<h2>{html.escape(pool)}</h2>" + "".join(_deployment_html(s) for s in states)
        for pool, states in router.pools.items()
    )

    running_jobs = [j for j in router.jobs.values() if j.status in ("running", "needs_tool_result")]
    jobs_html = "".join(_job_html(j) for j in running_jobs) or '<div class="job">no running jobs</div>'

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f"<title>Status</title><style>{_STYLE}</style></head><body>"
        "<h1>Model Status</h1>"
        f"{pools_html}"
        "<h2>Running Jobs</h2>"
        f"{jobs_html}"
        "</body></html>"
    )
