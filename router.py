from __future__ import annotations

import asyncio
import heapq
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from .config import CallActivity, DeploymentState, PoolsConfig
from .infer import infer

DEFAULT_MAX_TOOL_ROUNDS = 3
DEFAULT_FINAL_PROMPT = "Based on everything above, provide your final answer now."

# Exponential moving average weight for live per-request metric samples --
# low enough that one unusually slow/fast call doesn't swing scheduling
# estimates, high enough to track real drift (e.g. a model swap behind
# the same deployment) within a handful of requests.
_METRICS_EMA_ALPHA = 0.3

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


class NoHealthyDeploymentError(Exception):
    """Raised when every deployment in a pool is currently unhealthy."""


class SchemaValidationError(Exception):
    """Raised for either malformed JSON or schema-mismatched JSON."""


def validate(raw_text: str, schema: dict) -> dict:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON: {exc}") from exc

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError(f"schema validation failed: {exc.message}") from exc

    return data


@dataclass
class CompleteResult:
    success: bool
    content: str | None
    model_id: str | None
    attempts: int
    blocked_reason: str | None
    latency_seconds: float
    tool_calls: list[dict[str, Any]] | None = None


def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": tc.id,
            "type": tc.type,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in tool_calls
    ]


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Character-count heuristic -- good enough for routing/preflight, not exact."""
    text = " ".join(str(m.get("content") or "") for m in messages)
    return max(1, len(text) // 4)


@dataclass
class Job:
    job_id: str
    pool: str
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    final_response_format: dict[str, Any] | None = None
    final_prompt: str | None = None
    max_retries: int | None = None
    trace_id: str | None = None
    deployment_id: str | None = None
    rounds_done: int = 0
    status: str = "running"  # running | needs_tool_result | completed | failed | cancelled
    pending_tool_calls: list[dict[str, Any]] | None = None
    result: CompleteResult | None = None
    cancel_requested: bool = False
    task: asyncio.Task | None = None
    _resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    _supplied_tool_results: list[dict[str, Any]] | None = None


class Router:
    def __init__(
        self,
        config: PoolsConfig,
        metrics_path: str | Path,
        log_dir: str | Path,
    ):
        self.pools: dict[str, list[DeploymentState]] = {
            name: [DeploymentState(d) for d in deployments]
            for name, deployments in config.pools.items()
        }
        self.jobs: dict[str, Job] = {}
        self.metrics_path = Path(metrics_path)
        self.log_dir = Path(log_dir)

        self._apply_metrics(self._load_metrics())

    # -- routing -----------------------------------------------------

    def _select(self, pool: str, prompt_tokens: int) -> DeploymentState:
        candidates = [s for s in self.pools[pool] if s.healthy]
        if not candidates:
            raise NoHealthyDeploymentError(f"no healthy deployment in pool {pool!r}")

        best = min(candidates, key=lambda s: s.lanes[0])
        duration = best.estimate_duration(prompt_tokens)

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

    def _update_metrics(self, state: DeploymentState, prompt_tokens: int, activity: CallActivity) -> None:
        if activity.first_token_at is None or not activity.tokens_emitted:
            return  # errored out before producing anything usable

        prefill_elapsed = activity.first_token_at - activity.started_at
        decode_elapsed = (activity.last_chunk_at or activity.first_token_at) - activity.first_token_at

        def ema(prev: float | None, sample: float) -> float:
            return sample if prev is None else _METRICS_EMA_ALPHA * sample + (1 - _METRICS_EMA_ALPHA) * prev

        if prefill_elapsed > 0:
            state.prefill_tokens_per_second = ema(state.prefill_tokens_per_second, prompt_tokens / prefill_elapsed)
        if decode_elapsed > 0:
            state.decode_tokens_per_second = ema(
                state.decode_tokens_per_second, activity.tokens_emitted / decode_elapsed
            )
        state.avg_completion_tokens = ema(state.avg_completion_tokens, activity.tokens_emitted)

    async def _run_round(
        self,
        job: Job,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
    ) -> CompleteResult:
        prompt_tokens = estimate_tokens(messages)
        state = (
            self._get_state(job.pool, job.deployment_id)
            if job.deployment_id is not None
            else self._select(job.pool, prompt_tokens)
        )
        retries = state.get_max_retries(job.max_retries)
        request_timeout = state.get_timeout()

        # Registered before the semaphore is acquired so a call genuinely
        # waiting on capacity is visible as "queued", not absent --
        # display-only (`status.render()`), never fed back into
        # scheduling. Always removed on the way out, however this
        # returns.
        call_id = uuid.uuid4().hex
        activity = CallActivity(label={"job_id": job.job_id, "round": job.rounds_done + 1})
        state.active_calls[call_id] = activity

        start = time.monotonic()
        try:
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
                    activity.state = "inferring"
                    activity.started_at = time.monotonic()
                    activity.first_token_at = None
                    activity.tokens_emitted = 0
                    activity.last_chunk_at = None
                    try:
                        message = await asyncio.to_thread(
                            infer,
                            state.deployment,
                            messages,
                            response_format,
                            tools,
                            tool_choice,
                            request_timeout,
                            activity,
                        )
                    except Exception as exc:  # noqa: BLE001 -- surfaced via blocked_reason
                        last_error = exc
                        self._log_attempt(job.trace_id, state.deployment.id, attempt, None, exc)
                        continue

                    self._update_metrics(state, prompt_tokens, activity)
                    try:
                        await asyncio.to_thread(self._persist_metrics)
                    except OSError:
                        pass  # live scheduling already has the update; the disk copy is best-effort

                    # A tool-call turn is a normal, expected outcome -- not
                    # a validation failure -- so it's returned immediately,
                    # without retrying or running it through schema validation.
                    raw_tool_calls = getattr(message, "tool_calls", None)
                    if raw_tool_calls:
                        return CompleteResult(
                            success=True,
                            content=message.content,
                            model_id=state.deployment.id,
                            attempts=attempt,
                            blocked_reason=None,
                            latency_seconds=time.monotonic() - start,
                            tool_calls=_serialize_tool_calls(raw_tool_calls),
                        )

                    text = strip_code_fence(message.content or "")

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
                        validate(text, response_format["schema"])
                    except SchemaValidationError as exc:
                        last_error = exc
                        self._log_attempt(job.trace_id, state.deployment.id, attempt, text, exc)
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
        finally:
            state.active_calls.pop(call_id, None)
            if job.deployment_id is None:
                job.deployment_id = state.deployment.id

    # -- jobs: submit, poll, resume, cancel -----------------------------

    def submit(self, pool: str, items: list[dict[str, Any]]) -> list[str]:
        if pool not in self.pools:
            raise KeyError(f"unknown pool: {pool!r}")

        job_ids: list[str] = []
        for item in items:
            job = Job(
                job_id=f"job_{uuid.uuid4().hex[:12]}",
                pool=pool,
                messages=list(item["messages"]),
                response_format=item.get("response_format"),
                tools=item.get("tools"),
                tool_choice=item.get("tool_choice"),
                max_tool_rounds=(
                    item["max_tool_rounds"]
                    if item.get("max_tool_rounds") is not None
                    else DEFAULT_MAX_TOOL_ROUNDS
                ),
                final_response_format=item.get("final_response_format"),
                final_prompt=item.get("final_prompt"),
                max_retries=item.get("max_retries"),
                trace_id=item.get("trace_id"),
                deployment_id=item.get("pin_to_model_id"),
            )
            self.jobs[job.job_id] = job
            job.task = asyncio.create_task(self._run_job(job))
            job_ids.append(job.job_id)
        return job_ids

    async def _run_job(self, job: Job) -> None:
        try:
            while True:
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.result = CompleteResult(
                        success=False,
                        content=None,
                        model_id=job.deployment_id,
                        attempts=0,
                        blocked_reason="job cancelled",
                        latency_seconds=0.0,
                    )
                    return

                if job.tools is not None and job.rounds_done >= job.max_tool_rounds:
                    if job.final_response_format is not None:
                        job.messages.append(
                            {"role": "user", "content": job.final_prompt or DEFAULT_FINAL_PROMPT}
                        )
                        job.result = await self._run_round(
                            job, job.messages, job.final_response_format, None, None
                        )
                    else:
                        job.result = CompleteResult(
                            success=False,
                            content=None,
                            model_id=job.deployment_id,
                            attempts=0,
                            blocked_reason="tool round limit exceeded with an unresolved tool call",
                            latency_seconds=0.0,
                        )
                    job.status = "completed" if job.result.success else "failed"
                    return

                use_tools = job.tools is not None
                result = await self._run_round(
                    job,
                    job.messages,
                    job.response_format if not use_tools else None,
                    job.tools if use_tools else None,
                    job.tool_choice if use_tools else None,
                )

                if use_tools and result.tool_calls:
                    job.rounds_done += 1
                    job.pending_tool_calls = result.tool_calls
                    job.status = "needs_tool_result"
                    job._resume_event.clear()
                    await job._resume_event.wait()
                    job.messages.append(
                        {"role": "assistant", "content": result.content, "tool_calls": result.tool_calls}
                    )
                    job.messages.extend(
                        {"role": "tool", "tool_call_id": tr["tool_call_id"], "content": tr["content"]}
                        for tr in (job._supplied_tool_results or [])
                    )
                    job.status = "running"
                    continue

                # Tool phase ended cleanly (model didn't ask for a tool this
                # round), or there was no tool capability to begin with.
                if use_tools and job.final_response_format is not None:
                    job.messages.append(
                        {"role": "user", "content": job.final_prompt or DEFAULT_FINAL_PROMPT}
                    )
                    job.result = await self._run_round(
                        job, job.messages, job.final_response_format, None, None
                    )
                else:
                    job.result = result
                job.status = "completed" if job.result.success else "failed"
                return
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.result = CompleteResult(
                success=False,
                content=None,
                model_id=job.deployment_id,
                attempts=0,
                blocked_reason="job cancelled",
                latency_seconds=0.0,
            )
        except Exception as exc:
            job.status = "failed"
            job.result = CompleteResult(
                success=False,
                content=None,
                model_id=job.deployment_id,
                attempts=0,
                blocked_reason=str(exc),
                latency_seconds=0.0,
            )

    def describe(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "pool": job.pool,
            "status": job.status,
            "model_id": job.deployment_id,
            "pending_tool_calls": job.pending_tool_calls if job.status == "needs_tool_result" else None,
            "result": asdict(job.result) if job.result is not None else None,
        }

    def resume(self, job_id: str, tool_results: list[dict[str, Any]]) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id!r}")
        if job.status != "needs_tool_result":
            raise ValueError(f"job {job_id!r} is not awaiting a tool result (status={job.status!r})")

        supplied_ids = {tr["tool_call_id"] for tr in tool_results}
        expected_ids = {tc["id"] for tc in (job.pending_tool_calls or [])}
        if supplied_ids != expected_ids:
            raise ValueError(
                f"job {job_id!r}: supplied tool_call_ids {sorted(supplied_ids)} "
                f"do not match pending {sorted(expected_ids)}"
            )

        job._supplied_tool_results = tool_results
        job._resume_event.set()

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return True
        job.cancel_requested = True
        if job.task is not None and not job.task.done():
            job.task.cancel()
        return True

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

    def _persist_metrics(self) -> None:
        store = {
            self._metrics_key(pool, state.deployment.id): {
                "prefill_tokens_per_second": state.prefill_tokens_per_second,
                "decode_tokens_per_second": state.decode_tokens_per_second,
                "avg_completion_tokens": state.avg_completion_tokens,
            }
            for pool, states in self.pools.items()
            for state in states
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(store, indent=2))
