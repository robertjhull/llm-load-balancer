from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import status
from .config import load_pools
from .router import Router


class JobItemRequest(BaseModel):
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    max_tool_rounds: int | None = None
    final_response_format: dict[str, Any] | None = None
    final_prompt: str | None = None
    max_retries: int | None = None
    trace_id: str | None = None
    pin_to_model_id: str | None = None


class SubmitRequest(BaseModel):
    pool: str
    items: list[JobItemRequest]


class ResumeRequest(BaseModel):
    tool_results: list[dict[str, Any]]


def create_app(
    config_path: str,
    log_dir: str,
    metrics_path: str,
    health_interval: int,
) -> FastAPI:
    config = load_pools(config_path)
    router = Router(config=config, metrics_path=metrics_path, log_dir=log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await status.check_health(router.pools)
        health_task = asyncio.create_task(status.run_health_loop(router.pools, health_interval))
        try:
            yield
        finally:
            health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await health_task

    app = FastAPI(title="Switchboard", lifespan=lifespan)
    app.state.router = router

    @app.post("/jobs")
    async def submit_jobs(req: SubmitRequest) -> dict:
        try:
            job_ids = router.submit(req.pool, [item.model_dump() for item in req.items])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"job_ids": job_ids}

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str) -> dict:
        described = router.describe(job_id)
        if described is None:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id!r}")
        return described

    @app.post("/jobs/{job_id}/resume")
    async def job_resume(job_id: str, req: ResumeRequest) -> dict:
        try:
            router.resume(job_id, req.tool_results)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"resumed": True}

    @app.post("/jobs/{job_id}/cancel")
    async def job_cancel(job_id: str) -> dict:
        if not router.cancel(job_id):
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id!r}")
        return {"cancelled": True}

    @app.get("/status", response_class=HTMLResponse)
    async def status_endpoint() -> str:
        return status.render(router)

    return app
