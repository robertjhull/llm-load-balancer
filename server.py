"""FastAPI app: /complete /health /pools /benchmark."""
from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import benchmark
from .config import load_pools
from .router import NoHealthyDeploymentError, Router


class CompleteRequest(BaseModel):
    pool: str
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None = None
    max_retries: int | None = None
    trace_id: str | None = None


class CompleteResponse(BaseModel):
    success: bool
    content: str | None = None
    model_id: str | None = None
    attempts: int
    blocked_reason: str | None = None
    latency_seconds: float


class BenchmarkRequest(BaseModel):
    pool: str
    model_id: str | None = None
    prompts: list[dict[str, Any]] | None = None
    repeat: int = 1


def create_app(
    config_path: str,
    log_dir: str,
    metrics_path: str,
    health_interval: int,
) -> FastAPI:
    config = load_pools(config_path)
    router = Router(
        config=config,
        metrics_path=metrics_path,
        log_dir=log_dir,
        health_interval=health_interval,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await router.start()
        try:
            yield
        finally:
            await router.stop()

    app = FastAPI(title="Switchboard", lifespan=lifespan)
    app.state.router = router

    @app.post("/complete", response_model=CompleteResponse)
    async def complete(req: CompleteRequest) -> CompleteResponse:
        if req.pool not in router.pools:
            raise HTTPException(status_code=404, detail=f"unknown pool: {req.pool!r}")
        try:
            result = await router.complete(
                pool=req.pool,
                messages=req.messages,
                response_format=req.response_format,
                max_retries=req.max_retries,
                trace_id=req.trace_id,
            )
        except NoHealthyDeploymentError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return CompleteResponse(**dataclasses.asdict(result))

    @app.get("/health")
    async def health(pool: str | None = None) -> dict:
        if pool is not None and pool not in router.pools:
            raise HTTPException(status_code=404, detail=f"unknown pool: {pool!r}")
        return await router.check_health(pool)

    @app.get("/pools")
    async def pools_endpoint() -> dict:
        return router.describe_pools()

    @app.post("/benchmark")
    async def benchmark_endpoint(req: BenchmarkRequest) -> dict:
        if req.pool not in router.pools:
            raise HTTPException(status_code=404, detail=f"unknown pool: {req.pool!r}")
        if req.prompts is None:
            metrics = await benchmark.calibrate(router, req.pool, model_id=req.model_id)
            return {"mode": "calibration", "metrics": metrics}
        outputs = await benchmark.run_adhoc(
            router, req.pool, req.prompts, repeat=req.repeat, model_id=req.model_id
        )
        return {"mode": "adhoc", "results": outputs}

    return app
