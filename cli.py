from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_POOLS_CONFIG = "pools.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8800
DEFAULT_LOG_DIR = str(Path.home() / ".store" / "logs")
DEFAULT_METRICS_PATH = str(Path.home() / ".store" / "metrics.json")
DEFAULT_HEALTH_INTERVAL = 60

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llb")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP server")
    serve.add_argument("--config", default=DEFAULT_POOLS_CONFIG)
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    serve.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    serve.add_argument("--health-interval", type=int, default=DEFAULT_HEALTH_INTERVAL)

    list_ = sub.add_parser("list", help="List configured pools and their health")
    list_.add_argument("--config", default=DEFAULT_POOLS_CONFIG)
    list_.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    list_.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)

    return parser


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(
        config_path=args.config,
        log_dir=args.log_dir,
        metrics_path=args.metrics_path,
        health_interval=args.health_interval,
    )
    uvicorn.run(app, host=args.host, port=args.port)

def _list(args: argparse.Namespace) -> None:
    import asyncio

    from . import status
    from .config import load_pools
    from .router import Router

    config = load_pools(args.config)
    router = Router(config=config, metrics_path=args.metrics_path, log_dir=args.log_dir)
    asyncio.run(status.check_health(router.pools))
    print(status.render_text(router))

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            _serve(args)
        elif args.command == "list":
            _list(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
