"""argparse CLI; talks to a running server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import client

DEFAULT_SERVER = "http://127.0.0.1:8800"
DEFAULT_LOG_DIR = str(Path.home() / ".switchboard" / "logs")
DEFAULT_METRICS_PATH = str(Path.home() / ".switchboard" / "metrics.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="switchboard")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the switchboard HTTP server")
    serve.add_argument("--config", default="pools.json")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8800)
    serve.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    serve.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    serve.add_argument("--health-interval", type=int, default=60)

    pools = sub.add_parser("pools", help="List configured pools and their health")
    pools.add_argument("--server", default=DEFAULT_SERVER)

    health = sub.add_parser("health", help="Check deployment health")
    health.add_argument("--pool")
    health.add_argument("--server", default=DEFAULT_SERVER)

    bench = sub.add_parser("benchmark", help="Calibrate (default) or ad-hoc benchmark a pool")
    bench.add_argument("--pool", required=True)
    bench.add_argument("--prompts", help="JSONL file of {prompt, response_format?}; omit to calibrate")
    bench.add_argument("--repeat", type=int, default=1)
    bench.add_argument("--server", default=DEFAULT_SERVER)

    return parser


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(
        config_path=args.config,
        log_dir=args.log_dir,
        metrics_path=args.metrics_path,
        health_interval=args.health_interval,
    )
    uvicorn.run(app, host=args.host, port=args.port)


def _cmd_pools(args: argparse.Namespace) -> None:
    data = client.get(f"{args.server}/pools")
    print(json.dumps(data, indent=2))


def _cmd_health(args: argparse.Namespace) -> None:
    url = f"{args.server}/health"
    if args.pool:
        url += f"?pool={args.pool}"
    data = client.get(url)
    print(json.dumps(data, indent=2))


def _cmd_benchmark(args: argparse.Namespace) -> None:
    body: dict = {"pool": args.pool, "repeat": args.repeat}
    if args.prompts:
        prompts = []
        with open(args.prompts) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
        body["prompts"] = prompts
    data = client.post(f"{args.server}/benchmark", body)
    print(json.dumps(data, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            _cmd_serve(args)
        elif args.command == "pools":
            _cmd_pools(args)
        elif args.command == "health":
            _cmd_health(args)
        elif args.command == "benchmark":
            _cmd_benchmark(args)
    except client.ClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
