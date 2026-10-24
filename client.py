"""Thin stdlib-urllib client used by cli.py to talk to a running server."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class ClientError(Exception):
    """One clear error type for a non-2xx response or a connection failure."""


def _request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise ClientError(f"{exc.code} {exc.reason}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ClientError(f"could not reach {url}: {exc.reason}") from exc


def get(url: str) -> dict:
    return _request("GET", url)


def post(url: str, body: dict) -> dict:
    return _request("POST", url, body)
