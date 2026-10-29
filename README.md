# LLM Load Balancer

A capacity-aware HTTP router that sits in front of several LLM deployments
(local or hosted) and schedules jobs across them, with schema-validated
output, retries, tool-call loops, and health checks.

## Install

```
pip install -e .
```

Create a `pools.json` (see `pools.example.json`):

```json
{
  "<pool_name>": [
    {
      "model": "openai/gpt-4o",
      "api_base": "http://127.0.0.1:8000/v1",
      "api_key": "sk-...",
      "capacity": 4,
      "max_context_tokens": 8192,
      "max_retries": 3,
      "request_timeout": 120
    }
  ]
}
```

- `model` — required; a litellm model string (`<provider>/<model>`).
- `api_base` — override endpoint, e.g. for a local/self-hosted deployment.
  Omit for hosted providers litellm resolves by model name alone.
- `api_key` — per-deployment API key. Omit to fall back to the provider's
  usual env var.
- `capacity` — concurrent in-flight requests this deployment serves before
  new requests queue (default 1).
- `max_context_tokens` — optional; lets the router skip a request outright
  if the estimated prompt is already too large.
- `max_retries` / `request_timeout`

Logs and the metrics store default to
`~/.store/` (`--log-dir` / `--metrics-path` override).

## Run

```
llb serve --config pools.json
```

| flag | default |
|---|---|
| `--config` | `pools.json` |
| `--host` | `127.0.0.1` |
| `--port` | `8800` |
| `--log-dir` | `~/.store/logs` — per-attempt raw response/error, one subdir per `trace_id` |
| `--metrics-path` | `~/.store/metrics.json` — persisted throughput metrics, loaded on startup |
| `--health-interval` | `60` — seconds between background health checks |

`llb list` — print each pool's deployments with health, capacity, tok/s,
queue depth, and current status (same data as `/status`, as a plain-text
table). Takes `--config`, `--log-dir`, `--metrics-path`.

## HTTP API

- `POST /jobs` — submit jobs to a pool.
  ```json
  {
    "pool": "default_pool",
    "items": [
      {
        "messages": [{"role": "user", "content": "..."}],
        "response_format": null,
        "tools": null,
        "tool_choice": null,
        "max_tool_rounds": 3,
        "final_response_format": null,
        "final_prompt": null,
        "max_retries": null,
        "trace_id": null,
        "pin_to_model_id": null
      }
    ]
  }
  ```
  → `{"job_ids": ["..."]}`

- `GET /jobs/{job_id}` — poll a job's state.
  → `{"job_id", "pool", "status", "model_id", "pending_tool_calls", "result"}`
  `status` is one of `running | needs_tool_result | completed | failed | cancelled`.

- `POST /jobs/{job_id}/resume` — supply tool results for a job currently
  `needs_tool_result`.
  ```json
  {"tool_results": [{"tool_call_id": "...", "..."}]}
  ```
  → `{"resumed": true}`

- `POST /jobs/{job_id}/cancel` — cancel a running job.
  → `{"cancelled": true}`

- `GET /status` — HTML dashboard of pool/deployment health and in-flight calls.
