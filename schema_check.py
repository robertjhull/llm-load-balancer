"""JSON-schema validate-and-retry wrapper."""
from __future__ import annotations

import json

import jsonschema


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
