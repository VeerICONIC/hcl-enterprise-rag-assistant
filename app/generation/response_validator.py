"""Parsing and validating the model's JSON reply."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.generation.models import GenerationError, ModelOutput

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _extract_json(raw: str) -> dict:
    text = _FENCE.sub("", raw.strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise GenerationError("the model did not return a JSON object")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        raise GenerationError("the model returned malformed JSON") from None
    if not isinstance(data, dict):
        raise GenerationError("the model returned JSON that is not an object")
    return data


def parse_model_output(raw: str) -> ModelOutput:
    """Parse ``raw`` into a ``ModelOutput``; raise ``GenerationError`` if it is unusable."""
    if not raw or not raw.strip():
        raise GenerationError("the language model returned an empty response")
    data = _extract_json(raw)
    if isinstance(data.get("status"), str):
        data["status"] = data["status"].strip().lower().replace(" ", "_")
    if data.get("answer") is None:
        data["answer"] = ""
    claims = data.get("claims") or []
    data["claims"] = [{"text": c, "evidence": []} if isinstance(c, str) else c for c in claims] \
        if isinstance(claims, list) else []
    for c in data["claims"]:
        if isinstance(c, dict) and isinstance(c.get("evidence"), str):
            c["evidence"] = [c["evidence"]]
    try:
        return ModelOutput.model_validate(data)
    except ValidationError:
        raise GenerationError("the model's reply did not match the required format") from None
