"""Nova Lite through the Bedrock Converse API, using the shared M0 ``BedrockRuntime`` client.

The model ID comes from ``BEDROCK_LLM_MODEL_ID`` and may be a cross-region inference profile
(e.g. ``us.amazon.nova-lite-v1:0``). Bedrock/AWS errors (credentials, permissions, throttling,
bad model ID, inference-profile problems, timeouts) are raised as ``BedrockError`` subclasses.
"""

from __future__ import annotations

from typing import Optional

from app.core.bedrock import BedrockRuntime
from app.core.config import Settings, get_settings
from app.generation.models import GenerationError


class BedrockNovaClient:
    def __init__(self, runtime: Optional[BedrockRuntime] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.runtime = runtime or BedrockRuntime(self.settings)
        self.model_id = self.settings.bedrock_llm_model_id

    def converse(self, system: str, messages: list[dict], *, max_tokens: int, temperature: float) -> str:
        resp = self.runtime.converse(
            self.model_id,
            system=[{"text": system}],
            messages=[{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        if resp.get("stopReason") in ("content_filtered", "guardrail_intervened"):
            raise GenerationError("the model's safety filter blocked the response")
        try:
            parts = resp["output"]["message"]["content"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, TypeError, AttributeError):
            raise GenerationError("unexpected response shape from the language model") from None
        if not text:
            raise GenerationError("the language model returned an empty response")
        return text
