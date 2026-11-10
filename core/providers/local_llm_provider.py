"""Local GGUF LLM provider via an OpenAI-compatible llama.cpp server."""

from __future__ import annotations

from openai import OpenAI

from core.config import Settings
from core.llm_provider import LLMError, LLMResponse


class LocalLLMProvider:
    """LLM provider backed by a local llama.cpp OpenAI-compatible server."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.local_llm_base_url:
            raise LLMError("LOCAL_LLM_BASE_URL is not configured")
        if not settings.local_llm_model:
            raise LLMError("LOCAL_LLM_MODEL is not configured")

        self.client = OpenAI(
            base_url=settings.local_llm_base_url,
            api_key="not-needed",
            timeout=settings.local_llm_timeout_seconds,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.local_llm_model,
                temperature=self.settings.openai_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - convert provider errors
            raise LLMError(
                f"Local LLM request failed. Is the llama.cpp server running at "
                f"{self.settings.local_llm_base_url}? Error: {exc}"
            ) from exc

        if not response.choices:
            raise LLMError("Local LLM returned no choices.")

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )