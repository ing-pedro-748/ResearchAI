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
            message = str(exc).lower()
            if "timed out" in message or "timeout" in message:
                raise LLMError(
                    f"Local LLM request timed out after {self.settings.local_llm_timeout_seconds} seconds."
                ) from exc
            if "connection" in message or "refused" in message or "connect" in message:
                raise LLMError(
                    f"Local LLM server is unavailable at {self.settings.local_llm_base_url}."
                ) from exc
            status_code = getattr(exc, "status_code", None)
            if status_code == 404:
                raise LLMError(
                    f"Local LLM model '{self.settings.local_llm_model}' was not found on the server."
                ) from exc
            raise LLMError(f"Local LLM request failed: {exc}") from exc

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