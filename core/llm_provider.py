"""LLM provider abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from core.config import Settings


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMError(Exception):
    """Raised when an LLM call fails."""


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


class OpenAIProvider:
    """OpenAI chat completions provider."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not configured")

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=self.settings.openai_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - convert provider errors
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )