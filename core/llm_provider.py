"""LLM provider abstraction and factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


# Backward-compatible re-exports.
from core.providers.local_llm_provider import LocalLLMProvider  # noqa: E402
from core.providers.openai_provider import OpenAIProvider  # noqa: E402


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the configured LLM provider."""
    provider = (settings.llm_provider or "openai").strip().lower()
    if provider == "openai":
        return OpenAIProvider(settings)
    if provider == "local":
        return LocalLLMProvider(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")