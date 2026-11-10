"""Tests for configuration."""

from core.config import Settings


def test_local_llm_config_defaults() -> None:
    settings = Settings(llm_provider="local")
    assert settings.local_llm_base_url == "http://127.0.0.1:8001/v1"
    assert settings.local_llm_model == "TwIL-8B-Q4_K_M-balanced.gguf"
    assert settings.local_llm_timeout_seconds == 120