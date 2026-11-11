"""Tests for local LLM provider."""

from unittest.mock import MagicMock, patch

from core.config import Settings
from core.llm_provider import LLMResponse, create_llm_provider
from core.providers.local_llm_provider import LocalLLMProvider


def _local_settings() -> Settings:
    return Settings(
        llm_provider="local",
        openai_api_key="",
        local_llm_base_url="http://127.0.0.1:8001/v1",
        local_llm_model="TwIL-8B-Q4_K_M-balanced.gguf",
        local_llm_timeout_seconds=10,
    )


def test_local_provider_does_not_require_openai_key() -> None:
    with patch("core.providers.local_llm_provider.OpenAI"):
        provider = LocalLLMProvider(_local_settings())
        assert provider.settings.openai_api_key == ""


def test_local_provider_uses_local_endpoint() -> None:
    with patch("core.providers.local_llm_provider.OpenAI") as mock_openai:
        LocalLLMProvider(_local_settings())
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["base_url"] == "http://127.0.0.1:8001/v1"


def test_local_provider_sends_system_and_user_prompts() -> None:
    with patch("core.providers.local_llm_provider.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="local answer"))]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 7
        mock_response.usage.total_tokens = 12
        mock_client.chat.completions.create.return_value = mock_response

        provider = LocalLLMProvider(_local_settings())
        result = provider.generate("system text", "user text")

        assert isinstance(result, LLMResponse)
        assert result.text == "local answer"
        assert result.total_tokens == 12
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "TwIL-8B-Q4_K_M-balanced.gguf"
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ]


def test_local_provider_handles_server_unavailable() -> None:
    with patch("core.providers.local_llm_provider.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("connection refused")

        provider = LocalLLMProvider(_local_settings())
        try:
            provider.generate("s", "u")
        except Exception as exc:
            assert "unavailable" in str(exc).lower()
        else:
            raise AssertionError("Expected an error")


def test_factory_selects_openai_when_configured() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    with patch("core.providers.openai_provider.OpenAI"):
        provider = create_llm_provider(settings)
        assert type(provider).__name__ == "OpenAIProvider"


def test_factory_selects_local() -> None:
    with patch("core.providers.local_llm_provider.OpenAI"):
        provider = create_llm_provider(_local_settings())
        assert type(provider).__name__ == "LocalLLMProvider"


def test_local_provider_classifies_timeout() -> None:
    with patch("core.providers.local_llm_provider.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Request timed out")

        provider = LocalLLMProvider(_local_settings())
        try:
            provider.generate("s", "u")
        except Exception as exc:
            assert "timed out" in str(exc).lower()
        else:
            raise AssertionError("Expected an error")


def test_local_provider_classifies_connection_refused() -> None:
    with patch("core.providers.local_llm_provider.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        provider = LocalLLMProvider(_local_settings())
        try:
            provider.generate("s", "u")
        except Exception as exc:
            assert "unavailable" in str(exc).lower()
        else:
            raise AssertionError("Expected an error")


def test_local_provider_classifies_invalid_model() -> None:
    with patch("core.providers.local_llm_provider.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        exc = Exception("not found")
        exc.status_code = 404
        mock_client.chat.completions.create.side_effect = exc

        provider = LocalLLMProvider(_local_settings())
        try:
            provider.generate("s", "u")
        except Exception as exc:
            assert "not found" in str(exc).lower() or "model" in str(exc).lower()
        else:
            raise AssertionError("Expected an error")