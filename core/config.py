"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "openai"

    llm_context_window: int = 4096
    llm_reserved_answer_tokens: int = 512
    llm_prompt_safety_margin: int = 128

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    openai_timeout_seconds: int = 60

    local_llm_base_url: str = "http://127.0.0.1:8001/v1"
    local_llm_model: str = "TwIL-8B-Q4_K_M-balanced.gguf"
    local_llm_timeout_seconds: int = 120

    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.3

    vector_search_top_k: int = 10
    bm25_search_top_k: int = 10
    hybrid_final_top_k: int = 5
    rrf_k: int = 60

    chroma_persist_dir: str = "./data/chroma_db"
    documents_dir: str = "./data/documents"

    log_level: str = "INFO"
    max_agent_steps: int = 8

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings