"""Tests for RAG token budget handling."""


from core.config import Settings
from core.tokenizer import count_tokens, truncate_text
from rag.rag_pipeline import RAGPipeline
from rag.retriever import RetrievedChunk


class FakeProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str):
        from core.llm_provider import LLMResponse

        return LLMResponse(text="ok", total_tokens=1)


def test_truncate_text_respects_budget() -> None:
    text = "word " * 1000
    truncated = truncate_text(text, 10)
    assert count_tokens(truncated) <= 10


def test_rag_context_fits_budget() -> None:
    settings = Settings(llm_context_window=256, llm_reserved_answer_tokens=64, llm_prompt_safety_margin=32)
    provider = FakeProvider(settings)
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.llm_provider = provider
    pipeline._template = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    chunks = [
        RetrievedChunk(content="A " * 500, metadata={"source": "doc.pdf", "page": 1}, score=0.9),
        RetrievedChunk(content="B " * 500, metadata={"source": "doc.pdf", "page": 2}, score=0.8),
    ]
    context = pipeline._build_context(chunks, "What is this?")
    assert count_tokens(context) <= settings.llm_context_window