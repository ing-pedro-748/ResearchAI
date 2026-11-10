"""RAG generation using the local LLM provider (mocked)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf

from core.config import Settings
from core.container import build_services


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_rag_generates_answer_with_local_provider(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, "Retrieval augmented generation improves grounded answers.")

    settings = Settings(
        llm_provider="local",
        openai_api_key="",
        local_llm_base_url="http://127.0.0.1:8001/v1",
        local_llm_model="TwIL-8B-Q4_K_M-balanced.gguf",
        chroma_persist_dir=str(tmp_path / "chroma"),
        documents_dir=str(tmp_path / "documents"),
        chunk_size=200,
        chunk_overlap=20,
    )

    with patch("core.providers.local_llm_provider.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated with local model."))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        services = build_services(settings)
        from backend.services.document_service import DocumentService

        ds = DocumentService(settings, services.vector_store, services.bm25_index)
        ds.ingest_path(pdf_path)

        answer = services.rag_pipeline.answer("What improves grounded answers?")
        assert answer.answer == "Generated with local model."
        assert answer.total_tokens == 15