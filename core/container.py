"""Shared service container for backend and frontend."""

from __future__ import annotations

from dataclasses import dataclass

from core.config import Settings, get_settings
from core.llm_provider import OpenAIProvider

from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from rag.rag_pipeline import RAGPipeline

from agents.research_agent import ResearchAgent


@dataclass
class ServiceContainer:
    settings: Settings
    embedding_model: EmbeddingModel
    vector_store: VectorStore
    retriever: Retriever
    llm_provider: OpenAIProvider
    rag_pipeline: RAGPipeline
    research_agent: ResearchAgent


def build_services(settings: Settings | None = None) -> ServiceContainer:
    settings = settings or get_settings()
    embedding_model = EmbeddingModel(settings.embedding_model)
    vector_store = VectorStore(settings.chroma_persist_dir, embedding_model)
    retriever = Retriever(
        vector_store,
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    llm_provider = OpenAIProvider(settings)
    rag_pipeline = RAGPipeline(retriever, llm_provider)
    research_agent = ResearchAgent(retriever, vector_store, llm_provider, settings)
    return ServiceContainer(
        settings=settings,
        embedding_model=embedding_model,
        vector_store=vector_store,
        retriever=retriever,
        llm_provider=llm_provider,
        rag_pipeline=rag_pipeline,
        research_agent=research_agent,
    )