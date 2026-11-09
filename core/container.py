"""Shared service container for backend and frontend."""

from __future__ import annotations

from dataclasses import dataclass

from agents.research_agent import ResearchAgent
from core.config import Settings, get_settings
from core.llm_provider import OpenAIProvider
from rag.bm25_index import BM25Index
from rag.embeddings import EmbeddingModel
from rag.hybrid_retriever import HybridRetriever
from rag.rag_pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.vector_store import VectorStore


@dataclass
class ServiceContainer:
    settings: Settings
    embedding_model: EmbeddingModel
    vector_store: VectorStore
    retriever: Retriever
    bm25_index: BM25Index
    hybrid_retriever: HybridRetriever
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
    bm25_index = BM25Index()
    bm25_index.build(vector_store.get_all_chunks())
    hybrid_retriever = HybridRetriever(vector_store, bm25_index, settings)
    llm_provider = OpenAIProvider(settings)
    rag_pipeline = RAGPipeline(retriever, llm_provider)
    research_agent = ResearchAgent(retriever, vector_store, llm_provider, settings)
    return ServiceContainer(
        settings=settings,
        embedding_model=embedding_model,
        vector_store=vector_store,
        retriever=retriever,
        bm25_index=bm25_index,
        hybrid_retriever=hybrid_retriever,
        llm_provider=llm_provider,
        rag_pipeline=rag_pipeline,
        research_agent=research_agent,
    )