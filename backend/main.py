"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from core.container import build_services
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = build_services()
    app.state.settings = services.settings
    app.state.vector_store = services.vector_store
    app.state.retriever = services.retriever
    app.state.llm_provider = services.llm_provider
    app.state.rag_pipeline = services.rag_pipeline
    app.state.research_agent = services.research_agent
    from backend.services.document_service import DocumentService

    app.state.document_service = DocumentService(services.settings, services.vector_store, services.bm25_index)
    app.state.hybrid_retriever = services.hybrid_retriever
    yield


app = FastAPI(title="ResearchAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")