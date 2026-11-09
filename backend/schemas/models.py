"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    source: str


class IngestResponse(BaseModel):
    document_id: str
    source: str
    chunks: int
    skipped: bool


class SourceChunk(BaseModel):
    chunk_id: str = ""
    content: str
    source: str = ""
    page: int | None = None
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None
    document_id: str | None = None


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None
    document_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sufficient_context: bool
    sources: list[SourceChunk]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0


class SummarizeRequest(BaseModel):
    document_id: str


class CompareRequest(BaseModel):
    document_ids: list[str]


class MethodologyRequest(BaseModel):
    document_id: str


class ReportRequest(BaseModel):
    query: str = Field(min_length=1)


class AgentRequest(BaseModel):
    request: str = Field(min_length=1)


class TextResponse(BaseModel):
    result: str


class HealthResponse(BaseModel):
    status: str = "ok"