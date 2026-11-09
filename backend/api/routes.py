"""API routes."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.schemas.models import (
    AgentRequest,
    CompareRequest,
    HealthResponse,
    IngestResponse,
    MethodologyRequest,
    QueryRequest,
    QueryResponse,
    ReportRequest,
    SourceChunk,
    SummarizeRequest,
    TextResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/documents")
async def list_documents(request: Request):
    service = request.app.state.document_service
    return {"documents": service.list_documents()}


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    service = request.app.state.document_service
    result = service.ingest_upload(file.filename, content)
    return IngestResponse(
        document_id=result.document_id,
        source=result.source,
        chunks=result.chunks,
        skipped=result.skipped,
    )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, request: Request):
    service = request.app.state.document_service
    service.delete_document(document_id)
    return {"deleted": document_id}


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest, request: Request):
    pipeline = request.app.state.rag_pipeline
    answer = pipeline.answer(payload.question, top_k=payload.top_k, document_id=payload.document_id)
    sources = [
        SourceChunk(
            content=chunk.content,
            source=chunk.metadata.get("source", ""),
            page=chunk.metadata.get("page"),
            score=chunk.score,
        )
        for chunk in answer.sources
    ]
    return QueryResponse(
        answer=answer.answer,
        sufficient_context=answer.sufficient_context,
        sources=sources,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        total_tokens=answer.total_tokens,
        latency_seconds=answer.latency_seconds,
    )


@router.post("/summarize", response_model=TextResponse)
async def summarize(payload: SummarizeRequest, request: Request):
    tools = request.app.state.research_agent.research_tools
    result = tools.summarize_document(payload.document_id)
    return TextResponse(result=result)


@router.post("/compare", response_model=TextResponse)
async def compare(payload: CompareRequest, request: Request):
    tools = request.app.state.research_agent.research_tools
    result = tools.compare_documents(payload.document_ids)
    return TextResponse(result=result)


@router.post("/methodology", response_model=TextResponse)
async def methodology(payload: MethodologyRequest, request: Request):
    tools = request.app.state.research_agent.research_tools
    result = tools.extract_methodology(payload.document_id)
    return TextResponse(result=result)


@router.post("/report", response_model=TextResponse)
async def report(payload: ReportRequest, request: Request):
    tools = request.app.state.research_agent.research_tools
    result = tools.generate_research_report(payload.query)
    return TextResponse(result=result)


@router.post("/agent", response_model=TextResponse)
async def agent(payload: AgentRequest, request: Request):
    agent = request.app.state.research_agent
    result = agent.final_answer(payload.request)
    return TextResponse(result=result)


@router.get("/evaluation")
async def evaluation_results():
    from pathlib import Path

    path = Path("evaluation/results/prompt_experiment.json")
    if not path.exists():
        return JSONResponse({"status": "pending", "results": []})
    return JSONResponse(__import__("json").loads(path.read_text(encoding="utf-8")))