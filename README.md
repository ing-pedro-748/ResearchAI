# ResearchAI – Intelligent Research Assistant

Portfolio project demonstrating LLM integration, Retrieval-Augmented Generation (RAG),
LangGraph-based agentic workflows, prompt engineering, and quantitative evaluation.

> Status: Phase 1 (project initialization) in progress.

## Overview
ResearchAI lets users upload scientific papers (PDF), ask grounded questions, compare
documents, extract methodologies, and generate research reports with citations.

## Architecture
See `docs/` (to be added) for diagrams. High-level modules:

- `core/` – configuration, LLM provider abstraction, logging
- `rag/` – document loading, chunking, embeddings, vector store, retrieval, RAG pipeline
- `agents/` – LangGraph research agent, tools, state, workflow
- `backend/` – FastAPI API, schemas, services
- `app/` – Streamlit frontend
- `prompts/` – versioned prompt templates
- `evaluation/` – datasets and metrics

## Tech stack
Python 3.11+, OpenAI API, LangChain/LangGraph, Sentence Transformers, ChromaDB,
PyMuPDF, FastAPI, Streamlit, Pydantic, Pytest, Ruff.

## Setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Run
Backend and frontend commands will be documented in later phases.

## Development phases
- [x] Phase 1: Project initialization
- [x] Phase 2: Document ingestion
- [x] Phase 3: RAG implementation
- [x] Phase 4: Prompt engineering
- [ ] Phase 5: Agentic workflows
- [ ] Phase 6: Backend and frontend
- [ ] Phase 7: Evaluation and documentation

## License
MIT