# ResearchAI – Intelligent Research Assistant

Portfolio project demonstrating LLM integration, Retrieval-Augmented Generation (RAG),
LangGraph-based agentic workflows, prompt engineering, and quantitative evaluation.

## Overview
ResearchAI allows users to upload scientific papers (PDF), ask grounded questions, compare
documents, extract methodologies, and generate research reports with citations.

## Architecture

```
ResearchAI/
├── app/                 # Streamlit frontend
├── backend/             # FastAPI API
├── core/                # Config, logging, LLM provider, service container
├── rag/                 # Document loading, chunking, embeddings, vector store, retrieval, RAG pipeline
├── agents/              # LangGraph research agent, tools, state, workflow
├── prompts/             # Versioned prompt templates
├── evaluation/          # Datasets and evaluation scripts
├── tests/               # Automated tests
└── data/                # Documents and ChromaDB (gitignored DB)
```

## Tech stack
- Python 3.11+
- OpenAI API (provider abstraction)
- LangChain / LangGraph
- Sentence Transformers
- ChromaDB
- PyMuPDF
- FastAPI
- Streamlit
- Pydantic
- Pytest
- Ruff

## Setup (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## Run
Backend:
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```
Frontend:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

## Usage
1. Upload PDFs in the sidebar.
2. Ask questions in Research Chat.
3. Summarize, compare, or extract methodology in Document Analysis.
4. Generate and download research reports.

## Hybrid Search
ResearchAI now combines ChromaDB vector search and BM25 lexical search using
Reciprocal Rank Fusion (RRF, k=60). Retrieval Explorer displays RRF scores,
component scores, and which method retrieved each chunk.

## Evaluation
Run retrieval evaluation:
```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate_retrieval
```
Run answer evaluation (requires API key):
```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate_answers
```
Run prompt experiment (requires API key for generation):
```powershell
.\.venv\Scripts\python.exe -m evaluation.prompt_experiment
```

### Retrieval results (actual)
From `evaluation/results/retrieval_eval.json`:
- Mean Precision@K: 1.0
- Mean Recall@K: 1.0
- MRR: 1.0

### Answer and prompt results
Pending until an OpenAI API key is configured. The scripts record pending status honestly.

## Development phases
- [x] Phase 1: Project initialization
- [x] Phase 2: Document ingestion
- [x] Phase 3: RAG implementation
- [x] Phase 4: Prompt engineering
- [x] Phase 5: Agentic workflows
- [x] Phase 6: Backend and frontend
- [x] Phase 7: Evaluation and documentation

## Limitations
- No OCR for scanned PDFs.
- Answer faithfulness requires LLM-as-judge or human review.
- Single-process service container (no distributed deployment).

## Future improvements
- Add reranking.
- Add hybrid search.
- Add multi-user support.
- Add Docker deployment.

## License
MIT