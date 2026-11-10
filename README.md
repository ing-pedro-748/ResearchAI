# ResearchAI – Intelligent Research Assistant

ResearchAI is a portfolio project that demonstrates practical skills in Large Language Models (LLMs),
Retrieval-Augmented Generation (RAG), agentic AI systems, prompt engineering, and software engineering.

The application allows users to upload scientific papers and PDF documents, ask questions about their
contents, compare multiple documents, extract methodologies, and generate structured research reports.
All answers are grounded in retrieved document fragments to reduce hallucinations.

## Main features

- Upload and index PDF documents.
- Semantic and lexical hybrid retrieval (ChromaDB + BM25 + RRF).
- Grounded question answering with citations and page references.
- Document summarization, comparison, and methodology extraction.
- Research report generation with Markdown export.
- Retrieval Explorer that works without any LLM API key.
- Quantitative evaluation of retrieval quality.
- FastAPI backend and Streamlit frontend.
- Automated tests and linting.

## Architecture

```
ResearchAI/
├── app/                 # Streamlit frontend
├── backend/             # FastAPI API and services
├── core/                # Configuration, logging, LLM provider, service container
├── rag/                 # Document loading, chunking, embeddings, vector store, retrieval
├── agents/              # LangGraph research agent, tools, state, workflow
├── prompts/             # Versioned prompt templates
├── evaluation/          # Datasets, metrics, evaluation scripts
├── tests/               # Automated tests
└── data/                # Uploaded documents and ChromaDB storage
```

## Technology stack

- Python 3.11+
- OpenAI API (optional, with provider abstraction)
- LangChain and LangGraph
- Sentence Transformers
- ChromaDB
- rank-bm25
- PyMuPDF
- FastAPI
- Streamlit
- Pydantic
- Pytest
- Ruff

## Setup (Windows)

Open a PowerShell terminal in the project folder:

```powershell
cd "C:\Users\Lenovo\Documents\Opencode\RAG"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and configure at least:

```env
OPENAI_API_KEY=your_api_key_here
```

The application can start and use Retrieval Explorer without an API key. Chat, summaries,
comparisons, methodology extraction, and reports require an API key.

## How to run

Use two separate terminals (consoles).

### Terminal 1: Backend (FastAPI)

```powershell
cd "C:\Users\Lenovo\Documents\Opencode\RAG"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The backend will be available at:

- Health check: `http://127.0.0.1:8000/api/health`
- API docs: `http://127.0.0.1:8000/docs`

### Terminal 2: Frontend (Streamlit)

```powershell
cd "C:\Users\Lenovo\Documents\Opencode\RAG"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Streamlit will open the web interface in your browser.

## Usage

1. Upload one or more PDF files using the sidebar.
2. Use **Research Chat** to ask questions about the indexed documents.
3. Use **Retrieval Explorer** to inspect semantic and lexical retrieval without an API key.
4. Use **Document Analysis** to summarize, compare, or extract methodologies.
5. Use **Research Reports** to generate and download Markdown reports.
6. Use **Evaluation** to inspect retrieval experiment results.

## Local LLM (llama.cpp)

ResearchAI can use a local GGUF model served by `llama-cpp-python` instead of OpenAI.

1. Start the server:
```powershell
python -m llama_cpp.server --model "TwIL-8B-Q4_K_M-balanced.gguf" --host 127.0.0.1 --port 8000
```

2. Configure `.env`:
```env
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=TwIL-8B-Q4_K_M-balanced.gguf
LOCAL_LLM_TIMEOUT_SECONDS=180
```

3. Run ResearchAI normally. No `OPENAI_API_KEY` is required.

The local provider uses the OpenAI-compatible API exposed by the server and does not load
the GGUF file inside ResearchAI.

## Hybrid Search

ResearchAI combines two independent retrieval mechanisms:

- Vector search using ChromaDB and Sentence Transformers embeddings.
- Lexical search using BM25 over the full text of each chunk.
- Reciprocal Rank Fusion (RRF) with `k = 60` to merge both rankings.

Both searches run concurrently. Duplicate chunks are consolidated, and each result keeps its
RRF score, vector score, BM25 score, and the list of methods that retrieved it.

## Evaluation

Run retrieval evaluation:

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate_retrieval
```

Run answer evaluation (requires an API key):

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate_answers
```

Run prompt configuration experiment (requires an API key):

```powershell
.\.venv\Scripts\python.exe -m evaluation.prompt_experiment
```

The retrieval evaluation compares vector search and hybrid search using the same questions and
documents. It reports Precision@K, Recall@K, MRR, and retrieval latency. Results are written to
`evaluation/results/`.

## Tests and code quality

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Run linting:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

## Configuration

Main environment variables:

- `OPENAI_API_KEY`: API key for generative features.
- `OPENAI_MODEL`: LLM model name.
- `OPENAI_TEMPERATURE`: generation temperature.
- `EMBEDDING_MODEL`: Sentence Transformers model.
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: chunking parameters.
- `RETRIEVAL_TOP_K`: number of results for the vector retriever.
- `VECTOR_SEARCH_TOP_K`: vector candidates for hybrid search.
- `BM25_SEARCH_TOP_K`: BM25 candidates for hybrid search.
- `HYBRID_FINAL_TOP_K`: final number of hybrid results.
- `RRF_K`: RRF constant.

## Limitations

- No OCR support for scanned PDFs.
- Answer faithfulness and relevance require LLM-as-judge or human review.
- The service container is single-process and intended for local or demo use.

## Future improvements

- Add reranking.
- Add hybrid search tuning and ablation studies.
- Add OCR support.
- Add Docker deployment.
- Add multi-user support.

## License

MIT