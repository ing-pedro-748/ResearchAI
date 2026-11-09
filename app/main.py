"""Streamlit frontend for ResearchAI."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from core.config import get_settings
from core.container import build_services
from core.llm_provider import LLMError

settings = get_settings()
services = build_services()

st.set_page_config(page_title="ResearchAI", layout="wide")
st.title("ResearchAI – Intelligent Research Assistant")

with st.sidebar:
    st.header("Document Management")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest"):
        from backend.services.document_service import DocumentService

        ds = DocumentService(settings, services.vector_store)
        ingest = ds.ingest_upload(uploaded.name, uploaded.read())
        st.success(f"Ingested {ingest.source}: {ingest.chunks} chunks (skipped={ingest.skipped})")

    docs = services.vector_store.list_documents()
    st.write("Indexed documents:")
    for doc in docs:
        col1, col2 = st.columns([4, 1])
        col1.write(f"{doc['source']} ({doc['document_id']})")
        if col2.button("Delete", key=f"del_{doc['document_id']}"):
            services.vector_store.delete_document(doc["document_id"])
            st.rerun()

doc_ids = [d["document_id"] for d in docs]
doc_labels = {d["document_id"]: d["source"] for d in docs}

tab_chat, tab_retrieval, tab_analysis, tab_reports, tab_eval = st.tabs(
    ["Research Chat", "Retrieval Explorer", "Document Analysis", "Research Reports", "Evaluation"]
)

with tab_chat:
    st.subheader("Ask a question")
    question = st.text_area("Question", height=100)
    filter_doc = st.selectbox("Filter by document (optional)", ["None"] + doc_ids, format_func=lambda x: doc_labels.get(x, x) if x != "None" else "None")
    top_k = st.slider("Top K", 1, 10, settings.retrieval_top_k)
    if st.button("Ask"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            doc_filter = None if filter_doc == "None" else filter_doc
            try:
                result = services.rag_pipeline.answer(question, top_k=top_k, document_id=doc_filter)
            except LLMError as exc:
                st.error(f"Generation requires an API key: {exc}")
            else:
                st.markdown("### Answer")
                st.write(result.answer)
                st.caption(f"Sufficient context: {result.sufficient_context} | Tokens: {result.total_tokens} | Latency: {result.latency_seconds:.2f}s")
                if result.sources:
                    st.markdown("### Sources")
                    for src in result.sources:
                        st.write(f"- {src.metadata.get('source')} (page {src.metadata.get('page')}) score={src.score:.3f}")

with tab_retrieval:
    st.subheader("Retrieval Explorer (no LLM required)")
    if not docs:
        st.info("No documents indexed. Upload a PDF first.")
    else:
        r_question = st.text_area("Question", height=80, key="retrieval_question")
        r_top_k = st.slider("Top K", 1, 10, settings.retrieval_top_k, key="retrieval_top_k")
        r_filter = st.selectbox(
            "Filter by document (optional)",
            ["None"] + doc_ids,
            format_func=lambda x: doc_labels.get(x, x) if x != "None" else "None",
            key="retrieval_filter",
        )
        if st.button("Search", key="retrieval_search"):
            if not r_question.strip():
                st.warning("Enter a question.")
            else:
                doc_filter = None if r_filter == "None" else r_filter
                results = services.hybrid_retriever.retrieve(r_question, top_k=r_top_k, document_id=doc_filter)
                if not results:
                    st.info("No results found.")
                else:
                    st.markdown(f"### {len(results)} results")
                    for idx, chunk in enumerate(results, start=1):
                        with st.expander(f"Result {idx} — RRF={chunk.rrf_score:.4f}"):
                            st.write("**Chunk ID:**", chunk.chunk_id)
                            st.write("**Source:**", chunk.metadata.get("source"))
                            st.write("**Page:**", chunk.metadata.get("page"))
                            st.write("**RRF Score:**", round(chunk.rrf_score, 6))
                            st.write("**Vector Score:**", chunk.vector_score)
                            st.write("**BM25 Score:**", chunk.bm25_score)
                            st.write("**Retrieved by:**", ", ".join(chunk.sources))
                            st.write("**Metadata:**", chunk.metadata)
                            st.write("**Content:**")
                            st.write(chunk.content)

with tab_analysis:
    st.subheader("Summarize document")
    if doc_ids:
        selected = st.selectbox("Document", doc_ids, format_func=lambda x: doc_labels[x], key="sum_doc")
        if st.button("Summarize"):
            try:
                st.write(services.research_agent.research_tools.summarize_document(selected))
            except LLMError as exc:
                st.error(f"Generation requires an API key: {exc}")
    else:
        st.info("No documents indexed.")

    st.subheader("Compare documents")
    selected_docs = st.multiselect("Documents", doc_ids, format_func=lambda x: doc_labels[x])
    if st.button("Compare") and selected_docs:
        try:
            st.write(services.research_agent.research_tools.compare_documents(selected_docs))
        except LLMError as exc:
            st.error(f"Generation requires an API key: {exc}")

    st.subheader("Extract methodology")
    if doc_ids:
        meth_doc = st.selectbox("Document", doc_ids, format_func=lambda x: doc_labels[x], key="meth_doc")
        if st.button("Extract"):
            try:
                st.write(services.research_agent.research_tools.extract_methodology(meth_doc))
            except LLMError as exc:
                st.error(f"Generation requires an API key: {exc}")

with tab_reports:
    st.subheader("Generate research report")
    report_query = st.text_area("Research question", height=100, key="report_q")
    if st.button("Generate Report"):
        try:
            report = services.research_agent.research_tools.generate_research_report(report_query)
        except LLMError as exc:
            st.error(f"Generation requires an API key: {exc}")
        else:
            st.markdown(report)
            st.download_button("Download Markdown", report, file_name="research_report.md")

with tab_eval:
    st.subheader("Prompt experiment results")
    eval_path = Path("evaluation/results/prompt_experiment.json")
    if eval_path.exists():
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        st.write(f"Status: {data.get('status')}")
        st.json(data.get("results", []))
    else:
        st.info("No evaluation results yet.")