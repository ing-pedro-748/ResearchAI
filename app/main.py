"""Streamlit frontend for ResearchAI."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from core.config import get_settings
from core.container import build_services

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

tab_chat, tab_analysis, tab_reports, tab_eval = st.tabs(
    ["Research Chat", "Document Analysis", "Research Reports", "Evaluation"]
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
            result = services.rag_pipeline.answer(question, top_k=top_k, document_id=doc_filter)
            st.markdown("### Answer")
            st.write(result.answer)
            st.caption(f"Sufficient context: {result.sufficient_context} | Tokens: {result.total_tokens} | Latency: {result.latency_seconds:.2f}s")
            if result.sources:
                st.markdown("### Sources")
                for src in result.sources:
                    st.write(f"- {src.metadata.get('source')} (page {src.metadata.get('page')}) score={src.score:.3f}")

with tab_analysis:
    st.subheader("Summarize document")
    if doc_ids:
        selected = st.selectbox("Document", doc_ids, format_func=lambda x: doc_labels[x], key="sum_doc")
        if st.button("Summarize"):
            st.write(services.research_agent.research_tools.summarize_document(selected))
    else:
        st.info("No documents indexed.")

    st.subheader("Compare documents")
    selected_docs = st.multiselect("Documents", doc_ids, format_func=lambda x: doc_labels[x])
    if st.button("Compare") and selected_docs:
        st.write(services.research_agent.research_tools.compare_documents(selected_docs))

    st.subheader("Extract methodology")
    if doc_ids:
        meth_doc = st.selectbox("Document", doc_ids, format_func=lambda x: doc_labels[x], key="meth_doc")
        if st.button("Extract"):
            st.write(services.research_agent.research_tools.extract_methodology(meth_doc))

with tab_reports:
    st.subheader("Generate research report")
    report_query = st.text_area("Research question", height=100, key="report_q")
    if st.button("Generate Report"):
        report = services.research_agent.research_tools.generate_research_report(report_query)
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