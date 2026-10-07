import os
from pathlib import Path

import streamlit as st

from myrag import pipeline
from myrag.config import default_home, index_dir, load_config, project_dir
from myrag.store import index_stats, open_collection

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

st.set_page_config(page_title="my-rag-toolkit", page_icon="📚", layout="wide")

home = default_home()
projects = pipeline.discover_projects(home)
if not projects:
    st.error(
        f"No projects found in {home}. Create one from a terminal:\n\n"
        "`uv run rag init <name>`"
    )
    st.stop()

default_project = os.environ.get("MYRAG_UI_PROJECT", "")
if default_project not in projects:
    default_project = projects[0]

if "project" not in st.session_state:
    st.session_state.project = default_project
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📚 my-rag-toolkit")
    selected = st.selectbox(
        "Project",
        projects,
        index=projects.index(st.session_state.project),
    )
    if selected != st.session_state.project:
        st.session_state.project = selected
        st.session_state.messages = []
        st.rerun()

    project_path = project_dir(home, selected)
    cfg = load_config(project_path)
    try:
        stats = index_stats(open_collection(selected, index_dir(project_path)))
        st.caption(f"{stats['chunks']} chunks from {len(stats['documents'])} documents")
    except Exception:
        st.caption("index not built yet")

    st.caption(f"generation: {cfg.generate.model} (Albert)")
    use_llm = st.toggle(
        "Answer with Albert",
        value=True,
        help="off = retrieval only, nothing leaves this machine",
    )
    top_k = st.slider("Top-k excerpts", 3, 15, cfg.retrieve.top_k)

    st.divider()
    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Building index (local parsing + local embeddings)..."):
            try:
                report = pipeline.build_project(home, selected, reset=True)
                st.toast(f"indexed {report.total_chunks} chunks")
            except Exception as exc:
                st.error(f"build failed: {exc}")
        st.rerun()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**Privacy model** — documents, chunks and embeddings stay on this machine. "
        "Only the selected excerpts + your question are sent to Albert "
        "(`albert.api.etalab.gouv.fr`, DINUM) to generate the answer."
    )

st.header(f"{st.session_state.project}")
if cfg.description:
    st.caption(cfg.description)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(src)


def _format_source(c) -> str:
    if c.page and c.page > 0:
        loc = f"page {c.page}"
    elif c.section:
        loc = f"section: {c.section}"
    else:
        loc = "document"
    excerpt = c.text[:600].replace("\n", " ")
    suffix = "…" if len(c.text) > 600 else ""
    return (
        f"**[{c.index}] {c.doc_id}** — {loc} (score {c.score:.3f})\n\n"
        f"> {excerpt}{suffix}"
    )


if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving locally (local embeddings + local index)..."):
            try:
                result = pipeline.query_project(
                    home, st.session_state.project, prompt, use_llm=use_llm, top_k=top_k
                )
            except Exception as exc:
                st.error(f"query failed: {exc}")
                st.stop()
        if not result.contexts:
            st.warning("No matching excerpts found.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "No matching excerpts found.", "sources": []}
            )
            st.stop()
        sources = [_format_source(c) for c in result.contexts]
        if result.generated:
            st.markdown(result.answer)
        else:
            st.info("Retrieval-only mode: nothing was sent to any API.")
        with st.expander(
            f"Sources ({len(result.contexts)})", expanded=not result.generated
        ):
            for src in sources:
                st.markdown(src)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer or "(retrieval-only mode)",
                "sources": sources,
            }
        )
