# AGENTS.md

Local-first RAG toolkit for sensitive documents. Read README.md first; this
file lists invariants and checklists only.

## Privacy invariants (critical)

- **Egress policy**: the ONLY permitted network call on document-derived data
  is the generation request to Albert (`generate.base_url`, default
  `https://albert.api.etalab.gouv.fr/v1`) — question + top-k excerpts, nothing
  else. Never add embeddings/reranking/OCR/API calls that ship chunk or
  document text anywhere else.
- **Local-only by construction**: parsing (pymupdf/python-docx), chunking,
  embeddings (sentence-transformers bge-m3), vector store (ChromaDB embedded)
  must stay local. Do not introduce cloud SDKs for retrieval.
- **Telemetry stays off**: chromadb `Settings(anonymized_telemetry=False)`;
  `.streamlit/config.toml` `gatherUsageStats = false`. Do not add analytics.
- **Secrets**: `ALBERT_API_KEY` from env / `.env` only (`{env}`-style
  substitution mindset); never print keys, never commit `.env`.
- **`rags/` is runtime data** (corpora + indexes, may contain sensitive
  documents): gitignored, never committed, never printed in logs or tests.
- Every `ask` must keep printing the `[privacy] … sent N excerpt(s)` line, and
  usage logging to `rags/<name>/usage.jsonl` must be preserved.

## Architecture invariants

- Thin explicit pipeline (config → parse → chunk → embed → store → retrieve →
  generate → pipeline → cli/ui). No LangChain/agent frameworks: auditability
  is a feature.
- Per-project isolation: `rags/<name>/{config.yaml,docs/,index/}`; collection
  name derived via `config.collection_name`; embedding model is recorded in
  collection metadata and mismatch must raise (vectors are not interchangeable).
- `build` defaults to full reset+rebuild; `--no-reset` appends and must check
  the embed-model mismatch.
- Scanned PDFs: detect + skip with explicit warning (OCR is phase 2, local
  tesseract; do NOT use Albert's LightOnOCR for sensitive corpora).
- Chunker filters (references, boilerplate, digit-garbage, repeating loops)
  must stay conservative — false-dropping legit repetitive text (meeting
  notes) is worse than keeping some noise. Current tuning: repeating-loop =
  50-char substring ≥ 5 non-overlapping repeats, text ≥ 250 chars.

## Commands

- `uv sync` — install/refresh the venv (Python 3.12 via uv).
- `uv run pytest` — unit tests (model-free); `RUN_MODEL_TESTS=1` adds the live
  bge-m3 build+query test.
- Demo corpus recipe: `/var/folders/.../opencode/make_demo_docs.py` script
  pattern — generate FR/EN pdf/docx/md/txt fixtures, then
  `rag init/add/build/ask`.
- Update the README (commands table, config reference, roadmap) whenever
  commands or defaults change.

## Phase 2 (documented, not started)

Docling parser + section-aware chunking port from
`~/git/cmip6_gpt_fork/rag/chunk_papers.py`; local tesseract OCR; hybrid
BM25+dense retrieval with RRF; incremental indexing.
