# AGENTS.md

Local-first RAG toolkit for sensitive documents. Read README.md first; this
file lists invariants and checklists only.

## Privacy invariants (critical)

- **Egress policy**: the ONLY permitted network call on document-derived data
  is the generation request to the configured sovereign/institutional endpoint
  (`generate.base_url`, default `https://albert.api.etalab.gouv.fr/v1`;
  per-project `profiles:` may point at other sovereign/institutional
  OpenAI-compatible endpoints) — question + top-k excerpts, nothing else.
  Never add embeddings/reranking/OCR/API calls that ship chunk or document
  text anywhere else.
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

## GitHub confidentiality guardrails (critical)

This repo is PUBLIC. Everything that reaches GitHub — tracked files, commit
messages, branch/tag names, PR/issue text, release notes, repo description —
is world-readable. Strictly ZERO reference there to the topic, identity or
content of any real corpus processed with this tool:

- Forbidden in any pushed artifact: real corpus or project names, committee
  or institution names, session identifiers, researcher/person names,
  laboratory names and numbers, real document filenames or titles, and any
  excerpt, summary or query result derived from a real corpus.
- All corpus state lives under `rags/` (gitignored, local-only): project
  names, docs, indexes, usage logs. Keep `rags/` ignored forever; never
  `git add -f` it; never reference its contents in tracked files. CLI outputs
  (`rag list`, `rag status`, build reports, answers) carry corpus
  identifiers — never paste them into commits, issues or docs.
- Commit/PR messages describe toolkit changes only, never corpus activity.
- Demo/test fixtures committed to the repo must remain fictional.
- Asking for help on GitHub (issues, discussions): abstract the use case
  ("a corpus of committee evaluation reports"), zero identifying detail.
- The guardrail text itself stays generic — never write a real identifier
  into this file or any tracked file (naming the actual terms would itself
  be the leak).
- Mechanical guard: `scripts/push-guard.sh` runs as a pre-push hook and
  blocks pushes whose outgoing commits match the denylist at
  `~/.config/myrag/push-guard-denylist.txt` — a LOCAL file, never committed,
  holding the real identifiers of your corpora. Keep it current whenever a
  new corpus project is created.
- If a leak is found in local or pushed history: STOP, tell the owner
  immediately. Remediation (history rewrite, repo privacy) is a human
  decision, never performed unilaterally.

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
- `scripts/push-guard.sh` — pre-push leak guard; install once per clone:
  `cp scripts/push-guard.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push`.
  Denylist lives at `~/.config/myrag/push-guard-denylist.txt` (local-only).
- Demo/test corpora must be fictional (e.g. the FjordHydro-2027 sample):
  generate FR/EN pdf/docx/md/txt fixtures locally, then `rag init/add/build/ask`.
- Update the README (commands table, config reference, roadmap) whenever
  commands or defaults change.

## Phase 2 (documented, not started)

Docling parser + section-aware chunking port from
`~/git/cmip6_gpt_fork/rag/chunk_papers.py`; local tesseract OCR; hybrid
BM25+dense retrieval with RRF; incremental indexing.
