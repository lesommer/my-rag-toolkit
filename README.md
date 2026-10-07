# my-rag-toolkit

Local-first RAG (Retrieval-Augmented Generation) for **sensitive documents**.
Build a small, self-contained question-answering index over a corpus of
documents (< 100 docs per project), then query it from a CLI or a local web UI.

**Privacy model (the design driver):** parsing, chunking, embeddings and the
vector index all run **on your machine**. The *only* network egress is the final
answer-generation call: the retrieved excerpts + your question go to the
**Albert API** (`albert.api.etalab.gouv.fr`, operated by DINUM, the French
State's digital directorate). Documents never leave the machine; no US/cloud
provider is involved anywhere.

```
                 LOCAL (your Mac)                        SOVEREIGN (Albert/DINUM)
┌─────────────────────────────────────────────┐        ┌────────────────────────┐
│ pdf/docx/md/txt ─ parse ─ chunk ─ embed      │        │                        │
│                       (bge-m3, on-device)    │        │   gpt-oss-120b         │
│                     ▼                        │ ─────► │   chat completion      │
│              ChromaDB index (per project)    │ top-k  │   (excerpts+question)  │
│ query ─ embed ─ top-k ─ optional local rerank│ ◄───── │   cited answer         │
└─────────────────────────────────────────────┘        └────────────────────────┘
        Typer CLI (`rag …`)  +  Streamlit web UI
```

## RAG in 60 seconds

1. **Parse** each document into text (page numbers / sections kept as metadata).
2. **Chunk** the text into ~1000-token passages with a small overlap.
3. **Embed** each chunk into a 1024-dim vector with `bge-m3` — a multilingual
   open-source model that runs **locally** (Apple Silicon GPU). Vectors capture
   meaning, so "budget du projet" matches "project funding" across FR/EN.
4. **Store** vectors + text in a local ChromaDB index (one per project).
5. At **query** time: embed the question, retrieve the top-k most similar
   chunks (cosine similarity, threshold + optional local cross-encoder
   reranking), then send *only those k excerpts + the question* to Albert,
   which answers **with inline citations** `[1]`, `[2]`.

## Install

```bash
git clone <this-repo> ~/git/my-rag-toolkit
cd ~/git/my-rag-toolkit
uv sync                       # creates .venv with Python 3.12 (uv-managed)
export ALBERT_API_KEY="sk-..."   # or cp .env.example .env and edit it
```

First `build` downloads the bge-m3 model (~2.3 GB, cached in
`~/.cache/huggingface`, no document data is sent).

## Quickstart

```bash
uv run rag init eval-panel-2027 -d "Documents for the 2027 evaluation panel"
uv run rag add eval-panel-2027 ~/Downloads/panel-docs/     # pdf, docx, md, txt
uv run rag build eval-panel-2027        # parse → chunk → embed → index (local)
uv run rag ask eval-panel-2027 "Quels sont les points forts du projet ?"
uv run rag ui                           # local web UI (chat + sources)
```

Every project is an **independent RAG** with its own `docs/`, `index/` and
`config.yaml` under `rags/<name>/` (gitignored). Reuse the same tool for
evaluation panels, article corpora, or your notes database.

## CLI reference

| Command | What it does |
|---|---|
| `rag init <name> [-d "…"]` | create a project |
| `rag add <name> <files-or-dir>` | copy documents into the project |
| `rag build <name> [--no-reset]` | (re)build the index — local embeddings; default = full rebuild |
| `rag ask <name> "question"` | retrieve + Albert answer with citations (`--show-context`, `--top-k N`) |
| `rag ask <name> "q" --no-llm` | retrieval only — **nothing leaves the machine** |
| `rag search <name> "query"` | retrieval only (always local) |
| `rag list` / `rag status <name>` | projects and index stats |
| `rag remove <name>` | delete a project (asks confirmation) |
| `rag ui [--project <name>]` | launch the Streamlit UI |

## Configuration (per project, `rags/<name>/config.yaml`)

```yaml
retrieve:
  top_k: 8                  # excerpts sent to the LLM
  score_threshold: 0.35     # min cosine similarity (keeps min_candidates anyway)
  rerank:
    enabled: false          # local bge-reranker-v2-m3 (extra ~2.3 GB download)
generate:
  model: gpt-oss-120b       # any Albert chat model (mistral-small-3.2-24b, …)
  base_url: https://albert.api.etalab.gouv.fr/v1
embed:
  model: BAAI/bge-m3        # local; changing it requires a full rebuild
```

## Privacy & network audit

| Data | Where it goes |
|---|---|
| Documents, parsed text, chunks, embeddings, index | **local only** (`rags/<name>/`) |
| Question + top-k excerpts | **Albert only** at answer time (`albert.api.etalab.gouv.fr`) |
| API key | `ALBERT_API_KEY` env / `.env` — never committed, never printed |

Other network contacts (no document data involved): one-time downloads of the
bge-m3 model (huggingface.co), the tiktoken tokenizer file (public static
file), and pip packages at install time. Chroma telemetry is disabled
(`anonymized_telemetry=False`) and Streamlit usage stats are disabled
(`.streamlit/config.toml`). Every `ask` prints exactly what was sent
(`[privacy] sent N excerpt(s) (~X tokens) …`), and each call is logged to
`rags/<name>/usage.jsonl`.

Manual audit while a command runs: `lsof -nP -iTCP -a -p <pid>` — you should
only ever see `albert.api.etalab.gouv.fr` during `ask`.

## Scan detection (OCR deferred to phase 2)

PDFs whose pages have no text layer (i.e. paper scans) are **detected and
skipped** at build time with a clear warning — nothing silently goes missing.
Local OCR via `tesseract` is the planned phase-2 addition; Albert's OCR model
(LightOnOCR) exists but would upload document images, contradicting the
privacy model.

## Phase-2 roadmap

- **Docling parsing + section-aware chunking** — port of the battle-tested
  chunker from `~/git/cmip6_gpt_fork/rag/chunk_papers.py` (V6) for
  research-article corpora (tables, captions, section paths, richer filters).
- **Local OCR** for scanned PDFs (tesseract).
- **Hybrid retrieval** (BM25 + dense, RRF fusion) for acronym-heavy queries.
- Incremental indexing (currently `build` rebuilds the whole project index).

## Testing

```bash
uv run pytest              # model-free unit tests
RUN_MODEL_TESTS=1 uv run pytest   # adds a live bge-m3 build+query test
```

## Relationship to cmip6_gpt_fork

This toolkit deliberately **adopts** the fork's proven patterns (uv + Python
3.12, `openai`-SDK gateway config, tiktoken budgets, per-project `config.yaml`
knobs, chunk-quality filtering ideas) and **diverges** where the fork's
choices conflict with sensitive-data constraints: local embeddings/reranker
instead of Gemini/Vertex, embedded Chroma instead of a Qdrant server, and a
thin auditable pipeline instead of LangChain/LangGraph.
