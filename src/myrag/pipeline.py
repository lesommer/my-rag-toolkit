from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from myrag.chunk import chunk_document
from myrag.config import (
    SUPPORTED_EXTENSIONS,
    ProjectConfig,
    apply_profile,
    default_home,
    docs_dir,
    index_dir,
    load_config,
    project_dir,
    save_config,
)
from myrag.embed import get_embedder
from myrag.generate import generate_answer
from myrag.parse import parse_file
from myrag.retrieve import Context, Reranker, retrieve
from myrag.store import add_chunks, index_stats, open_collection, reset_collection


@dataclass
class FileReport:
    file: str
    status: str
    pages: int = 0
    chunks: int = 0
    note: str = ""


@dataclass
class BuildReport:
    project: str
    files: list[FileReport] = field(default_factory=list)
    total_chunks: int = 0
    elapsed_s: float = 0.0

    @property
    def ok_files(self) -> int:
        return sum(1 for f in self.files if f.status == "ok")

    @property
    def skipped_scanned(self) -> list[FileReport]:
        return [f for f in self.files if f.status == "skipped-scanned"]


@dataclass
class QueryResult:
    question: str
    answer: str
    contexts: list[Context]
    model: str
    base_url: str
    generated: bool
    context_tokens: int = 0
    usage: dict | None = None


def discover_projects(home: Path) -> list[str]:
    if not home.exists():
        return []
    return sorted(p.parent.name for p in home.glob("*/config.yaml"))


def create_project(home: Path, name: str, description: str = "") -> Path:
    project = project_dir(home, name)
    if project.exists():
        raise FileExistsError(f"project already exists: {project}")
    (project / "docs").mkdir(parents=True)
    (project / "index").mkdir(parents=True)
    cfg = ProjectConfig(name=name, description=description)
    save_config(project, cfg)
    return project


def add_documents(home: Path, name: str, paths: list[str]) -> list[Path]:
    project = project_dir(home, name)
    if not project.exists():
        raise FileNotFoundError(f"unknown project: {name} (create it with `rag init {name}`)")
    dest = docs_dir(project)
    copied: list[Path] = []
    for raw in paths:
        src = Path(raw).expanduser()
        if src.is_dir():
            candidates = sorted(
                p for p in src.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not candidates:
                raise ValueError(f"no supported files found under: {src}")
        elif src.is_file():
            if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"unsupported file type: {src.name} (supported: {sorted(SUPPORTED_EXTENSIONS)})"
                )
            candidates = [src]
        else:
            raise FileNotFoundError(f"no such file or directory: {src}")
        for c in candidates:
            shutil.copy2(c, dest / c.name)
            copied.append(dest / c.name)
    return copied


def remove_project(home: Path, name: str) -> None:
    project = project_dir(home, name)
    if not project.exists():
        raise FileNotFoundError(f"unknown project: {name}")
    shutil.rmtree(project)


def build_project(home: Path, name: str, reset: bool = True, log=None) -> BuildReport:
    def emit(message: str) -> None:
        if log is not None:
            log(message)

    project = project_dir(home, name)
    cfg = load_config(project)
    report = BuildReport(project=name)
    start = time.time()

    if reset:
        col = reset_collection(name, index_dir(project), embed_model=cfg.embed.model)
    else:
        col = open_collection(name, index_dir(project), embed_model=cfg.embed.model)
        existing = (col.metadata or {}).get("embed_model")
        if existing and existing != cfg.embed.model:
            raise RuntimeError(
                f"index was built with embed model '{existing}' but config says "
                f"'{cfg.embed.model}' — embeddings are not interchangeable; "
                f"rebuild without --no-reset"
            )

    embedder = get_embedder(cfg.embed)
    files = sorted(p for p in docs_dir(project).iterdir() if p.is_file())
    if not files:
        emit(f"no documents in {docs_dir(project)} — add some with `rag add`")

    for path in files:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            emit(f"  SKIP {path.name}: unsupported format")
            continue
        parsed = parse_file(path)
        if parsed.is_scan:
            report.files.append(
                FileReport(
                    file=path.name,
                    status="skipped-scanned",
                    pages=parsed.n_pages,
                    note=(
                        f"{len(parsed.scanned_pages)}/{parsed.n_pages} pages have no text layer "
                        f"(scanned?) — OCR is a phase-2 feature"
                    ),
                )
            )
            emit(
                f"  SKIP {path.name}: looks like a scan "
                f"({len(parsed.scanned_pages)}/{parsed.n_pages} pages without text) — needs OCR (phase 2)"
            )
            continue
        chunks = chunk_document(
            parsed,
            max_tokens=cfg.chunk.max_tokens,
            min_tokens=cfg.chunk.min_tokens,
            overlap_ratio=cfg.chunk.overlap_ratio,
            quality_min_tokens=cfg.chunk.quality_min_tokens,
            drop_reference_blocks=cfg.chunk.drop_reference_blocks,
        )
        note = ""
        if parsed.scanned_pages:
            note = f"{len(parsed.scanned_pages)} scanned page(s) skipped (OCR is a phase-2 feature)"
        if not chunks:
            report.files.append(
                FileReport(file=path.name, status="empty", pages=parsed.n_pages, note="no usable text chunks")
            )
            emit(f"  EMPTY {path.name}: no usable text chunks")
            continue
        embeddings = embedder.encode_texts([c.text_with_prefix for c in chunks])
        add_chunks(col, chunks, embeddings)
        report.files.append(
            FileReport(file=path.name, status="ok", pages=parsed.n_pages, chunks=len(chunks), note=note)
        )
        emit(f"  OK   {path.name}: {len(chunks)} chunks")

    report.total_chunks = sum(f.chunks for f in report.files)
    report.elapsed_s = time.time() - start
    emit(
        f"indexed {report.total_chunks} chunks from {report.ok_files} file(s) "
        f"in {report.elapsed_s:.1f}s (embeddings computed locally)"
    )
    return report


_rerankers: dict[str, Reranker] = {}


def _get_reranker(model_name: str) -> Reranker:
    if model_name not in _rerankers:
        _rerankers[model_name] = Reranker(model_name)
    return _rerankers[model_name]


def query_project(
    home: Path,
    name: str,
    question: str,
    use_llm: bool = True,
    top_k: int | None = None,
    embedder=None,
    col=None,
    profile: str | None = None,
) -> QueryResult:
    project = project_dir(home, name)
    cfg = load_config(project)
    if profile:
        apply_profile(cfg, profile)
    rcfg = cfg.retrieve
    if top_k:
        rcfg.top_k = top_k
    if col is None:
        col = open_collection(name, index_dir(project))
    if embedder is None:
        embedder = get_embedder(cfg.embed)
    reranker = None
    if rcfg.rerank.enabled:
        reranker = _get_reranker(rcfg.rerank.model)
    contexts = retrieve(question, col, embedder, rcfg, reranker=reranker)
    context_tokens = sum(c.token_count for c in contexts)
    if not use_llm or not contexts:
        return QueryResult(
            question=question,
            answer="",
            contexts=contexts,
            model="",
            base_url="",
            generated=False,
            context_tokens=context_tokens,
        )
    usage_path = project / "usage.jsonl"
    answer, usage = generate_answer(
        question, contexts, cfg.generate, usage_path=usage_path, profile=profile
    )
    return QueryResult(
        question=question,
        answer=answer,
        contexts=contexts,
        model=cfg.generate.model,
        base_url=cfg.generate.base_url,
        generated=True,
        context_tokens=context_tokens,
        usage=usage,
    )
