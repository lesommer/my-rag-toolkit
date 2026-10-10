from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


@dataclass
class EmbedConfig:
    model: str = "BAAI/bge-m3"
    device: str = "auto"
    batch_size: int = 16


@dataclass
class ChunkConfig:
    max_tokens: int = 1000
    min_tokens: int = 80
    overlap_ratio: float = 0.05
    quality_min_tokens: int = 30


@dataclass
class RerankConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"


@dataclass
class RetrieveConfig:
    top_k: int = 8
    candidate_k: int = 24
    score_threshold: float = 0.35
    min_candidates: int = 3
    rerank: RerankConfig = field(default_factory=RerankConfig)


@dataclass
class GenerateConfig:
    base_url: str = ALBERT_BASE_URL
    model: str = "gpt-oss-120b"
    temperature: float = 0.2
    max_tokens: int = 2048
    api_key_env: str = "ALBERT_API_KEY"
    extra_body: dict | None = None


@dataclass
class ProjectConfig:
    name: str
    description: str = ""
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    profiles: dict[str, dict] = field(default_factory=dict)


def default_home() -> Path:
    env = os.environ.get("MYRAG_HOME")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "rags"


def project_dir(home: Path, name: str) -> Path:
    return home / name


def config_path(project: Path) -> Path:
    return project / "config.yaml"


def docs_dir(project: Path) -> Path:
    return project / "docs"


def index_dir(project: Path) -> Path:
    return project / "index"


def collection_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")
    s = s[:63]
    if len(s) < 3:
        s = f"rag-{s}"
    if not s[0].isalnum():
        s = "r" + s
    if not s[-1].isalnum():
        s = s + "0"
    return s


def load_config(project: Path) -> ProjectConfig:
    raw: dict = {}
    path = config_path(project)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    cfg = ProjectConfig(name=raw.get("name") or project.name, description=raw.get("description", ""))
    sections = {
        "embed": EmbedConfig,
        "chunk": ChunkConfig,
        "retrieve": RetrieveConfig,
        "generate": GenerateConfig,
    }
    for section, _cls in sections.items():
        data = raw.get(section) or {}
        target = getattr(cfg, section)
        for key, value in data.items():
            if key == "rerank":
                for rk, rv in (value or {}).items():
                    if hasattr(target.rerank, rk):
                        setattr(target.rerank, rk, rv)
            elif hasattr(target, key):
                setattr(target, key, value)
    raw_profiles = raw.get("profiles") or {}
    cfg.profiles = {k: (v if isinstance(v, dict) else {}) for k, v in raw_profiles.items()}
    return cfg


def apply_profile(cfg: ProjectConfig, profile: str) -> ProjectConfig:
    if profile not in cfg.profiles:
        available = ", ".join(sorted(cfg.profiles)) or "none"
        raise ValueError(f"unknown generate profile '{profile}' (defined: {available})")
    gen = asdict(cfg.generate)
    for key, value in cfg.profiles[profile].items():
        if key in gen:
            gen[key] = value
    cfg.generate = GenerateConfig(**gen)
    return cfg


def save_config(project: Path, cfg: ProjectConfig) -> None:
    header = (
        "# my-rag-toolkit project configuration\n"
        "# Every project is an independent RAG: its own corpus, index and settings.\n"
        "# Embeddings are always computed locally (bge-m3); generation calls the\n"
        "# configured sovereign endpoint (default: Albert).\n"
    )
    config_path(project).write_text(
        header + yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
