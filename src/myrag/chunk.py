from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from myrag.parse import ParsedDocument

_enc = None

_DIGIT_SYMBOLS = set("°±+-.,|<>[]{}()=×÷%")

_URL_ONLY_RE = re.compile(r"^\s*(?:https?://\S+|www\.\S+)\s*$")
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
_YEAR_RE = re.compile(r"\(\d{4}[a-z]?\)")
_REF_LINE_RE = re.compile(
    r"(?:[A-Z][a-z]+(?:,\s*[A-Z]\.?)+\s*(?:,|&|and)\s*){2,}"
    r"|(?:et\s+al\.?,?\s*\d{4})",
    re.IGNORECASE,
)
_AFFILIATION_RE = re.compile(
    r"(?:department|school|universit|institute|laborator|centre|center|faculty)"
    r"|(?:orcid\.org)",
    re.IGNORECASE,
)
_MD_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")

BOILERPLATE_MARKERS = (
    "copyright and moral rights",
    "take down policy",
    "citation for published version",
    "version of record",
    "research square preprints",
    "verify currency and authenticity",
    "general rights",
)

_SENT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9«\"'])"
    r"|(?<=[.!?])\s*\n"
)


def _encoding():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text)
    return [s.strip() for s in parts if s and s.strip()]


def is_url_only(text: str) -> bool:
    return bool(_URL_ONLY_RE.match(text.strip()))


def is_reference_block(text: str) -> bool:
    t = text.strip()
    if len(t) < 100:
        return False
    if len(_DOI_RE.findall(t)) >= 3:
        return True
    if len(_YEAR_RE.findall(t)) >= 6:
        return True
    if t.lower().count("et al") >= 6:
        return True
    lines = [l for l in t.split("\n") if l.strip()]
    if len(lines) >= 3:
        hits = sum(1 for l in lines if _REF_LINE_RE.search(l))
        if hits / len(lines) > 0.6:
            return True
    return False


def is_digit_heavy(text: str, threshold: float = 0.3) -> bool:
    t = text.strip()
    if len(t) < 50:
        return False
    lines = [l for l in t.split("\n") if l.strip()]
    if sum(1 for l in lines if _MD_TABLE_LINE_RE.match(l)) >= 2:
        return False
    symbols = sum(1 for c in t if c.isdigit() or c in _DIGIT_SYMBOLS)
    return symbols / len(t) > threshold


def is_boilerplate(text: str) -> bool:
    t = text.strip().lower()
    hits = sum(1 for m in BOILERPLATE_MARKERS if m in t)
    return hits >= 2 or (hits >= 1 and len(t) < 300)


def is_affiliation_fragment(text: str) -> bool:
    t = text.strip()
    if len(t) > 500:
        return False
    words = t.split()
    if len(words) < 3:
        return False
    matches = len(_AFFILIATION_RE.findall(t))
    return matches >= 2 and matches / max(len(words), 1) > 0.05


def has_repeating_loop(text: str, min_repeat: int = 5) -> bool:
    t = text.strip()
    if len(t) < 250:
        return False
    for start in range(0, min(len(t) - 50, 400), 10):
        sub = t[start:start + 50]
        if t.count(sub) >= min_repeat:
            return True
    return False


def is_noise_text(text: str) -> bool:
    t = text.strip()
    return len(t) < 3 or is_url_only(t)


def chunk_text_block(text: str, max_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in split_sentences(text):
        st = count_tokens(sent)
        if st > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            ids = _encoding().encode(sent, disallowed_special=())
            for i in range(0, len(ids), max_tokens):
                chunks.append(_encoding().decode(ids[i:i + max_tokens]))
            continue
        if current and current_tokens + st > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += st
    if current:
        chunks.append(" ".join(current))
    return chunks


def add_overlap(chunks: list[str], ratio: float, max_tokens: int) -> list[str]:
    if ratio <= 0 or len(chunks) < 2:
        return chunks
    budget = max(1, int(max_tokens * ratio))
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail: list[str] = []
        tail_tokens = 0
        for s in reversed(split_sentences(chunks[i - 1])):
            st = count_tokens(s)
            if tail_tokens + st > budget:
                break
            tail.insert(0, s)
            tail_tokens += st
        merged = (" ".join(tail) + " " + chunks[i]).strip() if tail else chunks[i]
        if count_tokens(merged) > max_tokens + 50:
            merged = chunks[i]
        out.append(merged)
    return out


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    fmt: str
    section: str
    page: int
    chunk_index: int
    token_count: int
    text: str
    text_with_prefix: str


def _group_blocks(doc: ParsedDocument):
    groups = []
    if doc.fmt == "pdf":
        for b in doc.blocks:
            if b.text.strip():
                groups.append((b.section, [b]))
        return groups
    current_section = None
    current: list = []
    for b in doc.blocks:
        if not b.text.strip():
            continue
        if current and b.section == current_section:
            current.append(b)
        else:
            if current:
                groups.append((current_section, current))
            current_section, current = b.section, [b]
    if current:
        groups.append((current_section, current))
    return groups


def _emit_chunk(
    doc: ParsedDocument, section: str, page: int, text: str, index: int, seen_hashes: set[str]
) -> Chunk | None:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    if digest in seen_hashes:
        return None
    seen_hashes.add(digest)
    prefix_parts = [f"Doc: {doc.title}"]
    if section:
        prefix_parts.append(f"Section: {section}")
    if page and page > 0:
        prefix_parts.append(f"Page: {page}")
    prefix = "\n".join(prefix_parts) + "\n---\n"
    text_with_prefix = prefix + text
    return Chunk(
        chunk_id=f"{doc.doc_id}__{digest[:12]}",
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        title=doc.title,
        fmt=doc.fmt,
        section=section,
        page=page,
        chunk_index=index,
        token_count=count_tokens(text_with_prefix),
        text=text,
        text_with_prefix=text_with_prefix,
    )


def chunk_document(
    doc: ParsedDocument,
    *,
    max_tokens: int = 1000,
    min_tokens: int = 80,
    overlap_ratio: float = 0.05,
    quality_min_tokens: int = 30,
    drop_reference_blocks: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()

    for section, blocks in _group_blocks(doc):
        kind = blocks[0].kind
        texts = [b.text.strip() for b in blocks if b.text.strip()]
        if kind != "urls":
            texts = [t for t in texts if not is_noise_text(t)]
        if not texts:
            continue
        full = "\n\n".join(texts)

        if kind == "urls":
            for piece in chunk_text_block(full, max_tokens):
                chunk = _emit_chunk(doc, section, blocks[0].page, piece, len(chunks), seen_hashes)
                if chunk:
                    chunks.append(chunk)
            continue

        if count_tokens(full) < quality_min_tokens:
            continue

        raw_chunks = chunk_text_block(full, max_tokens)
        if len(raw_chunks) > 1:
            raw_chunks = add_overlap(raw_chunks, overlap_ratio, max_tokens)
        capped = []
        for rc in raw_chunks:
            if count_tokens(rc) > max_tokens + 50:
                capped.extend(chunk_text_block(rc, max_tokens))
            else:
                capped.append(rc)

        kept: list[str] = []
        for rc in capped:
            ct = count_tokens(rc)
            if ct < quality_min_tokens:
                continue
            if (
                (drop_reference_blocks and is_reference_block(rc))
                or is_boilerplate(rc)
                or is_digit_heavy(rc)
                or is_affiliation_fragment(rc)
                or has_repeating_loop(rc)
            ):
                continue
            if kept and ct < min_tokens and count_tokens(kept[-1]) + ct <= max_tokens + 50:
                kept[-1] = kept[-1] + "\n\n" + rc
            else:
                kept.append(rc)

        for text in kept:
            chunk = _emit_chunk(doc, section, blocks[0].page, text, len(chunks), seen_hashes)
            if chunk:
                chunks.append(chunk)
    return chunks
