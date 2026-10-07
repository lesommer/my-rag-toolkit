from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI

from myrag.retrieve import Context

SYSTEM_PROMPT = (
    "You are a meticulous research assistant. Answer the user's question using ONLY "
    "the numbered context excerpts provided.\n"
    "- Cite sources inline as [1], [2], ... matching the excerpt numbers.\n"
    "- If the excerpts do not contain the answer, say so explicitly and do not speculate.\n"
    "- Reply in the same language as the question (a French question gets a French answer).\n"
    "- Be concise and factual; quote exact figures, names and terms when they appear in the excerpts."
)


def build_user_message(question: str, contexts: list[Context]) -> str:
    parts = ["Context excerpts:"]
    for c in contexts:
        origin = f"source: {c.doc_id}"
        if c.page and c.page > 0:
            origin += f", page {c.page}"
        if c.section:
            origin += f", section: {c.section}"
        parts.append(f"[{c.index}] ({origin})\n{c.text}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    contexts: list[Context],
    gcfg,
    usage_path: Path | None = None,
) -> tuple[str, dict]:
    api_key = os.environ.get(gcfg.api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"missing API key: export {gcfg.api_key_env} (see .env.example)"
        )
    client = OpenAI(base_url=gcfg.base_url, api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, contexts)},
    ]
    resp = client.chat.completions.create(
        model=gcfg.model,
        messages=messages,
        temperature=gcfg.temperature,
        max_tokens=gcfg.max_tokens,
    )
    answer = resp.choices[0].message.content or ""
    usage = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": gcfg.model,
        "base_url": gcfg.base_url,
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
        "contexts": len(contexts),
        "context_doc_ids": sorted({c.doc_id for c in contexts}),
        "question_chars": len(question),
    }
    if usage_path is not None:
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(usage, ensure_ascii=False) + "\n")
    return answer, usage
