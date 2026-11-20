"""The `ask()` function: question in, API contract body out (see docs/api.md)."""
from __future__ import annotations

import os
import time
from typing import Protocol

from .answer import NO_ANSWER, UpstreamError, answer_question
from .config import CLAUDE_MODEL
from .index import Hit, LocalIndex

DEFAULT_SETUP = "section-800__local-bge-small"
MAX_QUESTION_CHARS = 500


class ServiceError(Exception):
    """An error with an HTTP status and a contract error code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class Retriever(Protocol):
    def search(self, question: str, k: int) -> list[Hit]: ...


class LocalRetriever:
    """Retrieves from a `LocalIndex` (needs the index's embedder set)."""

    def __init__(self, index: LocalIndex):
        self.index = index

    def search(self, question: str, k: int) -> list[Hit]:
        return self.index.search(question, k)


def setup_name() -> str:
    return os.environ.get("DOCENT_SETUP", DEFAULT_SETUP)


def validate_question(question) -> str:
    q = question.strip() if isinstance(question, str) else ""
    if not 1 <= len(q) <= MAX_QUESTION_CHARS:
        raise ServiceError(400, "invalid_question", "Enter a question of 1 to 500 characters.")
    return q


def _passages(hits: list[Hit]) -> list[dict]:
    out = []
    for n, h in enumerate(hits, start=1):
        m = h.doc.metadata
        out.append({
            "n": n,
            "chunk_id": m["chunk_id"],
            "book_title": m["book_title"],
            "section_title": m["section_title"],
            "page_start": m["page_start"],
            "page_end": m["page_end"],
            "text": h.doc.page_content,
        })
    return out


def _is_timeout(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or type(exc).__name__ == "APITimeoutError"


def ask(question, retriever: Retriever, client, k: int = 5) -> dict:
    """Return the section 5 response body, or raise `ServiceError`."""
    q = validate_question(question)
    start = time.monotonic()
    try:
        hits = retriever.search(q, k)
    except Exception as exc:
        raise ServiceError(502, "upstream_error", "Search failed. Try again.") from exc
    try:
        answer = answer_question(client, q, hits)
    except UpstreamError as exc:
        raise ServiceError(502, "upstream_error", "The model could not finish the answer.") from exc
    except Exception as exc:
        if _is_timeout(exc):
            raise ServiceError(504, "timeout", "The model took too long to answer.") from exc
        raise ServiceError(502, "upstream_error", "The model request failed.") from exc
    if answer.safety_refusal:
        status, segments = "refused", []
    elif answer.no_answer:
        status, segments = "no_answer", [{"text": NO_ANSWER, "citations": []}]
    else:
        status = "answered"
        segments = [
            {"text": s.text, "citations": [{"passage": c.rank, "cited_text": c.cited_text} for c in s.citations]}
            for s in answer.segments
        ]
    return {
        "status": status,
        "segments": segments,
        "passages": _passages(hits),
        "meta": {
            "model": answer.served_by or CLAUDE_MODEL,
            "setup": setup_name(),
            "latency_ms": round((time.monotonic() - start) * 1000),
        },
    }
