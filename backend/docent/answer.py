"""Answer a question from retrieved chunks with Claude, citing the passages used."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CLAUDE_MODEL, FALLBACK_BETA
from .index import Hit

NO_ANSWER = "The documentation does not answer this question."

SYSTEM = f"""You answer questions about technical documentation using only the provided documents.
- Answer in 1 to 4 sentences, and cite the documents that support each claim.
- Use no outside knowledge. If the documents do not contain the answer, reply with exactly: {NO_ANSWER}
- A document that is only related to the topic does not count as an answer."""


@dataclass
class Citation:
    rank: int  # 1-based position of the cited chunk in the retrieved list
    chunk_id: str
    section_id: str
    section_title: str
    book_title: str
    page_start: int
    page_end: int
    cited_text: str


@dataclass
class Segment:
    """One text block of the answer, with the citations Claude attached to it."""

    text: str
    citations: list[Citation] = field(default_factory=list)


class UpstreamError(Exception):
    """Claude's reply cannot be used (for example, it hit max_tokens)."""


@dataclass
class Answer:
    text: str  # all segment text joined
    citations: list[Citation] = field(default_factory=list)  # every segment's citations, in order
    segments: list[Segment] = field(default_factory=list)
    no_answer: bool = False  # the model said the docs do not answer
    safety_refusal: bool = False  # stop_reason == "refusal" after fallbacks
    served_by: str | None = None
    usage: dict = field(default_factory=dict)


def document_blocks(hits: list[Hit]) -> list[dict]:
    blocks = []
    for h in hits:
        m = h.doc.metadata
        pages = f"p. {m['page_start']}" if m["page_start"] == m["page_end"] else f"pp. {m['page_start']}-{m['page_end']}"
        blocks.append(
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": h.doc.page_content},
                "title": f"{m['book_title']}, {m['section_title']} ({pages})",
                "citations": {"enabled": True},
            }
        )
    return blocks


def _citation(hits: list[Hit], c) -> Citation | None:
    idx = getattr(c, "document_index", None)
    if idx is None or not 0 <= idx < len(hits):
        return None
    m = hits[idx].doc.metadata
    return Citation(idx + 1, m["chunk_id"], m["section_id"], m["section_title"], m["book_title"],
                    m["page_start"], m["page_end"], c.cited_text)


def parse_response(resp, hits: list[Hit]) -> Answer:
    if resp.stop_reason == "refusal":
        return Answer("", safety_refusal=True, served_by=getattr(resp, "model", None))
    if resp.stop_reason == "max_tokens":
        raise UpstreamError("Claude hit max_tokens before finishing the answer")
    segments: list[Segment] = []
    for block in resp.content:
        if block.type != "text":
            continue
        seen: set[tuple[int, str]] = set()
        cites: list[Citation] = []
        for c in getattr(block, "citations", None) or []:
            cite = _citation(hits, c)
            if cite and (cite.rank, cite.cited_text) not in seen:
                seen.add((cite.rank, cite.cited_text))
                cites.append(cite)
        segments.append(Segment(block.text, cites))
    text = "".join(s.text for s in segments).strip()
    if not text:
        raise UpstreamError("Claude returned no text")
    no_answer = text.startswith(NO_ANSWER)
    if no_answer:
        segments = [Segment(NO_ANSWER)]
    usage = resp.usage
    return Answer(
        text=text,
        citations=[c for s in segments for c in s.citations],
        segments=segments,
        no_answer=no_answer,
        served_by=getattr(resp, "model", None),
        usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
    )


def answer_question(client, question: str, hits: list[Hit], effort: str = "low") -> Answer:
    """One Claude call. Passages are `document` blocks with citations on; safety fallbacks are on."""
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,  # thinking tokens count against this
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": [*document_blocks(hits), {"type": "text", "text": question}]}],
    )
    return parse_response(resp, hits)
