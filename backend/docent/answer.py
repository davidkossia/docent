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
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
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


def parse_response(resp, hits: list[Hit]) -> Answer:
    if resp.stop_reason == "refusal":
        return Answer("", safety_refusal=True, served_by=getattr(resp, "model", None))
    text_parts: list[str] = []
    citations: list[Citation] = []
    for block in resp.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            idx = getattr(c, "document_index", None)
            if idx is None or not 0 <= idx < len(hits):
                continue
            m = hits[idx].doc.metadata
            citations.append(
                Citation(idx + 1, m["chunk_id"], m["section_id"], m["section_title"], m["book_title"],
                         m["page_start"], m["page_end"], c.cited_text)
            )
    text = "".join(text_parts).strip()
    usage = resp.usage
    return Answer(
        text=text,
        citations=citations,
        no_answer=text.startswith(NO_ANSWER),
        served_by=getattr(resp, "model", None),
        usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
    )


def answer_question(client, question: str, hits: list[Hit], effort: str = "low") -> Answer:
    """One Claude call. Passages are `document` blocks with citations on; safety fallbacks are on."""
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": [*document_blocks(hits), {"type": "text", "text": question}]}],
    )
    return parse_response(resp, hits)
