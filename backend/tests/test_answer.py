from types import SimpleNamespace as NS

from langchain_core.documents import Document

from docent.answer import NO_ANSWER, answer_question, document_blocks, parse_response
from docent.index import Hit


def _hits():
    meta = {"chunk_id": "c:1", "section_id": "py-1.5", "section_title": "1.5 Number basics",
            "book_title": "Intro to Python", "page_start": 29, "page_end": 30}
    return [Hit(Document(page_content="Multiplication takes precedence.", metadata=meta), 0.9)]


def _resp(blocks, stop="end_turn"):
    return NS(stop_reason=stop, content=blocks, model="claude-opus-5-5",
              usage=NS(input_tokens=10, output_tokens=5))


def test_document_blocks_enable_citations():
    b = document_blocks(_hits())[0]
    assert b["citations"] == {"enabled": True}
    assert "pp. 29-30" in b["title"]


def test_parse_maps_citation_to_chunk():
    cite = NS(document_index=0, cited_text="Multiplication takes precedence.")
    resp = _resp([NS(type="text", text="Multiplication goes first.", citations=[cite])])
    a = parse_response(resp, _hits())
    assert a.citations[0].section_id == "py-1.5"
    assert not a.no_answer and not a.safety_refusal


def test_no_answer_and_safety_refusal_are_flagged():
    a = parse_response(_resp([NS(type="text", text=NO_ANSWER, citations=None)]), _hits())
    assert a.no_answer and not a.citations
    assert parse_response(_resp([], stop="refusal"), _hits()).safety_refusal


def test_request_turns_on_fallbacks_and_citations():
    seen = {}

    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    return _resp([NS(type="text", text=NO_ANSWER, citations=None)])

    answer_question(Client, "q?", _hits())
    assert seen["model"] == "claude-opus-5-5"
    assert seen["fallbacks"] == "default"
    assert seen["betas"] == ["server-side-fallback-2026-07-01"]
    assert seen["messages"][0]["content"][0]["type"] == "document"
