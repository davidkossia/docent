from types import SimpleNamespace as NS

import pytest
from langchain_core.documents import Document

from docent.answer import NO_ANSWER, UpstreamError, answer_question, document_blocks, parse_response
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


def test_one_segment_per_text_block_with_its_own_citations():
    c1 = NS(document_index=0, cited_text="Multiplication takes precedence.")
    resp = _resp([
        NS(type="text", text="Multiplication goes first.", citations=[c1, c1]),
        NS(type="thinking", thinking="..."),
        NS(type="text", text=" Parentheses override.", citations=None),
    ])
    a = parse_response(resp, _hits())
    assert [s.text for s in a.segments] == ["Multiplication goes first.", " Parentheses override."]
    assert len(a.segments[0].citations) == 1  # duplicate dropped
    assert a.segments[1].citations == []
    assert a.text == "Multiplication goes first. Parentheses override."


def test_max_tokens_is_an_upstream_error():
    with pytest.raises(UpstreamError):
        parse_response(_resp([NS(type="text", text="cut", citations=None)], stop="max_tokens"), _hits())


def test_end_turn_without_text_is_an_upstream_error():
    with pytest.raises(UpstreamError):
        parse_response(_resp([NS(type="thinking", thinking="...")]), _hits())
    with pytest.raises(UpstreamError):
        parse_response(_resp([]), _hits())
    assert parse_response(_resp([], stop="refusal"), _hits()).safety_refusal  # refusal still wins


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
    assert seen["max_tokens"] == 16000
    assert seen["output_config"] == {"effort": "low"}
    assert seen["fallbacks"] == "default"
    assert seen["betas"] == ["server-side-fallback-2026-07-01"]
    assert seen["messages"][0]["content"][0]["type"] == "document"
