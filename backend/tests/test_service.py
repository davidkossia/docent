import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest
from langchain_core.documents import Document

from docent.answer import NO_ANSWER
from docent.index import Hit
from docent.server import FIXTURES, fake_ask, make_handler
from docent.service import ServiceError, ask


def _hits(n=3):
    return [
        Hit(Document(page_content=f"Passage {i} text.", metadata={
            "chunk_id": f"section-800:{i}", "section_id": f"py-{i}", "section_title": f"{i}.1 Title {i}",
            "book_title": "Intro to Python", "page_start": i, "page_end": i + 1}), 0.9 - i / 10)
        for i in range(1, n + 1)
    ]


class Retriever:
    def search(self, question, k):
        return _hits()[:k]


def _client(blocks, stop="end_turn", raises=None):
    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    if raises:
                        raise raises
                    return NS(stop_reason=stop, content=blocks, model="claude-opus-5-5",
                              usage=NS(input_tokens=1, output_tokens=1))
    return Client


def _text(text, *doc_indexes):
    cites = [NS(document_index=i, cited_text=f"cited {i}") for i in doc_indexes]
    return NS(type="text", text=text, citations=cites or None)


def test_answered_citation_numbers_match_passage_numbers():
    client = _client([_text("A.", 0), _text(" B.", 2, 1), _text(" C.")])
    out = ask("  what?  ", Retriever(), client)
    assert out["status"] == "answered"
    assert [p["n"] for p in out["passages"]] == [1, 2, 3]
    assert "".join(s["text"] for s in out["segments"]) == "A. B. C."
    by_n = {p["n"]: p for p in out["passages"]}
    cited = [c["passage"] for s in out["segments"] for c in s["citations"]]
    assert cited == [1, 3, 2]
    assert all(c["passage"] in by_n for s in out["segments"] for c in s["citations"])
    assert out["segments"][0]["citations"] == [{"passage": 1, "cited_text": "cited 0"}]
    assert out["passages"][0]["chunk_id"] == "section-800:1"
    assert out["passages"][0]["text"] == "Passage 1 text."
    assert set(out["meta"]) == {"model", "setup", "latency_ms"}


def test_no_answer_has_one_segment_and_no_citations():
    out = ask("q", Retriever(), _client([_text(NO_ANSWER)]))
    assert out["status"] == "no_answer"
    assert out["segments"] == [{"text": NO_ANSWER, "citations": []}]
    assert len(out["passages"]) == 3


def test_refused_has_no_segments():
    out = ask("q", Retriever(), _client([], stop="refusal"))
    assert out["status"] == "refused" and out["segments"] == []


def test_setup_comes_from_env(monkeypatch):
    monkeypatch.setenv("DOCENT_SETUP", "section-800__pinecone-e5")
    assert ask("q", Retriever(), _client([_text("A.", 0)]))["meta"]["setup"] == "section-800__pinecone-e5"


@pytest.mark.parametrize("q", ["", "   ", None, 5, "x" * 501])
def test_invalid_question(q):
    with pytest.raises(ServiceError) as e:
        ask(q, Retriever(), _client([]))
    assert (e.value.status, e.value.code) == (400, "invalid_question")


def test_500_characters_is_valid():
    assert ask(" " + "x" * 500 + " ", Retriever(), _client([_text("A.", 0)]))["status"] == "answered"


def test_max_tokens_and_failures_map_to_contract_errors():
    with pytest.raises(ServiceError) as e:
        ask("q", Retriever(), _client([_text("cut")], stop="max_tokens"))
    assert (e.value.status, e.value.code) == (502, "upstream_error")
    with pytest.raises(ServiceError) as e:
        ask("q", Retriever(), _client([], raises=RuntimeError("boom")))
    assert e.value.code == "upstream_error"
    with pytest.raises(ServiceError) as e:
        ask("q", Retriever(), _client([], raises=TimeoutError()))
    assert (e.value.status, e.value.code) == (504, "timeout")


def test_fixtures_are_valid_contract_bodies():
    for name in ("answered", "no_answer", "refused"):
        body = json.loads((FIXTURES / f"{name}.json").read_text())
        n = {p["n"] for p in body["passages"]}
        assert n == set(range(1, len(n) + 1))
        assert all(c["passage"] in n for s in body["segments"] for c in s["citations"])
        for s in body["segments"]:
            for c in s["citations"]:
                assert c["cited_text"] in body["passages"][c["passage"] - 1]["text"]
    answered = json.loads((FIXTURES / "answered.json").read_text())
    assert len(answered["segments"]) == 3 and len(answered["passages"]) == 5
    assert fake_ask("daily_limit", None)[0] == 429


def _post(port, payload, path="/ask"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=payload, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers


def test_http_server_serves_fake_fixtures_and_errors():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lambda q: fake_ask(q, "auto")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, body, headers = _post(port, b'{"question":"x"}')
        assert status == 200 and body["status"] == "answered"
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        assert _post(port, b'{"question":"no_answer"}')[1]["status"] == "no_answer"
        assert _post(port, b'{"question":"refused"}')[1]["status"] == "refused"
        status, body, _ = _post(port, b'{"question":"daily_limit"}')
        assert status == 429 and body["error"]["code"] == "daily_limit"
        status, body, _ = _post(port, json.dumps({"question": "x" * 501}).encode())
        assert (status, body["error"]["code"]) == (400, "invalid_question")
        assert _post(port, b"not json")[0] == 400
        assert _post(port, b"{}", path="/nope")[0] == 404
    finally:
        server.shutdown()
