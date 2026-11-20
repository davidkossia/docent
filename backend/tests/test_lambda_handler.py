import base64
import json
import logging
import sys
import time
from types import SimpleNamespace as NS

import pytest
from langchain_core.documents import Document

from docent import lambda_handler as lh
from docent.index import Hit

ANTHROPIC_KEY = "sk-ant-test-0000"
PINECONE_KEY = "pcsk-test-0000"
QUESTION = "How does operator precedence work in Python expressions?"


def _hits():
    return [Hit(Document(page_content="Multiplication goes first.", metadata={
        "chunk_id": "section-800:1", "section_id": "py-1.5", "section_title": "1.5 Number basics",
        "book_title": "Intro to Python", "page_start": 29, "page_end": 30}), 0.9)]


class Retriever:
    def __init__(self, raises=None):
        self.raises, self.calls = raises, 0

    def search(self, question, k):
        self.calls += 1
        if self.raises:
            raise self.raises
        return _hits()


class Client:
    """Fake Anthropic client: records `with_options` and `create` calls."""

    def __init__(self, raises=None, delay=0.0):
        self.raises, self.delay = raises, delay
        self.options: list = []
        self.created = 0
        self.beta = NS(messages=NS(create=self._create))

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    def _create(self, **kw):
        self.created += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        cite = NS(document_index=0, cited_text="Multiplication goes first.")
        return NS(stop_reason="end_turn", model="claude-opus-5-5",
                  content=[NS(type="text", text="Multiplication comes before addition.", citations=[cite])],
                  usage=NS(input_tokens=1234, output_tokens=56))


class Table:
    def __init__(self, allow=True, raises=None):
        self.allow, self.raises, self.calls = allow, raises, 0

    def update_item(self, **kw):
        self.calls += 1
        if self.raises:
            raise self.raises
        if not self.allow:
            exc = Exception("condition failed")
            exc.response = {"Error": {"Code": "ConditionalCheckFailedException"}}
            raise exc


class Context:
    def __init__(self, remaining_ms=29_000):
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self):
        return self.remaining_ms


def _deps(table=None, retriever=None, client=None):
    return lh.Deps(table or Table(), retriever or Retriever(), client or Client())


def _event(body=None, b64=False, raw=None):
    text = raw if raw is not None else json.dumps(body)
    if b64:
        text = base64.b64encode(text.encode()).decode()
    return {"version": "2.0", "routeKey": "POST /ask", "body": text, "isBase64Encoded": b64}


def _call(event, deps, context=None):
    resp = lh.handle(event, context or Context(), deps)
    assert resp["headers"]["content-type"] == "application/json"
    return resp["statusCode"], json.loads(resp["body"])


@pytest.fixture(autouse=True)
def _reset_cache():
    lh._deps = None
    yield
    lh._deps = None


def test_answered_question_returns_the_contract_body(caplog):
    deps = _deps()
    with caplog.at_level(logging.INFO, logger="docent"):
        status, body = _call(_event({"question": QUESTION}), deps)
    assert status == 200
    assert body["status"] == "answered"
    assert body["segments"][0]["citations"] == [{"passage": 1, "cited_text": "Multiplication goes first."}]
    assert body["passages"][0]["n"] == 1
    assert deps.table.calls == 1 and deps.client.created == 1
    record = json.loads(caplog.records[-1].getMessage())
    assert record == {"event": "ask", "input_tokens": 1234, "output_tokens": 56, "model": "claude-opus-5-5",
                      "status_code": 200, "status": "answered", "latency_ms": record["latency_ms"]}


def test_model_call_gets_one_attempt_and_a_timeout_inside_the_lambda_budget():
    deps = _deps()
    _call(_event({"question": QUESTION}), deps, Context(remaining_ms=29_000))
    [opts] = deps.client.options
    assert opts["max_retries"] == 0
    assert 25 < opts["timeout"] <= 29 - lh.REPLY_MARGIN_S


def test_base64_body_is_decoded():
    status, body = _call(_event({"question": QUESTION}, b64=True), _deps())
    assert status == 200 and body["status"] == "answered"


def test_over_the_daily_limit_returns_429_and_never_calls_the_model():
    deps = _deps(table=Table(allow=False))
    status, body = _call(_event({"question": QUESTION}), deps)
    assert status == 429
    assert body["error"]["code"] == "daily_limit"
    assert deps.retriever.calls == 0 and deps.client.created == 0


@pytest.mark.parametrize("event", [
    _event(raw="{not json"),
    _event(raw=""),
    {"version": "2.0", "routeKey": "POST /ask"},
    {"version": "2.0", "body": "e30=!", "isBase64Encoded": True},
    _event(raw="[1, 2]"),
    _event({"question": 42}),
    _event({"text": QUESTION}),
    _event({"question": "   "}),
    _event({"question": "x" * 501}),
    _event(raw=json.dumps({"question": QUESTION, "pad": "y" * lh.MAX_BODY_BYTES})),
    _event(raw="[" * 3000 + "]" * 3000),
    "not an event",
], ids=["bad-json", "empty-body", "missing-body", "bad-base64", "not-object", "not-string",
        "no-question", "blank", "501-chars", "oversized-body", "deep-nesting", "not-a-dict"])
def test_invalid_requests_return_400_before_the_quota(event):
    deps = _deps()
    status, body = _call(event, deps)
    assert status == 400
    assert body["error"]["code"] == "invalid_question"
    assert deps.table.calls == 0 and deps.client.created == 0


def test_500_character_question_is_accepted():
    status, _ = _call(_event({"question": "x" * 500}), _deps())
    assert status == 200


def test_model_failure_returns_502_without_details():
    deps = _deps(client=Client(raises=RuntimeError(f"boom {ANTHROPIC_KEY}")))
    status, body = _call(_event({"question": QUESTION}), deps)
    assert status == 502
    assert body == {"error": {"code": "upstream_error", "message": "The model request failed."}}


def test_search_failure_returns_502_and_still_counts_the_question():
    deps = _deps(retriever=Retriever(raises=ConnectionError("pinecone down")))
    status, body = _call(_event({"question": QUESTION}), deps)
    assert status == 502 and body["error"]["code"] == "upstream_error"
    assert deps.table.calls == 1 and deps.client.created == 0


def test_model_timeout_returns_504():
    class APITimeoutError(Exception):
        pass

    status, body = _call(_event({"question": QUESTION}), _deps(client=Client(raises=APITimeoutError())))
    assert status == 504 and body["error"]["code"] == "timeout"


def test_no_time_left_returns_504_without_calling_the_model():
    deps = _deps()
    status, body = _call(_event({"question": QUESTION}), deps, Context(remaining_ms=2_500))
    assert status == 504 and body["error"]["code"] == "timeout"
    assert deps.client.created == 0


def test_alarm_backstop_turns_a_hung_call_into_504_and_drops_the_cached_clients(monkeypatch):
    monkeypatch.setattr(lh, "REPLY_MARGIN_S", 0.0)
    monkeypatch.setattr(lh, "ALARM_GRACE_S", 0.0)
    monkeypatch.setattr(lh, "MIN_MODEL_S", 0.0)
    deps = _deps(client=Client(delay=5))  # ignores its own timeout, like a hung socket read
    lh._deps = deps
    start = time.monotonic()
    status, body = _call(_event({"question": QUESTION}), None, Context(remaining_ms=300))
    assert time.monotonic() - start < 2
    assert status == 504 and body["error"]["code"] == "timeout"
    assert lh._deps is None


def test_alarm_repeats_when_a_library_retries_after_the_first_one(monkeypatch):
    monkeypatch.setattr(lh, "REPLY_MARGIN_S", 0.0)
    monkeypatch.setattr(lh, "ALARM_GRACE_S", 0.0)
    monkeypatch.setattr(lh, "ALARM_REPEAT_S", 0.1)

    class RetryingRetriever(Retriever):
        def search(self, question, k):
            try:
                time.sleep(5)
            except TimeoutError:
                time.sleep(5)  # a retry that hangs again
            return _hits()

    deps = _deps(retriever=RetryingRetriever())
    start = time.monotonic()
    status, body = _call(_event({"question": QUESTION}), deps, Context(remaining_ms=200))
    assert time.monotonic() - start < 2
    assert status == 502 and body["error"]["code"] == "upstream_error"
    assert deps.client.created == 0


def test_quota_table_failure_fails_closed():
    deps = _deps(table=Table(raises=TimeoutError("dynamodb")))
    status, body = _call(_event({"question": QUESTION}), deps)
    assert status == 502 and body["error"]["code"] == "upstream_error"
    assert deps.client.created == 0


def test_logs_hold_no_question_answer_or_key(caplog):
    deps = _deps(client=Client(raises=RuntimeError(f"failed for {ANTHROPIC_KEY} on {QUESTION}")))
    with caplog.at_level(logging.DEBUG):
        _call(_event({"question": QUESTION}), deps)
        _call(_event({"question": QUESTION}), _deps())
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert QUESTION not in text and ANTHROPIC_KEY not in text
    assert "Multiplication" not in text
    assert '"error_type": "RuntimeError"' in text


# Cold start: keys from SSM, clients built once per container.

class FakeSSM:
    def __init__(self, params):
        self.params, self.calls = params, []

    def get_parameters(self, **kw):
        self.calls.append(kw)
        return {"Parameters": [{"Name": n, "Value": v} for n, v in self.params.items()],
                "InvalidParameters": []}


def _fake_aws(monkeypatch, ssm):
    built = {}
    boto3 = NS(
        client=lambda name: ssm if name == "ssm" else pytest.fail(f"unexpected client {name}"),
        resource=lambda name: NS(Table=lambda table_name: NS(name=table_name, update_item=Table().update_item)),
    )

    class Anthropic:
        def __init__(self, **kw):
            built["anthropic"] = kw

    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "anthropic", NS(Anthropic=Anthropic))
    monkeypatch.setattr(lh, "_build_retriever", lambda key, setup: built.update(pinecone=key, setup=setup)
                        or Retriever())
    monkeypatch.setenv("QUOTA_TABLE", "docent-quota")
    monkeypatch.setenv("DOCENT_SETUP", "section-800__pinecone-e5")
    return built


def test_cold_start_reads_both_keys_from_ssm_once(monkeypatch):
    ssm = FakeSSM({lh.ANTHROPIC_PARAM: ANTHROPIC_KEY, lh.PINECONE_PARAM: PINECONE_KEY})
    built = _fake_aws(monkeypatch, ssm)
    monkeypatch.setattr(lh, "ask", lambda q, r, c: {"status": "answered", "meta": {"model": "m"}})
    for _ in range(3):
        resp = lh.handler(_event({"question": QUESTION}), Context())
        assert resp["statusCode"] == 200
    assert ssm.calls == [{"Names": [lh.ANTHROPIC_PARAM, lh.PINECONE_PARAM], "WithDecryption": True}]
    assert built["anthropic"] == {"api_key": ANTHROPIC_KEY, "max_retries": 0}
    assert built["pinecone"] == PINECONE_KEY
    assert built["setup"] == "section-800__pinecone-e5"
    assert lh._deps.table.name == "docent-quota"


def test_missing_parameter_returns_502_and_is_retried_next_request(monkeypatch):
    ssm = FakeSSM({lh.ANTHROPIC_PARAM: ANTHROPIC_KEY})
    _fake_aws(monkeypatch, ssm)
    for _ in range(2):
        resp = lh.handler(_event({"question": QUESTION}), Context())
        assert resp["statusCode"] == 502
        assert ANTHROPIC_KEY not in resp["body"] and "pinecone" not in resp["body"]
    assert len(ssm.calls) == 2 and lh._deps is None


def test_build_retriever_passes_the_key_and_namespace(monkeypatch):
    import langchain_pinecone
    import pinecone

    seen = {}

    class Embeddings:
        def __init__(self, **kw):
            seen["embeddings"] = kw

    class Pinecone:
        def __init__(self, **kw):
            seen["client"] = kw

        def Index(self, name):
            seen["index"] = name
            return "index"

    class Store:
        def __init__(self, **kw):
            seen["store"] = kw

    monkeypatch.setattr(langchain_pinecone, "PineconeEmbeddings", Embeddings)
    monkeypatch.setattr(langchain_pinecone, "PineconeVectorStore", Store)
    monkeypatch.setattr(pinecone, "Pinecone", Pinecone)
    retriever = lh._build_retriever(PINECONE_KEY, "section-800__pinecone-e5")
    assert seen["embeddings"] == {"model": "multilingual-e5-large", "pinecone_api_key": PINECONE_KEY}
    assert seen["client"] == {"api_key": PINECONE_KEY}
    assert seen["index"] == "docent"
    assert seen["store"]["namespace"] == "section-800__pinecone-e5"
    assert seen["store"]["index"] == "index" and seen["store"]["text_key"] == "text"
    assert isinstance(retriever.store, Store)


@pytest.mark.parametrize("setup", ["section-800__local-bge-small", "pinecone-e5", "__pinecone-e5"])
def test_setup_must_use_a_pinecone_hosted_embedder(setup):
    with pytest.raises(RuntimeError):
        lh.split_setup(setup)
