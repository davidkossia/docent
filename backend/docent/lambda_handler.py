"""AWS Lambda entry point: `POST /ask` behind an API Gateway HTTP API (payload format 2.0).

Each request runs in this order: parse and validate, then the daily quota, then `service.ask`.
No path reaches Claude without passing the quota. Logs hold the status, latency and token counts
only: never the question, the answer, or a key.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

from . import quota
from .service import ServiceError, ask, setup_name, validate_question

log = logging.getLogger("docent")
log.setLevel(logging.INFO)
for _name in ("anthropic", "httpx", "httpcore", "pinecone", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

ANTHROPIC_PARAM = "/docent/anthropic_api_key"
PINECONE_PARAM = "/docent/pinecone_api_key"
EMBED_MODELS = {"pinecone-e5": "multilingual-e5-large"}  # embedder name -> Pinecone Inference model
MAX_BODY_BYTES = 8 * 1024
DEFAULT_REMAINING_S = 29.0  # the Lambda timeout, used when no context is given
REPLY_MARGIN_S = 2.0  # time kept back to log and reply before the Lambda timeout
ALARM_GRACE_S = 0.5  # the backstop alarm fires this long after the model call's own timeout
ALARM_REPEAT_S = 0.5  # and again at this interval until the request returns
MIN_MODEL_S = 1.0  # do not start a Claude call with less time than this


@dataclass
class Deps:
    table: object  # boto3 DynamoDB Table for the quota
    retriever: object  # PineconeRetriever
    client: object  # anthropic.Anthropic


_deps: Deps | None = None  # built once per container (cold start)


def _bad_body() -> ServiceError:
    return ServiceError(400, "invalid_question", "Send JSON with a question of 1 to 500 characters.")


def _parse_question(event) -> object:
    raw = event.get("body") if isinstance(event, dict) else None
    if not isinstance(raw, str) or not raw:
        raise _bad_body()
    if event.get("isBase64Encoded"):
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise _bad_body() from None
    else:
        data = raw.encode("utf-8", "surrogatepass")
    if len(data) > MAX_BODY_BYTES:
        raise _bad_body()
    try:
        body = json.loads(data)
    except (ValueError, RecursionError):
        raise _bad_body() from None
    if not isinstance(body, dict):
        raise _bad_body()
    return body.get("question")


def _read_keys(ssm) -> tuple[str, str]:
    resp = ssm.get_parameters(Names=[ANTHROPIC_PARAM, PINECONE_PARAM], WithDecryption=True)
    values = {p["Name"]: p.get("Value") for p in resp.get("Parameters", [])}
    missing = [name for name in (ANTHROPIC_PARAM, PINECONE_PARAM) if not values.get(name)]
    if missing:
        raise RuntimeError(f"missing SSM parameters: {', '.join(missing)}")
    return values[ANTHROPIC_PARAM], values[PINECONE_PARAM]


def split_setup(setup: str) -> tuple[str, str]:
    """`<strategy>__<embedder>` -> (strategy, embedder). The Lambda only runs Pinecone-hosted embedders."""
    strategy, sep, embedder = setup.rpartition("__")
    if not sep or not strategy or embedder not in EMBED_MODELS:
        raise RuntimeError(f"DOCENT_SETUP must end in one of {sorted(EMBED_MODELS)}")
    return strategy, embedder


def _build_retriever(pinecone_key: str, setup: str):
    from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
    from pinecone import Pinecone

    from .pinecone_store import INDEX_NAME, PineconeRetriever, namespace_for

    strategy, embedder = split_setup(setup)
    embeddings = PineconeEmbeddings(model=EMBED_MODELS[embedder], pinecone_api_key=pinecone_key)
    index = Pinecone(api_key=pinecone_key).Index(INDEX_NAME)
    store = PineconeVectorStore(
        index=index, embedding=embeddings, namespace=namespace_for(strategy, embedder), text_key="text"
    )
    return PineconeRetriever(store)


def load_deps() -> Deps:
    """Read both keys from SSM SecureString parameters and build the clients. Keys stay in memory."""
    import anthropic
    import boto3

    anthropic_key, pinecone_key = _read_keys(boto3.client("ssm"))
    table = boto3.resource("dynamodb").Table(os.environ["QUOTA_TABLE"])
    retriever = _build_retriever(pinecone_key, setup_name())
    return Deps(table, retriever, anthropic.Anthropic(api_key=anthropic_key, max_retries=0))


def _cached_deps() -> Deps:
    global _deps
    if _deps is None:
        _deps = load_deps()  # a failure is not cached; the next request tries again
    return _deps


class _Messages:
    def __init__(self, client, deadline: float, usage: dict):
        self._client, self._deadline, self._usage = client, deadline, usage

    def create(self, **kwargs):
        budget = self._deadline - time.monotonic()
        if budget < MIN_MODEL_S:
            raise TimeoutError("no time left for the model call")
        resp = self._client.with_options(timeout=budget, max_retries=0).beta.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._usage["input_tokens"] = getattr(usage, "input_tokens", None)
            self._usage["output_tokens"] = getattr(usage, "output_tokens", None)
        return resp


class DeadlineClient:
    """The Anthropic client for one request: one attempt, a timeout that ends before the Lambda's,
    and the token counts for the log. Exposes only `beta.messages.create`, which `answer.py` uses."""

    def __init__(self, client, deadline: float):
        self.usage: dict = {}
        self.beta = SimpleNamespace(messages=_Messages(client, deadline, self.usage))


@contextmanager
def _alarm(seconds: float, fired: list):
    """Raise `TimeoutError` after `seconds` of wall time, then every `ALARM_REPEAT_S` until the block
    exits (in case a library retries after the first one): a backstop for a network read that no
    client timeout bounds. Main thread only, as signals require; elsewhere it does nothing."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(signum, frame):
        fired.append(True)
        raise TimeoutError("request deadline passed")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.05), ALARM_REPEAT_S)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _remaining_s(context) -> float:
    get = getattr(context, "get_remaining_time_in_millis", None)
    return get() / 1000 if callable(get) else DEFAULT_REMAINING_S


def _answer(event, context, deps: Deps | None, record: dict) -> dict:
    global _deps
    question = validate_question(_parse_question(event))
    deps = deps or _cached_deps()
    if not quota.check_and_count(deps.table, quota.utc_day(), quota.daily_limit()):
        raise ServiceError(429, "daily_limit", "The daily question limit is reached. Try again tomorrow.")
    budget = _remaining_s(context) - REPLY_MARGIN_S
    client = DeadlineClient(deps.client, time.monotonic() + budget)
    fired: list = []
    try:
        with _alarm(budget + ALARM_GRACE_S, fired):
            result = ask(question, deps.retriever, client)
    finally:
        record.update(client.usage)
        if fired:
            _deps = None  # an interrupted read may leave a connection pool broken; rebuild next time
            record["alarm"] = True
    record["model"] = result["meta"]["model"]
    return result


def handle(event, context=None, deps: Deps | None = None) -> dict:
    """Return an API Gateway proxy response. `deps` replaces the SSM, DynamoDB and model clients in tests."""
    start = time.monotonic()
    record: dict = {"event": "ask"}
    try:
        status_code, body = 200, _answer(event, context, deps, record)
    except ServiceError as exc:
        status_code, body = exc.status, exc.body()
        if exc.__cause__ is not None:
            record["error_type"] = type(exc.__cause__).__name__
    except Exception as exc:  # noqa: BLE001  unplanned, including the quota table: fail closed, say nothing
        status_code, body = 502, ServiceError(502, "upstream_error", "The service failed. Try again.").body()
        record["error_type"] = type(exc).__name__
    record.update(
        status_code=status_code,
        status=body["status"] if status_code == 200 else body["error"]["code"],
        latency_ms=round((time.monotonic() - start) * 1000),
    )
    log.info(json.dumps(record))
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


def handler(event, context):
    """The Lambda handler (`docent.lambda_handler.handler`)."""
    return handle(event, context)
