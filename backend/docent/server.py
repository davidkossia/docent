"""Local HTTP server for `POST /ask` (standard library only)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import ROOT
from .service import ServiceError, ask, validate_question

PORT = 8787
ORIGIN = "http://localhost:5173"
FIXTURES = ROOT.parent / "web" / "src" / "fixtures"
FIXTURE_NAMES = ("answered", "no_answer", "refused", "error_daily_limit")
MAX_BODY = 16_384
ERROR_STATUS = {"invalid_question": 400, "daily_limit": 429, "upstream_error": 502, "timeout": 504}


def fixture_for(question: str, forced: str | None = None) -> str:
    """Pick a fixture name: forced, else by a keyword in the question, else `answered`."""
    if forced and forced != "auto":
        return forced
    q = question.lower()
    if "daily_limit" in q:
        return "error_daily_limit"
    for name in ("no_answer", "refused"):
        if name in q:
            return name
    return "answered"


def fake_ask(question, forced: str | None = None) -> tuple[int, dict]:
    name = fixture_for(validate_question(question), forced)
    body = json.loads((FIXTURES / f"{name}.json").read_text())
    return (ERROR_STATUS[body["error"]["code"]] if "error" in body else 200), body


def make_handler(handle):
    """`handle(question) -> (status, body)`; `ServiceError` becomes the contract error body."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict | None = None) -> None:
            data = json.dumps(body).encode() if body is not None else b""
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", ORIGIN)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204)

        def do_POST(self):
            if self.path != "/ask":
                return self._send(404, ServiceError(404, "not_found", "No such route.").body())
            try:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if not 0 <= length <= MAX_BODY:
                    raise ServiceError(400, "invalid_question", "Enter a question of 1 to 500 characters.")
                try:
                    question = json.loads(self.rfile.read(length) or b"{}").get("question")
                except (ValueError, AttributeError):
                    question = None
                status, body = handle(question)
            except ServiceError as exc:
                status, body = exc.status, exc.body()
            except Exception:  # never leak internals
                status, body = 502, ServiceError(502, "upstream_error", "The request failed.").body()
            self._send(status, body)

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

    return Handler


def serve(fake: str | None = None, port: int = PORT) -> None:
    if fake:
        handle = lambda q: fake_ask(q, fake)  # noqa: E731
    else:
        from anthropic import Anthropic

        from .config import INDEX_DIR, env
        from .embeddings import LOCAL_MODELS, get_embedder
        from .index import LocalIndex
        from .service import LocalRetriever, setup_name

        api_key = env("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to .env, or run with --fake.")
        key = setup_name()
        embedder_name = key.split("__", 1)[-1]
        if embedder_name not in LOCAL_MODELS or not (INDEX_DIR / key / "vectors.npy").exists():
            raise SystemExit(
                f"DOCENT_SETUP={key!r} is not a local setup with a built index. "
                "Use <strategy>__local-bge-small or <strategy>__local-bge-base."
            )
        index = LocalIndex.load(INDEX_DIR / key, get_embedder(embedder_name))
        retriever = LocalRetriever(index)
        client = Anthropic(api_key=api_key, timeout=60, max_retries=0)
        handle = lambda q: (200, ask(q, retriever, client))  # noqa: E731
    print(f"docent API on http://localhost:{port}/ask" + (" (fake)" if fake else ""))
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(handle)).serve_forever()
