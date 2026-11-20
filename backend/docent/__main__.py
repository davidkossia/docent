"""Command line entry point. More commands land with each milestone."""
from __future__ import annotations

import argparse

from .chunking import STRATEGIES
from .corpus import download
from .embeddings import NAMES


def main() -> int:
    p = argparse.ArgumentParser(prog="docent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download the corpus PDFs")
    r = sub.add_parser("eval-retrieval", help="recall@5 across chunking x embedding setups")
    r.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    r.add_argument("--embedders", nargs="+", default=["local-bge-small"], choices=NAMES)
    a = sub.add_parser("eval-answers", help="grounded-answer rate and refusals (calls Claude, costs money)")
    a.add_argument("--strategies", nargs="+", default=["section-800"], choices=list(STRATEGIES))
    a.add_argument("--embedders", nargs="+", default=["local-bge-small"], choices=NAMES)
    a.add_argument("--yes", action="store_true", help="spend the money")
    sv = sub.add_parser("serve", help="run the local API on port 8787")
    sv.add_argument("--fake", nargs="?", const="auto", metavar="FIXTURE",
                    help="answer from web/src/fixtures without calling Claude")
    sv.add_argument("--port", type=int, default=8787)
    ix = sub.add_parser("index", help="embed chunks and upsert them into Pinecone")
    ix.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ix.add_argument("--embedder", required=True, choices=NAMES)
    ix.add_argument("--create", action="store_true", help="create the Pinecone index first")
    ix.add_argument("--dry-run", action="store_true", help="print the plan; no network calls")
    args = p.parse_args()
    if args.cmd == "download":
        download()
    elif args.cmd == "eval-retrieval":
        from .evalrun import run_retrieval

        run_retrieval(args.strategies, args.embedders)
    elif args.cmd == "index":
        from .chunking import build_chunks
        from .ingest import load_sections
        from .pinecone_store import DIMENSIONS, INDEX_NAME, index_docs, namespace_for

        docs = build_chunks(STRATEGIES[args.strategy], load_sections())
        ns = namespace_for(args.strategy, args.embedder)
        if args.dry_run:
            print(f"{len(docs)} chunks, {DIMENSIONS[args.embedder]} dimensions, index {INDEX_NAME}, namespace {ns}")
        else:
            from .embeddings import get_embedder

            index_docs(docs, get_embedder(args.embedder), args.strategy, args.embedder, args.create)
    elif args.cmd == "serve":
        from .server import serve

        serve(args.fake, args.port)
    elif args.cmd == "eval-answers":
        from .evalrun import run_answers

        run_answers(args.strategies, args.embedders, args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
