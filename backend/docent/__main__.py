"""Command line entry point. More commands land with each milestone."""
from __future__ import annotations

import argparse

from .chunking import STRATEGIES
from .corpus import download
from .embeddings import NAMES
from .server import FIXTURE_NAMES


def main() -> int:
    p = argparse.ArgumentParser(prog="docent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download the corpus PDFs")
    v = sub.add_parser("eval-validate", help="check the question file against the eval spec (exit 1 on errors)")
    v.add_argument("--questions", default=None, help="question file (default: eval/questions.jsonl)")
    v.add_argument("--no-counts", action="store_true", help="skip the 144-question count rules")
    r = sub.add_parser("eval-retrieval", help="evidence recall and MRR across chunking x embedding setups")
    r.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    r.add_argument("--embedders", nargs="+", default=["local-bge-small"], choices=NAMES)
    r.add_argument("--questions", default=None, help="question file (default: eval/questions.jsonl)")
    r.add_argument("--out", default=None, help="results file (default: eval/results/retrieval.json)")
    sub.add_parser("report", help="rewrite the eval block in README.md from the results files")
    a = sub.add_parser("eval-answers", help="grounded-answer rate and refusals (calls Claude, costs money)")
    a.add_argument("--strategies", nargs="+", default=["section-800"], choices=list(STRATEGIES))
    a.add_argument("--embedders", nargs="+", default=["local-bge-small"], choices=NAMES)
    a.add_argument("--yes", action="store_true", help="spend the money")
    sv = sub.add_parser("serve", help="run the local API on port 8787")
    sv.add_argument("--fake", nargs="?", const="auto", choices=["auto", *FIXTURE_NAMES], metavar="FIXTURE",
                    help="answer from web/src/fixtures without calling Claude")
    sv.add_argument("--port", type=int, default=8787)
    ix = sub.add_parser("index", help="embed chunks and upsert them into Pinecone")
    ix.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ix.add_argument("--embedder", required=True, choices=NAMES)
    ix.add_argument("--create", action="store_true", help="create the Pinecone index first")
    ix.add_argument("--yes", action="store_true", help="embed and upsert (default: print the plan only)")
    ix.add_argument("--dry-run", action="store_true", help="print the plan only, even with --yes")
    args = p.parse_args()
    if args.cmd == "download":
        download()
    elif args.cmd == "eval-validate":
        from .evalrun import QUESTIONS_PATH, validate_questions

        errors = validate_questions(args.questions or QUESTIONS_PATH, check_counts=not args.no_counts)
        for e in errors:
            print(e)
        print(f"{len(errors)} error(s)")
        return 1 if errors else 0
    elif args.cmd == "eval-retrieval":
        from .evalrun import QUESTIONS_PATH, run_retrieval

        run_retrieval(args.strategies, args.embedders, args.questions or QUESTIONS_PATH, args.out)
    elif args.cmd == "report":
        from .evalrun import write_report

        write_report()
    elif args.cmd == "index":
        from .chunking import build_chunks
        from .embeddings import get_embedder
        from .ingest import load_sections
        from .pinecone_store import run_index

        docs = build_chunks(STRATEGIES[args.strategy], load_sections())
        run_index(docs, args.strategy, args.embedder, args.create, args.yes and not args.dry_run, get_embedder)
    elif args.cmd == "serve":
        from .server import serve

        serve(args.fake, args.port)
    elif args.cmd == "eval-answers":
        from .evalrun import run_answers

        run_answers(args.strategies, args.embedders, args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
