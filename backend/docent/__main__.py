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
    args = p.parse_args()
    if args.cmd == "download":
        download()
    elif args.cmd == "eval-retrieval":
        from .evalrun import run_retrieval

        run_retrieval(args.strategies, args.embedders)
    elif args.cmd == "serve":
        from .server import serve

        serve(args.fake, args.port)
    elif args.cmd == "eval-answers":
        from .evalrun import run_answers

        run_answers(args.strategies, args.embedders, args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
