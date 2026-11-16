"""Command line entry point. More commands land with each milestone."""
from __future__ import annotations

import sys

from .corpus import download


def main(argv: list[str]) -> int:
    if argv[:1] == ["download"]:
        download()
        return 0
    print("usage: python -m docent download", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
