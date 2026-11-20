#!/usr/bin/env bash
# Build the Lambda bundle in backend/build/lambda: Linux arm64 wheels for the query path, installed
# only from the hash-locked infra/scripts/requirements-lambda.lock, plus the committed docent
# package. Fails if the unzipped bundle passes 200 MB (Lambda's limit is 250 MB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/backend/build/lambda"
LOCK="$ROOT/infra/scripts/requirements-lambda.lock"
MAX_MB=200
UV="$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")"
TARGET=(--target "$OUT" --python-version 3.12 --python-platform aarch64-manylinux_2_28)  # Lambda python3.12 runs Amazon Linux 2023, glibc 2.34

# Ship only committed code: an uncommitted edit in backend/docent would deploy unreviewed.
if [ -n "$(git -C "$ROOT" status --porcelain -- backend/docent)" ]; then
  echo "error: backend/docent has uncommitted changes; commit or stash them first" >&2
  git -C "$ROOT" status --short -- backend/docent >&2
  exit 1
fi

# The lock must pin every direct requirement at the same version.
while read -r pin; do
  if ! grep -qix "${pin} \\\\" "$LOCK"; then
    echo "error: $LOCK does not pin $pin; run infra/scripts/lock_lambda.sh" >&2
    exit 1
  fi
done < <(grep -E '^[A-Za-z0-9]' "$ROOT/backend/requirements-lambda.txt")

rm -rf "$OUT"
mkdir -p "$OUT"
"$UV" pip install --quiet "${TARGET[@]}" --only-binary :all: --require-hashes --no-deps -r "$LOCK"
"$UV" pip install --quiet "${TARGET[@]}" --no-deps "$ROOT/backend"

for banned in openai langchain_openai tiktoken fastembed onnxruntime pypdf boto3; do
  if [ -e "$OUT/$banned" ]; then
    echo "error: $banned is in the bundle; the query path must not need it" >&2
    exit 1
  fi
done
find "$OUT" -type d -name __pycache__ -prune -exec rm -rf {} +

python3 - "$OUT" "$MAX_MB" <<'PY'
import os, sys
out, limit = sys.argv[1], int(sys.argv[2])
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(out) for f in files)
mb = size / 1_000_000
print(f"Lambda bundle: {mb:.1f} MB unzipped (limit {limit} MB) in {out}")
sys.exit(1 if mb > limit else 0)
PY
