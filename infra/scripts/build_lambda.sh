#!/usr/bin/env bash
# Build the Lambda bundle in backend/build/lambda: Linux arm64 wheels for the query path plus the
# docent package. Fails if the unzipped bundle passes 200 MB (Lambda's limit is 250 MB).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/backend/build/lambda"
MAX_MB=200
UV="$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")"
TARGET=(--target "$OUT" --python-version 3.12 --python-platform aarch64-manylinux_2_28)  # Lambda python3.12 runs Amazon Linux 2023, glibc 2.34

rm -rf "$OUT"
mkdir -p "$OUT"
"$UV" pip install --quiet "${TARGET[@]}" --only-binary :all: \
  --overrides "$ROOT/infra/scripts/lambda-overrides.txt" \
  -r "$ROOT/backend/requirements-lambda.txt"
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
