#!/usr/bin/env bash
# Regenerate infra/scripts/requirements-lambda.lock from backend/requirements-lambda.txt:
# every package pinned with hashes, for Linux arm64 and Python 3.12. Run after changing the
# requirements, then commit the lock. build_lambda.sh installs only from the lock.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV="$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")"
cd "$ROOT"
"$UV" pip compile backend/requirements-lambda.txt \
  --overrides infra/scripts/lambda-overrides.txt \
  --python-version 3.12 --python-platform aarch64-manylinux_2_28 --only-binary :all: \
  --generate-hashes --quiet \
  --custom-compile-command "infra/scripts/lock_lambda.sh" \
  -o infra/scripts/requirements-lambda.lock
