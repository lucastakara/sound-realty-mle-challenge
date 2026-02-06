#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "Smoke test: $BASE_URL/health"
curl -fsS "$BASE_URL/health" | cat
echo

echo "Smoke test: $BASE_URL/"
curl -fsS "$BASE_URL/" | cat
echo

echo "Smoke test: predict one sample"
python tests/run_live_unseen_examples.py --api-url "$BASE_URL" --sample-size 1