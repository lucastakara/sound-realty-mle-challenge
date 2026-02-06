#!/usr/bin/env bash
set -euo pipefail

echo "== Containers =="
docker compose -f deploy/docker-compose.bluegreen.yml ps

echo
echo "== Active upstream.conf =="
cat deploy/nginx/upstream.conf

echo
echo "== Health check via NGINX (port 8000) =="
curl -s http://localhost:8000/health || true
echo