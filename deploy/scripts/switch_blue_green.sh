#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
if [[ "$TARGET" != "blue" && "$TARGET" != "green" ]]; then
  echo "Usage: $0 blue|green"
  exit 1
fi

# This is the file NGINX actually includes (mounted into /etc/nginx/conf.d)
UPSTREAM_CONF="deploy/nginx/conf.d/upstream.conf"

if [[ "$TARGET" == "blue" ]]; then
  printf 'upstream api_upstream { server api_blue:8000; }\n' > "${UPSTREAM_CONF}"
else
  printf 'upstream api_upstream { server api_green:8000; }\n' > "${UPSTREAM_CONF}"
fi

# Reload NGINX (no container restart required)
NGINX_CID="$(docker compose -f deploy/docker-compose.bluegreen.yml ps -q nginx)"
docker exec -t "${NGINX_CID}" nginx -t >/dev/null
docker exec -t "${NGINX_CID}" nginx -s reload >/dev/null

echo "Switched traffic to: ${TARGET}"
cat "${UPSTREAM_CONF}"