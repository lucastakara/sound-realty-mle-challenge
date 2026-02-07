SHELL := /bin/bash

# -------------------------
# Blue/Green (current)
# -------------------------
COMPOSE_FILE ?= deploy/docker-compose.bluegreen.yml
COMPOSE      ?= docker compose -f $(COMPOSE_FILE)

.PHONY: help \
        up down restart ps logs bluegreen \
        unit integration test live health

help:
	@echo ""
	@echo "Blue/Green:"
	@echo "  make up        -> start blue, green, nginx"
	@echo "  make down      -> stop blue/green stack"
	@echo "  make restart   -> down + up (blue/green)"
	@echo "  make ps        -> show container status (blue/green)"
	@echo "  make logs      -> follow logs (blue/green)"
	@echo "  make bluegreen -> run blue/green integration test"
	@echo ""
	@echo "Tests:"
	@echo "  make unit        -> run unit tests"
	@echo "  make integration -> run integration tests"
	@echo "  make test        -> run all tests"
	@echo ""
	@echo "Live:"
	@echo "  make live     -> run live unseen examples against BLUE/GREEN (nginx :8000)"
	@echo "  make health   -> check /health on nginx (:8000)"
	@echo ""

# -------------------------
# Blue/Green targets
# -------------------------
up:
	@$(COMPOSE) up -d --build --remove-orphans

down:
	@$(COMPOSE) down --remove-orphans

restart: down up

ps:
	@$(COMPOSE) ps

logs:
	@$(COMPOSE) logs -f --tail=200

bluegreen:
	@pytest -q -s tests/integration/test_blue_green_deployment.py

# -------------------------
# Test shortcuts
# -------------------------
unit:
	@pytest -q tests/unit

integration:
	@pytest -q -s tests/integration

test:
	@pytest -q

# -------------------------
# Live run against BLUE/GREEN via NGINX
# -------------------------
live: up
	@python tests/integration/test_run_live_unseen_examples.py \
		--api-url http://localhost:8000 \
		--csv-path data/future_unseen_examples.csv \
		--sample-size 20 \
		--timeout 10 \
		--require-served-by

# ---------------
# Test API health
# ----------------
health:
	curl -fsS http://localhost:8000/health | cat