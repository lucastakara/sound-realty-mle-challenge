SHELL := /bin/bash

# -------------------------
# Blue/Green (current)
# -------------------------
COMPOSE_FILE ?= deploy/docker-compose.bluegreen.yml
COMPOSE      ?= docker compose -f $(COMPOSE_FILE)

# -------------------------
# Single API (new)
# -------------------------
SINGLE_COMPOSE_FILE ?= deploy/docker-compose.single.yml
SINGLE_COMPOSE      ?= docker compose -f $(SINGLE_COMPOSE_FILE)

.PHONY: help \
        up down restart ps logs bluegreen \
        single-up single-down single-restart single-ps single-logs \
        unit integration test live

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
	@echo "Single API:"
	@echo "  make single-up      -> start single api only"
	@echo "  make single-down    -> stop single api"
	@echo "  make single-restart -> down + up (single)"
	@echo "  make single-ps      -> status (single)"
	@echo "  make single-logs    -> logs (single)"
	@echo ""
	@echo "Tests:"
	@echo "  make unit        -> run unit tests"
	@echo "  make integration -> run integration tests"
	@echo "  make test        -> run all tests"
	@echo "  make live        -> run live unseen examples against single api"
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
# Single API targets
# -------------------------
single-up:
	@$(SINGLE_COMPOSE) up -d --build --remove-orphans

single-down:
	@$(SINGLE_COMPOSE) down --remove-orphans

single-restart: single-down single-up

single-ps:
	@$(SINGLE_COMPOSE) ps

single-logs:
	@$(SINGLE_COMPOSE) logs -f --tail=200

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
# Live run against SINGLE API
# -------------------------
live: single-up
	@python -m tests.integration.run_live_unseen_examples \
		--api-url http://localhost:8000 \
		--csv-path data/future_unseen_examples.csv \
		--sample-size 20 \
		--timeout 10