# Single entry point for every workflow. Anything a contributor needs to run
# should be a target here, so the README never drifts from reality.

COMPOSE := docker compose
BACKEND := backend
VENV    := $(BACKEND)/.venv
PY312   := /usr/bin/python3.12

# Files a container generates into the bind mount must belong to the person who
# ran make, not to the image's non-root uid, or the migration alembic just
# wrote is uneditable on the host.
HOST_USER := $(shell id -u):$(shell id -g)

# Host-side tooling reaches Postgres on the published port, not by service name.
LOCAL_DB := postgresql+asyncpg://$${POSTGRES_USER:-uke}:$${POSTGRES_PASSWORD:-uke_dev_password}@localhost:$${POSTGRES_PORT:-5432}/$${POSTGRES_DB:-uke}

.DEFAULT_GOAL := help

.PHONY: help env up down restart build logs ps sh psql migrate migrate-local revision \
        downgrade test test-slow test-live test-local lint fmt typecheck check web-typecheck web-build \
        venv reset

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if it does not exist
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

up: env ## Build and start db, api and web in the background
	$(COMPOSE) up --build -d
	@echo
	@echo "api  	 http://localhost:8000/docs"
	@echo "web  	 http://localhost:3000"

down: ## Stop containers, keeping data volumes
	$(COMPOSE) down

restart: ## Restart the api service
	$(COMPOSE) restart api

build: env ## Rebuild images without starting them
	$(COMPOSE) build

logs: ## Follow logs for all services
	$(COMPOSE) logs -f

ps: ## Show container status
	$(COMPOSE) ps

sh: ## Open a shell in the api container
	$(COMPOSE) run --rm api bash

psql: ## Open psql against the running database
	$(COMPOSE) exec db psql -U $${POSTGRES_USER:-uke} -d $${POSTGRES_DB:-uke}

migrate: env ## Apply all migrations
	$(COMPOSE) run --rm api alembic upgrade head

migrate-local: ## Apply all migrations from the host venv against localhost
	cd $(BACKEND) && DATABASE_URL="$(LOCAL_DB)" .venv/bin/alembic upgrade head

revision: env ## Autogenerate a migration: make revision MSG="add courses"
	@test -n "$(MSG)" || { echo 'MSG is required, e.g. make revision MSG="add courses"'; exit 1; }
	$(COMPOSE) run --rm --user $(HOST_USER) api alembic revision --autogenerate -m "$(MSG)"
	@echo
	@echo "Read the generated file before applying it. Autogenerate does not detect"
	@echo "renames (it emits a drop plus an add), CREATE EXTENSION, or data migrations."

downgrade: env ## Roll back one revision: make downgrade [REV=-1]
	$(COMPOSE) run --rm api alembic downgrade $(or $(REV),-1)

test: env ## Run the backend suite in the container (integration tests included)
	$(COMPOSE) run --rm api pytest

test-slow: env ## Run the tests that load the real embedding model
	$(COMPOSE) run --rm api pytest -m slow

test-live: env ## Run the tests that call the real Claude API (spends money; needs ANTHROPIC_API_KEY)
	$(COMPOSE) run --rm api pytest -m live

lint: env ## ruff check + mypy
	$(COMPOSE) run --rm api sh -c "ruff check . && mypy app tests"

fmt: env ## Format and autofix the backend
	$(COMPOSE) run --rm api sh -c "ruff format . && ruff check --fix ."

typecheck: env ## mypy only
	$(COMPOSE) run --rm api mypy app tests

web-typecheck: env ## tsc --noEmit in the web container
	$(COMPOSE) run --rm web npm run typecheck

web-build: env ## Production build of the frontend
	$(COMPOSE) run --rm web npm run build

check: lint test web-typecheck ## Everything CI would run

# --- host-side tooling (editor integration, faster feedback) ----------------

venv: ## Create backend/.venv from python3.12 and install dev deps
	$(PY312) -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --upgrade pip
	$(VENV)/bin/pip install --quiet -r $(BACKEND)/requirements-dev.txt
	@echo "activate with: source $(VENV)/bin/activate"

test-local: ## Run the suite from the host venv against localhost Postgres
	cd $(BACKEND) && TEST_DATABASE_URL="$(LOCAL_DB)" .venv/bin/pytest

reset: ## DESTRUCTIVE: stop everything and delete the database, storage and model volumes
	@echo "This deletes the uni-knowledge-engine db_data, api_storage and hf_cache volumes."
	@echo "The next start re-downloads the embedding model."
	@read -p "Type 'yes' to continue: " ans && [ "$$ans" = "yes" ] || { echo "aborted"; exit 1; }
	$(COMPOSE) down -v
