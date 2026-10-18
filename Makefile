# Single entry point for every workflow. Anything a contributor needs to run
# should be a target here, so the README never drifts from reality.

COMPOSE := docker compose
BACKEND := backend
VENV    := $(BACKEND)/.venv
PY312   := /usr/bin/python3.12

.DEFAULT_GOAL := help
.PHONY: help env up down restart build logs ps sh psql migrate revision test test-local \
        lint fmt typecheck check web-typecheck web-build venv reset

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

revision: env ## Autogenerate a migration: make revision MSG="add courses"
	@test -n "$(MSG)" || { echo 'MSG is required, e.g. make revision MSG="add courses"'; exit 1; }
	$(COMPOSE) run --rm api alembic revision --autogenerate -m "$(MSG)"

test: env ## Run the backend suite in the container (integration tests included)
	$(COMPOSE) run --rm api pytest

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
	cd $(BACKEND) && TEST_DATABASE_URL="postgresql+asyncpg://$${POSTGRES_USER:-uke}:$${POSTGRES_PASSWORD:-uke_dev_password}@localhost:$${POSTGRES_PORT:-5432}/$${POSTGRES_DB:-uke}" .venv/bin/pytest

reset: ## DESTRUCTIVE: stop everything and delete the database and storage volumes
	@echo "This deletes the uni-knowledge-engine db_data and api_storage volumes."
	@read -p "Type 'yes' to continue: " ans && [ "$$ans" = "yes" ] || { echo "aborted"; exit 1; }
	$(COMPOSE) down -v
