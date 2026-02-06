.PHONY: help setup setup-dev engine cli lint format test clean

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest

ifeq ($(OS),Windows_NT)
	PYTHON := .venv\Scripts\python
	PIP := .venv\Scripts\pip
	RUFF := .venv\Scripts\ruff
	PYTEST := .venv\Scripts\pytest
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create Python virtual environment
	python3 -m venv .venv
	$(PIP) install --upgrade pip

setup: venv ## Install production dependencies
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete. Activate with: source .venv/bin/activate"

setup-dev: venv ## Install dev dependencies
	$(PIP) install -r requirements-dev.txt
	@echo ""
	@echo "Dev setup complete. Activate with: source .venv/bin/activate"

setup-desktop: ## Install desktop app dependencies
	cd apps/desktop && npm install

engine: ## Run the engine server
	$(PYTHON) -m engine.main

cli: ## Run CLI (use ARGS for extra flags, e.g. make cli ARGS="--model small --lang en")
	$(PYTHON) -m apps.cli.main $(ARGS)

lint: ## Run ruff linter
	$(RUFF) check .

format: ## Run ruff formatter
	$(RUFF) format .
	cd apps/desktop && npx prettier --write "src/**/*.{ts,tsx,json,css}"

test: ## Run pytest
	$(PYTEST) -v

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build htmlcov .coverage

env: ## Create .env from example
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env created from .env.example"; \
	else \
		echo ".env already exists"; \
	fi
