.PHONY: run run-prod install test test-cov clean

run:
	uv run uvicorn src.quorum.oai_proxy:app --host 0.0.0.0 --port 8001 --reload --reload-include "*.yaml"

run-prod:
	uv run uvicorn src.quorum.oai_proxy:app --host 0.0.0.0 --port 8000

install:
	uv sync --all-extras --dev

test: install
	uv run pytest

test-cov: install
	uv run pytest --cov=quorum --cov-report=html --cov-report=term-missing

clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
