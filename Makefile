.PHONY: check lint format typecheck test build publish clean

check: lint typecheck test

lint:
	uv run ruff check src tests eval
	uv run ruff format --check src tests eval

format:
	uv run ruff format src tests eval
	uv run ruff check --fix src tests eval

typecheck:
	uv run mypy --strict src/ditto

test:
	uv run pytest --cov=src/ditto --cov-report=term --cov-fail-under=80

build:
	rm -rf dist
	uv build

# Requires a PyPI token in UV_PUBLISH_TOKEN; see CONTRIBUTING.md "Releasing".
publish: build
	uv publish

clean:
	rm -rf dist build .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
