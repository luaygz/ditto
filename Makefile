.PHONY: check lint format typecheck test test-spawn docs report build install-check publish clean

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

# The suite with spawned worker processes (the macOS default) instead of fork.
test-spawn:
	DITTO_MP_START=spawn uv run pytest -q

# Regenerate docs/spec.schema.json and docs/spec-format.md from the Pydantic spec models.
docs:
	uv run python -m ditto.specgen.schema_export

# Charts into docs/img/ from the latest eval/results/.
report:
	uv run python eval/report.py

build:
	rm -rf dist
	uv build

# Install the freshly built wheel (with the synth extra) into a clean venv and smoke-test it.
install-check: build
	rm -rf /tmp/ditto-clean
	python3 -m venv /tmp/ditto-clean
	/tmp/ditto-clean/bin/pip install --quiet "$$(ls dist/*.whl)[synth]"
	/tmp/ditto-clean/bin/ditto --help
	cd /tmp && /tmp/ditto-clean/bin/python -c "from ditto.specgen import load_prompt; from ditto.llm import load_prices; print(load_prompt('specgen').id, len(load_prices().models), 'priced models')"

# Requires a PyPI token in UV_PUBLISH_TOKEN; see CONTRIBUTING.md "Releasing".
publish: build
	uv publish

clean:
	rm -rf dist build .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
