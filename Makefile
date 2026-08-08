.PHONY: install test lint typecheck build check

install:
	python -m pip install -e '.[dev]'

test:
	pytest --cov=microalpha --cov-report=term-missing --cov-fail-under=85

lint:
	ruff check .

typecheck:
	mypy src

build:
	python -m build

check: lint typecheck test build
