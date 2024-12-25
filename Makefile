PHONY: run install check clean runner
.DEFAULT_GOAL=runner

run:
	cd src; poetry run python runner.py

install: pyproject.toml
	poetry install

check:
	poetry run flake8 src/

clean:
	rm -rf $$(find . -type d -name __pycache__)

runner: install check run clean

