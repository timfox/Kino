# LTX-Trainer Makefile

# Common development tasks for LTX-2 training

# Use virtual environment Python from parent directory (kino/.venv)
# Note: ltx-trainer is at kino/packages/ltx-trainer, so venv is at ../../.venv
PYTHON = ../../.venv/bin/python
TESTS_DIR = tests

.PHONY: test test-all test-controllight lint format check clean

# Run tests using virtual environment Python
test:
	$(PYTHON) -m pytest $(TESTS_DIR)/test_controllight.py -v

# Run all ltx-trainer tests (not the entire repo - see test-all comment)
test-all:
	$(PYTHON) -m pytest tests/ -v
# Note: test-all runs tests in the current directory (tests/) only.
# Running pytest from the repo root would collect tools/vllm_chat_qt.py
# which has a sys.exit(1) at module level and breaks test collection.
# To run all repo tests, use: ../../.venv/bin/python -m pytest tests/ -v
# Run ControlLight-specific tests
test-controllight:
	$(PYTHON) -m pytest $(TESTS_DIR)/test_controllight.py -v

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest $(TESTS_DIR)/ --cov=ltx_trainer --cov-report=term-missing -v

# Run linter
lint:
	$(PYTHON) -m ruff check src/ tests/

# Format code
format:
	$(PYTHON) -m ruff format src/ tests/

# Run all checks (lint + format check + tests)
check: lint format test-all

# Clean up build artifacts and cache
clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov __pycache__ src/**/*.pyc tests/**/*.pyc
	echo "Cleaned build artifacts and cache"

# Help target
help:
	@echo "LTX-Trainer Development Commands"
	@echo "================================"
	@echo "make test       - Run ControlLight tests"
	@echo "make test-all   - Run all tests"
	@echo "make test-cov   - Run tests with coverage report"
	@echo "make lint       - Run ruff linter"
	@echo "make format     - Format code with ruff"
	@echo "make check      - Run all checks (lint + format + tests)"
	@echo "make clean      - Clean up build artifacts and cache"
	@echo "make help       - Show this help message"

