.PHONY: help install install-dev clean test test-cov lint format type-check docs serve-docs build setup-metal benchmark profile

# Default target
help:
	@echo "Shvayambhu LLM Development Commands"
	@echo "===================================="
	@echo "Setup:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make setup-metal   - Set up Metal development environment"
	@echo ""
	@echo "Development:"
	@echo "  make format        - Format code with black and isort"
	@echo "  make lint          - Run linting checks"
	@echo "  make type-check    - Run type checking with mypy"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Build documentation"
	@echo "  make serve-docs    - Serve documentation locally"
	@echo ""
	@echo "Performance:"
	@echo "  make benchmark     - Run performance benchmarks"
	@echo "  make profile       - Profile code execution"
	@echo "  make metal-test    - Run comprehensive Metal performance tests"
	@echo "  make metal-test-quick - Run quick Metal performance tests"
	@echo "  make verify-metal  - Verify Metal setup and capabilities"
	@echo "  make memory-benchmark - Run memory benchmarking suite"
	@echo ""
	@echo "Build:"
	@echo "  make build         - Build distribution packages"

# Installation targets
install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# Metal setup
setup-metal:
	@echo "Setting up Metal development environment..."
	@bash scripts/setup_environment.sh

# Code quality targets
format:
	black .
	isort .

lint:
	flake8 . --count --statistics
	pylint shvayambhu

type-check:
	mypy shvayambhu

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=shvayambhu --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/performance/ -v --benchmark-only

# Documentation targets
docs:
	cd docs && sphinx-build -b html . _build/html

serve-docs:
	cd docs && python -m http.server --directory _build/html 8000

# Build targets
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete

# Performance targets
benchmark:
	python scripts/benchmark_model.py

profile:
	py-spy record -o profile.svg -- python examples/basic_inference.py

# Metal performance testing
metal-test:
	python scripts/metal_performance_test.py

metal-test-quick:
	python scripts/metal_performance_test.py --quick

verify-metal:
	python scripts/verify_metal_setup.py

memory-benchmark:
	python -c "from utils.hardware.benchmarking import run_quick_benchmark; run_quick_benchmark()"

# Development workflow targets
dev-setup: install-dev setup-metal
	@echo "Development environment ready!"

check: format lint type-check test
	@echo "All checks passed!"

# Training targets
train-bootstrap:
	python scripts/train_model.py --phase bootstrap

train-synthetic:
	python scripts/train_model.py --phase synthetic

train-full:
	python scripts/train_model.py --phase full

# Model management
download-seed-data:
	python scripts/download_seed_data.py

# Git workflow
commit-check: format lint type-check test
	@echo "Ready to commit!"

# Virtual environment management
venv:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Apple Silicon specific
check-metal:
	@python -c "import platform; print('Architecture:', platform.machine())"
	@python -c "import torch; print('PyTorch Metal available:', torch.backends.mps.is_available())"

# Development server
dev-server:
	python -m shvayambhu.server --reload

# Database operations
init-db:
	python -m shvayambhu.utils.init_database

# Memory profiling
memory-profile:
	mprof run python examples/basic_inference.py
	mprof plot