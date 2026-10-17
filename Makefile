.PHONY: run dev test lint clean docker-build docker-up docker-down

# Start production server
run:
	uvicorn src.server.app:create_app --factory --host 0.0.0.0 --port 8000

# Start dev server with auto-reload
dev:
	uvicorn src.server.app:create_app --factory --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest -v --cov=src --cov-report=term-missing

# Run tests with HTML coverage report
test-html:
	pytest -v --cov=src --cov-report=html

# Lint code
lint:
	ruff check src/ tests/

# Format code
format:
	ruff format src/ tests/

# Type check
typecheck:
	mypy src/

# Clean artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info

# Docker
docker-build:
	docker build -t life-agent:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Install (including dev deps)
install:
	pip install -e ".[dev]"

# Run all checks
check: lint test
	@echo "All checks passed!"
