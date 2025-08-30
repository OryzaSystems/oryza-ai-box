# ==========================================
# AI BOX - Makefile
# Easy commands for development and deployment
# ==========================================

.PHONY: help install test lint format build deploy clean

# Default target
.DEFAULT_GOAL := help

# Colors
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

# Configuration
PYTHON := python3
PIP := pip3
VENV := venv
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# ==========================================
# Help
# ==========================================
help: ## 📚 Show this help message
	@echo "$(BLUE)🤖 AI Box - Development Commands$(NC)"
	@echo "=================================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install          # Setup development environment"
	@echo "  make test            # Run all tests"
	@echo "  make build           # Build Docker images"
	@echo "  make deploy-pi5      # Deploy to Raspberry Pi 5"

# ==========================================
# Development Environment
# ==========================================
install: ## 🔧 Install development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip setuptools wheel
	./$(VENV)/bin/pip install -e .[dev]
	@echo "$(GREEN)✅ Development environment ready!$(NC)"
	@echo "$(YELLOW)Activate with: source $(VENV)/bin/activate$(NC)"

install-gpu: ## 🎮 Install with GPU support
	@echo "$(BLUE)Installing with GPU support...$(NC)"
	./$(VENV)/bin/pip install -e .[gpu]
	@echo "$(GREEN)✅ GPU support installed!$(NC)"

install-edge: ## 📱 Install edge device dependencies
	@echo "$(BLUE)Installing edge dependencies...$(NC)"
	./$(VENV)/bin/pip install -e .[edge]
	@echo "$(GREEN)✅ Edge dependencies installed!$(NC)"

# ==========================================
# Testing & Quality
# ==========================================
test: ## 🧪 Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	./$(VENV)/bin/pytest --cov=oryza_ai_box --cov-report=html --cov-report=term

test-fast: ## ⚡ Run fast tests only
	@echo "$(BLUE)Running fast tests...$(NC)"
	./$(VENV)/bin/pytest -m "not slow" --tb=short

test-gpu: ## 🎮 Run GPU tests
	@echo "$(BLUE)Running GPU tests...$(NC)"
	./$(VENV)/bin/pytest -m gpu

test-edge: ## 📱 Run edge device tests
	@echo "$(BLUE)Running edge tests...$(NC)"
	./$(VENV)/bin/pytest -m edge

lint: ## 🔍 Run code linting
	@echo "$(BLUE)Running linters...$(NC)"
	./$(VENV)/bin/flake8 .
	./$(VENV)/bin/mypy .

format: ## 🎨 Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	./$(VENV)/bin/black .
	./$(VENV)/bin/isort .

check: lint test ## ✅ Run all checks (lint + test)

# ==========================================
# Environment Testing
# ==========================================
test-env: ## 🧪 Test development environment
	@echo "$(BLUE)Testing environment...$(NC)"
	./$(VENV)/bin/python tools/test_environment.py

benchmark: ## 📊 Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	./$(VENV)/bin/python tools/benchmark_models.py

# ==========================================
# Docker Operations
# ==========================================
build: ## 🐳 Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

build-api: ## 🚀 Build API Gateway image
	@echo "$(BLUE)Building API Gateway...$(NC)"
	$(DOCKER) build -f deployment/docker/Dockerfile.api-gateway -t aibox-api-gateway .

build-model: ## 🧠 Build Model Server image
	@echo "$(BLUE)Building Model Server...$(NC)"
	$(DOCKER) build -f deployment/docker/Dockerfile.model-server -t aibox-model-server .

build-data: ## 💾 Build Data Manager image
	@echo "$(BLUE)Building Data Manager...$(NC)"
	$(DOCKER) build -f deployment/docker/Dockerfile.data-manager -t aibox-data-manager .

# ==========================================
# Local Development
# ==========================================
dev: ## 🔧 Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up -d
	@echo "$(GREEN)✅ Development environment started!$(NC)"
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  • API Gateway: http://localhost:8000"
	@echo "  • Jupyter: http://localhost:8888 (token: aibox123)"
	@echo "  • PgAdmin: http://localhost:5050"
	@echo "  • Grafana: http://localhost:3000"

dev-logs: ## 📋 Show development logs
	$(DOCKER_COMPOSE) logs -f

dev-stop: ## ⏹️ Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	$(DOCKER_COMPOSE) --profile dev down

# ==========================================
# Production Operations
# ==========================================
prod: ## 🚀 Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Production environment started!$(NC)"

prod-logs: ## 📋 Show production logs
	$(DOCKER_COMPOSE) logs -f

prod-stop: ## ⏹️ Stop production environment
	@echo "$(BLUE)Stopping production environment...$(NC)"
	$(DOCKER_COMPOSE) down

# ==========================================
# Deployment Commands
# ==========================================
deploy-pi5: ## 🍓 Deploy to Raspberry Pi 5
	@echo "$(BLUE)Deploying to Raspberry Pi 5...$(NC)"
	@read -p "Enter Pi 5 IP address: " PI5_IP; \
	./deployment/scripts/deploy.sh -e production -p raspberry-pi-5 -h $$PI5_IP

deploy-rock5: ## 🪨 Deploy to Radxa Rock 5 ITX
	@echo "$(BLUE)Deploying to Radxa Rock 5 ITX...$(NC)"
	@read -p "Enter Rock 5 IP address: " ROCK5_IP; \
	./deployment/scripts/deploy.sh -e production -p radxa-rock-5 -h $$ROCK5_IP

deploy-jetson: ## 🚀 Deploy to Jetson Nano
	@echo "$(BLUE)Deploying to Jetson Nano...$(NC)"
	@read -p "Enter Jetson IP address: " JETSON_IP; \
	./deployment/scripts/deploy.sh -e production -p jetson-nano -h $$JETSON_IP

deploy-core-i5: ## 💻 Deploy to Core i5 machine
	@echo "$(BLUE)Deploying to Core i5 machine...$(NC)"
	@read -p "Enter Core i5 IP address: " CORE_I5_IP; \
	./deployment/scripts/deploy.sh -e production -p core-i5 -h $$CORE_I5_IP

# ==========================================
# Database Operations
# ==========================================
db-init: ## 🗄️ Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(DOCKER_COMPOSE) exec postgres psql -U aibox -d aibox -f /docker-entrypoint-initdb.d/init.sql

db-backup: ## 💾 Backup database
	@echo "$(BLUE)Creating database backup...$(NC)"
	$(DOCKER_COMPOSE) exec postgres pg_dump -U aibox aibox > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Database backup created!$(NC)"

db-restore: ## 🔄 Restore database from backup
	@echo "$(BLUE)Restoring database...$(NC)"
	@read -p "Enter backup file path: " BACKUP_FILE; \
	$(DOCKER_COMPOSE) exec -T postgres psql -U aibox -d aibox < $$BACKUP_FILE

# ==========================================
# Monitoring & Logs
# ==========================================
logs: ## 📋 Show all logs
	$(DOCKER_COMPOSE) logs -f

logs-api: ## 📋 Show API Gateway logs
	$(DOCKER_COMPOSE) logs -f api-gateway

logs-model: ## 📋 Show Model Server logs
	$(DOCKER_COMPOSE) logs -f model-server

logs-data: ## 📋 Show Data Manager logs
	$(DOCKER_COMPOSE) logs -f data-manager

status: ## 📊 Show service status
	@echo "$(BLUE)Service Status:$(NC)"
	$(DOCKER_COMPOSE) ps

health: ## 🏥 Check service health
	@echo "$(BLUE)Health Check:$(NC)"
	@curl -s http://localhost:8000/health | jq . || echo "API Gateway: $(RED)DOWN$(NC)"
	@curl -s http://localhost:8001/health | jq . || echo "Model Server: $(RED)DOWN$(NC)"
	@curl -s http://localhost:8002/health | jq . || echo "Data Manager: $(RED)DOWN$(NC)"

# ==========================================
# Cleanup Operations
# ==========================================
clean: ## 🧹 Clean up development environment
	@echo "$(BLUE)Cleaning up...$(NC)"
	$(DOCKER_COMPOSE) down -v
	$(DOCKER) system prune -f
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Cleanup completed!$(NC)"

clean-docker: ## 🐳 Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER) system prune -af
	$(DOCKER) volume prune -f

clean-logs: ## 📋 Clean log files
	@echo "$(BLUE)Cleaning log files...$(NC)"
	rm -rf logs/*
	mkdir -p logs

# ==========================================
# Git Operations
# ==========================================
commit: format lint ## 📝 Format, lint and prepare for commit
	@echo "$(GREEN)✅ Code is ready for commit!$(NC)"

push: test ## 🚀 Run tests and push to repository
	@echo "$(BLUE)Running tests before push...$(NC)"
	git push

# ==========================================
# Documentation
# ==========================================
docs: ## 📚 Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	./$(VENV)/bin/sphinx-build -b html docs docs/_build/html

docs-serve: ## 🌐 Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs/_build/html && python -m http.server 8080

# ==========================================
# Release Operations
# ==========================================
version: ## 🏷️ Show current version
	@./$(VENV)/bin/python -c "import oryza_ai_box; print(oryza_ai_box.__version__)"

release: ## 📦 Create a new release
	@echo "$(BLUE)Creating release...$(NC)"
	@read -p "Enter version (e.g., 1.0.0): " VERSION; \
	git tag -a v$$VERSION -m "Release v$$VERSION"; \
	git push origin v$$VERSION

# ==========================================
# Quick Start
# ==========================================
quickstart: install test-env dev ## 🚀 Complete quickstart setup
	@echo "$(GREEN)🎉 AI Box is ready for development!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Activate virtual environment: source $(VENV)/bin/activate"
	@echo "  2. Open Jupyter: http://localhost:8888 (token: aibox123)"
	@echo "  3. Check API: http://localhost:8000/docs"
	@echo "  4. Monitor with Grafana: http://localhost:3000"
