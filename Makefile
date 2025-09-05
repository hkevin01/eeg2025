# EEG Foundation Challenge 2025 - Comprehensive Makefile

# Variables
SHELL := /bin/bash
PYTHON := python
PIP := pip
DOCKER_IMAGE := eeg2025
DOCKER_TAG := latest
PROJECT_NAME := eeg2025
DATA_DIR := data
OUTPUTS_DIR := outputs
MODELS_DIR := models
LOGS_DIR := logs

# Colors for output
CYAN := \033[36m
RESET := \033[0m
BOLD := \033[1m

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message with categories
	@echo "$(BOLD)EEG Foundation Challenge 2025 - Available Commands$(RESET)"
	@echo ""
	@echo "$(CYAN)📊 Data Management$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*Data/ {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(CYAN)🧠 Model Training$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*Train/ {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(CYAN)🔬 Evaluation & Testing$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*(Eval|Test)/ {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(CYAN)📦 Submission & Deployment$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*(Submit|Deploy|Docker)/ {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(CYAN)🛠️ Development & Utilities$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*(Dev|Clean|Util|Install|Lint|Format)/ {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Environment Setup
# =============================================================================

.PHONY: install
install: ## Dev - Install dependencies and setup environment
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "✅ Environment setup complete"

.PHONY: install-gpu
install-gpu: ## Dev - Install with CUDA support
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "✅ GPU environment setup complete"

.PHONY: env-create
env-create: ## Dev - Create conda environment
	conda create -n $(PROJECT_NAME) python=3.9 -y
	@echo "✅ Environment created. Activate with: conda activate $(PROJECT_NAME)"

.PHONY: env-export
env-export: ## Dev - Export conda environment
	conda env export > environment.yml
	@echo "✅ Environment exported to environment.yml"

# =============================================================================
# Data Management
# =============================================================================

.PHONY: prepare-data
prepare-data: ## Data - Prepare complete dataset with indexing and splits
	$(PYTHON) scripts/prepare_hbn_bids.py --config configs/data.yaml
	$(PYTHON) scripts/make_splits.py --config configs/data.yaml
	@echo "✅ Data preparation complete"

.PHONY: validate-data
validate-data: ## Data - Validate data integrity and format
	$(PYTHON) -m src.data.validation --config configs/data.yaml
	@echo "✅ Data validation complete"

.PHONY: create-splits
create-splits: ## Data - Create train/val/test splits
	$(PYTHON) scripts/make_splits.py --config configs/data.yaml
	@echo "✅ Data splits created"

.PHONY: data-stats
data-stats: ## Data - Generate dataset statistics
	$(PYTHON) -m src.data.analysis --config configs/data.yaml --output-dir $(OUTPUTS_DIR)/data_stats
	@echo "✅ Dataset statistics generated"

# =============================================================================
# Model Training Pipeline
# =============================================================================

.PHONY: train-ssl
train-ssl: ## Train - Run SSL pretraining with all objectives
	$(PYTHON) -m src.training.train_ssl \
		--config configs/ssl_config.yaml \
		--output-dir $(OUTPUTS_DIR)/ssl_pretraining \
		--log-dir $(LOGS_DIR)/ssl
	@echo "✅ SSL pretraining completed"

.PHONY: train-ssl-quick
train-ssl-quick: ## Train - Quick SSL pretraining (5 epochs for testing)
	$(PYTHON) -m src.training.train_ssl \
		--config configs/ssl_config.yaml \
		--output-dir $(OUTPUTS_DIR)/ssl_quick \
		--epochs 5 \
		--log-dir $(LOGS_DIR)/ssl_quick
	@echo "✅ Quick SSL pretraining completed"

.PHONY: train-cross-task
train-cross-task: ## Train - Cross-task transfer learning
	$(PYTHON) -m src.training.train_cross_task \
		--config configs/cross_task_config.yaml \
		--ssl-checkpoint $(OUTPUTS_DIR)/ssl_pretraining/best_model.pth \
		--output-dir $(OUTPUTS_DIR)/cross_task \
		--log-dir $(LOGS_DIR)/cross_task
	@echo "✅ Cross-task training completed"

.PHONY: train-psych
train-psych: ## Train - Psychopathology training with DANN
	$(PYTHON) -m src.training.train_psych \
		--config configs/psych_config.yaml \
		--pretrained-backbone $(OUTPUTS_DIR)/cross_task/best_backbone.pth \
		--output-dir $(OUTPUTS_DIR)/psychopathology \
		--log-dir $(LOGS_DIR)/psych \
		--dann \
		--irm
	@echo "✅ Psychopathology training completed"

.PHONY: train-psych-baseline
train-psych-baseline: ## Train - Baseline psychopathology without domain adaptation
	$(PYTHON) -m src.training.train_psych \
		--config configs/psych_config.yaml \
		--output-dir $(OUTPUTS_DIR)/psych_baseline \
		--log-dir $(LOGS_DIR)/psych_baseline
	@echo "✅ Baseline psychopathology training completed"

.PHONY: train-full-pipeline
train-full-pipeline: ## Train - Complete SSL → Cross-task → DANN pipeline
	@echo "🚀 Starting full training pipeline..."
	$(MAKE) train-ssl
	$(MAKE) train-cross-task
	$(MAKE) train-psych
	@echo "✅ Full training pipeline completed"

.PHONY: train-quick-pipeline
train-quick-pipeline: ## Train - Quick pipeline for testing (reduced epochs)
	@echo "🚀 Starting quick training pipeline..."
	$(PYTHON) -m src.training.train_ssl --config configs/ssl_config.yaml --epochs 5 --output-dir $(OUTPUTS_DIR)/quick_ssl
	$(PYTHON) -m src.training.train_cross_task --config configs/cross_task_config.yaml --epochs 5 --ssl-checkpoint $(OUTPUTS_DIR)/quick_ssl/best_model.pth --output-dir $(OUTPUTS_DIR)/quick_cross
	$(PYTHON) -m src.training.train_psych --config configs/psych_config.yaml --epochs 5 --pretrained-backbone $(OUTPUTS_DIR)/quick_cross/best_backbone.pth --output-dir $(OUTPUTS_DIR)/quick_psych --dann
	@echo "✅ Quick pipeline completed"

# =============================================================================
# Evaluation & Testing
# =============================================================================

.PHONY: evaluate
evaluate: ## Eval - Run comprehensive model evaluation
	$(PYTHON) -m src.evaluation.evaluate \
		--config configs/eval_config.yaml \
		--model-path $(OUTPUTS_DIR)/psychopathology/best_model.pth \
		--output-dir $(OUTPUTS_DIR)/evaluation
	@echo "✅ Model evaluation completed"

.PHONY: test-models
test-models: ## Test - Run model architecture tests
	$(PYTHON) -m pytest tests/models/ -v
	@echo "✅ Model tests completed"

.PHONY: test-training
test-training: ## Test - Run training pipeline tests
	$(PYTHON) -m pytest tests/training/ -v
	@echo "✅ Training tests completed"

.PHONY: test-unit
test-unit: ## Test - Run all unit tests
	$(PYTHON) -m pytest tests/unit/ -v --cov=src --cov-report=html
	@echo "✅ Unit tests completed"

.PHONY: test-integration
test-integration: ## Test - Run integration tests
	$(PYTHON) -m pytest tests/integration/ -v -x
	@echo "✅ Integration tests completed"

.PHONY: test-all
test-all: ## Test - Run complete test suite
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ All tests completed"

.PHONY: test-gpu
test-gpu: ## Test - Run GPU-specific tests
	$(PYTHON) -m pytest tests/gpu/ -v
	@echo "✅ GPU tests completed"

.PHONY: test-reproducibility
test-reproducibility: ## Test - Verify reproducibility
	$(PYTHON) -c "from src.utils.reproducibility import SeedManager; sm = SeedManager(42); sm.test_reproducibility(); print('✅ Reproducibility test passed')"

# =============================================================================
# Performance Benchmarking
# =============================================================================

.PHONY: benchmark
benchmark: ## Eval - Run comprehensive performance benchmarks
	$(PYTHON) -m src.evaluation.benchmarking \
		--output-dir $(OUTPUTS_DIR)/benchmarks \
		--config-names baseline_cnn,ssl_only,dann_only,full_pipeline
	@echo "✅ Performance benchmarking completed"

.PHONY: benchmark-quick
benchmark-quick: ## Eval - Quick benchmark with reduced configurations
	$(PYTHON) -m src.evaluation.benchmarking \
		--output-dir $(OUTPUTS_DIR)/benchmarks_quick \
		--config-names baseline_cnn,full_pipeline \
		--max-epochs 10 \
		--num-seeds 1
	@echo "✅ Quick benchmarking completed"

.PHONY: profile-memory
profile-memory: ## Eval - Profile memory usage
	$(PYTHON) -c "
	import torch
	from memory_profiler import profile
	from src.models.backbone import TemporalCNN

	@profile
	def test_memory():
		model = TemporalCNN(input_channels=19, num_layers=5)
		x = torch.randn(32, 19, 1000)
		y = model(x)
		return y

	test_memory()
	"

.PHONY: benchmark-inference
benchmark-inference: ## Eval - Benchmark inference speed
	$(PYTHON) -c "
	import time
	import torch
	from src.models.backbone import TemporalCNN

	model = TemporalCNN(input_channels=19, num_layers=5)
	model.eval()
	x = torch.randn(32, 19, 1000)

	# Warmup
	for _ in range(10):
		_ = model(x)

	# Benchmark
	times = []
	for _ in range(100):
		start = time.time()
		with torch.no_grad():
			_ = model(x)
		times.append(time.time() - start)

	avg_time = sum(times) / len(times)
	print(f'Average inference time: {avg_time*1000:.2f}ms')
	print(f'Throughput: {32/avg_time:.1f} samples/sec')
	"

# =============================================================================
# Submission & Packaging
# =============================================================================

.PHONY: generate-submission
generate-submission: ## Submit - Generate challenge submission files
	$(PYTHON) -m src.evaluation.submission \
		--config configs/submission_config.yaml \
		--model-path $(OUTPUTS_DIR)/psychopathology/best_model.pth \
		--output-dir $(OUTPUTS_DIR)/submission
	@echo "✅ Submission files generated"

.PHONY: validate-submission
validate-submission: ## Submit - Validate submission format
	$(PYTHON) -m src.evaluation.submission \
		--validate-only \
		--submission-dir $(OUTPUTS_DIR)/submission
	@echo "✅ Submission validation completed"

.PHONY: package-submission
package-submission: ## Submit - Create final submission archive
	@echo "📦 Creating submission package..."
	mkdir -p $(OUTPUTS_DIR)/final_submission
	tar -czf $(OUTPUTS_DIR)/final_submission/eeg2025-challenge-submission.tar.gz \
		src/ configs/ docs/ requirements.txt README.md \
		$(OUTPUTS_DIR)/submission/ \
		--exclude="__pycache__" --exclude="*.pyc" --exclude=".pytest_cache" \
		--exclude="*.log" --exclude="wandb/" --exclude=".git/"
	@echo "✅ Submission package created: $(OUTPUTS_DIR)/final_submission/eeg2025-challenge-submission.tar.gz"

.PHONY: submission-report
submission-report: ## Submit - Generate comprehensive submission report
	$(PYTHON) -m src.evaluation.submission \
		--generate-report \
		--model-path $(OUTPUTS_DIR)/psychopathology/best_model.pth \
		--output-dir $(OUTPUTS_DIR)/submission_report
	@echo "✅ Submission report generated"

# =============================================================================
# Docker Operations
# =============================================================================

.PHONY: docker-build
docker-build: ## Docker - Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f docker/Dockerfile .
	@echo "✅ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

.PHONY: docker-run
docker-run: ## Docker - Run Docker container interactively
	docker run -it --rm \
		--gpus all \
		-v $(PWD):/workspace \
		-v $(PWD)/$(DATA_DIR):/workspace/$(DATA_DIR) \
		-v $(PWD)/$(OUTPUTS_DIR):/workspace/$(OUTPUTS_DIR) \
		$(DOCKER_IMAGE):$(DOCKER_TAG) /bin/bash

.PHONY: docker-train
docker-train: ## Docker - Run full training pipeline in Docker
	docker run --rm \
		--gpus all \
		-v $(PWD):/workspace \
		-v $(PWD)/$(DATA_DIR):/workspace/$(DATA_DIR) \
		-v $(PWD)/$(OUTPUTS_DIR):/workspace/$(OUTPUTS_DIR) \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		make train-full-pipeline

.PHONY: docker-compose-up
docker-compose-up: ## Docker - Start development environment with Docker Compose
	docker compose -f docker/compose.yml up -d --build
	@echo "✅ Development environment started"

.PHONY: docker-compose-down
docker-compose-down: ## Docker - Stop Docker Compose environment
	docker compose -f docker/compose.yml down
	@echo "✅ Development environment stopped"

.PHONY: docker-logs
docker-logs: ## Docker - Show container logs
	docker compose -f docker/compose.yml logs -f app

# =============================================================================
# Code Quality & Documentation
# =============================================================================

.PHONY: lint
lint: ## Lint - Run code linting with flake8
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "✅ Linting completed"

.PHONY: format
format: ## Format - Auto-format code with black and isort
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatting completed"

.PHONY: format-check
format-check: ## Format - Check code formatting without changes
	black --check --diff src/ tests/
	isort --check-only --diff src/ tests/
	@echo "✅ Format check completed"

.PHONY: type-check
type-check: ## Lint - Run type checking with mypy
	mypy src/ --ignore-missing-imports
	@echo "✅ Type checking completed"

.PHONY: security-scan
security-scan: ## Lint - Run security vulnerability scan
	bandit -r src/ -ll
	safety check
	@echo "✅ Security scan completed"

.PHONY: docs-build
docs-build: ## Dev - Build documentation
	cd docs && make html
	@echo "✅ Documentation built"

.PHONY: docs-serve
docs-serve: ## Dev - Serve documentation locally
	cd docs && python -m http.server 8000 --directory _build/html
	@echo "📖 Documentation served at http://localhost:8000"

.PHONY: docs-clean
docs-clean: ## Dev - Clean documentation build
	cd docs && make clean
	@echo "✅ Documentation cleaned"

# =============================================================================
# Hyperparameter Optimization
# =============================================================================

.PHONY: hyperopt-ssl
hyperopt-ssl: ## Train - SSL hyperparameter optimization
	$(PYTHON) -m src.training.hyperopt \
		--stage ssl \
		--config configs/hyperopt/ssl_search.yaml \
		--output-dir $(OUTPUTS_DIR)/hyperopt_ssl \
		--trials 50

.PHONY: hyperopt-dann
hyperopt-dann: ## Train - DANN hyperparameter optimization
	$(PYTHON) -m src.training.hyperopt \
		--stage dann \
		--config configs/hyperopt/dann_search.yaml \
		--output-dir $(OUTPUTS_DIR)/hyperopt_dann \
		--trials 30

.PHONY: hyperopt-full
hyperopt-full: ## Train - Full pipeline hyperparameter optimization
	$(PYTHON) -m src.training.hyperopt \
		--stage full \
		--config configs/hyperopt/full_search.yaml \
		--output-dir $(OUTPUTS_DIR)/hyperopt_full \
		--trials 20

# =============================================================================
# Utilities & Maintenance
# =============================================================================

.PHONY: clean
clean: ## Util - Clean build artifacts and temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf .mypy_cache/ .tox/
	@echo "✅ Build artifacts cleaned"

.PHONY: clean-outputs
clean-outputs: ## Util - Clean output directories (careful!)
	@echo "⚠️  Warning: This will remove all training outputs"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(OUTPUTS_DIR)/ $(LOGS_DIR)/ wandb/; \
		echo "✅ Output directories cleaned"; \
	else \
		echo "❌ Cancelled"; \
	fi

.PHONY: clean-data
clean-data: ## Util - Clean processed data cache (careful!)
	@echo "⚠️  Warning: This will remove processed data cache"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(DATA_DIR)/processed/ $(DATA_DIR)/.cache/; \
		echo "✅ Data cache cleaned"; \
	else \
		echo "❌ Cancelled"; \
	fi

.PHONY: clean-all
clean-all: clean clean-outputs ## Util - Clean everything except source data
	@echo "✅ Complete cleanup finished"

.PHONY: setup-dirs
setup-dirs: ## Util - Create necessary directories
	mkdir -p $(DATA_DIR)/{raw,processed,splits}
	mkdir -p $(OUTPUTS_DIR)/{ssl_pretraining,cross_task,psychopathology,evaluation,submission}
	mkdir -p $(LOGS_DIR)/{ssl,cross_task,psych}
	mkdir -p $(MODELS_DIR)
	@echo "✅ Directory structure created"

.PHONY: check-gpu
check-gpu: ## Util - Check GPU availability
	$(PYTHON) -c "
	import torch
	print(f'CUDA available: {torch.cuda.is_available()}')
	if torch.cuda.is_available():
		print(f'GPU count: {torch.cuda.device_count()}')
		for i in range(torch.cuda.device_count()):
			print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
	"

.PHONY: system-info
system-info: ## Util - Display system and environment information
	@echo "$(BOLD)System Information$(RESET)"
	@echo "Python: $(shell python --version)"
	@echo "PyTorch: $(shell python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA: $(shell python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "Not available")')"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"
	@echo "Working directory: $(PWD)"

.PHONY: requirements-check
requirements-check: ## Util - Check if all requirements are installed
	$(PYTHON) -c "
	import pkg_resources
	import sys

	with open('requirements.txt', 'r') as f:
		requirements = f.read().splitlines()

	missing = []
	for requirement in requirements:
		if requirement and not requirement.startswith('#'):
			try:
				pkg_resources.require(requirement)
			except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
				missing.append(requirement)

	if missing:
		print('❌ Missing requirements:')
		for req in missing:
			print(f'  - {req}')
		sys.exit(1)
	else:
		print('✅ All requirements satisfied')
	"

# =============================================================================
# Quick Start Commands
# =============================================================================

.PHONY: quickstart
quickstart: ## Dev - Complete quickstart: setup + quick training + evaluation
	@echo "🚀 EEG2025 Challenge Quickstart"
	$(MAKE) setup-dirs
	$(MAKE) install
	$(MAKE) prepare-data
	$(MAKE) train-quick-pipeline
	$(MAKE) evaluate
	$(MAKE) generate-submission
	@echo "✅ Quickstart completed! Check $(OUTPUTS_DIR)/ for results"

.PHONY: demo
demo: ## Dev - Run demonstration with synthetic data
	@echo "🎯 Running demonstration with synthetic data"
	$(PYTHON) -c "
	print('Creating synthetic demo...')
	from src.demo import run_synthetic_demo
	run_synthetic_demo()
	print('✅ Demo completed successfully')
	"
