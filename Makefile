# EEG Foundation Challenge 2025 - Makefile

# Variables
SHELL := /bin/bash
PYTHON := python
PIP := pip
DOCKER_IMAGE := eeg2025
DOCKER_TAG := latest

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "EEG Foundation Challenge 2025"
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Docker commands
.PHONY: up
up: ## Start Docker development environment
	docker compose -f docker/compose.yml up -d --build

.PHONY: down
down: ## Stop Docker containers
	docker compose -f docker/compose.yml down

.PHONY: logs
logs: ## Show container logs
	docker compose -f docker/compose.yml logs -f app

.PHONY: bash
bash: ## Access container shell
	docker compose -f docker/compose.yml exec app bash

# Data preparation
.PHONY: prepare
prepare: ## Prepare dataset index and splits
	docker compose -f docker/compose.yml exec app bash -lc "python scripts/prepare_hbn_bids.py --config configs/data.yaml && python scripts/make_splits.py --config configs/data.yaml"

.PHONY: prepare-local
prepare-local: ## Prepare data locally (non-Docker)
	$(PYTHON) scripts/prepare_hbn_bids.py --config configs/data.yaml
	$(PYTHON) scripts/make_splits.py --config configs/data.yaml

# Training commands
.PHONY: pretrain
pretrain: ## Run SSL pretraining
	docker compose -f docker/compose.yml exec app bash -lc "python -m src.training.pretrain_ssl --config configs/pretrain.yaml"

.PHONY: train-cross
train-cross: ## Run cross-task training
	docker compose -f docker/compose.yml exec app bash -lc "python -m src.training.train_cross_task --config configs/train_cross_task.yaml"

.PHONY: train-psych
train-psych: ## Run psychopathology training
	docker compose -f docker/compose.yml exec app bash -lc "python -m src.training.train_psych --config configs/train_psych.yaml"

# Local training (non-Docker)
.PHONY: pretrain-local
pretrain-local: ## Run SSL pretraining locally
	$(PYTHON) -m src.training.pretrain_ssl --config configs/pretrain.yaml

.PHONY: train-cross-local
train-cross-local: ## Run cross-task training locally
	$(PYTHON) -m src.training.train_cross_task --config configs/train_cross_task.yaml

.PHONY: train-psych-local
train-psych-local: ## Run psychopathology training locally
	$(PYTHON) -m src.training.train_psych --config configs/train_psych.yaml

# Evaluation
.PHONY: eval
eval: ## Run evaluation
	docker compose -f docker/compose.yml exec app bash -lc "python -m src.training.evaluate --config configs/eval.yaml"

.PHONY: eval-local
eval-local: ## Run evaluation locally
	$(PYTHON) -m src.training.evaluate --config configs/eval.yaml

# Submission
.PHONY: package-submission
package-submission: ## Package submission files
	docker compose -f docker/compose.yml exec app bash -lc "python scripts/export_submission.py --config configs/eval.yaml"

.PHONY: package-submission-local
package-submission-local: ## Package submission locally
	$(PYTHON) scripts/export_submission.py --config configs/eval.yaml

# Documentation
.PHONY: docs
docs: ## Generate documentation
	cd docs && make html

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	cd docs && make livehtml

# Cleaning
.PHONY: clean
clean: ## Clean build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf logs/ outputs/ wandb/

.PHONY: clean-data
clean-data: ## Clean data cache (careful!)
	@echo "Warning: This will remove processed data cache"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed/ data/.cache/; \
	fi

# Environment management
.PHONY: env-create
env-create: ## Create conda environment
	conda create -n eeg2025 python=3.11 -y
	@echo "Activate with: conda activate eeg2025"

.PHONY: env-export
env-export: ## Export conda environment
	conda env export > environment.yml

# Package submission
.PHONY: package-submission
package-submission: ## Package submission files
	$(PYTHON) scripts/export_submission.py

.PHONY: validate-submission
validate-submission: ## Validate submission format
	$(PYTHON) scripts/validate_submission.py

# Performance profiling
.PHONY: profile
profile: ## Profile model performance
	$(PYTHON) scripts/profile_model.py

.PHONY: benchmark
benchmark: ## Benchmark inference speed
	$(PYTHON) scripts/benchmark_inference.py
