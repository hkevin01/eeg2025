# EEG2025 Project Directory Structure
Generated: 2025-10-31 10:19:14

## Root Directory
Keep only essential files:
- Configuration files (requirements.txt, pyproject.toml, setup.py)
- Main documentation (README.md, LICENSE)
- Current submission (phase1_v9_submission.zip)

## Active Directories

### `/src/` - Source Code
Main codebase organized by functionality:
- `models/` - Model architectures
- `dataio/` - Data loading and processing
- `training/` - Training infrastructure
- `evaluation/` - Evaluation metrics
- `gpu/` - GPU optimization

### `/scripts/` - Utility Scripts
Organized by purpose:
- `training/` - Training scripts (train_c1_*.py, train_c2_*.py)
- `submissions/` - Submission creation scripts
- `monitoring/` - Monitoring and verification scripts
- `organize_project.py` - This cleanup script

### `/tests/` - Test Suite
Unit and integration tests

### `/configs/` - Configuration Files
YAML configuration files for experiments

### `/checkpoints/` - Model Checkpoints
Saved model weights (.pt files)
- `c1_v8_best.pt` - Best C1 model
- `c2_phase1_best.pt` - Best C2 model

### `/submissions/` - Submission Packages
Organized competition submissions:
- `phase1_v8/` - V8 submission (C1: 1.0002)
- `phase1_v9/` - V9 submission (C1: 1.0002, C2: 1.0055-1.0075)

### `/docs/` - Documentation
- `strategy/` - Planning and strategy documents
- `api/` - API documentation
- Technical documentation

### `/logs/` - Training Logs
- Active logs in root
- `archive/` - Historical logs

## Archive Directory

### `/archive/status_reports/` - Historical Status
Training status and progress reports from past sessions

### `/archive/submissions/` - Submission History
Documentation of past submissions

### `/archive/old_submissions/` - Old Submission Files
Downloaded results and old zip files

### `/archive/misc/` - Miscellaneous
Metadata and other archived files

## Ignored Directories
(See .gitignore)
- `/venv_*/` - Virtual environments
- `/__pycache__/` - Python cache
- `/.pytest_cache/` - Pytest cache
- `/.mypy_cache/` - MyPy cache
- `/data/` - Large datasets (not committed)
- `/weights/` - Large weight files (not committed)

## Key Files

### Configuration
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Project metadata
- `.gitignore` - Git ignore patterns
- `Makefile` - Build automation

### Documentation
- `README.md` - Main project documentation
- `LICENSE` - Project license
- `DIRECTORY_INDEX.md` - This file

### Current Submission
- `phase1_v9_submission.zip` - Latest submission package
