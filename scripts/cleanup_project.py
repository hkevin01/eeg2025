#!/usr/bin/env python3
"""
Comprehensive Project Cleanup and Organization Script
Organizes files by type and moves them to appropriate directories
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Project root
ROOT = Path("/home/kevin/Projects/eeg2025")

# File categories
CATEGORIES = {
    "status_docs": {
        "patterns": [
            "C2_*STATUS*.md", "C2_*COMPLETE*.md", "TRAINING_*STATUS*.md",
            "STATUS_*.md", "SESSION_*SUMMARY*.md", "FINAL_*STATUS*.md",
            "FINAL_*SUMMARY*.md", "ITERATION_COMPLETE*.md", "OVERNIGHT_PLAN.md",
            "ROOT_CAUSE_FOUND.md", "ENSEMBLE_*ANALYSIS*.md", "INVESTIGATION_*.md",
            "TRAINING_*SUMMARY*.md", "TRAINING_*RESULTS*.md", "TRAINING_*COMPARISON*.md"
        ],
        "destination": "archive/status_reports"
    },
    "submission_docs": {
        "patterns": [
            "SUBMISSION_*.md", "V*_SUBMISSION_*.md", "*_SUBMISSION_READY.md",
            "V*_SUCCESS_REPORT.md"
        ],
        "destination": "archive/submissions"
    },
    "strategy_docs": {
        "patterns": [
            "*_STRATEGY*.md", "*_PLAN*.md", "ACTION_PLAN_*.md", 
            "TOP3_*.md", "DUAL_*.md", "AGGRESSIVE_*.md"
        ],
        "destination": "docs/strategy"
    },
    "training_logs": {
        "patterns": [
            "training_*.log", "*.log"
        ],
        "destination": "logs/archive"
    },
    "training_scripts": {
        "patterns": [
            "train_c1_*.py", "train_c2_*.py", "finetune_*.py", 
            "ssl_pretrain*.py", "extract_*.py"
        ],
        "destination": "scripts/training"
    },
    "submission_scripts": {
        "patterns": [
            "submission_*.py", "create_submission*.py", "create_ensemble*.py"
        ],
        "destination": "scripts/submissions"
    },
    "monitoring_scripts": {
        "patterns": [
            "monitor_*.sh", "watch_*.sh", "morning_*.sh", "verify_*.sh",
            "run_training.sh", "start_*.sh"
        ],
        "destination": "scripts/monitoring"
    },
    "old_zips": {
        "patterns": [
            "*_result_*.zip", "submission_*_downloaded.zip"
        ],
        "destination": "archive/old_submissions"
    },
    "metadata_files": {
        "patterns": [
            "metadata", "scores.json"
        ],
        "destination": "archive/misc"
    }
}

# Files to keep in root (important docs)
KEEP_IN_ROOT = [
    "README.md", "LICENSE", "Makefile", "pyproject.toml",
    "setup.py", "requirements.txt", "requirements-dev.txt",
    ".env", ".gitignore", ".editorconfig"
]

# Current submission to keep
KEEP_CURRENT = [
    "phase1_v9_submission.zip"
]

def create_directories():
    """Create all necessary directories"""
    print("📁 Creating directory structure...")
    for category, config in CATEGORIES.items():
        dest = ROOT / config["destination"]
        dest.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {config['destination']}")

def should_keep_in_root(filename):
    """Check if file should stay in root"""
    return filename in KEEP_IN_ROOT or filename in KEEP_CURRENT

def match_pattern(filename, pattern):
    """Simple pattern matching"""
    from fnmatch import fnmatch
    return fnmatch(filename, pattern)

def organize_files():
    """Organize files by category"""
    print("\n📦 Organizing files...")
    
    moved_count = 0
    skipped_count = 0
    
    # Get all files in root
    root_files = [f for f in os.listdir(ROOT) if os.path.isfile(ROOT / f)]
    
    for filename in root_files:
        # Skip files that should stay in root
        if should_keep_in_root(filename):
            continue
            
        # Find matching category
        moved = False
        for category, config in CATEGORIES.items():
            for pattern in config["patterns"]:
                if match_pattern(filename, pattern):
                    source = ROOT / filename
                    dest_dir = ROOT / config["destination"]
                    dest = dest_dir / filename
                    
                    # Don't overwrite existing files
                    if dest.exists():
                        print(f"   ⚠️  Skipping {filename} (already exists in {config['destination']})")
                        skipped_count += 1
                    else:
                        shutil.move(str(source), str(dest))
                        print(f"   ✓ {filename} → {config['destination']}")
                        moved_count += 1
                    
                    moved = True
                    break
            
            if moved:
                break
    
    print(f"\n✅ Moved {moved_count} files, skipped {skipped_count} files")

def clean_pycache():
    """Remove __pycache__ directories"""
    print("\n🧹 Cleaning __pycache__ directories...")
    count = 0
    for root, dirs, files in os.walk(ROOT):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            shutil.rmtree(pycache_path)
            count += 1
            print(f"   ✓ Removed {pycache_path}")
    print(f"✅ Removed {count} __pycache__ directories")

def create_index():
    """Create directory index"""
    print("\n📋 Creating directory index...")
    
    index_content = f"""# EEG2025 Project Directory Structure
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
"""
    
    index_path = ROOT / "DIRECTORY_INDEX.md"
    with open(index_path, "w") as f:
        f.write(index_content)
    
    print(f"✅ Created {index_path}")

def main():
    print("=" * 60)
    print("🧹 EEG2025 Project Cleanup and Organization")
    print("=" * 60)
    
    create_directories()
    organize_files()
    clean_pycache()
    create_index()
    
    print("\n" + "=" * 60)
    print("✅ Cleanup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the organized structure")
    print("2. Update .gitignore if needed")
    print("3. Commit changes: git add . && git commit -m 'chore: organize project structure'")

if __name__ == "__main__":
    main()
