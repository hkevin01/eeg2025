#!/usr/bin/env python3
"""
Project organization script to move test files and clean up root folder.
"""
import os
import shutil
from pathlib import Path

def organize_project():
    """Move test files and validation scripts to proper subdirectories."""
    root = Path("/home/kevin/Projects/eeg2025")

    # Files to move to tests/
    test_files = [
        "test_cross_task_simple.py",
        "test_enhanced_model.py",
        "test_enhanced_starter_kit.py",
        "simple_validation.py",
        "validate_enhancements.py"
    ]

    # Ensure tests directory exists
    tests_dir = root / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Move test files
    for file in test_files:
        src = root / file
        if src.exists():
            dst = tests_dir / file
            print(f"Moving {src} -> {dst}")
            shutil.move(str(src), str(dst))

    # Move train_advanced.py to scripts/
    train_src = root / "train_advanced.py"
    if train_src.exists():
        train_dst = root / "scripts" / "train_advanced.py"
        print(f"Moving {train_src} -> {train_dst}")
        shutil.move(str(train_src), str(train_dst))

    print("Project organization complete!")

if __name__ == "__main__":
    organize_project()
