#!/usr/bin/env python3
"""
EEG2025 Project Cleanup and Reorganization Script
Removes excessive documentation and reorganizes project structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Root directory
ROOT = Path("/home/kevin/Projects/eeg2025")

# Files to KEEP (essential docs)
KEEP_DOCS = {
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
}

# Files to REMOVE (AI session artifacts, redundant docs)
REMOVE_PATTERNS = [
    "*SUMMARY*.md",
    "*SESSION*.md",
    "*ANALYSIS*.md",
    "*STRATEGY*.md",
    "*PROGRESS*.md",
    "*STATUS*.md",
    "*COMPLETE*.md",
    "*READY*.md",
    "*QUICKSTART*.md",
    "*GUIDE*.md",
    "*TODO*.md",
    "*FAILURE*.md",
    "*BREAKTHROUGH*.md",
    "*IMPROVEMENT*.md",
    "*UPLOAD*.md",
    "*SUBMISSION*.md",
    "*ENHANCEMENT*.md",
    "*VERIFICATION*.md",
    "*CHECKLIST*.md",
    "*DEPENDENCY*.md",
    "*TRAINING*.md",
    "*NEXT_STEPS*.md",
    "*QUICK_REF*.md",
    "*START_HERE*.md",
    "C1_*.md",
    "V1*.md",
]

# Scripts to move to scripts/training/
TRAINING_SCRIPTS = [
    "train_c1_multiseed_v16.py",
    "train_c1_quick_test.py",
]

# Scripts to move to scripts/submission/
SUBMISSION_SCRIPTS = [
    "create_v15_submission.py",
    "create_v16_submission.py",
]

# Temporary/build files to remove
TEMP_FILES = [
    "*.pyc",
    "__pycache__",
    "*.patch",
    "*EVAL_RESULTS.txt",
]

def create_archive_folder():
    """Create archive folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ROOT / "archive" / f"cleanup_{timestamp}"
    archive_path.mkdir(parents=True, exist_ok=True)
    return archive_path

def archive_file(file_path, archive_root):
    """Move file to archive"""
    rel_path = file_path.relative_to(ROOT)
    dest = archive_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file_path), str(dest))
    return dest

def clean_root_md_files(archive_root):
    """Remove excessive .md files from root"""
    print("\n�� Cleaning root .md files...")
    removed_count = 0
    
    for pattern in REMOVE_PATTERNS:
        for md_file in ROOT.glob(pattern):
            if md_file.is_file() and md_file.name not in KEEP_DOCS:
                dest = archive_file(md_file, archive_root)
                print(f"   Archived: {md_file.name} -> archive/")
                removed_count += 1
    
    print(f"   ✅ Archived {removed_count} excessive .md files")
    return removed_count

def reorganize_scripts(archive_root):
    """Move scripts to appropriate folders"""
    print("\n📜 Reorganizing scripts...")
    moved = 0
    
    # Training scripts
    training_dir = ROOT / "scripts" / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for script in TRAINING_SCRIPTS:
        src = ROOT / script
        if src.exists():
            dest = training_dir / script
            shutil.move(str(src), str(dest))
            print(f"   Moved: {script} -> scripts/training/")
            moved += 1
    
    # Submission scripts
    submission_dir = ROOT / "scripts" / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for script in SUBMISSION_SCRIPTS:
        src = ROOT / script
        if src.exists():
            dest = submission_dir / script
            shutil.move(str(src), str(dest))
            print(f"   Moved: {script} -> scripts/submission/")
            moved += 1
    
    print(f"   ✅ Moved {moved} scripts to organized folders")
    return moved

def clean_temp_files():
    """Remove temporary and build files"""
    print("\n🧹 Cleaning temporary files...")
    removed = 0
    
    for pattern in TEMP_FILES:
        for temp_file in ROOT.glob(pattern):
            if temp_file.is_file():
                temp_file.unlink()
                print(f"   Removed: {temp_file.name}")
                removed += 1
            elif temp_file.is_dir() and temp_file.name == "__pycache__":
                shutil.rmtree(temp_file)
                print(f"   Removed: {temp_file.name}/")
                removed += 1
    
    print(f"   ✅ Removed {removed} temporary files")
    return removed

def ensure_structure():
    """Ensure proper project structure exists"""
    print("\n📁 Ensuring proper project structure...")
    
    required_dirs = [
        "src",
        "tests", 
        "docs",
        "config",
        "scripts",
        "scripts/training",
        "scripts/submission",
        "scripts/infrastructure",
        "data",
        "checkpoints",
        "logs",
        "submissions",
        "archive",
    ]
    
    created = 0
    for dir_name in required_dirs:
        dir_path = ROOT / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_name}/")
            created += 1
    
    if created == 0:
        print(f"   ✅ All required directories exist")
    else:
        print(f"   ✅ Created {created} directories")
    
    return created

def clean_archive_folder():
    """Clean up old archive folders, keep only essential"""
    print("\n🗄️  Cleaning archive folder...")
    
    archive_dir = ROOT / "archive"
    if not archive_dir.exists():
        return 0
    
    # Keep these important archive items
    keep_items = {
        "COMPETITION_RULES.md",
        "README.md",
        "docs",
        "weights",
    }
    
    removed = 0
    for item in archive_dir.iterdir():
        if item.name not in keep_items and item.name.startswith("cleanup_"):
            # These are our new cleanup archives, keep them
            continue
        
        if item.name not in keep_items:
            if item.name.startswith(("old_", "README_OLD", "SESSION_", "SUBMISSION_", "TRAINING_")):
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                print(f"   Removed: archive/{item.name}")
                removed += 1
    
    print(f"   ✅ Cleaned {removed} items from archive/")
    return removed

def create_docs_readme():
    """Create a clean docs/README.md"""
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    readme = docs_dir / "README.md"
    if not readme.exists():
        content = """# EEG2025 Documentation

This directory contains project documentation.

## Structure

- `architecture/` - System architecture and design decisions
- `api/` - API documentation
- `guides/` - User guides and tutorials
- `development/` - Development notes and conventions

## Quick Links

- [Main README](../README.md)
- [Installation Guide](../README.md#installation)
- [Training Guide](../README.md#training)
- [Submission Guide](../README.md#submission)
"""
        readme.write_text(content)
        print(f"\n📝 Created docs/README.md")

def generate_report(stats):
    """Generate cleanup report"""
    print("\n" + "=" * 70)
    print("📊 CLEANUP SUMMARY")
    print("=" * 70)
    print(f"Archived .md files:    {stats['md_files']}")
    print(f"Moved scripts:         {stats['scripts']}")
    print(f"Removed temp files:    {stats['temp_files']}")
    print(f"Created directories:   {stats['dirs']}")
    print(f"Cleaned archive items: {stats['archive_items']}")
    print("=" * 70)
    print("\n✅ Project cleanup complete!")
    print(f"📦 Archived files saved in: archive/cleanup_{stats['timestamp']}/")
    print("\n🎯 Next steps:")
    print("   1. Review the cleaned project structure")
    print("   2. Update README.md if needed")
    print("   3. Commit changes to git")
    print("=" * 70)

def main():
    print("=" * 70)
    print("🧹 EEG2025 Project Cleanup & Reorganization")
    print("=" * 70)
    print(f"📂 Working directory: {ROOT}")
    print()
    
    # Create archive folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = create_archive_folder()
    print(f"📦 Archive folder: archive/cleanup_{timestamp}/")
    
    # Perform cleanup
    stats = {
        'timestamp': timestamp,
        'md_files': clean_root_md_files(archive_root),
        'scripts': reorganize_scripts(archive_root),
        'temp_files': clean_temp_files(),
        'dirs': ensure_structure(),
        'archive_items': clean_archive_folder(),
    }
    
    # Create documentation
    create_docs_readme()
    
    # Generate report
    generate_report(stats)

if __name__ == "__main__":
    main()
