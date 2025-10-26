#!/usr/bin/env python3
"""
Clean up root directory by moving files to appropriate subdirectories
"""
import os
import shutil
from pathlib import Path

def organize_files():
    """Move files from root to appropriate subdirectories"""
    root = Path('.')
    
    # File categorization rules
    rules = {
        'docs/': [
            '*_ANALYSIS.md', '*_STATUS.md', '*_SUMMARY.md', 
            'ROCM_*.md', 'GPU_*.md', 'DRIVER_*.md', 'ENVIRONMENT_*.md',
            'SESSION_*.md', 'CLEANUP_*.md', 'CONVOLUTION_*.md',
            'TODO_*.md', 'UPLOAD_*.md', 'COMPLETE_*.md', 'DAILY_*.md',
            'FINAL_*.md', 'ACTION_*.md', 'DEVICE_*.md', 'SDK_*.md',
            'TRAINING_*.md', 'READY_*.md', '*_GUIDE.md', 'CHANGELOG.md'
        ],
        'scripts/': [
            '*_analysis.py', 'gpu_*.py', 'driver_*.py', 
            'quick_*.py', 'test_*.py', 'minimal_*.py',
            'practical_*.py', 'rapid_*.py', 'simple_*.py',
            'monitor_*.sh', 'setup_*.sh', 'activate_*.sh',
            'start_*.sh', 'restart_*.sh', 'prepare_*.sh',
            'test_*.sh', 'ROCM_*.sh', '*.patch',
            'fix_*.sh', 'skip_*.patch'
        ],
        'training/': [
            'train_*.py', 'gpu_training_*.py'
        ],
        'weights/': [
            'weights_*.pt', '*.pth'
        ],
        'archive/old_files/': [
            '*_OLD_*.md', '*_ARCHIVE_*.md'
        ]
    }
    
    # Create directories if they don't exist
    for dir_path in rules.keys():
        os.makedirs(dir_path, exist_ok=True)
    
    moved_files = []
    
    # Process each rule
    for target_dir, patterns in rules.items():
        for pattern in patterns:
            # Find matching files in root
            matching_files = list(root.glob(pattern))
            
            for file_path in matching_files:
                if file_path.is_file() and file_path.parent == root:
                    target_path = Path(target_dir) / file_path.name
                    
                    # Don't overwrite existing files
                    if target_path.exists():
                        print(f"⚠️  Skipping {file_path.name} (already exists in {target_dir})")
                        continue
                    
                    try:
                        shutil.move(str(file_path), str(target_path))
                        moved_files.append(f"{file_path.name} → {target_dir}")
                        print(f"✅ Moved {file_path.name} → {target_dir}")
                    except Exception as e:
                        print(f"❌ Error moving {file_path.name}: {e}")
    
    # Special handling for submission.py (keep in root)
    submission_files = ['submission.py', 'setup.py', 'pyproject.toml', 
                       'requirements.txt', 'requirements-dev.txt', 'README.md',
                       'LICENSE', 'Makefile']
    
    print(f"\n📊 Summary:")
    print(f"   Moved {len(moved_files)} files")
    print(f"   Kept {len(submission_files)} important files in root")
    
    if moved_files:
        print(f"\n📁 Files moved:")
        for move in moved_files[:10]:  # Show first 10
            print(f"   {move}")
        if len(moved_files) > 10:
            print(f"   ... and {len(moved_files) - 10} more")
    
    print(f"\n🧹 Root directory cleaned!")

if __name__ == "__main__":
    organize_files()
