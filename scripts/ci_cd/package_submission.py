#!/usr/bin/env python3
"""
Submission Packager for NeurIPS 2025 EEG Foundation Challenge
Creates ZIP file from validated submission directory.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


class SubmissionPackager:
    def __init__(self, submission_dir: str, output_name: str = None):
        self.submission_dir = Path(submission_dir)
        self.output_name = output_name or self.submission_dir.name
        
    def create_zip(self) -> Path:
        """Create ZIP file from submission directory."""
        print("=" * 70)
        print("📦 EEG2025 Submission Packager")
        print("=" * 70)
        print(f"📂 Source: {self.submission_dir}")
        print()
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{self.output_name}.zip"
        zip_path = self.submission_dir.parent / zip_filename
        
        # Remove existing zip if present
        if zip_path.exists():
            print(f"🗑️  Removing existing: {zip_path}")
            zip_path.unlink()
        
        # Create ZIP
        print(f"📦 Creating: {zip_filename}")
        print()
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in self.submission_dir.iterdir():
                if file.is_file():
                    arcname = file.name
                    zipf.write(file, arcname)
                    size_kb = file.stat().st_size / 1024
                    print(f"  ✅ Added: {arcname} ({size_kb:.1f} KB)")
        
        # Get final size
        zip_size_kb = zip_path.stat().st_size / 1024
        
        print()
        print("=" * 70)
        print("✅ PACKAGING COMPLETE")
        print("=" * 70)
        print(f"📦 Output: {zip_path}")
        print(f"📊 Size: {zip_size_kb:.1f} KB")
        print("=" * 70)
        
        return zip_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python package_submission.py <submission_directory> [output_name]")
        print("Example: python package_submission.py submissions/phase1_v14")
        print("         python package_submission.py submissions/phase1_v14 phase1_v14_final")
        sys.exit(1)
    
    submission_dir = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    packager = SubmissionPackager(submission_dir, output_name)
    zip_path = packager.create_zip()
    
    print(f"\n🎉 Ready to submit: {zip_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
