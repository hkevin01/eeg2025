#!/usr/bin/env python3
"""
Pre-Submission Validation Script
Comprehensive checks before creating competition submission zip files
"""

import sys
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
import subprocess
import json


class SubmissionValidator:
    """Validates competition submission packages"""
    
    REQUIRED_FILES = [
        'submission.py',
        'weights_challenge_1.pt',
        'weights_challenge_2.pt'
    ]
    
    REQUIRED_CLASSES = [
        'CompactResponseTimeCNN',
        'EEGNeX'
    ]
    
    REQUIRED_FUNCTIONS = [
        'predict'
    ]
    
    MAX_ZIP_SIZE_MB = 100  # Competition size limit
    
    def __init__(self, submission_dir):
        self.submission_dir = Path(submission_dir)
        self.errors = []
        self.warnings = []
        self.checks_passed = []
        
    def error(self, msg):
        """Add error message"""
        self.errors.append(msg)
        print(f"❌ ERROR: {msg}")
        
    def warning(self, msg):
        """Add warning message"""
        self.warnings.append(msg)
        print(f"⚠️  WARNING: {msg}")
        
    def success(self, msg):
        """Add success message"""
        self.checks_passed.append(msg)
        print(f"✅ {msg}")
        
    def check_directory_exists(self):
        """Check if submission directory exists"""
        print("\n📁 Checking directory...")
        if not self.submission_dir.exists():
            self.error(f"Directory does not exist: {self.submission_dir}")
            return False
        self.success(f"Directory exists: {self.submission_dir}")
        return True
        
    def check_required_files(self):
        """Check all required files are present"""
        print("\n📄 Checking required files...")
        all_present = True
        
        for filename in self.REQUIRED_FILES:
            filepath = self.submission_dir / filename
            if not filepath.exists():
                self.error(f"Missing required file: {filename}")
                all_present = False
            else:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                self.success(f"Found {filename} ({size_mb:.2f} MB)")
                
        return all_present
        
    def check_no_extra_files(self):
        """Check for unwanted files that shouldn't be in submission"""
        print("\n🧹 Checking for unwanted files...")
        unwanted_patterns = [
            '__pycache__',
            '*.pyc',
            '.git',
            '.gitignore',
            '*.log',
            '.DS_Store',
            'README.md',
            'VALIDATION_REPORT.md',
            '*.ipynb'
        ]
        
        found_unwanted = False
        for item in self.submission_dir.iterdir():
            name = item.name
            
            # Check if it's an unwanted file
            is_unwanted = False
            for pattern in unwanted_patterns:
                if pattern.startswith('*') and name.endswith(pattern[1:]):
                    is_unwanted = True
                elif name == pattern or pattern in name:
                    is_unwanted = True
                    
            if is_unwanted:
                self.warning(f"Unwanted file found: {name} (should be removed)")
                found_unwanted = True
            elif name not in self.REQUIRED_FILES:
                self.warning(f"Extra file: {name} (not required, may cause issues)")
                
        if not found_unwanted:
            self.success("No unwanted files found")
            
        return True  # Not critical, just warnings
        
    def check_python_syntax(self):
        """Check Python syntax of submission.py"""
        print("\n🐍 Checking Python syntax...")
        
        submission_py = self.submission_dir / 'submission.py'
        if not submission_py.exists():
            return False  # Already reported in required files check
            
        try:
            with open(submission_py, 'r') as f:
                code = f.read()
            compile(code, 'submission.py', 'exec')
            self.success("Python syntax is valid")
            return True
        except SyntaxError as e:
            self.error(f"Syntax error in submission.py: {e}")
            return False
            
    def check_imports(self):
        """Check if submission.py imports successfully"""
        print("\n📦 Checking imports...")
        
        submission_py = self.submission_dir / 'submission.py'
        if not submission_py.exists():
            return False
            
        # Create temporary environment
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy submission.py to temp dir
            temp_submission = Path(tmpdir) / 'submission.py'
            shutil.copy(submission_py, temp_submission)
            
            # Try to import
            test_code = f"""
import sys
sys.path.insert(0, '{tmpdir}')
try:
    import submission
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
"""
            
            result = subprocess.run(
                [sys.executable, '-c', test_code],
                capture_output=True,
                text=True
            )
            
            if 'IMPORT_SUCCESS' in result.stdout:
                self.success("submission.py imports successfully")
                return True
            else:
                self.error(f"Import failed: {result.stdout}\n{result.stderr}")
                return False
                
    def check_required_classes(self):
        """Check if required model classes exist"""
        print("\n🧠 Checking model classes...")
        
        submission_py = self.submission_dir / 'submission.py'
        if not submission_py.exists():
            return False
            
        with open(submission_py, 'r') as f:
            content = f.read()
            
        all_found = True
        for class_name in self.REQUIRED_CLASSES:
            # Check if class is defined OR imported
            if f"class {class_name}" in content:
                self.success(f"Found class definition: {class_name}")
            elif f"import {class_name}" in content or f"from " in content and class_name in content:
                self.success(f"Found class import: {class_name}")
            else:
                self.error(f"Missing required class: {class_name} (neither defined nor imported)")
                all_found = False
                
        return all_found
        
    def check_required_functions(self):
        """Check if required functions exist"""
        print("\n⚙️  Checking required functions...")
        
        submission_py = self.submission_dir / 'submission.py'
        if not submission_py.exists():
            return False
            
        with open(submission_py, 'r') as f:
            content = f.read()
            
        all_found = True
        for func_name in self.REQUIRED_FUNCTIONS:
            if f"def {func_name}(" in content:
                self.success(f"Found function: {func_name}()")
            else:
                self.warning(f"Function not found: {func_name}() (may be auto-called)")
                
        return True  # Not critical if auto-execution works
        
    def check_weights_loadable(self):
        """Check if weight files can be loaded by PyTorch"""
        print("\n💾 Checking weight files...")
        
        try:
            import torch
        except ImportError:
            self.warning("PyTorch not available, skipping weight validation")
            return True
            
        for weight_file in ['weights_challenge_1.pt', 'weights_challenge_2.pt']:
            filepath = self.submission_dir / weight_file
            if not filepath.exists():
                continue
                
            try:
                # Try to load with weights_only=False (more permissive)
                state = torch.load(filepath, map_location='cpu', weights_only=False)
                
                if isinstance(state, dict):
                    if 'model_state_dict' in state:
                        self.success(f"{weight_file} is valid (checkpoint format)")
                    else:
                        self.success(f"{weight_file} is valid (state dict)")
                else:
                    self.warning(f"{weight_file} has unusual format")
                    
            except Exception as e:
                self.error(f"Cannot load {weight_file}: {e}")
                return False
                
        return True
        
    def check_zip_structure_preview(self):
        """Preview what the zip structure will look like"""
        print("\n📦 Previewing zip structure...")
        
        print("   Files that will be zipped:")
        for item in sorted(self.submission_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"      {item.name} ({size_mb:.2f} MB)")
                
        return True
        
    def check_estimated_zip_size(self):
        """Estimate final zip size"""
        print("\n📏 Checking estimated zip size...")
        
        total_size = sum(
            f.stat().st_size 
            for f in self.submission_dir.iterdir() 
            if f.is_file()
        )
        
        total_mb = total_size / (1024 * 1024)
        
        # Estimate compression (PyTorch files compress poorly, ~10-20%)
        estimated_zip_mb = total_mb * 0.9
        
        print(f"   Total uncompressed: {total_mb:.2f} MB")
        print(f"   Estimated zip size: {estimated_zip_mb:.2f} MB")
        
        if estimated_zip_mb > self.MAX_ZIP_SIZE_MB:
            self.error(f"Estimated zip size ({estimated_zip_mb:.2f} MB) exceeds limit ({self.MAX_ZIP_SIZE_MB} MB)")
            return False
        else:
            self.success(f"Size within limit ({estimated_zip_mb:.2f} MB < {self.MAX_ZIP_SIZE_MB} MB)")
            return True
            
    def create_test_zip(self):
        """Create a test zip and validate its structure"""
        print("\n🧪 Creating test zip to validate structure...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_zip = Path(tmpdir) / 'test_submission.zip'
            
            # Create zip with files at root level (correct structure)
            with zipfile.ZipFile(test_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for item in self.submission_dir.iterdir():
                    if item.is_file() and item.name in self.REQUIRED_FILES:
                        # Add at root level (no directory prefix)
                        zf.write(item, arcname=item.name)
                        
            # Validate zip structure
            with zipfile.ZipFile(test_zip, 'r') as zf:
                namelist = zf.namelist()
                
                print("   Zip contents:")
                for name in namelist:
                    info = zf.getinfo(name)
                    compressed_mb = info.compress_size / (1024 * 1024)
                    print(f"      {name} ({compressed_mb:.2f} MB compressed)")
                    
                # Check for nested directories (common mistake)
                has_nested = any('/' in name for name in namelist)
                if has_nested:
                    self.error("❌ CRITICAL: Zip has nested directories! Files must be at root level.")
                    self.error("   Current structure has folders like 'submissions/phase1_v9/'")
                    self.error("   Competition expects: submission.py (not submissions/phase1_v9/submission.py)")
                    return False
                else:
                    self.success("Zip structure is correct (flat, no nested directories)")
                    
                # Check all required files present in zip
                for required in self.REQUIRED_FILES:
                    if required not in namelist:
                        self.error(f"Required file missing from zip: {required}")
                        return False
                        
                self.success("All required files present in zip")
                
                # Check final size
                zip_size_mb = test_zip.stat().st_size / (1024 * 1024)
                print(f"   Final zip size: {zip_size_mb:.2f} MB")
                
                if zip_size_mb > self.MAX_ZIP_SIZE_MB:
                    self.error(f"Zip size exceeds limit: {zip_size_mb:.2f} MB > {self.MAX_ZIP_SIZE_MB} MB")
                    return False
                    
        return True
        
    def run_all_checks(self):
        """Run all validation checks"""
        print("=" * 60)
        print("🔍 SUBMISSION VALIDATION - PRE-ZIP CHECKS")
        print("=" * 60)
        print(f"Validating: {self.submission_dir}")
        
        checks = [
            self.check_directory_exists,
            self.check_required_files,
            self.check_no_extra_files,
            self.check_python_syntax,
            self.check_imports,
            self.check_required_classes,
            self.check_required_functions,
            self.check_weights_loadable,
            self.check_zip_structure_preview,
            self.check_estimated_zip_size,
            self.create_test_zip,
        ]
        
        all_passed = True
        for check in checks:
            try:
                result = check()
                if result is False:
                    all_passed = False
            except Exception as e:
                self.error(f"Check failed with exception: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
                
        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Checks passed: {len(self.checks_passed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning}")
                
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"   - {error}")
                
        print("\n" + "=" * 60)
        if all_passed and len(self.errors) == 0:
            print("✅ ALL CHECKS PASSED - READY TO ZIP!")
            print("=" * 60)
            return True
        else:
            print("❌ VALIDATION FAILED - DO NOT SUBMIT YET")
            print("=" * 60)
            return False
            
    def generate_report(self, output_file=None):
        """Generate validation report"""
        report = {
            'submission_dir': str(self.submission_dir),
            'checks_passed': len(self.checks_passed),
            'warnings': len(self.warnings),
            'errors': len(self.errors),
            'passed': len(self.errors) == 0,
            'details': {
                'checks_passed': self.checks_passed,
                'warnings': self.warnings,
                'errors': self.errors
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to: {output_file}")
            
        return report


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_directory>")
        print("\nExample:")
        print("  python validate_submission.py submissions/phase1_v9")
        sys.exit(1)
        
    submission_dir = sys.argv[1]
    validator = SubmissionValidator(submission_dir)
    
    success = validator.run_all_checks()
    
    # Generate report
    report_file = Path(submission_dir) / 'VALIDATION_REPORT.json'
    validator.generate_report(report_file)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
