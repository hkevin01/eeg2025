#!/usr/bin/env python3
"""
Submission Validator for NeurIPS 2025 EEG Foundation Challenge
Validates submission structure, format, and functionality before packaging.
"""

import os
import sys
import json
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
import torch


class SubmissionValidator:
    def __init__(self, submission_dir: str):
        self.submission_dir = Path(submission_dir)
        self.errors = []
        self.warnings = []
        self.results = {}
        
    def check_directory_structure(self) -> bool:
        """Check if submission directory exists and is accessible."""
        print("📁 Checking directory structure...")
        
        if not self.submission_dir.exists():
            self.errors.append(f"Submission directory not found: {self.submission_dir}")
            return False
            
        if not self.submission_dir.is_dir():
            self.errors.append(f"Path is not a directory: {self.submission_dir}")
            return False
            
        print(f"  ✅ Directory exists: {self.submission_dir}")
        return True
    
    def check_required_files(self) -> bool:
        """Check if all required files are present."""
        print("\n📄 Checking required files...")
        
        required_files = [
            "submission.py",
            "weights_challenge_1.pt",
            "weights_challenge_2.pt"
        ]
        
        all_present = True
        for filename in required_files:
            filepath = self.submission_dir / filename
            if not filepath.exists():
                self.errors.append(f"Missing required file: {filename}")
                all_present = False
                print(f"  ❌ Missing: {filename}")
            else:
                size_kb = filepath.stat().st_size / 1024
                print(f"  ✅ Found: {filename} ({size_kb:.1f} KB)")
                
        return all_present
    
    def check_submission_py_format(self) -> bool:
        """Validate submission.py has correct format and required classes."""
        print("\n🔍 Validating submission.py format...")
        
        submission_file = self.submission_dir / "submission.py"
        
        # Read file content
        try:
            with open(submission_file, 'r') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read submission.py: {e}")
            return False
        
        # Check for two possible submission formats:
        # Format 1: Challenge1Submission and Challenge2Submission classes
        # Format 2: Single Submission class with challenge_1 and challenge_2 methods
        
        format1 = "class Challenge1Submission" in content and "class Challenge2Submission" in content
        format2 = "class Submission" in content and "def challenge_1" in content and "def challenge_2" in content
        
        if format1:
            print("  ✅ Found format: Challenge1Submission + Challenge2Submission classes")
            self.results['format'] = 'separate_classes'
        elif format2:
            print("  ✅ Found format: Submission class with challenge_1/challenge_2 methods")
            self.results['format'] = 'unified_class'
        else:
            self.errors.append("Invalid submission format. Expected either:\n" +
                             "  - Challenge1Submission and Challenge2Submission classes, OR\n" +
                             "  - Submission class with challenge_1() and challenge_2() methods")
            print("  ❌ Invalid submission format")
            return False
        
        # Check for __init__ method
        if "def __init__" not in content:
            self.warnings.append("No __init__ method found")
            print("  ⚠️  Warning: No __init__ method found")
        
        print("  ✅ submission.py format valid")
        return True
    
    def test_model_loading(self, challenge_num: int) -> bool:
        """Test if model weights can be loaded."""
        print(f"\n🧪 Testing Challenge {challenge_num} model loading...")
        
        weights_file = self.submission_dir / f"weights_challenge_{challenge_num}.pt"
        
        try:
            # Try loading weights
            checkpoint = torch.load(weights_file, map_location='cpu')
            
            # Check structure
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  ✅ Loaded checkpoint with keys: {list(checkpoint.keys())}")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"  ✅ Loaded checkpoint with keys: {list(checkpoint.keys())}")
                else:
                    state_dict = checkpoint
                    print(f"  ✅ Loaded state dict directly")
            else:
                state_dict = checkpoint
                print(f"  ✅ Loaded model directly")
            
            # Count parameters
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            print(f"  📊 Total parameters: {total_params:,}")
            
            # Check for NaN values
            has_nan = any(torch.isnan(p).any() for p in state_dict.values() if isinstance(p, torch.Tensor))
            if has_nan:
                self.errors.append(f"Challenge {challenge_num} weights contain NaN values")
                print(f"  ❌ Weights contain NaN values")
                return False
            
            print(f"  ✅ Challenge {challenge_num} weights loaded successfully")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load Challenge {challenge_num} weights: {e}")
            print(f"  ❌ Failed to load weights: {e}")
            traceback.print_exc()
            return False
    
    def test_submission_inference(self) -> bool:
        """Test if submission.py can run inference."""
        print("\n🚀 Testing submission inference...")
        
        # Add submission directory to path
        sys.path.insert(0, str(self.submission_dir))
        
        try:
            # Import submission module
            spec = importlib.util.spec_from_file_location(
                "submission", 
                str(self.submission_dir / "submission.py")
            )
            submission_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(submission_module)
            
            test_input = torch.randn(4, 129, 200)
            
            # Detect submission format
            submission_format = self.results.get('format', 'unknown')
            
            if submission_format == 'separate_classes':
                # Format 1: Challenge1Submission and Challenge2Submission
                print("  Testing Challenge 1...")
                c1_model = submission_module.Challenge1Submission()
                
                with torch.no_grad():
                    c1_output = c1_model(test_input)
                
                if c1_output.shape != (4,):
                    self.errors.append(f"Challenge 1 output shape mismatch: expected (4,), got {c1_output.shape}")
                    print(f"  ❌ C1 output shape: {c1_output.shape} (expected (4,))")
                    return False
                
                print(f"  ✅ Challenge 1 inference successful")
                print(f"     Input: {test_input.shape}, Output: {c1_output.shape}")
                
                # Test Challenge 2
                print("  Testing Challenge 2...")
                c2_model = submission_module.Challenge2Submission()
                
                with torch.no_grad():
                    c2_output = c2_model(test_input)
                
                if c2_output.shape != (4,):
                    self.errors.append(f"Challenge 2 output shape mismatch: expected (4,), got {c2_output.shape}")
                    print(f"  ❌ C2 output shape: {c2_output.shape} (expected (4,))")
                    return False
                
                print(f"  ✅ Challenge 2 inference successful")
                print(f"     Input: {test_input.shape}, Output: {c2_output.shape}")
                
            elif submission_format == 'unified_class':
                # Format 2: Unified Submission class
                print("  Testing unified Submission class...")
                SFREQ = 100
                DEVICE = 'cpu'
                submission = submission_module.Submission(SFREQ, DEVICE)
                
                # Test Challenge 1
                print("  Testing challenge_1()...")
                with torch.no_grad():
                    c1_output = submission.challenge_1(test_input)
                
                if c1_output.shape != (4,):
                    self.errors.append(f"Challenge 1 output shape mismatch: expected (4,), got {c1_output.shape}")
                    print(f"  ❌ C1 output shape: {c1_output.shape} (expected (4,))")
                    return False
                
                print(f"  ✅ Challenge 1 inference successful")
                print(f"     Input: {test_input.shape}, Output: {c1_output.shape}")
                
                # Test Challenge 2
                print("  Testing challenge_2()...")
                with torch.no_grad():
                    c2_output = submission.challenge_2(test_input)
                
                if c2_output.shape != (4,):
                    self.errors.append(f"Challenge 2 output shape mismatch: expected (4,), got {c2_output.shape}")
                    print(f"  ❌ C2 output shape: {c2_output.shape} (expected (4,))")
                    return False
                
                print(f"  ✅ Challenge 2 inference successful")
                print(f"     Input: {test_input.shape}, Output: {c2_output.shape}")
            else:
                self.errors.append(f"Unknown submission format: {submission_format}")
                return False
            
            return True
            
        except Exception as e:
            self.errors.append(f"Inference test failed: {e}")
            print(f"  ❌ Inference failed: {e}")
            traceback.print_exc()
            return False
        
        finally:
            # Clean up path
            if str(self.submission_dir) in sys.path:
                sys.path.remove(str(self.submission_dir))
    
    def check_file_sizes(self) -> bool:
        """Check file sizes against competition limits."""
        print("\n📊 Checking file sizes...")
        
        max_size_mb = 5  # Competition limit
        
        total_size = 0
        for file in self.submission_dir.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  📄 {file.name}: {size_mb:.2f} MB")
        
        print(f"\n  📦 Total size: {total_size:.2f} MB (limit: {max_size_mb} MB)")
        
        if total_size > max_size_mb:
            self.errors.append(f"Submission exceeds size limit: {total_size:.2f} MB > {max_size_mb} MB")
            print(f"  ❌ Exceeds size limit")
            return False
        
        print(f"  ✅ Size within limits")
        return True
    
    def generate_report(self) -> Dict:
        """Generate validation report."""
        report = {
            "submission_dir": str(self.submission_dir),
            "validation_passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "results": self.results
        }
        return report
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("=" * 70)
        print("🧪 EEG2025 Submission Validator")
        print("=" * 70)
        print(f"📂 Validating: {self.submission_dir}")
        print()
        
        checks = [
            ("Directory Structure", self.check_directory_structure),
            ("Required Files", self.check_required_files),
            ("submission.py Format", self.check_submission_py_format),
            ("Challenge 1 Weights", lambda: self.test_model_loading(1)),
            ("Challenge 2 Weights", lambda: self.test_model_loading(2)),
            ("Inference Test", self.test_submission_inference),
            ("File Sizes", self.check_file_sizes),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                result = check_func()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"\n❌ {check_name} failed with exception: {e}")
                traceback.print_exc()
                self.errors.append(f"{check_name} failed: {e}")
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 70)
        print("📋 VALIDATION SUMMARY")
        print("=" * 70)
        
        if all_passed:
            print("✅ All checks PASSED")
            print("🎉 Submission is ready for packaging!")
        else:
            print("❌ Validation FAILED")
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print("=" * 70)
        
        return all_passed


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_directory>")
        print("Example: python validate_submission.py submissions/phase1_v14")
        sys.exit(1)
    
    submission_dir = sys.argv[1]
    validator = SubmissionValidator(submission_dir)
    
    success = validator.run_all_checks()
    
    # Save report
    report = validator.generate_report()
    report_file = Path(submission_dir) / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_file}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
