"""
Version Compatibility Verification Script
Compares local package versions with competition platform requirements
"""

import sys
from typing import Dict, List, Tuple
from packaging import version as pkg_version


class VersionCompatibilityChecker:
    """Check if local versions are compatible with competition platform."""
    
    # Competition platform package versions (from V10 success + assumptions)
    COMPETITION_VERSIONS = {
        'torch': {
            'min': '1.13.0',
            'max': '2.5.99',
            'notes': 'V10 uses weights_only=False (requires 1.13+)',
            'evidence': 'V10 submission works'
        },
        'numpy': {
            'min': '1.20.0',
            'max': '1.99.99',
            'notes': 'Standard scientific computing, avoid NumPy 2.0',
            'evidence': 'Standard dependency'
        },
        'braindecode': {
            'min': '0.8.0',
            'max': '1.99.99',
            'notes': 'EEGNeX model (added in 0.8), CRITICAL for C2',
            'evidence': 'V10 Challenge 2 works'
        },
        'sklearn': {
            'min': '0.24.0',
            'max': '1.99.99',
            'notes': 'Ridge regression for calibration (V13)',
            'evidence': 'Standard ML library'
        },
    }
    
    # Known incompatibilities
    INCOMPATIBILITIES = {
        'numpy': {
            '2.0.0': 'Breaking changes in NumPy 2.0 - avoid!',
            '2.0.1': 'Breaking changes in NumPy 2.0 - avoid!',
        },
        'torch': {
            '1.12.1': 'weights_only parameter not available',
            '1.12.0': 'weights_only parameter not available',
            '1.11.0': 'weights_only parameter not available',
        }
    }
    
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []
    
    def get_installed_version(self, package_name: str) -> str:
        """Get installed version of a package."""
        try:
            module = __import__(package_name)
            return getattr(module, '__version__', 'unknown')
        except ImportError:
            return 'NOT_INSTALLED'
    
    def check_version_compatibility(self, package: str, installed: str, 
                                   min_ver: str, max_ver: str) -> Tuple[bool, str]:
        """Check if installed version is within acceptable range."""
        if installed == 'NOT_INSTALLED':
            return False, 'Not installed'
        if installed == 'unknown':
            return None, 'Version unknown - cannot verify'
        
        try:
            installed_v = pkg_version.parse(installed)
            min_v = pkg_version.parse(min_ver)
            max_v = pkg_version.parse(max_ver)
            
            if installed_v < min_v:
                return False, f'Too old (need >={min_ver})'
            if installed_v > max_v:
                return False, f'Too new (need <={max_ver})'
            
            return True, 'Compatible'
        except Exception as e:
            return None, f'Version parse error: {e}'
    
    def check_known_incompatibilities(self, package: str, version: str) -> List[str]:
        """Check for known incompatible versions."""
        issues = []
        if package in self.INCOMPATIBILITIES:
            if version in self.INCOMPATIBILITIES[package]:
                issues.append(self.INCOMPATIBILITIES[package][version])
        return issues
    
    def verify_all_packages(self) -> bool:
        """Verify all packages and return overall status."""
        print("=" * 80)
        print("VERSION COMPATIBILITY VERIFICATION")
        print("Comparing local versions with competition platform requirements")
        print("=" * 80)
        print()
        
        all_compatible = True
        
        for package, requirements in self.COMPETITION_VERSIONS.items():
            installed = self.get_installed_version(package)
            min_ver = requirements['min']
            max_ver = requirements['max']
            notes = requirements['notes']
            evidence = requirements['evidence']
            
            compatible, message = self.check_version_compatibility(
                package, installed, min_ver, max_ver
            )
            
            # Check for known incompatibilities
            incompatibility_issues = self.check_known_incompatibilities(package, installed)
            
            # Determine status
            if compatible is False:
                status = "❌ INCOMPATIBLE"
                all_compatible = False
                self.errors.append(f"{package}: {message}")
            elif compatible is None:
                status = "⚠️  UNKNOWN"
                self.warnings.append(f"{package}: {message}")
            elif incompatibility_issues:
                status = "❌ KNOWN ISSUE"
                all_compatible = False
                self.errors.extend([f"{package}: {issue}" for issue in incompatibility_issues])
            else:
                status = "✅ COMPATIBLE"
            
            # Print package info
            print(f"{status} {package}")
            print(f"  Installed:  {installed}")
            print(f"  Required:   >={min_ver}, <={max_ver}")
            print(f"  Notes:      {notes}")
            print(f"  Evidence:   {evidence}")
            
            if incompatibility_issues:
                for issue in incompatibility_issues:
                    print(f"  ⚠️  Issue:     {issue}")
            
            print()
        
        return all_compatible
    
    def print_summary(self, all_compatible: bool):
        """Print verification summary."""
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print()
        if all_compatible and not self.errors:
            print("✅ ALL PACKAGES COMPATIBLE WITH COMPETITION PLATFORM")
            print("   Safe to submit!")
            return 0
        else:
            print("❌ COMPATIBILITY ISSUES DETECTED")
            print("   Review errors above before submitting.")
            return 1


def compare_with_requirements_files():
    """Compare versions specified in different requirements files."""
    print("\n" + "=" * 80)
    print("REQUIREMENTS FILES COMPARISON")
    print("=" * 80)
    print()
    
    files = [
        'requirements.txt',
        'requirements-submission.txt',
    ]
    
    for filename in files:
        try:
            with open(filename, 'r') as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                print(f"📄 {filename}:")
                for line in lines[:10]:  # Show first 10 packages
                    print(f"  {line}")
                if len(lines) > 10:
                    print(f"  ... and {len(lines) - 10} more")
                print()
        except FileNotFoundError:
            print(f"⚠️  {filename} not found")
            print()


def test_critical_features():
    """Test critical features that depend on specific versions."""
    print("=" * 80)
    print("CRITICAL FEATURE TESTS")
    print("=" * 80)
    print()
    
    tests_passed = []
    tests_failed = []
    
    # Test 1: torch.load with weights_only parameter
    try:
        import torch
        import inspect
        sig = inspect.signature(torch.load)
        if 'weights_only' in sig.parameters:
            print("✅ torch.load(weights_only=...) supported")
            tests_passed.append("torch.load weights_only")
        else:
            print("❌ torch.load(weights_only=...) NOT supported")
            print("   PyTorch version too old (need 1.13+)")
            tests_failed.append("torch.load weights_only")
    except Exception as e:
        print(f"❌ torch.load test failed: {e}")
        tests_failed.append("torch.load test")
    
    # Test 2: braindecode.models.EEGNeX import
    try:
        from braindecode.models import EEGNeX
        print("✅ braindecode.models.EEGNeX available")
        tests_passed.append("EEGNeX import")
    except ImportError as e:
        print(f"❌ braindecode.models.EEGNeX import failed: {e}")
        tests_failed.append("EEGNeX import")
    
    # Test 3: NumPy API compatibility
    try:
        import numpy as np
        # Test a common operation
        arr = np.array([1, 2, 3])
        result = np.mean(arr)
        print("✅ NumPy API compatible")
        tests_passed.append("NumPy API")
    except Exception as e:
        print(f"❌ NumPy API test failed: {e}")
        tests_failed.append("NumPy API")
    
    # Test 4: sklearn Ridge regression (for V13)
    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
        print("✅ sklearn.linear_model.Ridge available")
        tests_passed.append("Ridge regression")
    except ImportError as e:
        print(f"⚠️  sklearn.linear_model.Ridge import failed: {e}")
        print("   (Optional - only needed for V13 calibration)")
    
    print()
    print(f"Passed: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)}")
    
    return len(tests_failed) == 0


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("🔍 COMPETITION PLATFORM VERSION COMPATIBILITY CHECK")
    print("=" * 80)
    print()
    
    # Version compatibility check
    checker = VersionCompatibilityChecker()
    all_compatible = checker.verify_all_packages()
    exit_code = checker.print_summary(all_compatible)
    
    # Requirements files comparison
    compare_with_requirements_files()
    
    # Critical feature tests
    features_ok = test_critical_features()
    
    if not features_ok:
        exit_code = 1
    
    print("\n" + "=" * 80)
    print("FINAL STATUS")
    print("=" * 80)
    
    if exit_code == 0:
        print("\n✅ ALL CHECKS PASSED")
        print("   Local environment is compatible with competition platform")
        print("   Safe to proceed with submission!")
    else:
        print("\n❌ COMPATIBILITY ISSUES DETECTED")
        print("   Review errors above and update packages before submission")
        print("\n   Recommendations:")
        print("   1. Check DEPENDENCY_VERIFICATION.md for details")
        print("   2. Update incompatible packages")
        print("   3. Re-run this test")
    
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
