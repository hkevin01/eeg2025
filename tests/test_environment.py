"""
Environment Verification Script
Tests all dependencies required for competition submission
"""

import sys
from typing import Tuple, List


def test_dependency(module_name: str, min_version: str = None) -> Tuple[bool, str, str]:
    """Test if a dependency is available and check version."""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            # Simple version comparison (works for most cases)
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, version, f"Version {version} < required {min_version}"
        
        return True, version, "OK"
    except ImportError as e:
        return False, "NOT INSTALLED", str(e)


def test_specific_imports() -> List[Tuple[str, bool, str]]:
    """Test specific imports used in submission.py."""
    results = []
    
    # Test torch.load with weights_only parameter
    try:
        import torch
        import inspect
        sig = inspect.signature(torch.load)
        has_weights_only = 'weights_only' in sig.parameters
        results.append(
            ('torch.load(weights_only=...)', has_weights_only, 
             'Supported' if has_weights_only else 'Not supported (PyTorch < 1.13)')
        )
    except Exception as e:
        results.append(('torch.load check', False, str(e)))
    
    # Test braindecode.models.EEGNeX import
    try:
        from braindecode.models import EEGNeX
        results.append(('braindecode.models.EEGNeX', True, 'Available'))
    except ImportError as e:
        results.append(('braindecode.models.EEGNeX', False, str(e)))
    
    # Test torch.nn availability
    try:
        import torch.nn as nn
        results.append(('torch.nn', True, 'Available'))
    except ImportError as e:
        results.append(('torch.nn', False, str(e)))
    
    return results


def main():
    """Run all environment tests."""
    print("=" * 80)
    print("COMPETITION SUBMISSION ENVIRONMENT VERIFICATION")
    print("=" * 80)
    print()
    
    # Core dependencies used in submission.py
    dependencies = [
        ('torch', '1.13.0'),
        ('numpy', '1.20.0'),
        ('braindecode', '0.8.0'),
    ]
    
    # Optional dependencies
    optional_deps = [
        ('sklearn', '1.0.0'),  # For calibration
        ('scipy', None),
    ]
    
    print("CORE DEPENDENCIES (Required for submission):")
    print("-" * 80)
    all_passed = True
    for module, min_ver in dependencies:
        success, version, message = test_dependency(module, min_ver)
        status = "✅" if success else "❌"
        print(f"{status} {module:20s} {version:15s} {message}")
        if not success:
            all_passed = False
    
    print()
    print("OPTIONAL DEPENDENCIES:")
    print("-" * 80)
    for module, min_ver in optional_deps:
        success, version, message = test_dependency(module, min_ver)
        status = "✅" if success else "⚠️"
        print(f"{status} {module:20s} {version:15s} {message}")
    
    print()
    print("SPECIFIC IMPORT TESTS:")
    print("-" * 80)
    import_results = test_specific_imports()
    for name, success, message in import_results:
        status = "✅" if success else "❌"
        print(f"{status} {name:40s} {message}")
        if not success and 'braindecode' in name:
            all_passed = False
    
    print()
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print("=" * 80)
    
    if all_passed:
        print("\n✅ ALL REQUIRED DEPENDENCIES VERIFIED")
        print("   Submission environment is ready!")
        return 0
    else:
        print("\n❌ SOME DEPENDENCIES FAILED")
        print("   Review errors above before submitting.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST SCRIPT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
