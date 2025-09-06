#!/usr/bin/env python3
"""
Quick Repository Validation Script
==================================

Lightweight validation that works without heavy dependencies.
Checks core repository structure and basic functionality.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, Dict]:
    """Check Python version compatibility."""
    details = {
        'python_version': sys.version,
        'version_info': list(sys.version_info[:3])
    }

    if sys.version_info < (3, 8):
        return False, {**details, 'error': 'Python 3.8+ required'}

    return True, details


def check_project_structure() -> Tuple[bool, Dict]:
    """Check that essential project files exist."""
    required_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'src/__init__.py',
        'tests/__init__.py',
        'configs/training/base_config.yaml',
        'scripts/health_check.py',
        '.github/workflows/ci.yml'
    ]

    required_dirs = [
        'src/',
        'src/models/',
        'src/models/backbone/',
        'src/models/adapters/',
        'src/models/heads/',
        'src/models/invariance/',
        'src/models/compression_ssl/',
        'src/training/',
        'src/gpu/',
        'tests/',
        'configs/',
        'scripts/',
        'web/',
        '.github/workflows/'
    ]

    missing_files = []
    missing_dirs = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    details = {
        'total_files_checked': len(required_files),
        'total_dirs_checked': len(required_dirs),
        'missing_files': missing_files,
        'missing_dirs': missing_dirs,
        'files_present': len(required_files) - len(missing_files),
        'dirs_present': len(required_dirs) - len(missing_dirs)
    }

    success = len(missing_files) == 0 and len(missing_dirs) == 0
    return success, details


def check_configuration_files() -> Tuple[bool, Dict]:
    """Check configuration files are valid."""
    config_files = [
        'configs/gpu/enhanced_gpu.yaml',
        'configs/cpu/cpu_fallback.yaml',
        'configs/training/base_config.yaml'
    ]

    try:
        import yaml
        yaml_available = True
    except ImportError:
        yaml_available = False

    config_status = {}
    issues = []

    for config_file in config_files:
        config_path = Path(config_file)

        if not config_path.exists():
            config_status[config_file] = 'MISSING'
            issues.append(f"Missing: {config_file}")
            continue

        if not yaml_available:
            config_status[config_file] = 'YAML_UNAVAILABLE'
            continue

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Config must be a dictionary")

            # Check for key sections
            expected_sections = ['model', 'training'] if 'training' in config_file else ['model']
            missing_sections = [section for section in expected_sections if section not in config]

            if missing_sections:
                config_status[config_file] = f'MISSING_SECTIONS: {missing_sections}'
                issues.append(f"{config_file}: missing sections {missing_sections}")
            else:
                config_status[config_file] = 'VALID'

        except Exception as e:
            config_status[config_file] = f'INVALID: {e}'
            issues.append(f"{config_file}: {e}")

    details = {
        'yaml_available': yaml_available,
        'config_status': config_status,
        'issues': issues,
        'total_configs': len(config_files)
    }

    success = len(issues) == 0
    return success, details


def check_test_files() -> Tuple[bool, Dict]:
    """Check that test files exist and have content."""
    test_files = [
        'tests/test_dann_multi.py',
        'tests/test_adapters.py',
        'tests/test_compression_ssl.py',
        'tests/test_gpu_ops.py',
        'tests/test_heads.py'
    ]

    test_status = {}
    issues = []

    for test_file in test_files:
        test_path = Path(test_file)

        if not test_path.exists():
            test_status[test_file] = 'MISSING'
            issues.append(f"Missing: {test_file}")
            continue

        try:
            with open(test_path) as f:
                content = f.read()

            # Check for test content
            has_test_functions = 'def test_' in content
            has_test_classes = 'class Test' in content
            has_imports = 'import' in content

            if not (has_test_functions or has_test_classes):
                test_status[test_file] = 'NO_TESTS'
                issues.append(f"{test_file}: no test functions/classes found")
            elif not has_imports:
                test_status[test_file] = 'NO_IMPORTS'
                issues.append(f"{test_file}: no import statements found")
            else:
                test_status[test_file] = 'VALID'

        except Exception as e:
            test_status[test_file] = f'ERROR: {e}'
            issues.append(f"{test_file}: {e}")

    details = {
        'test_status': test_status,
        'issues': issues,
        'total_tests': len(test_files)
    }

    success = len(issues) == 0
    return success, details


def check_source_files() -> Tuple[bool, Dict]:
    """Check that source files exist and have basic structure."""
    source_files = [
        'src/models/backbone/eeg_transformer.py',
        'src/models/adapters/task_aware.py',
        'src/models/heads/temporal_regression.py',
        'src/models/invariance/dann_multi.py',
        'src/models/compression_ssl/augmentation.py',
        'src/training/trainers/ssl_trainer.py',
        'src/gpu/triton/fused_ops.py',
        'src/gpu/cupy/perceptual_quant.py'
    ]

    source_status = {}
    issues = []

    for source_file in source_files:
        source_path = Path(source_file)

        if not source_path.exists():
            source_status[source_file] = 'MISSING'
            issues.append(f"Missing: {source_file}")
            continue

        try:
            with open(source_path) as f:
                content = f.read()

            # Check for basic Python structure
            has_classes = 'class ' in content
            has_functions = 'def ' in content
            has_imports = 'import ' in content
            has_docstring = '"""' in content or "'''" in content

            if not (has_classes or has_functions):
                source_status[source_file] = 'NO_CODE'
                issues.append(f"{source_file}: no classes or functions found")
            elif not has_imports:
                source_status[source_file] = 'NO_IMPORTS'
                issues.append(f"{source_file}: no import statements found")
            else:
                quality_score = sum([has_classes, has_functions, has_imports, has_docstring])
                source_status[source_file] = f'VALID (quality: {quality_score}/4)'

        except Exception as e:
            source_status[source_file] = f'ERROR: {e}'
            issues.append(f"{source_file}: {e}")

    details = {
        'source_status': source_status,
        'issues': issues,
        'total_sources': len(source_files)
    }

    success = len(issues) == 0
    return success, details


def check_scripts() -> Tuple[bool, Dict]:
    """Check that essential scripts exist."""
    script_files = [
        'scripts/train.py',
        'scripts/inference.py',
        'scripts/bench_inference.py',
        'scripts/validate_repository.py',
        'scripts/health_check.py'
    ]

    script_status = {}
    issues = []

    for script_file in script_files:
        script_path = Path(script_file)

        if not script_path.exists():
            script_status[script_file] = 'MISSING'
            issues.append(f"Missing: {script_file}")
            continue

        try:
            with open(script_path) as f:
                content = f.read()

            # Check for script structure
            has_main = 'if __name__' in content
            has_argparse = 'argparse' in content
            has_imports = 'import ' in content

            if not has_main:
                script_status[script_file] = 'NO_MAIN'
                issues.append(f"{script_file}: no main entry point found")
            elif not has_imports:
                script_status[script_file] = 'NO_IMPORTS'
                issues.append(f"{script_file}: no import statements found")
            else:
                features = sum([has_main, has_argparse, has_imports])
                script_status[script_file] = f'VALID (features: {features}/3)'

        except Exception as e:
            script_status[script_file] = f'ERROR: {e}'
            issues.append(f"{script_file}: {e}")

    details = {
        'script_status': script_status,
        'issues': issues,
        'total_scripts': len(script_files)
    }

    success = len(issues) == 0
    return success, details


def check_documentation() -> Tuple[bool, Dict]:
    """Check documentation files."""
    doc_files = [
        'README.md',
        'web/README.md'
    ]

    doc_status = {}
    issues = []

    for doc_file in doc_files:
        doc_path = Path(doc_file)

        if not doc_path.exists():
            doc_status[doc_file] = 'MISSING'
            issues.append(f"Missing: {doc_file}")
            continue

        try:
            with open(doc_path) as f:
                content = f.read()

            # Check content quality
            has_headers = '#' in content
            has_code_blocks = '```' in content
            is_substantial = len(content) > 500

            if not is_substantial:
                doc_status[doc_file] = 'TOO_SHORT'
                issues.append(f"{doc_file}: content too short")
            elif not has_headers:
                doc_status[doc_file] = 'NO_STRUCTURE'
                issues.append(f"{doc_file}: no headers found")
            else:
                quality = sum([has_headers, has_code_blocks, is_substantial])
                doc_status[doc_file] = f'VALID (quality: {quality}/3)'

        except Exception as e:
            doc_status[doc_file] = f'ERROR: {e}'
            issues.append(f"{doc_file}: {e}")

    details = {
        'doc_status': doc_status,
        'issues': issues,
        'total_docs': len(doc_files)
    }

    success = len(issues) == 0
    return success, details


def run_validation():
    """Run all validation checks."""
    print("🔍 Quick Repository Validation")
    print("=" * 40)

    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Configuration Files", check_configuration_files),
        ("Test Files", check_test_files),
        ("Source Files", check_source_files),
        ("Scripts", check_scripts),
        ("Documentation", check_documentation),
    ]

    results = {
        'timestamp': time.time(),
        'validation_type': 'quick_check',
        'checks': {}
    }

    passed = 0
    total = len(checks)

    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 30)

        try:
            start_time = time.time()
            success, details = check_func()
            elapsed = time.time() - start_time

            if success:
                print(f"✅ PASS ({elapsed:.2f}s)")
                passed += 1
                status = "PASS"
            else:
                print(f"❌ FAIL ({elapsed:.2f}s)")
                status = "FAIL"

                # Show issues
                if 'issues' in details and details['issues']:
                    for issue in details['issues'][:3]:  # Show first 3 issues
                        print(f"   • {issue}")
                    if len(details['issues']) > 3:
                        print(f"   ... and {len(details['issues']) - 3} more")

            results['checks'][check_name] = {
                'status': status,
                'elapsed_time': elapsed,
                'details': details
            }

        except Exception as e:
            print(f"💥 ERROR: {e}")
            results['checks'][check_name] = {
                'status': 'ERROR',
                'error': str(e)
            }

    # Summary
    success_rate = passed / total
    print(f"\n{'='*40}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*40}")
    print(f"Passed: {passed}/{total} ({success_rate:.1%})")

    if success_rate >= 0.8:
        print("🎉 Repository is in GOOD SHAPE!")
        overall_status = "GOOD"
        exit_code = 0
    elif success_rate >= 0.6:
        print("⚠️  Repository has MINOR ISSUES")
        overall_status = "MINOR_ISSUES"
        exit_code = 1
    else:
        print("🚨 Repository has MAJOR ISSUES")
        overall_status = "MAJOR_ISSUES"
        exit_code = 2

    results['summary'] = {
        'overall_status': overall_status,
        'passed': passed,
        'total': total,
        'success_rate': success_rate
    }

    # Save results
    with open('quick_validation.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Results saved to: quick_validation.json")

    return exit_code


if __name__ == "__main__":
    sys.exit(run_validation())
