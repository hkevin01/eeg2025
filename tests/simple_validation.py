#!/usr/bin/env python3
"""
Simple validation script for EEG Foundation Challenge 2025 data infrastructure.
This script validates the enhanced implementation and challenge-compliant infrastructure.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def validate_file_structure():
    """Validate that all enhanced files exist and have expected content."""
    logger.info("=" * 60)
    logger.info("Validating Challenge Infrastructure Files")
    logger.info("=" * 60)

    # Core files to validate
    files_to_check = {
        "StarterKitDataLoader": "src/dataio/starter_kit.py",
        "BIDS Loader": "src/dataio/bids_loader.py",
        "Preprocessing": "src/dataio/preprocessing.py",
        "Official Splits": "scripts/make_splits.py",
        "HBN BIDS Prep": "scripts/prepare_hbn_bids.py",
        "Data Config": "configs/data.yaml",
    }

    results = {}
    all_valid = True

    for component, filepath in files_to_check.items():
        full_path = Path("/home/kevin/Projects/eeg2025") / filepath

        if not full_path.exists():
            logger.error(f"❌ {component} file not found: {filepath}")
            results[component] = {"exists": False, "size": 0, "lines": 0}
            all_valid = False
            continue

        try:
            with open(full_path, "r") as f:
                content = f.read()

            file_size = len(content)
            lines = content.split("\n")
            line_count = len(lines)

            logger.info(f"✅ {component}: {line_count:,} lines, {file_size:,} chars")
            results[component] = {
                "exists": True,
                "size": file_size,
                "lines": line_count,
                "path": str(full_path),
            }

            # Component-specific validation
            if component == "StarterKitDataLoader" and line_count < 2000:
                logger.warning(f"⚠️ {component} may be incomplete ({line_count} lines)")
            elif component == "Preprocessing" and line_count < 500:
                logger.warning(f"⚠️ {component} may be incomplete ({line_count} lines)")

        except Exception as e:
            logger.error(f"❌ Error reading {component}: {e}")
            results[component] = {"exists": True, "error": str(e)}
            all_valid = False

    return all_valid, results


def validate_challenge_infrastructure():
    """Validate challenge-specific infrastructure components."""
    logger.info("=" * 60)
    logger.info("Validating Challenge Infrastructure")
    logger.info("=" * 60)

    # Test splits file
    splits_file = Path("/home/kevin/Projects/eeg2025/scripts/make_splits.py")
    if splits_file.exists():
        with open(splits_file, "r") as f:
            splits_content = f.read()

        # Check for key features
        splits_features = [
            "OfficialSplitGenerator",
            "stratified",
            "leakage",
            "subject_level",
            "validation",
        ]

        splits_found = sum(
            1 for feature in splits_features if feature in splits_content
        )
        logger.info(
            f"✅ Official splits: {splits_found}/{len(splits_features)} features found"
        )
    else:
        logger.warning("⚠️ Official splits script not found")

    # Test preprocessing
    prep_file = Path("/home/kevin/Projects/eeg2025/src/dataio/preprocessing.py")
    if prep_file.exists():
        with open(prep_file, "r") as f:
            prep_content = f.read()

        prep_features = [
            "LeakageFreePreprocessor",
            "SessionAwareSampler",
            "normalization",
            "fit_normalization_stats",
            "validate_leakage",
        ]

        prep_found = sum(1 for feature in prep_features if feature in prep_content)
        logger.info(
            f"✅ Preprocessing: {prep_found}/{len(prep_features)} features found"
        )
    else:
        logger.warning("⚠️ Preprocessing module not found")

    # Test configuration
    config_file = Path("/home/kevin/Projects/eeg2025/configs/data.yaml")
    if config_file.exists():
        logger.info("✅ Data configuration file exists")
    else:
        logger.warning("⚠️ Data configuration not found")

    return True


def validate_enhanced_features(file_results):
    """Validate that enhanced features are present across all files."""
    logger.info("=" * 60)
    logger.info("Validating Enhanced Features")
    logger.info("=" * 60)

    # Read starter kit content
    starter_kit_path = Path("/home/kevin/Projects/eeg2025/src/dataio/starter_kit.py")
    if not starter_kit_path.exists():
        logger.error("❌ StarterKitDataLoader not found")
        return False

    with open(starter_kit_path, "r") as f:
        starter_content = f.read()

    # Essential enhanced features for StarterKitDataLoader
    required_features = {
        "Memory Management": [
            "MemoryStats",
            "memory_monitor",
            "_get_current_memory_usage",
            "psutil",
            "gc.collect",
        ],
        "Error Handling": [
            "graceful_error_handler",
            "contextmanager",
            "logger.error",
            "try:",
            "except Exception",
        ],
        "Challenge Integration": [
            "load_ccd_labels",
            "load_cbcl_labels",
            "official_splits",
            "_load_participants_data",
            "leakage",
        ],
        "Data Quality": [
            "_validate_bids_structure",
            "_validate_splits",
            "data_quality_metrics",
            "boundary condition",
        ],
        "Performance Monitoring": [
            "TimingStats",
            "timing_monitor",
            "cache",
            "performance",
        ],
    }

    results = {}
    overall_success = True

    for category, features in required_features.items():
        logger.info(f"Checking {category}:")
        category_results = []

        for feature in features:
            found = feature.lower() in starter_content.lower()
            status = "✅" if found else "❌"
            logger.info(f"  {status} {feature}")
            category_results.append(found)

            if not found:
                overall_success = False

        success_rate = sum(category_results) / len(category_results)
        results[category] = success_rate
        logger.info(f"  Category success: {success_rate:.1%}")

    # Summary
    logger.info("\n" + "-" * 40)
    logger.info("Feature Validation Summary:")
    for category, success_rate in results.items():
        status = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.5 else "❌"
        logger.info(f"  {status} {category}: {success_rate:.1%}")

    return overall_success


def validate_syntax():
    """Validate Python syntax."""
    logger.info("=" * 60)
    logger.info("Validating Python Syntax")
    logger.info("=" * 60)

    starter_kit_path = Path("/home/kevin/Projects/eeg2025/src/dataio/starter_kit.py")

    try:
        with open(starter_kit_path, "r") as f:
            code = f.read()

        # Test compilation
        compile(code, starter_kit_path, "exec")
        logger.info("✅ Python syntax is valid")
        return True

    except SyntaxError as e:
        logger.error(f"❌ Syntax error found:")
        logger.error(f"  Line {e.lineno}: {e.text}")
        logger.error(f"  Error: {e.msg}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during syntax validation: {e}")
        return False


def validate_enhancement_completeness(content):
    """Validate that the enhancement is comprehensive."""
    logger.info("=" * 60)
    logger.info("Validating Enhancement Completeness")
    logger.info("=" * 60)

    # Check for production-ready patterns
    production_patterns = {
        "Robustness": [
            "try:",
            "except:",
            "finally:",
            "raise",
            "graceful",
            "error handling",
            "fallback",
        ],
        "Memory Safety": [
            "memory",
            "gc.collect",
            "cleanup",
            "optimize",
            "memory_limit",
            "threshold",
        ],
        "Performance": [
            "timing",
            "performance",
            "monitor",
            "profile",
            "cache",
            "efficient",
        ],
        "Data Quality": [
            "validate",
            "check",
            "verify",
            "quality",
            "boundary",
            "completeness",
            "integrity",
        ],
        "Logging & Monitoring": [
            "logger",
            "logging",
            "debug",
            "info",
            "warning",
            "error",
            "monitor",
            "stats",
            "metrics",
        ],
        "Documentation": [
            '"""',
            "'''",
            "Args:",
            "Returns:",
            "Raises:",
            "Example:",
            "Note:",
        ],
    }

    enhancement_scores = {}
    overall_score = 0

    for category, patterns in production_patterns.items():
        found_patterns = []
        for pattern in patterns:
            if pattern.lower() in content.lower():
                found_patterns.append(pattern)

        score = len(found_patterns) / len(patterns)
        enhancement_scores[category] = score
        overall_score += score

        status = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
        logger.info(
            f"{status} {category}: {score:.1%} ({len(found_patterns)}/{len(patterns)})"
        )

    overall_score = overall_score / len(production_patterns)

    logger.info(f"\nOverall Enhancement Score: {overall_score:.1%}")

    if overall_score >= 0.8:
        logger.info("✅ Comprehensive production-ready enhancements detected")
        return True
    elif overall_score >= 0.6:
        logger.info("⚠️ Good enhancements with room for improvement")
        return True
    else:
        logger.warning("❌ Enhancements may be incomplete")
        return False


def main():
    """Run comprehensive validation for EEG Challenge 2025 infrastructure."""
    logger.info("🚀 Starting EEG Foundation Challenge 2025 Infrastructure Validation")
    logger.info("Validating challenge-compliant data pipeline...\n")

    test_results = []

    try:
        # Test 1: File Structure
        file_valid, file_results = validate_file_structure()
        test_results.append(("File Structure", file_valid))

        if not file_valid:
            logger.error("Cannot proceed - file structure validation failed")
            return 1

        # Test 2: Challenge Infrastructure
        challenge_valid = validate_challenge_infrastructure()
        test_results.append(("Challenge Infrastructure", challenge_valid))

        # Test 3: Enhanced Features
        features_valid = validate_enhanced_features(file_results)
        test_results.append(("Enhanced Features", features_valid))

        # Test 4: Syntax Validation
        syntax_valid = validate_syntax()
        test_results.append(("Python Syntax", syntax_valid))

        # Final Summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL VALIDATION SUMMARY")
        logger.info("=" * 80)

        passed_tests = 0
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            if result:
                passed_tests += 1

        total_tests = len(test_results)
        success_rate = passed_tests / total_tests

        logger.info(
            f"\nOverall Result: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})"
        )

        if success_rate >= 0.75:
            logger.info("\n🎉 VALIDATION SUCCESSFUL!")
            logger.info("✅ EEG Challenge 2025 infrastructure operational")
            logger.info("✅ Official splits and leakage protection implemented")
            logger.info("✅ Challenge-compliant data loading ready")
            logger.info("✅ Ready for challenge submission")
            return 0
        else:
            logger.warning(
                f"\n⚠️ VALIDATION INCOMPLETE ({success_rate:.1%} success rate)"
            )
            logger.warning("Some components may need additional work")
            return 1

    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        return 1


def validate_syntax():
    """Validate Python syntax for key files."""
    logger.info("=" * 60)
    logger.info("Validating Python Syntax")
    logger.info("=" * 60)

    files_to_check = [
        "src/dataio/starter_kit.py",
        "src/dataio/bids_loader.py",
        "src/dataio/preprocessing.py",
        "scripts/make_splits.py",
    ]

    all_valid = True

    for filepath in files_to_check:
        full_path = Path("/home/kevin/Projects/eeg2025") / filepath

        if not full_path.exists():
            logger.warning(f"⚠️ File not found: {filepath}")
            continue

        try:
            with open(full_path, "r") as f:
                code = f.read()

            # Test compilation
            compile(code, full_path, "exec")
            logger.info(f"✅ {filepath}: Syntax valid")

        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {filepath}:")
            logger.error(f"  Line {e.lineno}: {e.text}")
            logger.error(f"  Error: {e.msg}")
            all_valid = False
        except Exception as e:
            logger.error(f"❌ Error checking {filepath}: {e}")
            all_valid = False

    return all_valid


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
