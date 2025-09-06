"""
Enhanced starter kit test suite for EEG2025 Challenge submissions.

This test suite validates submission format compliance, model integration,
and end-to-end pipeline functionality for both cross-task transfer and
psychopathology prediction tracks.

Usage:
    python test_enhanced_starter_kit.py --submission_dir ./my_submission
    python test_enhanced_starter_kit.py --track psychopathology --validate_only
"""

import os
import sys
import argparse
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
import pandas as pd
from datetime import datetime

from evaluation.submission import (
    SubmissionValidator, SubmissionPackager, SubmissionMetadata,
    CrossTaskPrediction, PsychopathologyPrediction,
    create_sample_predictions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class EnhancedStarterKitTester:
    """Comprehensive test suite for EEG2025 challenge submissions."""

    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize test suite."""
        self.test_dir = test_dir or Path("test_submission_validation")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.validator = SubmissionValidator()
        self.test_results = {}
        self.test_errors = []

        logger.info(f"Initialized enhanced starter kit tester at {self.test_dir}")

    def test_submission_schema_compliance(self) -> bool:
        """Test CSV schema compliance for both tracks."""
        logger.info("Testing submission schema compliance...")

        try:
            # Create sample predictions
            ct_preds, psych_preds = create_sample_predictions(20)

            # Create test packager
            packager = SubmissionPackager(self.test_dir, "test_team")

            # Export and validate cross-task predictions
            ct_file = packager.export_cross_task_predictions(ct_preds, "test_cross_task.csv")
            ct_valid = self.validator.validate_cross_task_csv(ct_file)

            if not ct_valid:
                self.test_errors.append(f"Cross-task CSV validation failed: {self.validator.get_validation_report()}")
                return False

            # Export and validate psychopathology predictions
            psych_file = packager.export_psychopathology_predictions(psych_preds, "test_psych.csv")
            psych_valid = self.validator.validate_psychopathology_csv(psych_file)

            if not psych_valid:
                self.test_errors.append(f"Psychopathology CSV validation failed: {self.validator.get_validation_report()}")
                return False

            logger.info("✓ Schema compliance tests passed")
            return True

        except Exception as e:
            self.test_errors.append(f"Schema compliance test error: {e}")
            logger.error(f"Schema compliance test failed: {e}")
            return False

    def test_data_integrity_validation(self) -> bool:
        """Test data integrity validation (duplicates, missing values, ranges)."""
        logger.info("Testing data integrity validation...")

        try:
            # Test duplicate detection
            duplicate_ct_data = [
                CrossTaskPrediction("sub-001", "ses-1", "task1", 0.5, 0.8),
                CrossTaskPrediction("sub-001", "ses-1", "task1", 0.6, 0.9),  # Duplicate
                CrossTaskPrediction("sub-002", "ses-1", "task1", 0.7, 0.85)
            ]

            packager = SubmissionPackager(self.test_dir, "integrity_test")
            duplicate_file = packager.export_cross_task_predictions(
                duplicate_ct_data, "test_duplicates.csv"
            )

            # This should fail validation
            is_valid = self.validator.validate_cross_task_csv(duplicate_file)
            if is_valid:
                self.test_errors.append("Duplicate detection failed - duplicates not caught")
                return False

            # Test invalid ranges
            extreme_psych_data = [
                PsychopathologyPrediction("sub-001", 10.0, -10.0, 15.0, -8.0),  # Extreme values
                PsychopathologyPrediction("sub-002", 0.5, 0.3, -0.2, 0.8)
            ]

            extreme_file = packager.export_psychopathology_predictions(
                extreme_psych_data, "test_extremes.csv"
            )

            # Should validate but generate warnings
            is_valid = self.validator.validate_psychopathology_csv(extreme_file)
            report = self.validator.get_validation_report()

            if not is_valid or len(report['warnings']) == 0:
                self.test_errors.append("Range validation failed - extreme values not flagged")
                return False

            logger.info("✓ Data integrity validation tests passed")
            return True

        except Exception as e:
            self.test_errors.append(f"Data integrity test error: {e}")
            logger.error(f"Data integrity test failed: {e}")
            return False

    def test_submission_packaging(self) -> bool:
        """Test complete submission packaging workflow."""
        logger.info("Testing submission packaging...")

        try:
            # Create realistic predictions
            ct_preds, psych_preds = create_sample_predictions(50)

            # Create metadata
            metadata = SubmissionMetadata(
                team_name="test_team",
                submission_id="test_sub_001",
                timestamp=datetime.now().isoformat(),
                challenge_track="both",
                model_description="Test CNN with DANN",
                training_duration_hours=5.5,
                num_parameters=1_200_000,
                cross_validation_folds=5,
                best_validation_score=0.68
            )

            # Package submission
            packager = SubmissionPackager(self.test_dir, "packaging_test")
            archive_path, validation_results = packager.package_full_submission(
                cross_task_predictions=ct_preds,
                psychopathology_predictions=psych_preds,
                metadata=metadata
            )

            # Verify archive was created
            if not archive_path.exists():
                self.test_errors.append("Submission archive not created")
                return False

            # Check validation results
            for filename, result in validation_results.items():
                if not result['valid']:
                    self.test_errors.append(f"File {filename} failed validation: {result['report']['errors']}")
                    return False

            # Verify archive contents
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                archive_files = zipf.namelist()

                expected_files = {
                    'cross_task_submission.csv',
                    'psychopathology_submission.csv',
                    'submission_manifest.json'
                }

                missing_files = expected_files - set(archive_files)
                if missing_files:
                    self.test_errors.append(f"Missing files in archive: {missing_files}")
                    return False

            logger.info("✓ Submission packaging tests passed")
            return True

        except Exception as e:
            self.test_errors.append(f"Submission packaging test error: {e}")
            logger.error(f"Submission packaging test failed: {e}")
            return False

    def test_model_integration_apis(self) -> bool:
        """Test model integration APIs and interfaces."""
        logger.info("Testing model integration APIs...")

        try:
            # Test DANN model creation
            from models.invariance.dann import create_dann_model, GRLScheduler
            from unittest.mock import MagicMock

            # Create mock components
            backbone = MagicMock()
            backbone.return_value = torch.randn(4, 100, 128)

            task_head = MagicMock()
            task_head.return_value = torch.randn(4, 2)

            # Test DANN model creation
            dann_model = create_dann_model(
                backbone=backbone,
                task_head=task_head,
                num_domains=3,
                lambda_schedule_config={
                    "strategy": "linear_warmup",
                    "initial_lambda": 0.0,
                    "final_lambda": 0.2,
                    "warmup_steps": 1000
                }
            )

            # Test forward pass
            test_input = torch.randn(4, 64, 1000)
            outputs = dann_model(test_input)

            required_outputs = {'task_output', 'domain_output', 'lambda'}
            if not required_outputs.issubset(outputs.keys()):
                self.test_errors.append(f"DANN model missing required outputs: {required_outputs - outputs.keys()}")
                return False

            # Test lambda scheduling
            initial_lambda = outputs['lambda']
            for _ in range(10):
                outputs = dann_model(test_input)
            final_lambda = outputs['lambda']

            if final_lambda <= initial_lambda:
                self.test_errors.append("Lambda scheduling not working - lambda not increasing")
                return False

            # Test training configuration loading
            import yaml

            config_path = Path(__file__).parent / "configs" / "train_psych.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Check for required configuration sections
                required_sections = {
                    'model', 'training', 'dann_schedule', 'uncertainty_weighting'
                }

                if not required_sections.issubset(config.keys()):
                    self.test_errors.append(f"Missing config sections: {required_sections - config.keys()}")
                    return False

            logger.info("✓ Model integration API tests passed")
            return True

        except Exception as e:
            self.test_errors.append(f"Model integration test error: {e}")
            logger.error(f"Model integration test failed: {e}")
            traceback.print_exc()
            return False

    def test_reproducibility_features(self) -> bool:
        """Test reproducibility and environment capture features."""
        logger.info("Testing reproducibility features...")

        try:
            # Test that we can capture environment info
            import platform
            import torch

            env_info = {
                'python_version': platform.python_version(),
                'pytorch_version': torch.__version__,
                'platform': platform.platform(),
                'cuda_available': torch.cuda.is_available()
            }

            # Test deterministic operations
            torch.manual_seed(42)
            np.random.seed(42)

            # Generate some deterministic outputs
            x1 = torch.randn(10, 20)
            y1 = torch.mean(x1)

            # Reset seeds and generate again
            torch.manual_seed(42)
            np.random.seed(42)

            x2 = torch.randn(10, 20)
            y2 = torch.mean(x2)

            # Should be identical
            if not torch.allclose(y1, y2):
                self.test_errors.append("Reproducibility test failed - outputs not deterministic")
                return False

            # Test that we can save environment manifest
            env_manifest = {
                'environment': env_info,
                'seeds': {'torch': 42, 'numpy': 42},
                'timestamp': datetime.now().isoformat()
            }

            manifest_path = self.test_dir / "test_env_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(env_manifest, f, indent=2)

            # Verify we can read it back
            with open(manifest_path, 'r') as f:
                loaded_manifest = json.load(f)

            if loaded_manifest['environment']['python_version'] != platform.python_version():
                self.test_errors.append("Environment manifest save/load failed")
                return False

            logger.info("✓ Reproducibility feature tests passed")
            return True

        except Exception as e:
            self.test_errors.append(f"Reproducibility test error: {e}")
            logger.error(f"Reproducibility test failed: {e}")
            return False

    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarking capabilities."""
        logger.info("Testing performance benchmarks...")

        try:
            # Test basic timing capabilities
            import time
            from models.invariance.dann import GRLScheduler

            # Test scheduler performance
            scheduler = GRLScheduler(
                strategy="linear_warmup",
                initial_lambda=0.0,
                final_lambda=0.2,
                warmup_steps=1000
            )

            start_time = time.time()

            # Run many scheduler steps
            for _ in range(1000):
                scheduler.step()

            end_time = time.time()
            duration = end_time - start_time

            # Should be fast (< 1 second for 1000 steps)
            if duration > 1.0:
                self.test_errors.append(f"Scheduler performance too slow: {duration:.3f}s for 1000 steps")
                return False

            # Test memory usage tracking
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()

                # Allocate some tensors
                large_tensor = torch.randn(1000, 1000, device='cuda')
                peak_memory = torch.cuda.memory_allocated()

                # Clean up
                del large_tensor
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()

                # Memory should have increased then decreased
                if peak_memory <= initial_memory:
                    self.test_errors.append("CUDA memory tracking not working")
                    return False

            # Test that we can measure inference time
            mock_input = torch.randn(8, 64, 1000)
            mock_model = torch.nn.Conv1d(64, 32, 3)

            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = mock_model(mock_input)
            end_time = time.time()

            inference_time = (end_time - start_time) / 10  # Average per inference

            # Should be reasonably fast
            if inference_time > 1.0:
                self.test_errors.append(f"Inference time too slow: {inference_time:.3f}s")
                return False

            logger.info("✓ Performance benchmark tests passed")
            return True

        except Exception as e:
            self.test_errors.append(f"Performance benchmark test error: {e}")
            logger.error(f"Performance benchmark test failed: {e}")
            return False

    def test_end_to_end_pipeline(self) -> bool:
        """Test complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")

        try:
            # Create a minimal end-to-end test
            from evaluation.submission import create_sample_predictions

            # Generate predictions
            ct_preds, psych_preds = create_sample_predictions(10)

            # Create submission
            packager = SubmissionPackager(self.test_dir, "e2e_test")

            metadata = SubmissionMetadata(
                team_name="e2e_test",
                submission_id="e2e_001",
                timestamp=datetime.now().isoformat(),
                challenge_track="both",
                model_description="End-to-end test submission",
                training_duration_hours=1.0,
                num_parameters=100_000,
                cross_validation_folds=3
            )

            # Package complete submission
            archive_path, validation_results = packager.package_full_submission(
                cross_task_predictions=ct_preds,
                psychopathology_predictions=psych_preds,
                metadata=metadata
            )

            # Verify submission
            all_valid = all(result['valid'] for result in validation_results.values())
            if not all_valid:
                self.test_errors.append("End-to-end submission validation failed")
                return False

            # Test that archive is reasonable size
            archive_size = archive_path.stat().st_size
            if archive_size < 1000 or archive_size > 10_000_000:  # 1KB to 10MB
                self.test_errors.append(f"Archive size unreasonable: {archive_size} bytes")
                return False

            logger.info("✓ End-to-end pipeline tests passed")
            return True

        except Exception as e:
            self.test_errors.append(f"End-to-end pipeline test error: {e}")
            logger.error(f"End-to-end pipeline test failed: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all test suites."""
        logger.info("Running all enhanced starter kit tests...")

        test_suites = [
            ("Schema Compliance", self.test_submission_schema_compliance),
            ("Data Integrity", self.test_data_integrity_validation),
            ("Submission Packaging", self.test_submission_packaging),
            ("Model Integration", self.test_model_integration_apis),
            ("Reproducibility", self.test_reproducibility_features),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("End-to-End Pipeline", self.test_end_to_end_pipeline)
        ]

        results = {}

        for test_name, test_func in test_suites:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {test_name} Tests")
            logger.info(f"{'='*50}")

            try:
                success = test_func()
                results[test_name] = success

                if success:
                    logger.info(f"✅ {test_name} tests PASSED")
                else:
                    logger.error(f"❌ {test_name} tests FAILED")

            except Exception as e:
                results[test_name] = False
                self.test_errors.append(f"{test_name} test suite error: {e}")
                logger.error(f"❌ {test_name} tests FAILED with exception: {e}")

        return results

    def generate_test_report(self, results: Dict[str, bool]) -> str:
        """Generate comprehensive test report."""
        total_tests = len(results)
        passed_tests = sum(results.values())
        failed_tests = total_tests - passed_tests

        report = f"""
Enhanced Starter Kit Test Report
{'='*60}

Summary:
  Total Test Suites: {total_tests}
  Passed: {passed_tests} ✅
  Failed: {failed_tests} ❌
  Success Rate: {passed_tests/total_tests*100:.1f}%

Test Results:
"""

        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            report += f"  {test_name:<25} {status}\n"

        if self.test_errors:
            report += f"\nErrors Encountered ({len(self.test_errors)}):\n"
            for i, error in enumerate(self.test_errors, 1):
                report += f"  {i}. {error}\n"

        report += f"""
Test Environment:
  Python: {sys.version.split()[0]}
  PyTorch: {torch.__version__}
  Platform: {sys.platform}
  Test Directory: {self.test_dir}

"""

        return report


def validate_external_submission(submission_path: Path) -> Dict[str, Any]:
    """Validate an external submission directory or archive."""
    logger.info(f"Validating external submission: {submission_path}")

    validator = SubmissionValidator()
    results = {}

    if submission_path.is_file() and submission_path.suffix == '.zip':
        # Extract and validate zip archive
        import zipfile

        extract_dir = submission_path.parent / f"{submission_path.stem}_extracted"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(submission_path, 'r') as zipf:
            zipf.extractall(extract_dir)

        submission_path = extract_dir

    # Find CSV files
    csv_files = list(submission_path.glob("*.csv"))

    if not csv_files:
        results['error'] = "No CSV files found in submission"
        return results

    # Validate each CSV
    for csv_file in csv_files:
        filename = csv_file.name

        if 'cross_task' in filename.lower():
            is_valid = validator.validate_cross_task_csv(csv_file)
            results[filename] = {
                'valid': is_valid,
                'type': 'cross_task',
                'report': validator.get_validation_report()
            }
        elif 'psychopathology' in filename.lower() or 'psych' in filename.lower():
            is_valid = validator.validate_psychopathology_csv(csv_file)
            results[filename] = {
                'valid': is_valid,
                'type': 'psychopathology',
                'report': validator.get_validation_report()
            }
        else:
            logger.warning(f"Unknown CSV file type: {filename}")

    return results


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Enhanced Starter Kit Test Suite")

    parser.add_argument(
        '--submission_dir',
        type=Path,
        help="Path to submission directory or zip file to validate"
    )

    parser.add_argument(
        '--track',
        choices=['cross_task', 'psychopathology', 'both'],
        default='both',
        help="Challenge track to test"
    )

    parser.add_argument(
        '--validate_only',
        action='store_true',
        help="Only validate submission, don't run full test suite"
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path("test_results"),
        help="Directory for test outputs"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate external submission if provided
    if args.submission_dir:
        if not args.submission_dir.exists():
            logger.error(f"Submission path does not exist: {args.submission_dir}")
            return 1

        validation_results = validate_external_submission(args.submission_dir)

        print(f"\nSubmission Validation Results for {args.submission_dir}:")
        print("="*60)

        for filename, result in validation_results.items():
            if 'error' in result:
                print(f"❌ ERROR: {result}")
                return 1

            status = "✅ VALID" if result['valid'] else "❌ INVALID"
            print(f"{filename}: {status} ({result['type']})")

            if result['report']['errors']:
                print(f"  Errors: {result['report']['errors']}")
            if result['report']['warnings']:
                print(f"  Warnings: {result['report']['warnings']}")

        if args.validate_only:
            return 0

    # Run full test suite
    tester = EnhancedStarterKitTester(args.output_dir)
    results = tester.run_all_tests()

    # Generate and save report
    report = tester.generate_test_report(results)

    report_path = args.output_dir / "test_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"Full test report saved to: {report_path}")

    # Return appropriate exit code
    if all(results.values()):
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
