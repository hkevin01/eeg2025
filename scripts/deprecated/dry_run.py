#!/usr/bin/env python3
"""
Dry-run script for EEG Foundation Challenge 2025.

This script performs a quick test of the data loading pipeline,
generates sample CSVs, and validates against Starter Kit requirements.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import traceback

import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataio.hbn_dataset import HBNDataset, create_hbn_datasets
from dataio.starter_kit import StarterKitDataLoader, OfficialMetrics, SubmissionValidator
from utils.submission import create_starter_kit_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_starter_kit_integration(bids_root: str) -> Dict[str, Any]:
    """Test basic Starter Kit integration."""
    logger.info("Testing Starter Kit integration...")

    results = {
        "starter_kit_loader": False,
        "ccd_labels": False,
        "cbcl_labels": False,
        "official_metrics": False,
        "submission_validator": False
    }

    try:
        # Test StarterKitDataLoader
        starter_kit = StarterKitDataLoader(bids_root)
        results["starter_kit_loader"] = True
        logger.info("✅ StarterKitDataLoader initialized")

        # Test CCD labels loading
        try:
            ccd_labels = starter_kit.load_ccd_labels("train")
            results["ccd_labels"] = len(ccd_labels) > 0
            logger.info(f"✅ CCD labels loaded: {len(ccd_labels)} samples")
        except Exception as e:
            logger.warning(f"⚠️ CCD labels loading failed: {e}")

        # Test CBCL labels loading
        try:
            cbcl_labels = starter_kit.load_cbcl_labels("train")
            results["cbcl_labels"] = len(cbcl_labels) > 0
            logger.info(f"✅ CBCL labels loaded: {len(cbcl_labels)} samples")
        except Exception as e:
            logger.warning(f"⚠️ CBCL labels loading failed: {e}")

        # Test official metrics
        try:
            metrics_dummy = {
                "response_time": np.random.randn(100),
                "success": np.random.rand(100)
            }
            targets_dummy = {
                "response_time": np.random.randn(100),
                "success": (np.random.rand(100) > 0.5).astype(int)
            }

            challenge1_metrics = OfficialMetrics.compute_challenge1_metrics(metrics_dummy, targets_dummy)
            results["official_metrics"] = len(challenge1_metrics) > 0
            logger.info(f"✅ Official metrics computed: {list(challenge1_metrics.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ Official metrics test failed: {e}")

        # Test submission validator
        try:
            validator = SubmissionValidator()
            results["submission_validator"] = True
            logger.info("✅ SubmissionValidator initialized")
        except Exception as e:
            logger.warning(f"⚠️ SubmissionValidator test failed: {e}")

    except Exception as e:
        logger.error(f"❌ Starter Kit integration failed: {e}")
        traceback.print_exc()

    return results


def test_dataset_creation(bids_root: str, task_type: str = "cross_task") -> Dict[str, Any]:
    """Test enhanced dataset creation."""
    logger.info(f"Testing dataset creation for {task_type}...")

    results = {
        "datasets_created": False,
        "data_loading": False,
        "label_extraction": False,
        "batch_structure": {}
    }

    try:
        # Create datasets
        datasets = create_hbn_datasets(
            bids_root=bids_root,
            task_type=task_type,
            window_length=2.0,
            overlap=0.0,
            sample_rate=128,
            use_official_splits=True,
            enable_compression_aug=False
        )

        results["datasets_created"] = len(datasets) > 0
        logger.info(f"✅ Datasets created: {list(datasets.keys())}")

        # Test data loading
        if "train" in datasets:
            train_dataset = datasets["train"]

            if len(train_dataset) > 0:
                # Test single sample
                sample = train_dataset[0]
                results["data_loading"] = True
                results["batch_structure"] = {
                    "keys": list(sample.keys()),
                    "eeg_shape": sample["eeg"].shape if "eeg" in sample else None,
                    "sample_keys": list(sample.keys())
                }
                logger.info(f"✅ Data loading successful. Sample keys: {list(sample.keys())}")

                # Check for task-specific labels
                if task_type == "cross_task":
                    expected_keys = ["response_time_target", "success_target"]
                elif task_type == "psychopathology":
                    expected_keys = ["p_factor_target", "binary_target"]
                else:
                    expected_keys = []

                found_labels = [key for key in expected_keys if key in sample]
                results["label_extraction"] = len(found_labels) > 0
                logger.info(f"✅ Found labels: {found_labels}")

            else:
                logger.warning("⚠️ Empty dataset")

    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        traceback.print_exc()

    return results


def generate_sample_predictions(task_type: str, num_samples: int = 100) -> pd.DataFrame:
    """Generate sample predictions for testing."""
    logger.info(f"Generating {num_samples} sample predictions for {task_type}...")

    # Base metadata
    data = {
        "subject": [f"sub-{i:04d}" for i in range(num_samples)],
        "session": ["ses-1"] * num_samples,
    }

    if task_type == "cross_task":
        # Challenge 1 predictions
        data.update({
            "task": np.random.choice(["Nback", "ASSR", "WM"], num_samples),
            "trial": np.random.randint(1, 100, num_samples),
            "response_time": np.random.uniform(0.3, 2.0, num_samples),
            "success": np.random.uniform(0.0, 1.0, num_samples)
        })

    elif task_type == "psychopathology":
        # Challenge 2 predictions
        data.update({
            "p_factor": np.random.uniform(-2.0, 2.0, num_samples),
            "internalizing": np.random.uniform(-2.0, 2.0, num_samples),
            "externalizing": np.random.uniform(-2.0, 2.0, num_samples),
            "attention": np.random.uniform(-2.0, 2.0, num_samples),
            "binary_label": np.random.uniform(0.0, 1.0, num_samples)
        })

    df = pd.DataFrame(data)
    logger.info(f"✅ Generated predictions with shape {df.shape}")
    return df


def test_submission_creation(task_type: str, output_dir: str) -> Dict[str, Any]:
    """Test submission file creation and validation."""
    logger.info(f"Testing submission creation for {task_type}...")

    results = {
        "csv_creation": False,
        "submission_validation": False,
        "file_path": None
    }

    try:
        # Generate sample predictions
        predictions_df = generate_sample_predictions(task_type)

        # Create submission
        submission_path = create_starter_kit_submission(
            predictions_df=predictions_df,
            task_type=task_type,
            output_dir=output_dir
        )

        results["csv_creation"] = Path(submission_path).exists()
        results["file_path"] = submission_path
        logger.info(f"✅ Submission created: {submission_path}")

        # Validate submission
        try:
            validator = SubmissionValidator()
            validator.validate_submission(submission_path, task_type)
            results["submission_validation"] = True
            logger.info("✅ Submission validation passed")
        except Exception as e:
            logger.warning(f"⚠️ Submission validation failed: {e}")

    except Exception as e:
        logger.error(f"❌ Submission creation failed: {e}")
        traceback.print_exc()

    return results


def test_official_metrics(task_type: str) -> Dict[str, Any]:
    """Test official metrics computation."""
    logger.info(f"Testing official metrics for {task_type}...")

    results = {
        "metrics_computed": False,
        "metrics": {}
    }

    try:
        if task_type == "cross_task":
            # Challenge 1 test data
            predictions = {
                "response_time": np.random.uniform(0.3, 2.0, 1000),
                "success": np.random.uniform(0.0, 1.0, 1000)
            }
            targets = {
                "response_time": np.random.uniform(0.3, 2.0, 1000),
                "success": (np.random.uniform(0.0, 1.0, 1000) > 0.5).astype(int)
            }

            metrics = OfficialMetrics.compute_challenge1_metrics(predictions, targets)

        elif task_type == "psychopathology":
            # Challenge 2 test data
            predictions = {
                "p_factor": np.random.uniform(-2.0, 2.0, 1000),
                "internalizing": np.random.uniform(-2.0, 2.0, 1000),
                "externalizing": np.random.uniform(-2.0, 2.0, 1000),
                "attention": np.random.uniform(-2.0, 2.0, 1000),
                "binary_label": np.random.uniform(0.0, 1.0, 1000)
            }
            targets = {
                "p_factor": np.random.uniform(-2.0, 2.0, 1000),
                "internalizing": np.random.uniform(-2.0, 2.0, 1000),
                "externalizing": np.random.uniform(-2.0, 2.0, 1000),
                "attention": np.random.uniform(-2.0, 2.0, 1000),
                "binary_label": (np.random.uniform(0.0, 1.0, 1000) > 0.5).astype(int)
            }

            metrics = OfficialMetrics.compute_challenge2_metrics(predictions, targets)

        else:
            metrics = {}

        results["metrics_computed"] = len(metrics) > 0
        results["metrics"] = metrics

        logger.info(f"✅ Metrics computed: {list(metrics.keys())}")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    except Exception as e:
        logger.error(f"❌ Metrics computation failed: {e}")
        traceback.print_exc()

    return results


def run_dry_run(bids_root: str, output_dir: str = "dry_run_results") -> Dict[str, Any]:
    """Run complete dry-run test."""
    logger.info("="*70)
    logger.info("EEG FOUNDATION CHALLENGE 2025 - DRY RUN")
    logger.info("="*70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Test Starter Kit integration
    logger.info("\n" + "="*50)
    logger.info("1. TESTING STARTER KIT INTEGRATION")
    logger.info("="*50)
    results["starter_kit"] = test_starter_kit_integration(bids_root)

    # 2. Test dataset creation for both tasks
    for task_type in ["cross_task", "psychopathology"]:
        logger.info(f"\n" + "="*50)
        logger.info(f"2. TESTING DATASET CREATION - {task_type.upper()}")
        logger.info("="*50)
        results[f"dataset_{task_type}"] = test_dataset_creation(bids_root, task_type)

    # 3. Test submission creation
    for task_type in ["cross_task", "psychopathology"]:
        logger.info(f"\n" + "="*50)
        logger.info(f"3. TESTING SUBMISSION CREATION - {task_type.upper()}")
        logger.info("="*50)
        task_output_dir = output_path / task_type
        task_output_dir.mkdir(exist_ok=True)
        results[f"submission_{task_type}"] = test_submission_creation(task_type, str(task_output_dir))

    # 4. Test official metrics
    for task_type in ["cross_task", "psychopathology"]:
        logger.info(f"\n" + "="*50)
        logger.info(f"4. TESTING OFFICIAL METRICS - {task_type.upper()}")
        logger.info("="*50)
        results[f"metrics_{task_type}"] = test_official_metrics(task_type)

    # 5. Summary
    logger.info("\n" + "="*70)
    logger.info("DRY RUN SUMMARY")
    logger.info("="*70)

    total_tests = 0
    passed_tests = 0

    for category, test_results in results.items():
        if isinstance(test_results, dict):
            for test_name, test_result in test_results.items():
                total_tests += 1
                if test_result:
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                logger.info(f"{category}.{test_name}: {status}")

    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    logger.info(f"\nOverall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1%})")

    if success_rate > 0.8:
        logger.info("🎉 DRY RUN SUCCESSFUL - Ready for training!")
    elif success_rate > 0.5:
        logger.info("⚠️ DRY RUN PARTIAL - Some issues need attention")
    else:
        logger.info("❌ DRY RUN FAILED - Major issues need resolution")

    # Save results
    import json
    results_path = output_path / "dry_run_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            return obj

        json_results = convert_numpy(results)
        json.dump(json_results, f, indent=2)

    logger.info(f"\n📁 Results saved to: {results_path}")

    return results


def main():
    """Main dry-run function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run dry-run test for EEG Foundation Challenge")
    parser.add_argument("--bids-root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument("--output-dir", type=str, default="dry_run_results", help="Output directory")

    args = parser.parse_args()

    # Run dry-run
    results = run_dry_run(args.bids_root, args.output_dir)

    return results


if __name__ == "__main__":
    main()
