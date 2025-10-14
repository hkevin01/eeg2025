#!/usr/bin/env python3
"""
Cross-Site Validation for EEG Foundation Challenge

Implements leave-one-site-out and cross-site validation to ensure
model generalization across different recording sites/equipment.

For HBN dataset, sites can be inferred from:
- Release numbers (R1, R2, R3, etc.)
- Recording dates
- Equipment metadata

Usage:
    python scripts/cross_site_validation.py --model baseline --challenge 1
    python scripts/cross_site_validation.py --model foundation --challenge 2 --cv-folds 5
"""

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Site Assignment
# ============================================================================


def assign_sites_from_metadata(bids_root: Path, subjects: List[str]) -> Dict[str, str]:
    """
    Assign site/group labels to subjects based on metadata.

    For HBN:
    - Use release_number as proxy for site/batch
    - Release numbers indicate different data collection waves

    Args:
        bids_root: BIDS dataset root
        subjects: List of subject IDs

    Returns:
        Dictionary mapping subject_id -> site_label
    """
    # Load participants data
    participants_file = bids_root / "participants.tsv"
    participants = pd.read_csv(participants_file, sep="\t")

    site_map = {}

    for subject in subjects:
        subject_id = f"sub-{subject}"
        subject_row = participants[participants["participant_id"] == subject_id]

        if len(subject_row) > 0:
            # Use release_number as site identifier
            release = subject_row["release_number"].values[0]
            site_map[subject] = str(release)
        else:
            # Default site if not found
            site_map[subject] = "unknown"
            logger.warning(f"Subject {subject} not found in participants.tsv")

    # Log site distribution
    site_counts = defaultdict(int)
    for site in site_map.values():
        site_counts[site] += 1

    logger.info(f"Site distribution:")
    for site, count in sorted(site_counts.items()):
        logger.info(f"  {site}: {count} subjects")

    return site_map


# ============================================================================
# Cross-Site Validation Strategies
# ============================================================================


def leave_one_site_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    model_fn,
    task_type: str = "regression",
) -> Dict[str, Any]:
    """
    Leave-One-Site-Out Cross-Validation.

    Train on all sites except one, test on the held-out site.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        sites: Site labels (n_samples,)
        model_fn: Function that returns a model instance
        task_type: 'classification' or 'regression'

    Returns:
        Results dictionary with per-site and overall metrics
    """
    logo = LeaveOneGroupOut()
    unique_sites = np.unique(sites)

    logger.info(f"Running Leave-One-Site-Out CV with {len(unique_sites)} sites...")

    results = {
        "strategy": "leave_one_site_out",
        "n_sites": len(unique_sites),
        "task_type": task_type,
        "per_site": {},
        "overall": {},
    }

    all_predictions = []
    all_true = []

    for train_idx, test_idx in logo.split(X, y, groups=sites):
        test_site = sites[test_idx[0]]
        logger.info(f"Testing on site: {test_site}")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model = model_fn()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        all_predictions.extend(y_pred)
        all_true.extend(y_test)

        # Evaluate
        site_results = {}
        if task_type == "classification":
            site_results["accuracy"] = np.mean(y_pred == y_test)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                site_results["auroc"] = roc_auc_score(y_test, y_pred_proba)
        else:
            site_results["mse"] = mean_squared_error(y_test, y_pred)
            site_results["rmse"] = np.sqrt(site_results["mse"])
            site_results["r2"] = r2_score(y_test, y_pred)
            site_results["pearson_r"] = np.corrcoef(y_test, y_pred)[0, 1]

        results["per_site"][test_site] = site_results
        logger.info(f"Site {test_site} results: {site_results}")

    # Overall results
    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)

    if task_type == "classification":
        results["overall"]["accuracy"] = np.mean(all_predictions == all_true)
    else:
        results["overall"]["mse"] = mean_squared_error(all_true, all_predictions)
        results["overall"]["rmse"] = np.sqrt(results["overall"]["mse"])
        results["overall"]["r2"] = r2_score(all_true, all_predictions)
        results["overall"]["pearson_r"] = np.corrcoef(all_true, all_predictions)[0, 1]

    logger.info(f"Overall results: {results['overall']}")

    return results


def grouped_k_fold_cv(
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    model_fn,
    n_splits: int = 5,
    task_type: str = "regression",
) -> Dict[str, Any]:
    """
    Grouped K-Fold Cross-Validation.

    Ensures subjects from the same site stay together in train/test splits.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        sites: Site labels (n_samples,)
        model_fn: Function that returns a model instance
        n_splits: Number of folds
        task_type: 'classification' or 'regression'

    Returns:
        Results dictionary
    """
    gkf = GroupKFold(n_splits=n_splits)

    logger.info(f"Running Grouped {n_splits}-Fold CV...")

    results = {
        "strategy": "grouped_k_fold",
        "n_splits": n_splits,
        "task_type": task_type,
        "per_fold": {},
        "overall": {},
    }

    all_predictions = []
    all_true = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=sites)):
        test_sites = np.unique(sites[test_idx])
        logger.info(f"Fold {fold+1}/{n_splits}: Testing sites {test_sites}")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model = model_fn()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        all_predictions.extend(y_pred)
        all_true.extend(y_test)

        # Evaluate
        fold_results = {}
        if task_type == "classification":
            fold_results["accuracy"] = np.mean(y_pred == y_test)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fold_results["auroc"] = roc_auc_score(y_test, y_pred_proba)
        else:
            fold_results["mse"] = mean_squared_error(y_test, y_pred)
            fold_results["rmse"] = np.sqrt(fold_results["mse"])
            fold_results["r2"] = r2_score(y_test, y_pred)
            fold_results["pearson_r"] = np.corrcoef(y_test, y_pred)[0, 1]

        results["per_fold"][f"fold_{fold+1}"] = fold_results
        logger.info(f"Fold {fold+1} results: {fold_results}")

    # Overall results
    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)

    if task_type == "classification":
        results["overall"]["accuracy"] = np.mean(all_predictions == all_true)
    else:
        results["overall"]["mse"] = mean_squared_error(all_true, all_predictions)
        results["overall"]["rmse"] = np.sqrt(results["overall"]["mse"])
        results["overall"]["r2"] = r2_score(all_true, all_predictions)
        results["overall"]["pearson_r"] = np.corrcoef(all_true, all_predictions)[0, 1]

    # Calculate mean and std across folds
    if task_type == "classification":
        fold_accs = [r["accuracy"] for r in results["per_fold"].values()]
        results["overall"]["accuracy_mean"] = np.mean(fold_accs)
        results["overall"]["accuracy_std"] = np.std(fold_accs)
    else:
        fold_r2s = [r["r2"] for r in results["per_fold"].values()]
        fold_pearsons = [r["pearson_r"] for r in results["per_fold"].values()]
        results["overall"]["r2_mean"] = np.mean(fold_r2s)
        results["overall"]["r2_std"] = np.std(fold_r2s)
        results["overall"]["pearson_r_mean"] = np.mean(fold_pearsons)
        results["overall"]["pearson_r_std"] = np.std(fold_pearsons)

    logger.info(f"Overall results: {results['overall']}")

    return results


# ============================================================================
# Data Loading (reuse from baseline script)
# ============================================================================


def load_baseline_data(
    bids_root: Path, challenge: int, target: str = "p_factor"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load preprocessed baseline data and subject IDs.

    Returns:
        X: Features
        y: Labels
        subjects: Subject IDs
    """
    # For now, use dummy data
    # In practice, load from baseline training results
    logger.warning("Using dummy data - integrate with actual baseline data")

    # Get available subjects
    subjects = [
        d.name.replace("sub-", "")
        for d in bids_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ]

    n_subjects = len(subjects)
    n_features = 640  # 128 channels * 5 frequency bands

    X = np.random.randn(n_subjects, n_features)
    y = np.random.randn(n_subjects) if challenge == 2 else np.random.randint(0, 2, n_subjects)

    return X, y, subjects


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Cross-site validation")
    parser.add_argument(
        "--bids-root", type=str, default="data/raw/hbn", help="BIDS dataset root"
    )
    parser.add_argument(
        "--challenge", type=int, choices=[1, 2], required=True, help="Challenge number"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["leave_one_site_out", "grouped_k_fold"],
        default="leave_one_site_out",
        help="CV strategy",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (for grouped k-fold)")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["logistic", "random_forest", "linear"],
        help="Model type",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="p_factor",
        choices=["p_factor", "attention", "internalizing", "externalizing"],
        help="Target for Challenge 2",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/cross_site", help="Output directory"
    )

    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    X, y, subjects = load_baseline_data(bids_root, args.challenge, args.target)

    # Assign sites
    logger.info("Assigning sites...")
    site_map = assign_sites_from_metadata(bids_root, subjects)
    sites = np.array([site_map[s] for s in subjects])

    # Define model function
    def model_fn():
        if args.model == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000, random_state=42)
        elif args.model == "random_forest":
            if args.challenge == 1:
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(n_estimators=100, random_state=42)
        elif args.model == "linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()

    # Run cross-site validation
    task_type = "classification" if args.challenge == 1 else "regression"

    if args.strategy == "leave_one_site_out":
        results = leave_one_site_out_cv(X, y, sites, model_fn, task_type)
    else:
        results = grouped_k_fold_cv(X, y, sites, model_fn, args.cv_folds, task_type)

    # Save results
    output_file = output_dir / f"cross_site_challenge{args.challenge}_{args.model}_{args.strategy}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-SITE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Strategy: {results['strategy']}")
    print(f"Task Type: {results['task_type']}")
    print(f"\nOverall Results:")
    for metric, value in results["overall"].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()
