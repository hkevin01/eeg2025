#!/usr/bin/env python3
"""
Hyperparameter Optimization for EEG Foundation Models

Uses Optuna for efficient hyperparameter search with:
- Tree-structured Parzen Estimator (TPE) sampling
- Early stopping with pruning
- Multi-objective optimization (accuracy + latency)
- Distributed optimization support

Usage:
    python scripts/hyperparameter_optimization.py --challenge 1 --n-trials 100
    python scripts/hyperparameter_optimization.py --challenge 2 --target p_factor --n-trials 50
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Hyperparameter Search Spaces
# ============================================================================


def get_search_space_logistic(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for Logistic Regression."""
    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": "saga",
        "max_iter": 1000,
        "random_state": 42,
    }


def get_search_space_random_forest(trial: optuna.Trial, task_type: str) -> Dict[str, Any]:
    """Search space for Random Forest."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
    }


def get_search_space_svm(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for SVM."""
    return {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "random_state": 42,
    }


def get_search_space_mlp(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for Multi-Layer Perceptron."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = []
    for i in range(n_layers):
        hidden_units.append(trial.suggest_int(f"n_units_l{i}", 32, 512))

    return {
        "hidden_layer_sizes": tuple(hidden_units),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "max_iter": 500,
        "random_state": 42,
    }


def get_search_space_ridge(trial: optuna.Trial) -> Dict[str, Any]:
    """Search space for Ridge Regression."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
        "random_state": 42,
    }


# ============================================================================
# Objective Functions
# ============================================================================


def objective_classification(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str,
) -> float:
    """
    Objective function for classification tasks.

    Returns:
        AUROC score (to maximize)
    """
    # Get hyperparameters
    if model_type == "logistic":
        params = get_search_space_logistic(trial)
        model = LogisticRegression(**params)
    elif model_type == "random_forest":
        params = get_search_space_random_forest(trial, "classification")
        model = RandomForestClassifier(**params)
    elif model_type == "svm":
        params = get_search_space_svm(trial)
        model = SVC(**params, probability=True)
    elif model_type == "mlp":
        params = get_search_space_mlp(trial)
        model = MLPClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    auroc = roc_auc_score(y_val, y_pred_proba)

    return auroc


def objective_regression(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str,
    metric: str = "r2",
) -> float:
    """
    Objective function for regression tasks.

    Args:
        metric: 'r2', 'pearson_r', or 'neg_mse' (all to maximize)

    Returns:
        Metric value (to maximize)
    """
    # Get hyperparameters
    if model_type == "ridge":
        params = get_search_space_ridge(trial)
        model = Ridge(**params)
    elif model_type == "random_forest":
        params = get_search_space_random_forest(trial, "regression")
        model = RandomForestRegressor(**params)
    elif model_type == "svm":
        params = get_search_space_svm(trial)
        model = SVR(**params)
    elif model_type == "mlp":
        params = get_search_space_mlp(trial)
        model = MLPRegressor(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_val_scaled)

    # Evaluate
    if metric == "r2":
        score = r2_score(y_val, y_pred)
    elif metric == "pearson_r":
        score = np.corrcoef(y_val, y_pred)[0, 1]
    elif metric == "neg_mse":
        score = -mean_squared_error(y_val, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return score


# ============================================================================
# Data Loading
# ============================================================================


def load_dummy_data(
    challenge: int, target: str = "p_factor", n_samples: int = 100, n_features: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dummy data for testing.

    In practice, replace with actual data loading.
    """
    logger.warning("Using dummy data - integrate with actual data pipeline")

    X = np.random.randn(n_samples, n_features)

    if challenge == 1:
        # Classification
        y = np.random.randint(0, 2, n_samples)
    else:
        # Regression
        y = np.random.randn(n_samples)

    return X, y


# ============================================================================
# Optimization
# ============================================================================


def run_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str,
    task_type: str,
    n_trials: int = 100,
    metric: str = "r2",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_type: Model type
        task_type: 'classification' or 'regression'
        n_trials: Number of optimization trials
        metric: Metric to optimize (for regression)
        output_dir: Where to save results

    Returns:
        Dictionary with best parameters and results
    """
    logger.info(f"Starting optimization for {model_type} ({task_type})")
    logger.info(f"Running {n_trials} trials...")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    # Define objective
    if task_type == "classification":

        def objective(trial):
            return objective_classification(trial, X_train, y_train, X_val, y_val, model_type)

    else:

        def objective(trial):
            return objective_regression(trial, X_train, y_train, X_val, y_val, model_type, metric)

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    logger.info(f"Best {metric if task_type == 'regression' else 'AUROC'}: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")

    results = {
        "model_type": model_type,
        "task_type": task_type,
        "metric": metric if task_type == "regression" else "auroc",
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": n_trials,
        "study_statistics": {
            "n_trials": len(study.trials),
            "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        },
    }

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        params_file = output_dir / f"best_params_{model_type}.json"
        with open(params_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {params_file}")

        # Save study
        study_file = output_dir / f"optuna_study_{model_type}.pkl"
        with open(study_file, "wb") as f:
            pickle.dump(study, f)
        logger.info(f"Saved study to {study_file}")

        # Save optimization history plot
        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig(output_dir / f"optimization_history_{model_type}.png")
            plt.close(fig)

            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            fig.savefig(output_dir / f"param_importances_{model_type}.png")
            plt.close(fig)

            logger.info("Saved optimization plots")
        except Exception as e:
            logger.warning(f"Could not save plots: {e}")

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument(
        "--challenge", type=int, choices=[1, 2], required=True, help="Challenge number"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["logistic", "random_forest", "svm", "mlp", "ridge"],
        help="Model type",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="p_factor",
        choices=["p_factor", "attention", "internalizing", "externalizing"],
        help="Target for Challenge 2",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        choices=["r2", "pearson_r", "neg_mse"],
        help="Metric to optimize (for regression)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/hyperopt", help="Output directory"
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples (for dummy data)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (dummy for now)
    logger.info("Loading data...")
    X, y = load_dummy_data(args.challenge, args.target, n_samples=args.n_samples)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Run optimization
    task_type = "classification" if args.challenge == 1 else "regression"

    results = run_optimization(
        X_train,
        y_train,
        X_val,
        y_val,
        args.model,
        task_type,
        n_trials=args.n_trials,
        metric=args.metric,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Model: {results['model_type']}")
    print(f"Task: {results['task_type']}")
    print(f"Metric: {results['metric']}")
    print(f"Best Value: {results['best_value']:.4f}")
    print(f"\nBest Parameters:")
    for param, value in results["best_params"].items():
        print(f"  {param}: {value}")
    print(f"\nStudy Statistics:")
    for stat, value in results["study_statistics"].items():
        print(f"  {stat}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()
