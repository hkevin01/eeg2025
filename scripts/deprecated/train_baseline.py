#!/usr/bin/env python3
"""
Baseline Model Training Script for EEG Foundation Challenge 2025

This script trains simple baseline models for both challenge tracks:
1. Cross-Task Transfer (SuS → CCD)
2. Psychopathology Prediction (Resting State → P-factor, etc.)

Baselines include:
- Logistic Regression
- Random Forest
- Simple CNN
- LSTM

Usage:
    python scripts/train_baseline.py --challenge 1 --model logistic
    python scripts/train_baseline.py --challenge 2 --model random_forest
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mne_bids import BIDSPath, read_raw_bids
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Simple Neural Network Models
# ============================================================================


class SimpleCNN(nn.Module):
    """Simple CNN baseline for EEG classification/regression."""

    def __init__(self, n_channels: int = 128, seq_len: int = 1000, n_outputs: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size
        self.flat_size = 64 * (seq_len // 8)  # 3 pooling layers

        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc2 = nn.Linear(256, n_outputs)

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimpleLSTM(nn.Module):
    """Simple LSTM baseline for EEG time series."""

    def __init__(self, n_channels: int = 128, hidden_size: int = 128, n_outputs: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        # x: (batch, channels, time) -> transpose to (batch, time, channels)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x


# ============================================================================
# Data Loading
# ============================================================================


def load_eeg_data(
    bids_root: Path,
    subject: str,
    task: str,
    run: Optional[int] = None,
    session: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Load EEG data for a subject/task.

    Returns:
        np.ndarray: (n_channels, n_samples) or None if loading fails
    """
    try:
        bids_path = BIDSPath(
            root=bids_root,
            subject=subject,
            task=task,
            run=run,
            session=session,
            datatype="eeg",
            suffix="eeg",
            extension=".set",
        )

        raw = read_raw_bids(bids_path, verbose=False)

        # Basic preprocessing
        raw.load_data()
        raw.filter(l_freq=0.1, h_freq=40.0, verbose=False)
        raw.notch_filter(freqs=60.0, verbose=False)

        # Get data
        data = raw.get_data()  # (n_channels, n_samples)
        return data

    except Exception as e:
        logger.warning(f"Failed to load {subject}/{task}: {e}")
        return None


def extract_features(data: np.ndarray, method: str = "psd") -> np.ndarray:
    """
    Extract features from EEG data.

    Args:
        data: (n_channels, n_samples)
        method: 'psd', 'stats', 'raw_mean'

    Returns:
        Feature vector
    """
    if method == "psd":
        # Power spectral density features
        from scipy.signal import welch

        features = []
        for ch in range(data.shape[0]):
            freqs, psd = welch(data[ch], fs=500, nperseg=256)
            # Extract band powers
            delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
            theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
            alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
            beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])
            gamma = np.mean(psd[(freqs >= 30) & (freqs < 40)])
            features.extend([delta, theta, alpha, beta, gamma])
        return np.array(features)

    elif method == "stats":
        # Statistical features
        features = []
        for ch in range(data.shape[0]):
            ch_data = data[ch]
            features.extend(
                [
                    np.mean(ch_data),
                    np.std(ch_data),
                    np.max(ch_data),
                    np.min(ch_data),
                    np.median(ch_data),
                ]
            )
        return np.array(features)

    elif method == "raw_mean":
        # Simple mean over time
        return np.mean(data, axis=1)

    else:
        raise ValueError(f"Unknown feature extraction method: {method}")


def load_participants_data(bids_root: Path) -> pd.DataFrame:
    """Load participants.tsv file."""
    participants_file = bids_root / "participants.tsv"
    if not participants_file.exists():
        raise FileNotFoundError(f"No participants.tsv at {participants_file}")
    return pd.read_csv(participants_file, sep="\t")


# ============================================================================
# Challenge 1: Cross-Task Transfer (SuS → CCD)
# ============================================================================


def prepare_challenge1_data(
    bids_root: Path, subjects: List[str], feature_method: str = "psd"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Challenge 1: Cross-Task Transfer.

    Returns:
        X: Features from SuS task
        y: Dummy labels (for now, we'd need actual CCD performance labels)
    """
    X_list = []
    y_list = []

    logger.info(f"Loading Challenge 1 data for {len(subjects)} subjects...")

    for subject in tqdm(subjects):
        # Load SuS task data
        sus_data = load_eeg_data(bids_root, subject, task="surroundSupp", run=1)

        if sus_data is not None:
            # Extract features
            features = extract_features(sus_data, method=feature_method)
            X_list.append(features)

            # For now, use dummy labels (0 or 1)
            # In real challenge, these would be CCD task performance metrics
            y_list.append(np.random.randint(0, 2))

    if len(X_list) == 0:
        raise ValueError("No data loaded for Challenge 1!")

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info(f"Challenge 1 data shape: X={X.shape}, y={y.shape}")
    return X, y


# ============================================================================
# Challenge 2: Psychopathology Prediction
# ============================================================================


def prepare_challenge2_data(
    bids_root: Path, subjects: List[str], feature_method: str = "psd", target: str = "p_factor"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Challenge 2: Psychopathology Prediction.

    Args:
        target: 'p_factor', 'attention', 'internalizing', or 'externalizing'

    Returns:
        X: Features from Resting State EEG
        y: Target psychopathology scores
    """
    # Load participants data
    participants = load_participants_data(bids_root)

    X_list = []
    y_list = []

    logger.info(f"Loading Challenge 2 data ({target}) for {len(subjects)} subjects...")

    for subject in tqdm(subjects):
        # Load Resting State data
        rs_data = load_eeg_data(bids_root, subject, task="RestingState")

        if rs_data is not None:
            # Get target label
            subject_id = f"sub-{subject}"
            subject_row = participants[participants["participant_id"] == subject_id]

            if len(subject_row) > 0 and not pd.isna(subject_row[target].values[0]):
                # Extract features
                features = extract_features(rs_data, method=feature_method)
                X_list.append(features)

                # Get target
                y_list.append(subject_row[target].values[0])

    if len(X_list) == 0:
        raise ValueError(f"No data loaded for Challenge 2 ({target})!")

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info(f"Challenge 2 data shape: X={X.shape}, y={y.shape}")
    return X, y


# ============================================================================
# Training Functions
# ============================================================================


def train_sklearn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    task_type: str,
) -> Dict[str, Any]:
    """
    Train sklearn baseline model.

    Args:
        model_type: 'logistic', 'random_forest', 'linear'
        task_type: 'classification' or 'regression'
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    logger.info(f"Training {model_type} model...")
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    results = {"model_type": model_type, "task_type": task_type}

    if task_type == "classification":
        # Get predicted probabilities
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            results["auroc"] = roc_auc_score(y_test, y_pred_proba)

        results["accuracy"] = np.mean(y_pred == y_test)

    else:  # regression
        results["mse"] = mean_squared_error(y_test, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["r2"] = r2_score(y_test, y_pred)

        # Pearson correlation
        results["pearson_r"] = np.corrcoef(y_test, y_pred)[0, 1]

    logger.info(f"Results: {results}")

    return {
        "model": model,
        "scaler": scaler,
        "results": results,
    }


def train_neural_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    task_type: str,
    epochs: int = 50,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Train neural network baseline model.

    Args:
        model_type: 'cnn' or 'lstm'
        task_type: 'classification' or 'regression'
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    if task_type == "classification":
        y_train_t = y_train_t.long()
        y_test_t = y_test_t.long()

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    n_channels = X_train.shape[1]
    seq_len = X_train.shape[2] if len(X_train.shape) > 2 else 1

    if model_type == "cnn":
        model = SimpleCNN(n_channels=n_channels, seq_len=seq_len, n_outputs=1).to(device)
    elif model_type == "lstm":
        model = SimpleLSTM(n_channels=n_channels, n_outputs=1).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Loss and optimizer
    if task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    logger.info(f"Training {model_type} neural network...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).squeeze().cpu().numpy()

    results = {"model_type": model_type, "task_type": task_type}

    if task_type == "classification":
        y_pred_binary = (torch.sigmoid(torch.FloatTensor(y_pred)) > 0.5).numpy().astype(int)
        results["accuracy"] = np.mean(y_pred_binary == y_test)
        results["auroc"] = roc_auc_score(y_test, torch.sigmoid(torch.FloatTensor(y_pred)).numpy())
    else:
        results["mse"] = mean_squared_error(y_test, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["r2"] = r2_score(y_test, y_pred)
        results["pearson_r"] = np.corrcoef(y_test, y_pred)[0, 1]

    logger.info(f"Results: {results}")

    return {"model": model, "results": results}


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--challenge", type=int, choices=[1, 2], required=True, help="Challenge number (1 or 2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "random_forest", "linear", "cnn", "lstm"],
        default="logistic",
        help="Model type",
    )
    parser.add_argument(
        "--bids-root", type=str, default="data/raw/hbn", help="BIDS dataset root"
    )
    parser.add_argument("--output-dir", type=str, default="outputs/baselines", help="Output directory")
    parser.add_argument("--feature-method", type=str, default="psd", help="Feature extraction method")
    parser.add_argument(
        "--target",
        type=str,
        default="p_factor",
        choices=["p_factor", "attention", "internalizing", "externalizing"],
        help="Target for Challenge 2",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for neural models")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Setup
    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get available subjects
    subjects = [
        d.name.replace("sub-", "")
        for d in bids_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ]
    logger.info(f"Found {len(subjects)} subjects")

    # Prepare data
    if args.challenge == 1:
        X, y = prepare_challenge1_data(bids_root, subjects, args.feature_method)
        task_type = "classification"
    else:
        X, y = prepare_challenge2_data(bids_root, subjects, args.feature_method, args.target)
        task_type = "regression"

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train model
    if args.model in ["logistic", "random_forest", "linear"]:
        result = train_sklearn_model(X_train, y_train, X_test, y_test, args.model, task_type)
    else:
        # For neural models, we need raw time series data, not features
        logger.warning("Neural models need raw data - using feature-based approach for now")
        result = train_sklearn_model(X_train, y_train, X_test, y_test, "random_forest", task_type)

    # Save results
    output_file = output_dir / f"baseline_challenge{args.challenge}_{args.model}.json"
    with open(output_file, "w") as f:
        json.dump(result["results"], f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Save model
    model_file = output_dir / f"baseline_challenge{args.challenge}_{args.model}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"Model saved to {model_file}")


if __name__ == "__main__":
    main()
