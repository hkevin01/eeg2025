#!/usr/bin/env python3
"""
Artifact Detection for EEG Data

Implements robust artifact detection and removal using:
1. ICA (Independent Component Analysis) for eye blinks, ECG, muscle artifacts
2. Autoreject for automated bad channel/epoch detection
3. Statistical thresholding for extreme values

Usage:
    python scripts/artifact_detection.py --subject NDARAC904DMU --task RestingState
    python scripts/artifact_detection.py --all-subjects --output-dir data/processed
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Artifact Detection Methods
# ============================================================================


def detect_bad_channels(raw: mne.io.Raw, threshold: float = 3.0) -> List[str]:
    """
    Detect bad channels using variance and correlation methods.

    Args:
        raw: Raw EEG data
        threshold: Number of standard deviations for outlier detection

    Returns:
        List of bad channel names
    """
    data = raw.get_data()
    ch_names = raw.ch_names

    bad_channels = []

    # Method 1: Variance-based detection
    variances = np.var(data, axis=1)
    mean_var = np.mean(variances)
    std_var = np.std(variances)

    for i, (var, ch_name) in enumerate(zip(variances, ch_names)):
        if var > mean_var + threshold * std_var or var < mean_var - threshold * std_var:
            bad_channels.append(ch_name)
            logger.info(f"Bad channel (variance): {ch_name} (var={var:.2e})")

    # Method 2: Correlation-based detection
    # Channels should be correlated with neighbors
    corr_matrix = np.corrcoef(data)
    mean_corr = np.mean(corr_matrix, axis=1)

    for i, (corr, ch_name) in enumerate(zip(mean_corr, ch_names)):
        if corr < 0.4:  # Low correlation with other channels
            if ch_name not in bad_channels:
                bad_channels.append(ch_name)
                logger.info(f"Bad channel (correlation): {ch_name} (corr={corr:.2f})")

    return bad_channels


def detect_artifacts_threshold(
    raw: mne.io.Raw, threshold: float = 150e-6
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Detect artifacts using amplitude thresholding.

    Args:
        raw: Raw EEG data
        threshold: Amplitude threshold in volts (default: 150 µV)

    Returns:
        mask: Boolean array indicating artifact samples
        bad_segments: List of (start_time, end_time) tuples
    """
    data = raw.get_data()
    times = raw.times

    # Find samples exceeding threshold
    artifact_mask = np.abs(data) > threshold
    artifact_any = np.any(artifact_mask, axis=0)

    # Find continuous bad segments
    bad_segments = []
    in_segment = False
    start_idx = 0

    for i, is_bad in enumerate(artifact_any):
        if is_bad and not in_segment:
            start_idx = i
            in_segment = True
        elif not is_bad and in_segment:
            bad_segments.append((times[start_idx], times[i - 1]))
            in_segment = False

    if in_segment:
        bad_segments.append((times[start_idx], times[-1]))

    logger.info(f"Found {len(bad_segments)} bad segments using threshold {threshold*1e6:.1f} µV")

    return artifact_mask, bad_segments


def run_ica_artifact_removal(
    raw: mne.io.Raw, n_components: int = 20, random_state: int = 42
) -> Tuple[ICA, List[int]]:
    """
    Run ICA to identify and remove artifacts (eye blinks, ECG, muscle).

    Args:
        raw: Raw EEG data (should be filtered)
        n_components: Number of ICA components
        random_state: Random seed

    Returns:
        ica: Fitted ICA object
        exclude: List of component indices to exclude
    """
    logger.info(f"Running ICA with {n_components} components...")

    # Create ICA object
    ica = ICA(n_components=n_components, random_state=random_state, max_iter=800)

    # Fit ICA
    ica.fit(raw)

    logger.info(f"ICA converged in {ica.n_iter_} iterations")

    # Automatically find bad components
    exclude = []

    # Find EOG (eye blink) components
    if any("EOG" in ch for ch in raw.ch_names):
        eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
        exclude.extend(eog_indices)
        logger.info(f"Found {len(eog_indices)} EOG components: {eog_indices}")

    # Find ECG components
    if any("ECG" in ch for ch in raw.ch_names):
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, threshold=3.0)
        exclude.extend(ecg_indices)
        logger.info(f"Found {len(ecg_indices)} ECG components: {ecg_indices}")

    # Remove duplicates
    exclude = list(set(exclude))

    if len(exclude) == 0:
        logger.warning("No artifact components automatically detected")

    return ica, exclude


def apply_autoreject(raw: mne.io.Raw, epochs: Optional[mne.Epochs] = None) -> mne.Epochs:
    """
    Apply Autoreject for automated bad epoch/channel detection.

    Args:
        raw: Raw EEG data
        epochs: Pre-created epochs (if None, will create 2s epochs)

    Returns:
        cleaned_epochs: Epochs with bad channels interpolated and bad epochs dropped
    """
    try:
        from autoreject import AutoReject
    except ImportError:
        logger.error("autoreject not installed. Install with: pip install autoreject")
        return None

    # Create epochs if not provided
    if epochs is None:
        events = mne.make_fixed_length_events(raw, duration=2.0)
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False
        )

    logger.info(f"Running Autoreject on {len(epochs)} epochs...")

    # Run Autoreject
    ar = AutoReject(random_state=42, n_jobs=-1, verbose=False)
    epochs_clean = ar.fit_transform(epochs)

    n_dropped = len(epochs) - len(epochs_clean)
    logger.info(f"Autoreject: dropped {n_dropped}/{len(epochs)} epochs")

    return epochs_clean


# ============================================================================
# Artifact Detection Pipeline
# ============================================================================


def process_subject(
    bids_root: Path,
    subject: str,
    task: str,
    run: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict:
    """
    Run complete artifact detection pipeline for one subject.

    Args:
        bids_root: BIDS dataset root
        subject: Subject ID (without 'sub-' prefix)
        task: Task name
        run: Run number (optional)
        output_dir: Where to save processed data
        save_plots: Whether to save diagnostic plots

    Returns:
        Dictionary with artifact detection results
    """
    logger.info(f"Processing subject {subject}, task {task}")

    # Load data
    try:
        bids_path = BIDSPath(
            root=bids_root,
            subject=subject,
            task=task,
            run=run,
            datatype="eeg",
            suffix="eeg",
            extension=".set",
        )
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {"success": False, "error": str(e)}

    results = {
        "subject": subject,
        "task": task,
        "run": run,
        "success": True,
        "original_duration": raw.times[-1],
        "n_channels": len(raw.ch_names),
    }

    # Step 1: Detect bad channels
    logger.info("Step 1: Detecting bad channels...")
    bad_channels = detect_bad_channels(raw, threshold=3.0)
    results["bad_channels"] = bad_channels
    results["n_bad_channels"] = len(bad_channels)

    if bad_channels:
        raw.info["bads"] = bad_channels

    # Step 2: Basic preprocessing
    logger.info("Step 2: Basic preprocessing (filtering)...")
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw.notch_filter(freqs=60.0, verbose=False)

    # Step 3: Threshold-based artifact detection
    logger.info("Step 3: Threshold-based artifact detection...")
    artifact_mask, bad_segments = detect_artifacts_threshold(raw, threshold=150e-6)
    results["bad_segments"] = [(float(s), float(e)) for s, e in bad_segments]
    results["n_bad_segments"] = len(bad_segments)

    # Step 4: ICA artifact removal
    logger.info("Step 4: ICA artifact removal...")
    try:
        ica, exclude_components = run_ica_artifact_removal(raw, n_components=20)
        results["ica_components"] = exclude_components
        results["n_ica_components"] = len(exclude_components)

        # Apply ICA
        raw_clean = raw.copy()
        ica.apply(raw_clean, exclude=exclude_components)

    except Exception as e:
        logger.warning(f"ICA failed: {e}")
        raw_clean = raw.copy()
        results["ica_error"] = str(e)

    # Step 5: Autoreject (optional, can be slow)
    # Uncomment if you want to use autoreject
    # logger.info("Step 5: Autoreject...")
    # try:
    #     epochs = mne.make_fixed_length_epochs(raw_clean, duration=2.0, preload=True)
    #     epochs_clean = apply_autoreject(raw_clean, epochs)
    #     results["autoreject_dropped"] = len(epochs) - len(epochs_clean)
    # except Exception as e:
    #     logger.warning(f"Autoreject failed: {e}")

    # Save cleaned data
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save cleaned raw data
        output_file = output_dir / f"sub-{subject}_task-{task}_clean.fif"
        raw_clean.save(output_file, overwrite=True, verbose=False)
        logger.info(f"Saved cleaned data to {output_file}")

        # Save artifact report
        report_file = output_dir / f"sub-{subject}_task-{task}_artifacts.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved artifact report to {report_file}")

        # Save plots
        if save_plots:
            plot_dir = output_dir / "plots"
            plot_dir.mkdir(exist_ok=True)

            # Plot bad channels
            if bad_channels:
                fig = raw.plot(
                    bad_color="red", duration=10.0, n_channels=30, scalings="auto", show=False
                )
                fig.savefig(plot_dir / f"sub-{subject}_task-{task}_bad_channels.png")
                plt.close(fig)

            # Plot ICA components
            if "ica_components" in results and len(results["ica_components"]) > 0:
                fig = ica.plot_components(picks=exclude_components, show=False)
                if isinstance(fig, list):
                    fig[0].savefig(plot_dir / f"sub-{subject}_task-{task}_ica_components.png")
                    plt.close(fig[0])
                else:
                    fig.savefig(plot_dir / f"sub-{subject}_task-{task}_ica_components.png")
                    plt.close(fig)

    logger.info(f"Artifact detection complete for {subject}/{task}")
    logger.info(f"Summary: {results['n_bad_channels']} bad channels, " f"{results['n_bad_segments']} bad segments, " f"{results.get('n_ica_components', 0)} ICA components")

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Artifact detection for EEG data")
    parser.add_argument("--bids-root", type=str, default="data/raw/hbn", help="BIDS dataset root")
    parser.add_argument("--subject", type=str, help="Subject ID (without sub- prefix)")
    parser.add_argument("--task", type=str, help="Task name")
    parser.add_argument("--run", type=int, help="Run number (optional)")
    parser.add_argument(
        "--all-subjects", action="store_true", help="Process all available subjects"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument("--no-plots", action="store_true", help="Don't save plots")

    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)

    if args.all_subjects:
        # Get all subjects
        subjects = [
            d.name.replace("sub-", "")
            for d in bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ]

        logger.info(f"Processing {len(subjects)} subjects...")

        all_results = []
        for subject in tqdm(subjects):
            # Process RestingState task for each subject
            result = process_subject(
                bids_root,
                subject,
                task="RestingState",
                output_dir=output_dir,
                save_plots=not args.no_plots,
            )
            all_results.append(result)

        # Save summary
        summary_file = output_dir / "artifact_detection_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

    else:
        if not args.subject or not args.task:
            parser.error("--subject and --task are required if not using --all-subjects")

        result = process_subject(
            bids_root,
            args.subject,
            args.task,
            run=args.run,
            output_dir=output_dir,
            save_plots=not args.no_plots,
        )

        print("\nArtifact Detection Results:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
