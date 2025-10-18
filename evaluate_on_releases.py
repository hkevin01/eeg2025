#!/usr/bin/env python3
"""
Evaluate a submission zip on releases R1..R6 using the local scoring utilities.

This script extracts the submission, imports `Submission` from it, then
evaluates the models using the same ingestion pipeline from
`starter_kit_integration/local_scoring.py`, iterating releases R1..R6.

Outputs a JSON with per-release and overall NRMSE scores.
"""
import json
import pickle
import zipfile
from pathlib import Path
import sys
import numpy as np

from starter_kit_integration import local_scoring


def evaluate_submission_on_releases(submission_zip: str, data_dir: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract submission
    with zipfile.ZipFile(submission_zip, 'r') as z:
        z.extractall(out_dir)
    sys.path.insert(0, str(out_dir))
    from submission import Submission  # type: ignore

    releases = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
    results = {}

    for rel in releases:
        print(f"\n=== Evaluating on {rel} ===")

        # Use ingestion logic but swapping dataset release
        # We'll call a modified ingestion that accepts release name
        ingestion_fn = local_scoring.ingestion
        # local_scoring.ingestion uses release='R5' hardcoded; we'll emulate its logic

        # For simplicity, call the main local_scoring with fast_dev_run=False but
        # override environment by setting EEGChallengeDataset calls is not trivial here.
        # Instead, we call local_scoring.main but that expects a submission zip and will
        # run ingestion for R5. To avoid editing local_scoring, we will temporarily
        # monkeypatch EEGChallengeDataset in its module.

        # Monkeypatch EEGChallengeDataset to force release
        try:
            import importlib
            import eegdash
            real_dataset = importlib.import_module('eegdash')
        except Exception:
            real_dataset = None

        # Call the helper that runs ingestion but passing release via environment variable
        # The simplest and robust approach: copy local_scoring.ingestion into this script's scope
        # and adapt it to use the requested release. We'll reuse much of local_scoring code by
        # importing internals.
        print("Running ingestion for release", rel)
        ingestion_output = run_ingestion_for_release(Submission, data_dir, rel)

        scores = local_scoring.scoring(ingestion_output)
        results[rel] = scores

    # Save results
    out_path = out_dir / 'per_release_scores.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved per-release scores to {out_path}")
    return results


def run_ingestion_for_release(SubmissionClass, data_dir, release_name):
    """A reduced ingestion adapted from local_scoring.ingestion that accepts a release name."""
    # Import locally to avoid circular imports
    from eegdash import EEGChallengeDataset
    from braindecode.preprocessing import (
        Preprocessor,
        preprocess,
        create_windows_from_events,
        create_fixed_length_windows,
    )
    from eegdash.hbn.windows import (
        annotate_trials_with_target,
        add_aux_anchors,
        add_extras_columns,
        keep_only_recordings_with,
    )
    from torch.utils.data import DataLoader, SequentialSampler
    from braindecode.datasets.base import BaseConcatDataset
    import torch
    import math

    SFREQ = local_scoring.SFREQ
    EPOCH_LEN_S = local_scoring.EPOCH_LEN_S
    DEVICE = local_scoring.DEVICE

    # Challenge 1
    sub = SubmissionClass(SFREQ, DEVICE)
    model_1 = sub.get_model_challenge_1()
    # If submission returned a wrapper (e.g., TTAPredictor) that holds the real model
    # call eval() on the inner model. Otherwise call eval() on the model itself.
    if hasattr(model_1, 'model'):
        try:
            model_1.model.eval()
        except Exception:
            pass
    else:
        try:
            model_1.eval()
        except Exception:
            pass

    dataset_1 = EEGChallengeDataset(
        release=release_name,
        mini=False,
        query=dict(task='contrastChangeDetection'),
        cache_dir=data_dir,
    )

    preprocessors = [
        Preprocessor(
            annotate_trials_with_target,
            apply_on_array=False,
            target_field='rt_from_stimulus',
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset_1, preprocessors, n_jobs=-1)

    SHIFT_AFTER_STIM = 0.5
    WINDOW_LEN = 2.0
    dataset_2 = keep_only_recordings_with('stimulus_anchor', dataset_1)
    dataset_3 = create_windows_from_events(
        dataset_2,
        mapping={'stimulus_anchor': 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    dataset_3 = add_extras_columns(dataset_3, dataset_2, desc='stimulus_anchor', keys=(
        'target', 'rt_from_stimulus', 'rt_from_trialstart', 'stimulus_onset', 'response_onset', 'correct', 'response_type'
    ))

    dataloader_1 = DataLoader(dataset_3, batch_size=1, sampler=SequentialSampler(dataset_3), shuffle=False, drop_last=False)

    y_preds = []
    y_trues = []
    with torch.inference_mode():
        for batch in dataloader_1:
            X, y, infos = batch
            X = X.to(dtype=torch.float32, device=DEVICE)
            y = y.to(dtype=torch.float32, device=DEVICE).unsqueeze(1)
            # Use predict() for wrappers (TTA) or forward() for standard nn.Module
            if hasattr(model_1, 'predict'):
                y_pred = model_1.predict(X)
            else:
                y_pred = model_1.forward(X)
            y_preds.append(y_pred.detach().cpu().numpy()[0][0])
            y_trues.append(y.detach().cpu().numpy()[0][0])

    challenge_1_y_preds = np.array(y_preds)
    challenge_1_y_trues = np.squeeze(np.array(y_trues))

    # Challenge 2
    sub = SubmissionClass(SFREQ, DEVICE)
    model_2 = sub.get_model_challenge_2()
    if hasattr(model_2, 'model'):
        try:
            model_2.model.eval()
        except Exception:
            pass
    else:
        try:
            model_2.eval()
        except Exception:
            pass

    dataset_4 = EEGChallengeDataset(
        release=release_name,
        mini=False,
        query=dict(task='contrastChangeDetection'),
        description_fields=['externalizing'],
        cache_dir=data_dir,
    )

    dataset_5 = BaseConcatDataset([ds for ds in dataset_4.datasets if ds.raw.n_times >= 4 * SFREQ and not math.isnan(ds.description['externalizing'])])
    dataset_6 = create_fixed_length_windows(dataset_5, window_size_samples=4 * SFREQ, window_stride_samples=2 * SFREQ, drop_last_window=True)
    dataset_6 = BaseConcatDataset([local_scoring.DatasetWrapper(ds, crop_size_samples=2 * SFREQ, seed=42) for ds in dataset_6.datasets])

    dataloader_2 = DataLoader(dataset_6, batch_size=1, sampler=SequentialSampler(dataset_6), shuffle=False, drop_last=False)

    y_preds = []
    y_trues = []
    with torch.inference_mode():
        for batch in dataloader_2:
            X, y, crop_inds, infos = batch
            X = X.to(dtype=torch.float32, device=DEVICE)
            y = y.to(dtype=torch.float32, device=DEVICE).unsqueeze(1)
            if hasattr(model_2, 'predict'):
                y_pred = model_2.predict(X)
            else:
                y_pred = model_2.forward(X)
            y_preds.append(y_pred.detach().cpu().numpy()[0][0])
            y_trues.append(y.detach().cpu().numpy()[0][0])

    challenge_2_y_preds = np.array(y_preds)
    challenge_2_y_trues = np.array(y_trues)

    return {
        'challenge_1': {'y_preds': challenge_1_y_preds, 'y_trues': challenge_1_y_trues},
        'challenge_2': {'y_preds': challenge_2_y_preds, 'y_trues': challenge_2_y_trues},
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission-zip', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='.')
    args = parser.parse_args()

    results = evaluate_submission_on_releases(args.submission_zip, args.data_dir, args.output_dir)
    print('\nFinal per-release results:')
    import pprint
    pprint.pprint(results)
