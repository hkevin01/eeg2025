#!/usr/bin/env python3
"""
Challenge 2 Training - R1 & R2 Only
====================================
Start training with available cache (R1, R2)
While R3-R5 are still being created.
"""

import os
import sys
import time
import warnings
import traceback
import argparse
import multiprocessing as mp
from queue import Empty
from functools import partial
from contextlib import nullcontext

print("=" * 80)
print("🚀 CHALLENGE 2: Training with R1 & R2 (Start Fast!)")
print("=" * 80)
print()
print("🔍 Starting imports...")
sys.stdout.flush()

try:
    import h5py
    print("  ✅ h5py imported")
except Exception as e:
    print(f"  ❌ Failed to import h5py: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("  ✅ numpy imported")
except Exception as e:
    print(f"  ❌ Failed to import numpy: {e}")
    sys.exit(1)

try:
    import sqlite3
    print("  ✅ sqlite3 imported")
except Exception as e:
    print(f"  ❌ Failed to import sqlite3: {e}")
    sys.exit(1)

try:
    import json
    print("  ✅ json imported")
except Exception as e:
    print(f"  ❌ Failed to import json: {e}")
    sys.exit(1)

try:
    from datetime import datetime
    print("  ✅ datetime imported")
except Exception as e:
    print(f"  ❌ Failed to import datetime: {e}")
    sys.exit(1)

try:
    from pathlib import Path
    print("  ✅ pathlib imported")
except Exception as e:
    print(f"  ❌ Failed to import pathlib: {e}")
    sys.exit(1)

try:
    import torch
    print("  ✅ torch imported")
except Exception as e:
    print(f"  ❌ Failed to import torch: {e}")
    sys.exit(1)

try:
    from torch import optim
    print("  ✅ torch.optim imported")
except Exception as e:
    print(f"  ❌ Failed to import torch.optim: {e}")
    sys.exit(1)

try:
    from torch.nn.functional import l1_loss
    print("  ✅ torch.nn.functional.l1_loss imported")
except Exception as e:
    print(f"  ❌ Failed to import l1_loss: {e}")
    sys.exit(1)

try:
    from torch.utils.data import DataLoader, Dataset, random_split
    print("  ✅ torch.utils.data imported")
except Exception as e:
    print(f"  ❌ Failed to import torch.utils.data utilities: {e}")
    sys.exit(1)

try:
    from braindecode.models import EEGNeX
    print("  ✅ braindecode EEGNeX imported")
except Exception as e:
    print(f"  ❌ Failed to import EEGNeX: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# Detect compute device (ROCm exposes as torch.cuda)
DEVICE = torch.device("cpu")

GPU_HEALTH_TIMEOUT = 30  # seconds


def _get_mp_context():
    """Return a multiprocessing context suited for the current platform."""

    if hasattr(mp, "get_context"):
        return mp.get_context("spawn")
    return mp
print(f"🖥️  Using device: {DEVICE}")
sys.stdout.flush()

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cached"
DB_FILE = REPO_ROOT / "data" / "metadata.db"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "challenge2_r1r2"
LOG_DIR = REPO_ROOT / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
MAX_EPOCHS = 20
LEARNING_RATE = 0.002
PATIENCE = 5
SFREQ = 100
CROP_SIZE = 2.0  # seconds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train EEGNeX on Challenge 2 cached windows (R1 & R2)."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size for training and validation.")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="Maximum number of epochs to train.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for Adamax optimizer.")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience on validation loss.")
    parser.add_argument(
        "--crop-size",
        type=float,
        default=CROP_SIZE,
        help="Window crop size in seconds used for training windows.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Select a specific CUDA device index when multiple GPUs are available.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for DataLoader instances. Leave unset to auto-tune.",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Force DataLoader pin_memory=True regardless of device.",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Force DataLoader pin_memory=False even on GPU.",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override DataLoader prefetch factor (default PyTorch value when workers > 0)",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (requires num_workers > 0).",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Force DataLoader workers to respawn each epoch.",
    )
    parser.set_defaults(persistent_workers=None)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable random temporal augmentation (center crop instead).",
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help="Enable automatic mixed precision with GradScaler on GPU.",
    )
    parser.add_argument(
        "--note",
        default="Training with R1+R2 only (R3-R5 still creating)",
        help="Free-form note stored with the run metadata.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device preference. 'auto' tries GPU first with health-check fallback to CPU, 'cuda' forces GPU, 'cpu' forces CPU.",
    )
    return parser.parse_args(argv)


def recommend_num_workers():
    """Return a conservative default for multi-process data loading."""

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return max(2, min(8, cpu_count // 2))


def _gpu_probe_worker(device_index: int, crop_samples: int, result_queue):
    """Worker executed in a child process to validate GPU functionality."""

    try:
        torch.cuda.set_device(device_index)
        torch.manual_seed(0)

        probe_model = EEGNeX(
            n_outputs=1,
            n_chans=129,
            n_times=crop_samples,
            sfreq=SFREQ,
        ).to(f"cuda:{device_index}")
        probe_model.eval()

        dummy_input = torch.randn(2, 129, crop_samples, device=f"cuda:{device_index}")

        with torch.no_grad():
            output = probe_model(dummy_input)
            _ = float(output.mean().cpu())

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        result_queue.put(("ok", None))
    except Exception as exc:  # pragma: no cover - best effort logging
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        result_queue.put(("error", str(exc)))


def run_gpu_health_check(device_index: int, timeout: int = GPU_HEALTH_TIMEOUT):
    """Run a short GPU probe in a separate process to avoid ROCm hangs."""

    if not torch.cuda.is_available():
        return False, "CUDA not available"

    ctx = _get_mp_context()
    result_queue = ctx.Queue()

    crop_samples = int(CROP_SIZE * SFREQ)
    process = ctx.Process(
        target=_gpu_probe_worker,
        args=(device_index, crop_samples, result_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"GPU health check timed out after {timeout}s"

    status, message = ("error", "GPU health check produced no result")
    try:
        status, message = result_queue.get_nowait()
    except Empty:
        pass
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    return status == "ok", message


class CachedEEGDataset(Dataset):
    """Dataset that streams windows directly from HDF5 cache files."""

    def __init__(self, cache_files, crop_size_samples=200, augment=True):
        self.crop_size = crop_size_samples
        self.augment = augment
        self.cache_file_paths = [Path(p).resolve() for p in cache_files]
        self.cumulative_lengths = []
        self.file_sample_counts = []
        self._worker_handles = None  # Lazily initialised per worker process

        print(f"Loading metadata from {len(cache_files)} cache files...")
        sys.stdout.flush()

        total_windows = 0
        try:
            for idx, cache_file in enumerate(self.cache_file_paths, start=1):
                print(f"  [{idx}/{len(cache_files)}] Opening {cache_file.name} (metadata)")
                sys.stdout.flush()

                with h5py.File(cache_file, "r") as f:
                    if "data" not in f or "targets" not in f:
                        raise KeyError(
                            f"File {cache_file} must contain 'data' and 'targets' datasets"
                        )

                    data = f["data"]
                    targets = f["targets"]

                    windows = data.shape[0]
                    total_windows += windows
                    self.file_sample_counts.append(windows)
                    self.cumulative_lengths.append(total_windows)

                    print(
                        f"       Windows: {windows:,} | Shape: {data.shape} | dtype: {data.dtype}"
                    )
                    if targets.shape[0] != windows:
                        raise ValueError(
                            f"Targets dataset in {cache_file} mismatches data windows"
                        )
                    sys.stdout.flush()
        except Exception:
            raise

        if total_windows == 0:
            raise RuntimeError("No windows found across provided cache files.")

        self.total_windows = total_windows

        print(f"✅ Total windows available: {total_windows:,}")
        sys.stdout.flush()

    def __len__(self):
        return self.total_windows

    def _locate_index(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        prev_cum = 0 if file_idx == 0 else self.cumulative_lengths[file_idx - 1]
        local_idx = idx - prev_cum
        return file_idx, local_idx

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_windows}")

        # Debug for first few items
        if idx < 5:
            print(f"   Dataset.__getitem__({idx}): Starting data retrieval...")
            sys.stdout.flush()

        file_idx, local_idx = self._locate_index(idx)

        if idx < 5:
            print(f"   Dataset.__getitem__({idx}): Located at file_idx={file_idx}, local_idx={local_idx}")
            sys.stdout.flush()

        data_sets, target_sets = self._get_worker_datasets()

        if idx < 5:
            print(f"   Dataset.__getitem__({idx}): Got worker datasets")
            sys.stdout.flush()

        data_window = data_sets[file_idx][local_idx]
        target_value = target_sets[file_idx][local_idx]

        if idx < 5:
            print(f"   Dataset.__getitem__({idx}): Raw data shape={data_window.shape}, target={target_value}")
            sys.stdout.flush()

        if self.augment and data_window.shape[1] > self.crop_size:
            start = np.random.randint(0, data_window.shape[1] - self.crop_size + 1)
            data_window = data_window[:, start : start + self.crop_size]
        elif data_window.shape[1] > self.crop_size:
            start = (data_window.shape[1] - self.crop_size) // 2
            data_window = data_window[:, start : start + self.crop_size]

        if idx < 5:
            print(f"   Dataset.__getitem__({idx}): Cropped data shape={data_window.shape}")
            sys.stdout.flush()

        data_tensor = torch.from_numpy(np.asarray(data_window, dtype=np.float32))
        target_tensor = torch.tensor([float(target_value)], dtype=torch.float32)

        if idx < 5:
            print(f"   Dataset.__getitem__({idx}): Final tensors data={data_tensor.shape}, target={target_tensor.shape}")
            sys.stdout.flush()

        return data_tensor, target_tensor

    def close(self):
        if self._worker_handles is not None:
            files, _, _ = self._worker_handles
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass
            self._worker_handles = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_worker_datasets(self):
        """Open HDF5 handles lazily per worker process."""

        if self._worker_handles is None:
            files = []
            data_dsets = []
            target_dsets = []
            for path in self.cache_file_paths:
                f = h5py.File(path, "r")
                files.append(f)
                data_dsets.append(f["data"])
                target_dsets.append(f["targets"])
            self._worker_handles = (files, data_dsets, target_dsets)

        _, data_dsets, target_dsets = self._worker_handles
        return data_dsets, target_dsets

    def __getstate__(self):
        state = self.__dict__.copy()
        # HDF5 handles cannot be pickled; they are reopened lazily per worker.
        state["_worker_handles"] = None
        return state

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def create_training_run(challenge, model_name, config):
    """Register a new training run in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO training_runs (challenge, model_name, start_time, status, config)
        VALUES (?, ?, ?, ?, ?)
    ''', (challenge, model_name, datetime.now().isoformat(), 'running', json.dumps(config)))

    run_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return run_id

def log_epoch(run_id, epoch, train_loss, val_loss, lr, duration):
    """Log epoch metrics to database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO epoch_history (run_id, epoch, train_loss, val_loss, learning_rate, duration_seconds, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (run_id, epoch, train_loss, val_loss, lr, duration, datetime.now().isoformat()))

    conn.commit()
    conn.close()

def save_checkpoint_info(run_id, epoch, val_loss, file_path, is_best=False):
    """Register checkpoint in database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO model_checkpoints (run_id, epoch, val_loss, file_path, is_best, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (run_id, epoch, val_loss, str(file_path), is_best, datetime.now().isoformat()))

    conn.commit()
    conn.close()

def update_run_status(run_id, status, best_val_loss=None):
    """Update training run status."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if best_val_loss is not None:
        cursor.execute('''
            UPDATE training_runs
            SET status = ?, end_time = ?, best_val_loss = ?
            WHERE id = ?
        ''', (status, datetime.now().isoformat(), best_val_loss, run_id))
    else:
        cursor.execute('''
            UPDATE training_runs
            SET status = ?, end_time = ?
            WHERE id = ?
        ''', (status, datetime.now().isoformat(), run_id))

    conn.commit()
    conn.close()

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main(args=None):
    print("🔍 Inside main() function")
    sys.stdout.flush()

    if args is None:
        args = argparse.Namespace(
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            patience=PATIENCE,
            crop_size=CROP_SIZE,
            device_index=None,
            num_workers=0,
            pin_memory=None,
            no_augment=False,
            prefetch_factor=None,
            persistent_workers=None,
            use_amp=False,
            note="Training with R1+R2 only (R3-R5 still creating)",
            device="auto",
        )

    requested_device = getattr(args, "device", "auto").lower()
    prefer_gpu = requested_device in {"auto", "cuda"}
    device_index = args.device_index if args.device_index is not None else 0

    global DEVICE
    DEVICE = torch.device("cpu")

    if prefer_gpu:
        if not torch.cuda.is_available():
            if requested_device == "cuda":
                raise RuntimeError("CUDA/ROCm device requested but none are available.")
            print("⚠️  CUDA/ROCm device not available; continuing on CPU.")
            sys.stdout.flush()
        else:
            print(
                f"🧪 Running GPU health check on cuda:{device_index} (timeout={GPU_HEALTH_TIMEOUT}s)..."
            )
            sys.stdout.flush()
            healthy, failure_reason = run_gpu_health_check(device_index, GPU_HEALTH_TIMEOUT)
            if healthy:
                if args.device_index is not None:
                    torch.cuda.set_device(args.device_index)
                    DEVICE = torch.device(f"cuda:{args.device_index}")
                else:
                    DEVICE = torch.device("cuda")
                print("✅ GPU health check passed; using GPU for training.")
                sys.stdout.flush()
            else:
                print(f"⚠️  GPU health check failed: {failure_reason}")
                if requested_device == "cuda":
                    raise RuntimeError(
                        "GPU was explicitly requested but health check failed; see logs for details."
                    )
                print("   Falling back to CPU execution.")
                sys.stdout.flush()
    else:
        print("🧠 CPU-only mode requested; skipping GPU checks.")
        sys.stdout.flush()

    print(f"🖥️  Using device: {DEVICE}")
    sys.stdout.flush()

    batch_size = args.batch_size
    max_epochs = args.max_epochs
    learning_rate = args.learning_rate
    patience = args.patience
    crop_size = args.crop_size
    if args.num_workers is None:
        num_workers = recommend_num_workers()
        print(
            f"⚙️  Auto-selected num_workers={num_workers} based on cpu_count={os.cpu_count() or 'unknown'}"
        )
    else:
        num_workers = max(0, args.num_workers)
    pin_memory = args.pin_memory
    augment = not args.no_augment
    prefetch_factor = args.prefetch_factor
    persistent_workers = args.persistent_workers
    use_amp = args.use_amp
    note = args.note

    if pin_memory is None:
        pin_memory = DEVICE.type == "cuda"
    elif pin_memory and DEVICE.type != "cuda":
        print("⚠️  pin_memory=True is ineffective on CPU; forcing pin_memory=False.")
        pin_memory = False
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    elif persistent_workers and num_workers == 0:
        print("⚠️  Ignoring --persistent-workers because num_workers == 0")
        persistent_workers = False

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Check cache files
    print("📁 Checking cache files...")
    sys.stdout.flush()

    cache_candidates = [
        ("R1", CACHE_DIR / "challenge2_R1_windows.h5"),
        ("R2", CACHE_DIR / "challenge2_R2_windows.h5"),
        ("R3", CACHE_DIR / "challenge2_R3_windows.h5"),
        ("R4", CACHE_DIR / "challenge2_R4_windows.h5"),
        ("R5", CACHE_DIR / "challenge2_R5_windows.h5"),
    ]

    cache_files = []
    missing_required = []
    missing_optional = []

    for idx, (label, path) in enumerate(cache_candidates):
        if path.exists():
            cache_files.append(path)
        else:
            if idx < 2:
                missing_required.append((label, path))
            else:
                missing_optional.append((label, path))

    if missing_required:
        print("❌ Missing required cache files:")
        for label, path in missing_required:
            print(f"   {label}: {path}")
        print("Cannot proceed until the required caches are available.")
        return

    print(f"✅ Cache files detected: {len(cache_files)} total")
    for path in cache_files:
        size_gb = path.stat().st_size / (1024**3)
        print(f"   {path.name}: {size_gb:.1f} GB")

    if missing_optional:
        print("ℹ️ Optional cache files not yet available (will be included automatically once present):")
        for label, path in missing_optional:
            print(f"   {label}: {path}")

    print()
    sys.stdout.flush()

    # Register training run
    print("📊 Registering training run in database...")
    sys.stdout.flush()

    try:
        config = {
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'learning_rate': learning_rate,
            'patience': patience,
            'crop_size': crop_size,
            'model': 'EEGNeX',
            'optimizer': 'Adamax',
            'loss': 'L1',
            'note': note,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': persistent_workers,
            'amp': use_amp,
        }

        run_id = create_training_run(challenge=2, model_name='EEGNeX_R1R2', config=config)
        print(f"✅ Training run registered: ID = {run_id}")
    except Exception as e:
        print(f"⚠️  Database registration failed: {e}")
        print("   Continuing without database logging...")
        run_id = None
    print()
    sys.stdout.flush()


    # Load data
    print("="*80)
    print("PHASE 1: DATA LOADING")
    print("="*80)
    print()
    sys.stdout.flush()

    load_start = time.time()
    full_dataset = None

    try:
        print("🔄 Loading combined dataset (R1 + R2)...")
        print("   This may take 1-3 minutes for 23GB of data...")
        print(f"   Cache files to load: {len(cache_files)}")
        for i, cache_file in enumerate(cache_files):
            file_size = cache_file.stat().st_size / (1024**3)
            print(f"     {i+1}. {cache_file.name} ({file_size:.1f} GB)")
        sys.stdout.flush()

        dataset_start = time.time()
        full_dataset = CachedEEGDataset(
            cache_files,
            crop_size_samples=int(crop_size * SFREQ),
            augment=augment,
        )
        dataset_time = time.time() - dataset_start

        print("✅ Dataset loaded successfully!")
        print(f"   Loading took: {dataset_time:.2f} seconds")
        print(f"   Total windows: {len(full_dataset):,}")
        print(f"   Augmentation: {'enabled' if augment else 'disabled'}")
        print(f"   Crop size: {int(crop_size * SFREQ)} samples ({crop_size}s)")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        traceback.print_exc()
        return

    try:
        # Split 80/20 for train/val
        print("\n🔄 Splitting dataset (80% train, 20% val)...")
        sys.stdout.flush()

        n_total = len(full_dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train

        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

        print(f"✅ Data loaded in {time.time() - load_start:.1f}s")
        print(f"   Train: {len(train_dataset)} windows")
        print(f"   Val:   {len(val_dataset)} windows")
        print()
        sys.stdout.flush()

        # Create dataloaders
        print("🔄 Creating data loaders...")
        sys.stdout.flush()

        try:
            print("   Creating train DataLoader with:")
            print(f"     - batch_size: {batch_size}")
            print(f"     - num_workers: {num_workers}")
            print(f"     - pin_memory: {pin_memory}")
            print(f"     - prefetch_factor: {prefetch_factor}")
            print(f"     - persistent_workers: {persistent_workers}")
            sys.stdout.flush()

            train_loader_kwargs = {
                "dataset": train_dataset,
                "batch_size": batch_size,
                "shuffle": True,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": False,
            }
            val_loader_kwargs = {
                "dataset": val_dataset,
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": False,
            }

            if num_workers > 0:
                if prefetch_factor is not None:
                    train_loader_kwargs["prefetch_factor"] = prefetch_factor
                    val_loader_kwargs["prefetch_factor"] = prefetch_factor
                if persistent_workers is not None:
                    train_loader_kwargs["persistent_workers"] = persistent_workers
                    val_loader_kwargs["persistent_workers"] = persistent_workers

            print("   Creating train DataLoader object...")
            sys.stdout.flush()
            train_loader = DataLoader(**train_loader_kwargs)

            print("   Creating validation DataLoader object...")
            sys.stdout.flush()
            val_loader = DataLoader(**val_loader_kwargs)

            print("✅ DataLoaders created")
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Val batches: {len(val_loader)}")

            # Test first batch loading
            print("   Testing first batch loading...")
            sys.stdout.flush()
            test_start = time.time()
            train_iter = iter(train_loader)
            first_batch = next(train_iter)
            test_time = time.time() - test_start
            print(f"   ✅ First batch loaded in {test_time:.3f}s")
            print(f"   First batch shapes: X={first_batch[0].shape}, y={first_batch[1].shape}")
            print(f"   First batch dtypes: X={first_batch[0].dtype}, y={first_batch[1].dtype}")
            del train_iter, first_batch  # Clean up
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Failed to create dataloaders: {e}")
            traceback.print_exc()
            return
        print()
        sys.stdout.flush()

        # Create model
        print("=" * 80)
        print("PHASE 2: MODEL CREATION")
        print("=" * 80)
        print()

        print("Creating EEGNeX model...")
        print(f"  Input shape: (batch_size, {129}, {int(crop_size * SFREQ)})")
        print(f"  Sampling frequency: {SFREQ} Hz")
        print(f"  Crop size: {crop_size} seconds")
        sys.stdout.flush()

        model = EEGNeX(
            n_outputs=1,
            n_chans=129,
            n_times=int(crop_size * SFREQ),
            sfreq=SFREQ,
        )

        print("✅ Model created successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Model device placement with detailed info
        print(f"Moving model to device: {DEVICE}")
        sys.stdout.flush()

        model_move_start = time.time()
        model = model.to(DEVICE)
        model_move_time = time.time() - model_move_start

        print("✅ Model moved to device successfully")
        print(f"  Device placement took: {model_move_time:.3f}s")
        if DEVICE.type == "cuda":
            print("  Device type: GPU (ROCm)")
            # Clear GPU cache before checking memory
            torch.cuda.empty_cache()
            print(f"  GPU memory after model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            print("  Device type: CPU")


        # Test model with dummy input (DISABLED - causes ROCm crash)
        print("⚠️ Skipping model test due to ROCm compatibility issues...")
        print("   Will test model during first training batch instead.")
        print()
        sys.stdout.flush()

        print("Creating optimizer and loss function...")
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
        criterion = l1_loss
        print(f"✅ Optimizer: Adamax (lr={learning_rate})")
        print("✅ Loss function: L1 Loss")
        print()
        sys.stdout.flush()

        # Training loop
        print("=" * 80)
        print("PHASE 3: TRAINING")
        print("=" * 80)
        print()

        best_val_loss = float('inf')
        patience_counter = 0
        non_blocking = DEVICE.type == "cuda"
        amp_enabled = use_amp and DEVICE.type == "cuda"

        def get_autocast_context():
            """Get appropriate autocast context based on current device"""
            if DEVICE.type == "cuda" and amp_enabled:
                return partial(torch.amp.autocast, device_type="cuda", dtype=torch.float16)
            else:
                return nullcontext

        autocast_context = get_autocast_context()
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        def handle_rocm_failure(error_msg: str):
            """Gracefully fall back to CPU execution when ROCm misbehaves."""

            nonlocal non_blocking, amp_enabled, autocast_context, scaler, model
            global DEVICE

            if DEVICE.type != "cuda":
                print(f"\n❌ ROCm error after GPU fallback: {error_msg}")
                print("   Unable to recover because execution is already on CPU.")
                sys.stdout.flush()
                raise RuntimeError(error_msg)

            print(f"\n⚠️ ROCm GPU error detected: {error_msg}")
            print("   Switching execution to CPU, disabling AMP, and clearing GPU cache.")
            sys.stdout.flush()

            DEVICE = torch.device("cpu")
            model = model.cpu()
            non_blocking = False
            amp_enabled = False

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            autocast_context = get_autocast_context()
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            print("✅ CPU fallback ready; continuing training without GPU acceleration.")
            print("   Tip: verify PYTORCH_ROCM_ARCH / HSA_OVERRIDE_GFX_VERSION match your GPU (see docs).")
            sys.stdout.flush()

        for epoch in range(max_epochs):
            epoch_start = time.time()

            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{max_epochs}")
            print(f"{'='*80}")
            sys.stdout.flush()

            # Train
            model.train()
            train_losses = []

            print("🔄 Training phase...")
            print(f"   Preparing to process {len(train_loader)} batches...")
            sys.stdout.flush()

            log_interval = max(1, min(25, len(train_loader) // 20 or 1))
            recent_loss_sum = 0.0
            recent_time_sum = 0.0

            print(f"   Starting batch loop (will log every {log_interval} batches)...")
            sys.stdout.flush()

            for batch_idx, (X, y) in enumerate(train_loader):
                if batch_idx == 0:
                    first_msg = "GPU initialization" if DEVICE.type == "cuda" else "initial CPU execution"
                    print(f"   Processing first batch (this may take 10-30s for {first_msg})...")
                    prefetch_display = getattr(train_loader, "prefetch_factor", None)
                    if prefetch_display is None:
                        prefetch_display = "default"
                    print(
                        f"   DataLoader settings: workers={train_loader.num_workers}, prefetch={prefetch_display}"
                    )
                    current_model_device = next(model.parameters()).device
                    print(f"   Model device: {current_model_device}")
                    if current_model_device.type == "cuda":
                        print(f"   GPU memory before batch: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    else:
                        print("   GPU path unavailable; running on CPU after ROCm fallback.")
                    sys.stdout.flush()
                try:
                    batch_start = time.time()

                    # Detailed timing for first batch
                    if batch_idx == 0:
                        print(f"   Batch {batch_idx+1}: Input shapes X={X.shape}, y={y.shape}")
                        print(f"   Batch {batch_idx+1}: Data types X={X.dtype}, y={y.dtype}")
                        print(f"   Batch {batch_idx+1}: Data ranges X=[{X.min():.4f}, {X.max():.4f}], y=[{y.min():.4f}, {y.max():.4f}]")
                        sys.stdout.flush()

                    # Move to device with timing
                    move_start = time.time()
                    X = X.to(DEVICE, non_blocking=non_blocking)
                    y = y.to(DEVICE, non_blocking=non_blocking)
                    move_time = time.time() - move_start
                    if batch_idx == 0:
                        print(f"   Batch {batch_idx+1}: Data movement took {move_time:.3f}s")
                        print(f"   Batch {batch_idx+1}: Target device: {DEVICE}")
                        sys.stdout.flush()

                    # Zero gradients
                    zero_start = time.time()
                    optimizer.zero_grad()
                    zero_time = time.time() - zero_start
                    if batch_idx == 0:
                        print(f"   Batch {batch_idx+1}: Zero gradients took {zero_time:.3f}s")
                        sys.stdout.flush()

                    # Forward with ROCm-aware fallback
                    forward_start = time.time()
                    try:
                        with autocast_context():
                            y_pred = model(X)
                            loss = criterion(y_pred, y)
                        forward_time = time.time() - forward_start
                        if batch_idx == 0:
                            print(f"   Batch {batch_idx+1}: Forward pass took {forward_time:.3f}s")
                            print(f"   Batch {batch_idx+1}: Output shape {y_pred.shape}, loss {loss.item():.6f}")
                            sys.stdout.flush()
                    except RuntimeError as e:
                        error_message = str(e)
                        if "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION" in error_message or "rocdevice" in error_message.lower():
                            handle_rocm_failure(error_message)

                            # Ensure data tensors reside on CPU before retrying
                            X = X.cpu()
                            y = y.cpu()

                            with autocast_context():
                                y_pred = model(X)
                                loss = criterion(y_pred, y)
                            forward_time = time.time() - forward_start
                            print(f"   Batch {batch_idx+1}: CPU forward pass took {forward_time:.3f}s")
                            sys.stdout.flush()
                        else:
                            raise  # Re-raise if it's not ROCm-related

                    # Backward
                    backward_start = time.time()
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    backward_time = time.time() - backward_start
                    if batch_idx == 0:
                        print(f"   Batch {batch_idx+1}: Backward+step took {backward_time:.3f}s")
                        if DEVICE.type == "cuda":
                            print(f"   GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                        else:
                            print("   Running on CPU (no GPU memory usage).")
                        sys.stdout.flush()

                    loss_value = loss.detach().item()
                    batch_time = time.time() - batch_start
                    train_losses.append(loss_value)
                    recent_loss_sum += loss_value
                    recent_time_sum += batch_time

                    # Extra logging for first few batches
                    if batch_idx in [0, 1, 2, 3, 4]:
                        print(f"   Batch {batch_idx+1}: Total time {batch_time:.3f}s, Loss {loss_value:.6f}")
                        sys.stdout.flush()

                    should_log = (
                        batch_idx == 0
                        or (batch_idx + 1) % log_interval == 0
                        or (batch_idx + 1) == len(train_loader)
                    )

                    if should_log:
                        recent_count = min(log_interval, batch_idx + 1)
                        avg_recent = recent_loss_sum / max(1, recent_count)
                        avg_time = recent_time_sum / max(1, recent_count)
                        print(
                            f"  Train [{batch_idx+1:4d}/{len(train_loader)}] "
                            f"loss={loss_value:.6f} | avg_last={avg_recent:.6f} | "
                            f"batch_time={batch_time:.2f}s | avg_time={avg_time:.2f}s"
                        )
                        sys.stdout.flush()
                        recent_loss_sum = 0.0
                        recent_time_sum = 0.0

                except Exception as e:
                    print(f"\n❌ Error in training batch {batch_idx+1}: {e}")
                    traceback.print_exc()
                    raise

            # Validate
            print("\n🔄 Validation phase...")
            print(f"   Processing {len(val_loader)} validation batches...")
            sys.stdout.flush()

            model.eval()
            val_losses = []
            val_log_interval = max(1, min(20, len(val_loader) // 10 or 1))
            val_recent_loss_sum = 0.0
            val_recent_time_sum = 0.0

            print(f"   Will log every {val_log_interval} batches...")
            sys.stdout.flush()

            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(val_loader):
                    if batch_idx == 0:
                        print("   Processing first validation batch...")
                        sys.stdout.flush()
                    try:
                        batch_start = time.time()
                        X = X.to(DEVICE, non_blocking=non_blocking)
                        y = y.to(DEVICE, non_blocking=non_blocking)
                        with autocast_context():
                            y_pred = model(X)
                            loss = criterion(y_pred, y)
                        loss_value = loss.item()
                        batch_time = time.time() - batch_start
                        val_losses.append(loss_value)
                        val_recent_loss_sum += loss_value
                        val_recent_time_sum += batch_time

                        should_log = (
                            batch_idx == 0
                            or (batch_idx + 1) % val_log_interval == 0
                            or (batch_idx + 1) == len(val_loader)
                        )

                        if should_log:
                            recent_count = min(val_log_interval, batch_idx + 1)
                            avg_recent = val_recent_loss_sum / max(1, recent_count)
                            avg_time = val_recent_time_sum / max(1, recent_count)
                            print(
                                f"  Val   [{batch_idx+1:4d}/{len(val_loader)}] "
                                f"loss={loss_value:.6f} | avg_last={avg_recent:.6f} | "
                                f"batch_time={batch_time:.2f}s | avg_time={avg_time:.2f}s"
                            )
                            sys.stdout.flush()
                            val_recent_loss_sum = 0.0
                            val_recent_time_sum = 0.0

                    except Exception as e:
                        print(f"\n❌ Error in validation batch {batch_idx+1}: {e}")
                        traceback.print_exc()
                        raise

            # Metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            epoch_time = time.time() - epoch_start

            # Log
            if run_id is not None:
                log_epoch(run_id, epoch + 1, train_loss, val_loss, learning_rate, epoch_time)

            print(f"\n📊 Epoch {epoch+1}/{max_epochs}")
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss:   {val_loss:.6f}")
            print(f"   Time:       {epoch_time:.1f}s")

            # Save checkpoint
            checkpoint_path = CHECKPOINT_DIR / f"challenge2_r1r2_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)

            # Check if best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_path = CHECKPOINT_DIR / "challenge2_r1r2_best.pth"
                torch.save(model.state_dict(), best_path)
                print(f"   ⭐ New best model! Saved to {best_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"   Patience: {patience_counter}/{patience}")

            if run_id is not None:
                save_checkpoint_info(run_id, epoch + 1, val_loss, checkpoint_path, is_best)

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered (patience={patience})")
                break

            print()

        # Complete
        if run_id is not None:
            update_run_status(run_id, 'completed', best_val_loss)

        print("=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print("Best Model:    checkpoints/challenge2_r1r2_best.pth")
        print()
    finally:
        if full_dataset is not None:
            try:
                print("🔚 Closing HDF5 cache file handles...")
                sys.stdout.flush()
                full_dataset.close()
            except Exception as close_error:
                print(f"⚠️  Failed to close cache files cleanly: {close_error}")

if __name__ == '__main__':
    print("🔍 Starting main function...")
    sys.stdout.flush()

    try:
        print("📞 Calling main()...")
        sys.stdout.flush()
        cli_args = parse_args()
        main(cli_args)
        print("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print("\n\n❌ ERROR OCCURRED!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        sys.exit(1)
