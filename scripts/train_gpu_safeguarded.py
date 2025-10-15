#!/usr/bin/env python3
"""
GPU Training with Maximum Crash Prevention
- Progressive GPU load testing
- Memory overflow protection
- Driver stability checks
- Automatic fallback to CPU
- Graceful error recovery
"""

import os
import sys
import signal
import traceback
from pathlib import Path

# Set environment variables BEFORE importing torch
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['ROCR_VISIBLE_DEVICES'] = '0'
os.environ['GPU_MAX_HEAP_SIZE'] = '50'
os.environ['GPU_MAX_ALLOC_PERCENT'] = '50'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

print("🛡️  GPU Training with Maximum Safeguards")
print("=" * 70)

# Ultra-safe configuration
CONFIG = {
    'data_dir': Path(__file__).parent.parent / "data" / "raw" / "hbn",
    'output_dir': Path(__file__).parent.parent / "outputs",
    'checkpoint_dir': Path(__file__).parent.parent / "checkpoints",
    'log_dir': Path(__file__).parent.parent / "logs",

    # Data - Very limited for testing
    'max_subjects': 2,
    'max_windows_per_subject': 50,
    'train_split': 0.8,

    # Model - Minimal size
    'hidden_dim': 32,
    'n_heads': 2,
    'n_layers': 1,
    'dropout': 0.1,

    # Training - Ultra-conservative
    'batch_size': 1,  # One sample at a time
    'epochs': 2,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,

    # Safety
    'gradient_clip': 0.5,
    'save_every': 1,
    'max_memory_fraction': 0.3,  # Only use 30% of GPU memory
    'gpu_test_timeout': 5,  # Timeout for GPU tests
    'enable_checkpointing': True,
    'auto_fallback_cpu': True,  # Fallback to CPU if GPU fails
}

class GPUSafetyChecker:
    """Progressive GPU testing to catch issues early"""

    @staticmethod
    def test_gpu_available():
        """Test 1: Is GPU even available?"""
        print("\n🔍 Test 1: Checking GPU availability...")
        try:
            available = torch.cuda.is_available()
            if available:
                print(f"   ✅ GPU available: {torch.cuda.get_device_name(0)}")
                return True
            else:
                print("   ❌ No GPU available")
                return False
        except Exception as e:
            print(f"   ❌ Error checking GPU: {e}")
            return False

    @staticmethod
    def test_gpu_memory():
        """Test 2: Can we query GPU memory?"""
        print("\n🔍 Test 2: Checking GPU memory access...")
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_mem = props.total_memory / (1024**3)
                print(f"   ✅ Total memory: {total_mem:.2f} GB")
                return True
            return False
        except Exception as e:
            print(f"   ❌ Cannot access GPU memory: {e}")
            return False

    @staticmethod
    def test_tensor_creation(timeout=5):
        """Test 3: Can we create a tensor on GPU?"""
        print("\n🔍 Test 3: Testing tensor creation...")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(10, 10, device=device)
            print(f"   ✅ Created tensor on GPU: {x.shape}")
            del x
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"   ❌ Tensor creation failed: {e}")
            return False

    @staticmethod
    def test_computation(timeout=5):
        """Test 4: Can we do basic computation?"""
        print("\n🔍 Test 4: Testing GPU computation...")
        try:
            device = torch.device('cuda:0')
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            print(f"   ✅ Matrix multiplication successful: {c.shape}")
            del a, b, c
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"   ❌ Computation failed: {e}")
            return False

    @staticmethod
    def test_gradient_flow(timeout=5):
        """Test 5: Can we do backprop?"""
        print("\n🔍 Test 5: Testing gradient computation...")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(10, 10, device=device, requires_grad=True)
            y = (x ** 2).sum()
            y.backward()
            torch.cuda.synchronize()
            print(f"   ✅ Gradient computation successful")
            del x, y
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"   ❌ Gradient computation failed: {e}")
            return False

    @staticmethod
    def test_model_forward(model_class, n_channels, seq_len, timeout=10):
        """Test 6: Can we run a model forward pass?"""
        print("\n🔍 Test 6: Testing model forward pass...")
        try:
            device = torch.device('cuda:0')
            model = model_class(
                n_channels=n_channels,
                seq_len=seq_len,
                hidden_dim=16,
                n_heads=2,
                n_layers=1,
            ).to(device)

            x = torch.randn(1, n_channels, seq_len, device=device)
            y = model(x)
            torch.cuda.synchronize()
            print(f"   ✅ Model forward pass successful: {y.shape}")
            del model, x, y
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"   ❌ Model forward failed: {e}")
            return False

    @staticmethod
    def test_backward_pass(model_class, n_channels, seq_len, timeout=10):
        """Test 7: Can we run backprop through model?"""
        print("\n🔍 Test 7: Testing model backward pass...")
        try:
            device = torch.device('cuda:0')
            model = model_class(
                n_channels=n_channels,
                seq_len=seq_len,
                hidden_dim=16,
                n_heads=2,
                n_layers=1,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()

            x = torch.randn(1, n_channels, seq_len, device=device)
            target = torch.tensor([0], device=device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            print(f"   ✅ Backward pass successful, loss: {loss.item():.4f}")
            del model, optimizer, x, target, output, loss
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"   ❌ Backward pass failed: {e}")
            return False

    @classmethod
    def run_all_tests(cls, model_class=None, n_channels=129, seq_len=1000):
        """Run all safety tests progressively"""
        print("\n" + "=" * 70)
        print("🛡️  Running GPU Safety Tests")
        print("=" * 70)

        tests = [
            ("GPU Available", lambda: cls.test_gpu_available()),
            ("Memory Access", lambda: cls.test_gpu_memory()),
            ("Tensor Creation", lambda: cls.test_tensor_creation()),
            ("Basic Computation", lambda: cls.test_computation()),
            ("Gradient Flow", lambda: cls.test_gradient_flow()),
        ]

        if model_class:
            tests.append(("Model Forward", lambda: cls.test_model_forward(model_class, n_channels, seq_len)))
            tests.append(("Model Backward", lambda: cls.test_backward_pass(model_class, n_channels, seq_len)))

        passed = 0
        failed_test = None

        for test_name, test_fn in tests:
            try:
                if test_fn():
                    passed += 1
                else:
                    failed_test = test_name
                    break
            except Exception as e:
                print(f"   ❌ {test_name} exception: {e}")
                failed_test = test_name
                break

            # Small delay between tests
            time.sleep(0.5)

        print("\n" + "=" * 70)
        print(f"📊 Safety Tests: {passed}/{len(tests)} passed")

        if failed_test:
            print(f"❌ Failed at: {failed_test}")
            print("⚠️  GPU is not stable - will use CPU instead")
            return False
        else:
            print("✅ All tests passed - GPU appears stable")
            return True

class TinyTransformerEEG(nn.Module):
    """Minimal transformer for maximum stability"""
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=32,
                 n_heads=2, n_layers=1, n_classes=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

class SafeGPUTrainer:
    """Trainer with automatic crash recovery"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.crash_count = 0
        self.max_crashes = 3

    def safe_batch_process(self, data, target, criterion, optimizer):
        """Process one batch with error handling"""
        try:
            # Move data
            data = data.to(self.device, non_blocking=False)
            target = target.to(self.device, non_blocking=False)

            # Forward
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            optimizer.step()

            # Sync
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # Stats
            _, predicted = output.max(1)
            correct = predicted.eq(target).sum().item()

            return loss.item(), correct, target.size(0)

        except RuntimeError as e:
            error_msg = str(e)

            if "out of memory" in error_msg.lower():
                print(f"\n⚠️  OOM detected! Clearing cache...")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                return None, 0, 0

            elif "hip" in error_msg.lower() or "rocm" in error_msg.lower():
                print(f"\n⚠️  ROCm error detected: {error_msg}")
                self.crash_count += 1
                if self.crash_count >= self.max_crashes:
                    raise Exception("Too many ROCm errors - GPU unstable")
                return None, 0, 0

            else:
                print(f"\n❌ Unknown error: {error_msg}")
                raise

        finally:
            # Always cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

def train_epoch_safe(trainer, dataloader, criterion, optimizer, epoch):
    """Safe training epoch with error recovery"""
    trainer.model.train()
    total_loss = 0
    correct = 0
    total = 0
    failed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (data, target) in enumerate(pbar):
        result = trainer.safe_batch_process(data, target, criterion, optimizer)

        if result[0] is None:
            failed_batches += 1
            if failed_batches > 10:
                raise Exception("Too many failed batches - stopping")
            continue

        loss, batch_correct, batch_total = result
        total_loss += loss
        correct += batch_correct
        total += batch_total

        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'acc': f'{100.*correct/total:.1f}%',
            'fails': failed_batches
        })

    successful_batches = len(dataloader) - failed_batches
    avg_loss = total_loss / successful_batches if successful_batches > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0

    return avg_loss, accuracy, failed_batches

def validate_safe(model, dataloader, criterion, device, epoch):
    """Safe validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ")
        for data, target in pbar:
            try:
                data = data.to(device, non_blocking=False)
                target = target.to(device, non_blocking=False)

                output = model(data)
                loss = criterion(output, target)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })

            except Exception as e:
                print(f"\n⚠️  Validation error: {e}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy

def main():
    start_time = time.time()

    # Create directories
    CONFIG['output_dir'].mkdir(exist_ok=True)
    CONFIG['checkpoint_dir'].mkdir(exist_ok=True)
    CONFIG['log_dir'].mkdir(exist_ok=True)

    # Load minimal dataset first
    print(f"\n{'='*70}")
    print("📂 Loading Minimal Dataset")
    print(f"{'='*70}")

    try:
        dataset = SimpleEEGDataset(
            data_dir=CONFIG['data_dir'],
            max_subjects=CONFIG['max_subjects'],
            window_size=1000,
            verbose=True
        )

        if len(dataset) == 0:
            print("❌ No data loaded!")
            return

        # Limit windows
        max_windows = CONFIG['max_windows_per_subject'] * CONFIG['max_subjects']
        if len(dataset) > max_windows:
            indices = list(range(max_windows))
            dataset = Subset(dataset, indices)
            print(f"   Limited to {len(dataset)} windows")

        # Get data shape
        if isinstance(dataset, Subset):
            sample_data, _ = dataset.dataset[0]
        else:
            sample_data, _ = dataset[0]
        n_channels, seq_len = sample_data.shape
        print(f"   Data shape: {n_channels} channels, {seq_len} timepoints")

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return

    # Run GPU safety tests
    gpu_safe = GPUSafetyChecker.run_all_tests(
        model_class=TinyTransformerEEG,
        n_channels=n_channels,
        seq_len=seq_len
    )

    # Choose device
    if gpu_safe and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(CONFIG['max_memory_fraction'], device=device)
        print(f"\n✅ Using GPU with {CONFIG['max_memory_fraction']*100}% memory limit")
    else:
        device = torch.device('cpu')
        print(f"\n⚠️  Using CPU (GPU not stable or unavailable)")

    # Split dataset
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\n📊 Splits: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    # Create model
    print(f"\n{'='*70}")
    print("🧠 Creating Model")
    print(f"{'='*70}")

    try:
        model = TinyTransformerEEG(
            n_channels=n_channels,
            seq_len=seq_len,
            hidden_dim=CONFIG['hidden_dim'],
            n_heads=CONFIG['n_heads'],
            n_layers=CONFIG['n_layers'],
            dropout=CONFIG['dropout']
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")
        print(f"   Device: {device}")

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Create safe trainer
    trainer = SafeGPUTrainer(model, device, CONFIG)

    # Training
    print(f"\n{'='*70}")
    print("🏋️  Safe Training")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    try:
        for epoch in range(CONFIG['epochs']):
            epoch_start = time.time()

            # Train
            train_loss, train_acc, failed = train_epoch_safe(
                trainer, train_loader, criterion, optimizer, epoch
            )

            # Validate
            val_loss, val_acc = validate_safe(
                model, val_loader, criterion, device, epoch
            )

            epoch_time = time.time() - epoch_start

            # Record
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print
            print(f"\n📈 Epoch {epoch+1}/{CONFIG['epochs']}:")
            print(f"   Train: loss={train_loss:.4f}, acc={train_acc:.1f}% (failed={failed})")
            print(f"   Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
            print(f"   Time: {epoch_time:.1f}s")

            if device.type == 'cuda':
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                print(f"   GPU: {mem_alloc:.2f}GB")

            # Save
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = CONFIG['checkpoint_dir'] / f"safeguarded_best_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'device': str(device),
                }, checkpoint_path)
                print(f"   💾 Saved: {checkpoint_path.name}")

            # Cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # Complete
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("✅ Training Complete!")
        print(f"{'='*70}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print(f"   Device used: {device}")
        print(f"   Crashes: {trainer.crash_count}")

        # Save history
        history_path = CONFIG['log_dir'] / f"safeguarded_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            json.dump({
                'history': history,
                'config': {k: str(v) for k, v in CONFIG.items()},
                'device': str(device),
                'crashes': trainer.crash_count,
            }, f, indent=2)
        print(f"   History: {history_path.name}")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ Training failed: {e}")
        print(f"{'='*70}")
        traceback.print_exc()

        # Save partial results
        try:
            emergency_path = CONFIG['checkpoint_dir'] / f"emergency_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'error': str(e),
            }, emergency_path)
            print(f"\n💾 Emergency save: {emergency_path}")
        except:
            pass

    finally:
        # Final cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
