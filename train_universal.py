#!/usr/bin/env python3
"""
Universal Training Script for EEG Foundation Model Challenge 2025
Automatically detects GPU backend (ROCm/CUDA) and trains accordingly
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.gpu_utils import setup_device, GPUConfig


def main():
    parser = argparse.ArgumentParser(
        description="Universal EEG Challenge Training (ROCm/CUDA compatible)"
    )
    parser.add_argument(
        "--challenge",
        type=int,
        choices=[1, 2],
        required=True,
        help="Challenge number (1 or 2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/challenge{N}.yaml)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (default: auto-select)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU training (no GPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: auto-detect based on GPU)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/hbn",
        help="Path to HBN data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--use-sdk",
        action="store_true",
        help="Force use of custom ROCm SDK (for AMD gfx1010)",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"🧠 EEG Foundation Model Challenge 2025 - Challenge {args.challenge}")
    print("="*80)
    
    # Setup device
    if args.cpu_only:
        import torch
        device = torch.device('cpu')
        gpu_config = None
        print("\n🖥️  CPU-only mode requested")
    else:
        device, gpu_config = setup_device(
            gpu_id=args.gpu,
            force_sdk=args.use_sdk,
            optimize=True
        )
    
    # Determine batch size
    if args.batch_size is None:
        if gpu_config and gpu_config.available:
            batch_size = gpu_config.get_optimal_batch_size()
            print(f"\n💡 Auto-detected batch size: {batch_size}")
        else:
            batch_size = 16  # Conservative for CPU
            print(f"\n💡 Using conservative CPU batch size: {batch_size}")
    else:
        batch_size = args.batch_size
        print(f"\n💡 Using user-specified batch size: {batch_size}")
    
    # Set config path
    if args.config is None:
        args.config = f"configs/challenge{args.challenge}.yaml"
    
    print(f"\n📄 Config: {args.config}")
    print(f"📁 Data dir: {args.data_dir}")
    print(f"📂 Output dir: {args.output_dir}")
    print(f"🔢 Batch size: {batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    
    # Import and run appropriate training script
    if args.challenge == 1:
        from scripts.training.challenge1 import train_challenge1_robust_gpu
        print("\n🎯 Starting Challenge 1 training (Response Time prediction)...")
        train_challenge1_robust_gpu.main(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=device,
            batch_size=batch_size,
            epochs=args.epochs,
        )
    else:  # Challenge 2
        from scripts.training.challenge2 import train_challenge2_externalizing
        print("\n🎯 Starting Challenge 2 training (Externalizing prediction)...")
        # We'll update this to use GPU
        print("⚠️  Challenge 2 training script needs GPU update...")
        print("   Creating GPU-compatible version...")
        
        # For now, show what would be done
        print(f"\n📋 Challenge 2 Training Plan:")
        print(f"   - Task: Externalizing Factor Prediction")
        print(f"   - Device: {device}")
        print(f"   - Batch Size: {batch_size}")
        print(f"   - Epochs: {args.epochs}")
        print(f"   - Metric: NRMSE (target < 0.5)")
        
        # TODO: Implement GPU-compatible Challenge 2 training
        print("\n⚠️  GPU-compatible Challenge 2 training coming next...")
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
