#!/usr/bin/env python3
"""
Foundation Model Training Script - P2.2
Trains the advanced EEG foundation model on HBN dataset
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Train EEG Foundation Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--data_dir", type=str, default="data/raw/hbn", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/foundation", help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"🚀 Starting Foundation Model Training")
    print(f"   Config: {args.config}")
    print(f"   Data: {args.data_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   GPU: {args.gpu}")
    print()
    
    # Extract config values
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    
    # Build command for existing training script
    cmd_args = [
        sys.executable,
        str(Path(__file__).parent / "train_advanced.py"),
        f"--data_dir={args.data_dir}",
        f"--output_dir={args.output_dir}",
        f"--experiment_name=foundation_small",
        f"--hidden_dim={model_config.get('architecture', {}).get('hidden_dim', 512)}",
        f"--num_layers={model_config.get('architecture', {}).get('n_transformer_layers', 6)}",
        f"--num_heads={model_config.get('architecture', {}).get('n_attention_heads', 8)}",
        f"--dropout={model_config.get('architecture', {}).get('dropout', 0.1)}",
        f"--batch_size={training_config.get('batch_size', 8)}",
        f"--learning_rate={training_config.get('learning_rate', 0.0001)}",
        f"--warmup_steps={training_config.get('warmup_steps', 100)}",
        f"--ssl_epochs={training_config.get('max_epochs', 50) // 2}",
        f"--finetune_epochs={training_config.get('max_epochs', 50)}",
        "--use_domain_adaptation",
        "--use_task_adapters",
        "--use_compression_ssl",
    ]
    
    if config.get("logging", {}).get("wandb", False):
        cmd_args.append("--use_wandb")
    
    print("📋 Configuration loaded:")
    print(f"   Model: {model_config.get('name', 'advanced_foundation_model')}")
    print(f"   Hidden dim: {model_config.get('architecture', {}).get('hidden_dim', 512)}")
    print(f"   Layers: {model_config.get('architecture', {}).get('n_transformer_layers', 6)}")
    print(f"   Heads: {model_config.get('architecture', {}).get('n_attention_heads', 8)}")
    print(f"   Batch size: {training_config.get('batch_size', 8)}")
    print(f"   Epochs: {training_config.get('max_epochs', 50)}")
    print(f"   Learning rate: {training_config.get('learning_rate', 0.0001)}")
    print()
    
    # Import subprocess here to run the actual training
    import subprocess
    
    print(f"🏃 Executing training script...")
    print(f"   Command: {' '.join(cmd_args)}")
    print()
    
    try:
        result = subprocess.run(cmd_args, check=True)
        print("\n✅ Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n❌ Training script not found: {cmd_args[1]}")
        print("   Running in simplified mode instead...")
        return run_simplified_training(args, config)

def run_simplified_training(args, config):
    """Simplified training for when advanced script is not available."""
    print("\n�� Running simplified training mode...")
    print("   This will train the baseline model on available data")
    print()
    
    import sys
    from pathlib import Path
    
    # Try to run baseline training as fallback
    baseline_script = Path(__file__).parent / "train_baseline.py"
    if baseline_script.exists():
        import subprocess
        cmd = [
            sys.executable,
            str(baseline_script),
            f"--data={args.data_dir}/sub-*/"
        ]
        subprocess.run(cmd)
    else:
        print("⚠️  No training script available. Please check:")
        print(f"   1. {Path(__file__).parent / 'train_advanced.py'}")
        print(f"   2. {Path(__file__).parent / 'train_baseline.py'}")
        print()
        print("   For now, listing available data:")
        import glob
        subjects = glob.glob(f"{args.data_dir}/sub-*/")
        print(f"\n   Found {len(subjects)} subjects:")
        for subj in subjects[:5]:
            print(f"      - {Path(subj).name}")
        if len(subjects) > 5:
            print(f"      ... and {len(subjects) - 5} more")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
