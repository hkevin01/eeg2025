#!/usr/bin/env python3
"""
Interactive MongoDB Database Demo
==================================
Demonstrates the database structure and capabilities for EEG2025 AI/ML pipeline.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.nosql_backend import MongoExperimentTracker
from datetime import datetime, timedelta
import json


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"🔍 {text}")
    print("="*80)


def print_subheader(text):
    """Print a formatted subheader."""
    print(f"\n📊 {text}")
    print("-"*80)


def demo_create_experiment():
    """Demo: Create a sample experiment."""
    print_header("DEMO 1: Create Experiment")
    
    tracker = MongoExperimentTracker()
    
    # Create a demo experiment
    exp_id = tracker.create_experiment(
        experiment_name="demo_experiment_20251020",
        challenge=2,
        model={
            'name': 'EEGNeX',
            'architecture': 'transformer',
            'parameters': 2457821,
            'version': '1.0'
        },
        config={
            'batch_size': 16,
            'max_epochs': 20,
            'learning_rate': 0.002,
            'optimizer': 'Adamax',
            'loss': 'L1',
            'patience': 5,
            'device': 'cpu'
        },
        dataset={
            'releases': ['R1', 'R2'],
            'train_windows': 103724,
            'val_windows': 25931,
            'cache_files': [
                'data/cached/challenge2_R1_windows.h5',
                'data/cached/challenge2_R2_windows.h5'
            ]
        },
        tags=['demo', 'baseline', 'cpu'],
        note='Demo experiment to showcase MongoDB structure'
    )
    
    print(f"\n✅ Experiment created with ID: {exp_id}")
    print(f"\nExperiment document structure:")
    print(json.dumps({
        '_id': 'ObjectId(...)',
        'experiment_name': 'demo_experiment_20251020',
        'challenge': 2,
        'status': 'running',
        'model': {'name': 'EEGNeX', 'parameters': 2457821},
        'config': {'batch_size': 16, 'learning_rate': 0.002},
        'dataset': {'releases': ['R1', 'R2'], 'train_windows': 103724},
        'tags': ['demo', 'baseline', 'cpu'],
        'start_time': datetime.utcnow().isoformat()
    }, indent=2))
    
    tracker.close()
    return exp_id


def demo_log_epochs(exp_id):
    """Demo: Log epoch metrics."""
    print_header("DEMO 2: Log Training Epochs")
    
    tracker = MongoExperimentTracker()
    
    print("\nSimulating training loop with 5 epochs...")
    
    # Simulate 5 epochs
    for epoch in range(5):
        # Simulated losses (decreasing over time)
        train_loss = 0.15 - (epoch * 0.02)
        val_loss = 0.18 - (epoch * 0.015)
        lr = 0.002 * (0.9 ** epoch)  # Decaying learning rate
        
        tracker.log_epoch(
            experiment_id=exp_id,
            epoch=epoch,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': lr,
                'val_mae': val_loss,
                'val_rmse': val_loss * 1.5
            },
            timing={
                'duration_seconds': 3200 + (epoch * 50),
                'samples_per_second': 32.5
            }
        )
        
        print(f"  ✓ Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    print(f"\n✅ Logged 5 epochs to MongoDB")
    print(f"\nEpoch document structure:")
    print(json.dumps({
        '_id': 'ObjectId(...)',
        'experiment_id': 'ObjectId(...)',
        'epoch': 4,
        'metrics': {
            'train_loss': 0.07,
            'val_loss': 0.12,
            'learning_rate': 0.0013,
            'val_mae': 0.12,
            'val_rmse': 0.18
        },
        'timing': {
            'duration_seconds': 3400,
            'samples_per_second': 32.5
        },
        'timestamp': datetime.utcnow().isoformat()
    }, indent=2))
    
    tracker.close()


def demo_save_checkpoints(exp_id):
    """Demo: Save checkpoint information."""
    print_header("DEMO 3: Save Checkpoints")
    
    tracker = MongoExperimentTracker()
    
    # Save regular checkpoint
    tracker.save_checkpoint_info(
        experiment_id=exp_id,
        epoch=3,
        metrics={'val_loss': 0.135, 'train_loss': 0.09},
        file={
            'path': f'checkpoints/demo/epoch_3.pt',
            'size_mb': 9.8,
            'checksum': 'sha256:abc123...',
            'format': 'pytorch'
        },
        is_best=False
    )
    print("  ✓ Saved checkpoint for epoch 3")
    
    # Save best checkpoint
    tracker.save_checkpoint_info(
        experiment_id=exp_id,
        epoch=4,
        metrics={'val_loss': 0.12, 'train_loss': 0.07},
        file={
            'path': f'checkpoints/demo/best_model.pt',
            'size_mb': 9.8,
            'checksum': 'sha256:def456...',
            'format': 'pytorch'
        },
        is_best=True
    )
    print("  ✓ Saved BEST checkpoint for epoch 4")
    
    print(f"\n✅ Checkpoint tracking enabled")
    print(f"\nCheckpoint document structure:")
    print(json.dumps({
        '_id': 'ObjectId(...)',
        'experiment_id': 'ObjectId(...)',
        'epoch': 4,
        'is_best': True,
        'metrics': {'val_loss': 0.12, 'train_loss': 0.07},
        'file': {
            'path': 'checkpoints/demo/best_model.pt',
            'size_mb': 9.8,
            'checksum': 'sha256:def456...'
        },
        'timestamp': datetime.utcnow().isoformat()
    }, indent=2))
    
    tracker.close()


def demo_update_experiment(exp_id):
    """Demo: Update experiment status."""
    print_header("DEMO 4: Complete Experiment")
    
    tracker = MongoExperimentTracker()
    
    tracker.update_experiment_status(
        experiment_id=exp_id,
        status='completed',
        metrics={
            'best_val_loss': 0.12,
            'best_epoch': 4,
            'total_epochs': 5,
            'final_train_loss': 0.07,
            'final_val_loss': 0.12,
            'training_time_hours': 4.5
        }
    )
    
    print(f"✅ Experiment marked as completed")
    print(f"\nFinal experiment document includes:")
    print(json.dumps({
        'status': 'completed',
        'end_time': datetime.utcnow().isoformat(),
        'metrics': {
            'best_val_loss': 0.12,
            'best_epoch': 4,
            'total_epochs': 5,
            'training_time_hours': 4.5
        }
    }, indent=2))
    
    tracker.close()


def demo_query_experiments():
    """Demo: Query experiments."""
    print_header("DEMO 5: Query Database")
    
    tracker = MongoExperimentTracker()
    
    # Query 1: All experiments
    print_subheader("Query 1: All Experiments")
    all_exps = tracker.search_experiments(
        query={},
        projection={'experiment_name': 1, 'status': 1, 'challenge': 1, '_id': 0},
        limit=10
    )
    print(f"Found {len(all_exps)} experiments")
    for exp in all_exps[:3]:
        print(f"  • {exp.get('experiment_name', 'N/A')} - {exp.get('status', 'N/A')} (Challenge {exp.get('challenge', 'N/A')})")
    
    # Query 2: Running experiments
    print_subheader("Query 2: Running Experiments")
    running = tracker.search_experiments(
        query={'status': 'running'},
        sort=[('start_time', -1)]
    )
    print(f"Found {len(running)} running experiments")
    
    # Query 3: Demo experiments
    print_subheader("Query 3: Demo Experiments (by tag)")
    demos = tracker.search_experiments(
        query={'tags': 'demo'},
        projection={'experiment_name': 1, 'tags': 1, '_id': 0}
    )
    print(f"Found {len(demos)} demo experiments")
    for exp in demos:
        print(f"  • {exp.get('experiment_name', 'N/A')} - tags: {exp.get('tags', [])}")
    
    # Query 4: Best models for Challenge 2
    print_subheader("Query 4: Best Models for Challenge 2")
    best = tracker.get_best_models(challenge=2, n=3)
    print(f"Top {len(best)} models:")
    for i, model in enumerate(best, 1):
        exp_name = model.get('experiment_name', 'N/A')
        val_loss = model.get('metrics', {}).get('best_val_loss', 'N/A')
        print(f"  {i}. {exp_name} - Val Loss: {val_loss}")
    
    tracker.close()


def demo_experiment_history(exp_id):
    """Demo: Get complete experiment history."""
    print_header("DEMO 6: Complete Experiment History")
    
    tracker = MongoExperimentTracker()
    
    history = tracker.get_experiment_history(exp_id)
    
    if history:
        exp = history['experiment']
        epochs = history['epochs']
        checkpoints = history['checkpoints']
        
        print(f"\n📄 Experiment: {exp.get('experiment_name')}")
        print(f"   Status: {exp.get('status')}")
        print(f"   Model: {exp.get('model', {}).get('name')}")
        print(f"   Challenge: {exp.get('challenge')}")
        
        print(f"\n📈 Training History ({len(epochs)} epochs):")
        for epoch in epochs[:5]:  # Show first 5
            e = epoch.get('epoch', 0)
            metrics = epoch.get('metrics', {})
            print(f"   Epoch {e}: train_loss={metrics.get('train_loss', 'N/A'):.4f}, "
                  f"val_loss={metrics.get('val_loss', 'N/A'):.4f}")
        
        print(f"\n💾 Checkpoints ({len(checkpoints)} saved):")
        for ckpt in checkpoints:
            epoch = ckpt.get('epoch', 0)
            is_best = ckpt.get('is_best', False)
            val_loss = ckpt.get('metrics', {}).get('val_loss', 'N/A')
            marker = "⭐ BEST" if is_best else ""
            print(f"   Epoch {epoch}: val_loss={val_loss:.4f} {marker}")
    else:
        print("❌ Experiment not found")
    
    tracker.close()


def show_database_structure():
    """Show the complete database structure."""
    print_header("DATABASE STRUCTURE")
    
    print("""
MongoDB: eeg2025
├── experiments     → Training experiments
│   ├── _id (ObjectId)
│   ├── experiment_name (string)
│   ├── challenge (int: 1 or 2)
│   ├── status (string: running|completed|failed|stopped)
│   ├── model (object)
│   │   ├── name (string)
│   │   ├── architecture (string)
│   │   ├── parameters (int)
│   │   └── version (string)
│   ├── config (object)
│   │   ├── batch_size (int)
│   │   ├── learning_rate (float)
│   │   ├── optimizer (string)
│   │   └── ... (flexible schema)
│   ├── dataset (object)
│   │   ├── releases (array of strings)
│   │   ├── train_windows (int)
│   │   └── cache_files (array of strings)
│   ├── metrics (object)
│   │   ├── best_val_loss (float)
│   │   ├── best_epoch (int)
│   │   └── total_epochs (int)
│   ├── tags (array of strings)
│   ├── start_time (ISODate)
│   └── end_time (ISODate)
│
├── epochs          → Per-epoch metrics
│   ├── _id (ObjectId)
│   ├── experiment_id (ObjectId reference)
│   ├── epoch (int)
│   ├── metrics (object)
│   │   ├── train_loss (float)
│   │   ├── val_loss (float)
│   │   └── learning_rate (float)
│   ├── timing (object)
│   │   └── duration_seconds (float)
│   └── timestamp (ISODate)
│
├── checkpoints     → Model checkpoints
│   ├── _id (ObjectId)
│   ├── experiment_id (ObjectId reference)
│   ├── epoch (int)
│   ├── is_best (boolean)
│   ├── metrics (object)
│   ├── file (object)
│   │   ├── path (string)
│   │   ├── size_mb (float)
│   │   └── checksum (string)
│   └── timestamp (ISODate)
│
└── subjects        → Subject metadata (optional)
    ├── _id (ObjectId)
    ├── subject_id (string)
    ├── age (float)
    ├── sex (string)
    ├── releases (array of strings)
    └── phenotypic (object)

Indices (for fast queries):
• experiments: challenge, status, start_time, tags
• epochs: (experiment_id, epoch) [unique], timestamp
• checkpoints: experiment_id, is_best, (experiment_id, epoch)
• subjects: releases, age
    """)


def main():
    """Run all demos."""
    print("\n")
    print("🗄️ " + "="*78)
    print("   MONGODB DATABASE DESIGN DEMO - EEG2025 AI/ML PIPELINE")
    print("="*82)
    
    # Show structure first
    show_database_structure()
    
    input("\n🎯 Press Enter to start interactive demos...")
    
    # Demo 1: Create experiment
    exp_id = demo_create_experiment()
    input("\n⏯️  Press Enter to continue...")
    
    # Demo 2: Log epochs
    demo_log_epochs(exp_id)
    input("\n⏯️  Press Enter to continue...")
    
    # Demo 3: Save checkpoints
    demo_save_checkpoints(exp_id)
    input("\n⏯️  Press Enter to continue...")
    
    # Demo 4: Complete experiment
    demo_update_experiment(exp_id)
    input("\n⏯️  Press Enter to continue...")
    
    # Demo 5: Query database
    demo_query_experiments()
    input("\n⏯️  Press Enter to continue...")
    
    # Demo 6: Get history
    demo_experiment_history(exp_id)
    
    # Summary
    print_header("SUMMARY")
    print("""
✅ MongoDB Setup Complete!

📊 Collections Created:
   • experiments  - Training runs with full configuration
   • epochs       - Per-epoch metrics for analysis
   • checkpoints  - Model checkpoint tracking
   
🔍 Query Capabilities:
   • Find best models by validation loss
   • Filter by challenge, status, tags
   • Get complete training history
   • Compare experiments side-by-side
   • Real-time monitoring (via change streams)

🌐 Web UI: http://localhost:8082 (admin/pass123)

📚 Full Documentation: docs/DATABASE_DESIGN.md

🚀 Next Steps:
   1. Integrate into training scripts (replace SQLite calls)
   2. Explore data via Web UI
   3. Build custom queries for your analysis
   4. (Optional) Migrate historical SQLite data
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
