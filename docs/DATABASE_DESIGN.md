# 🗄️ MongoDB Database Design for EEG2025 AI/ML Pipeline

**Status**: ✅ Production Ready  
**Version**: 1.0  
**Last Updated**: October 20, 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Database Architecture](#database-architecture)
3. [Collections Schema](#collections-schema)
4. [AI/ML Integration Flow](#aiml-integration-flow)
5. [Usage Examples](#usage-examples)
6. [Query Patterns](#query-patterns)
7. [Performance Optimization](#performance-optimization)
8. [Migration from SQLite](#migration-from-sqlite)

---

## Overview

### Purpose

MongoDB serves as the **central experiment tracking system** for the EEG2025 Foundation Model project. It replaces SQLite to provide:

- **Concurrent access**: Multiple training jobs can write simultaneously
- **Distributed queries**: Access experiment data from any machine
- **Rich queries**: Complex aggregations and filtering
- **Real-time monitoring**: Change streams for live updates
- **Flexible schema**: Evolving metadata without schema migrations

### Technology Stack

```yaml
Database: MongoDB 7.0
Driver: PyMongo 4.6.0+
UI: Mongo Express (Web-based)
Storage: WiredTiger Engine (1.5GB cache)
Network: Docker bridge network
Ports:
  - MongoDB: 27017
  - Web UI: 8082
```

---

## Database Architecture

### High-Level Structure

```
eeg2025 (Database)
├── experiments    → Training experiments (N experiments)
├── epochs         → Per-epoch metrics (N × epochs records)
├── checkpoints    → Model checkpoints (N × saved models)
└── subjects       → Subject metadata (1,600+ subjects)
```

### Design Philosophy

1. **Denormalization**: Store frequently-accessed data together
2. **Embedded documents**: Nest related data (config within experiment)
3. **References**: Use ObjectId for parent-child relationships
4. **Indices**: Strategic indices for common query patterns
5. **Atomic updates**: Use MongoDB operators for concurrent safety

---

## Collections Schema

### 1. `experiments` Collection

**Purpose**: Primary collection tracking each training run

```javascript
{
  // Unique identifier (auto-generated)
  _id: ObjectId("671234567890abcdef123456"),
  
  // Experiment metadata
  experiment_name: "eegnex_r1r2_20251020_164100",
  challenge: 2,                    // Challenge number (1 or 2)
  status: "running",               // running | completed | failed | stopped
  
  // Model information
  model: {
    name: "EEGNeX",
    architecture: "transformer",
    parameters: 2457821,
    version: "1.0"
  },
  
  // Training configuration (embedded document)
  config: {
    batch_size: 16,
    max_epochs: 20,
    learning_rate: 0.002,
    optimizer: "Adamax",
    loss: "L1",
    patience: 5,
    crop_size: 2.0,
    num_workers: 2,
    prefetch_factor: 2,
    persistent_workers: false,
    amp: false,
    device: "cpu"
  },
  
  // Dataset information
  dataset: {
    releases: ["R1", "R2"],
    train_windows: 103724,
    val_windows: 25931,
    cache_files: [
      "data/cached/challenge2_R1_windows.h5",
      "data/cached/challenge2_R2_windows.h5"
    ]
  },
  
  // Performance metrics (updated during training)
  metrics: {
    best_val_loss: 0.0452,
    best_epoch: 15,
    total_epochs: 20,
    final_train_loss: 0.0389,
    final_val_loss: 0.0452,
    training_time_hours: 18.5
  },
  
  // Timestamps
  start_time: ISODate("2025-10-20T16:41:00.000Z"),
  end_time: ISODate("2025-10-21T11:15:00.000Z"),
  
  // User and environment
  created_by: "kevin",
  hostname: "gpu-workstation",
  
  // Organization
  tags: ["baseline", "cpu", "phase1"],
  note: "CPU training with 20 epochs",
  
  // References
  parent_experiment_id: null,      // For experiment continuation
  checkpoint_dir: "checkpoints/eegnex_r1r2_20251020_164100/"
}
```

**Indices**:
```javascript
experiments.createIndex({ challenge: 1 })
experiments.createIndex({ status: 1 })
experiments.createIndex({ start_time: -1 })
experiments.createIndex({ tags: 1 })
experiments.createIndex({ experiment_name: 1, start_time: -1 })
```

---

### 2. `epochs` Collection

**Purpose**: Per-epoch training metrics for detailed analysis

```javascript
{
  // Unique identifier
  _id: ObjectId("671234567890abcdef654321"),
  
  // Parent reference
  experiment_id: ObjectId("671234567890abcdef123456"),
  
  // Epoch information
  epoch: 15,
  
  // Training metrics
  metrics: {
    train_loss: 0.0389,
    val_loss: 0.0452,
    learning_rate: 0.002,
    
    // Optional: Task-specific metrics
    val_mae: 0.0452,
    val_rmse: 0.0687,
    val_r2: 0.7834,
    
    // Optional: Per-dataset metrics
    val_loss_r1: 0.0445,
    val_loss_r2: 0.0459
  },
  
  // Timing information
  timing: {
    duration_seconds: 3240.5,
    samples_per_second: 31.98,
    epoch_start: ISODate("2025-10-20T16:41:00.000Z"),
    epoch_end: ISODate("2025-10-20T17:35:00.000Z")
  },
  
  // Timestamp
  timestamp: ISODate("2025-10-20T17:35:00.000Z"),
  
  // Optional: Memory usage
  memory: {
    gpu_allocated_mb: 0,
    gpu_cached_mb: 0,
    cpu_rss_mb: 4096
  }
}
```

**Indices**:
```javascript
epochs.createIndex({ experiment_id: 1, epoch: 1 }, { unique: true })
epochs.createIndex({ timestamp: -1 })
```

**Unique Constraint**: Each (experiment_id, epoch) pair is unique

---

### 3. `checkpoints` Collection

**Purpose**: Track saved model checkpoints

```javascript
{
  // Unique identifier
  _id: ObjectId("671234567890abcdef789012"),
  
  // Parent reference
  experiment_id: ObjectId("671234567890abcdef123456"),
  
  // Checkpoint information
  epoch: 15,
  is_best: true,                   // Only one per experiment
  
  // Performance at checkpoint
  metrics: {
    val_loss: 0.0452,
    train_loss: 0.0389,
    val_mae: 0.0452
  },
  
  // File information
  file: {
    path: "checkpoints/eegnex_r1r2_20251020_164100/best_model.pt",
    size_mb: 9.8,
    checksum: "sha256:a3f5e8d9c2b1...",
    format: "pytorch"
  },
  
  // Model state
  model_state: {
    optimizer: "Adamax",
    scheduler: "ReduceLROnPlateau",
    epoch: 15,
    step: 97405
  },
  
  // Timestamp
  timestamp: ISODate("2025-10-20T17:35:00.000Z"),
  
  // Optional: Validation results
  validation: {
    n_samples: 25931,
    inference_time_ms: 15234.5,
    predictions_file: "outputs/predictions_epoch15.csv"
  }
}
```

**Indices**:
```javascript
checkpoints.createIndex({ experiment_id: 1 })
checkpoints.createIndex({ is_best: 1 })
checkpoints.createIndex({ experiment_id: 1, epoch: -1 })
```

---

### 4. `subjects` Collection

**Purpose**: Subject demographics and metadata (optional, for analysis)

```javascript
{
  // Unique identifier
  _id: ObjectId("671234567890abcdef345678"),
  
  // Subject information
  subject_id: "sub-NDARAA075AMK",
  
  // Demographics
  age: 12.5,
  sex: "F",
  handedness: "R",
  
  // Data availability
  releases: ["R1", "R2", "R3"],
  has_eeg: true,
  has_demographics: true,
  has_phenotypic: true,
  
  // Phenotypic data
  phenotypic: {
    adhd_score: 65,
    anxiety_score: 58,
    depression_score: 52,
    
    // CBCL scores
    cbcl_anxious_depressed: 58,
    cbcl_withdrawn_depressed: 52,
    cbcl_somatic_complaints: 55,
    // ... more scores
  },
  
  // EEG quality metrics
  eeg_quality: {
    n_channels: 128,
    sampling_rate: 500,
    duration_seconds: 600,
    artifact_ratio: 0.12,
    
    // Per-release metrics
    r1_windows: 38,
    r2_windows: 42,
    r3_windows: 45
  },
  
  // Dataset splits
  splits: {
    challenge1_split: "train",     // train | val | test
    challenge2_split: "val"
  },
  
  // Timestamps
  date_added: ISODate("2025-06-15T00:00:00.000Z"),
  last_processed: ISODate("2025-10-15T12:30:00.000Z")
}
```

**Indices**:
```javascript
subjects.createIndex({ releases: 1 })
subjects.createIndex({ age: 1 })
subjects.createIndex({ splits.challenge1_split: 1 })
subjects.createIndex({ splits.challenge2_split: 1 })
```

---

## AI/ML Integration Flow

### Training Lifecycle

```python
from src.data.nosql_backend import MongoExperimentTracker

# 1. Initialize tracker
tracker = MongoExperimentTracker()

# 2. Create experiment (start of training)
exp_id = tracker.create_experiment(
    experiment_name="eegnex_r1r2_20251020",
    challenge=2,
    model={
        'name': 'EEGNeX',
        'parameters': 2457821
    },
    config={
        'batch_size': 16,
        'max_epochs': 20,
        'learning_rate': 0.002
    },
    dataset={
        'releases': ['R1', 'R2'],
        'train_windows': 103724
    },
    tags=['baseline', 'cpu']
)

# 3. Training loop
for epoch in range(max_epochs):
    # Train
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    # Log epoch metrics
    tracker.log_epoch(
        experiment_id=exp_id,
        epoch=epoch,
        metrics={
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        },
        timing={
            'duration_seconds': epoch_time,
            'samples_per_second': len(train_loader.dataset) / epoch_time
        }
    )
    
    # Save checkpoint
    if val_loss < best_val_loss:
        save_checkpoint(model, f'checkpoints/best_model.pt')
        
        tracker.save_checkpoint_info(
            experiment_id=exp_id,
            epoch=epoch,
            metrics={'val_loss': val_loss},
            file={
                'path': 'checkpoints/best_model.pt',
                'size_mb': get_file_size('checkpoints/best_model.pt')
            },
            is_best=True
        )

# 4. Complete experiment
tracker.update_experiment_status(
    experiment_id=exp_id,
    status='completed',
    metrics={
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_epochs': max_epochs
    }
)

# 5. Close connection
tracker.close()
```

### Integration Points

**Current SQLite functions → MongoDB methods**:

```python
# OLD (SQLite)
run_id = create_training_run(challenge=2, model_name='EEGNeX', config={})
log_epoch(run_id, epoch, train_loss, val_loss, lr, duration)
save_checkpoint_info(run_id, epoch, val_loss, file_path, is_best)
update_run_status(run_id, status='completed', best_val_loss=0.045)

# NEW (MongoDB)
exp_id = tracker.create_experiment(challenge=2, model={'name': 'EEGNeX'}, config={})
tracker.log_epoch(exp_id, epoch, {'train_loss': train_loss, 'val_loss': val_loss})
tracker.save_checkpoint_info(exp_id, epoch, {'val_loss': val_loss}, file={...}, is_best=True)
tracker.update_experiment_status(exp_id, status='completed', metrics={'best_val_loss': 0.045})
```

---

## Usage Examples

### Example 1: Simple Training Run

```python
from src.data.nosql_backend import get_tracker

# Get tracker instance
tracker = get_tracker()

# Start experiment
exp_id = tracker.create_experiment(
    experiment_name="test_run",
    challenge=1,
    model={'name': 'EEGNeX'},
    config={'batch_size': 32, 'epochs': 10}
)

# Training loop
for epoch in range(10):
    # ... train model ...
    
    tracker.log_epoch(
        experiment_id=exp_id,
        epoch=epoch,
        metrics={'train_loss': 0.5, 'val_loss': 0.6}
    )

# Complete
tracker.update_experiment_status(exp_id, 'completed')
tracker.close()
```

### Example 2: Query Best Models

```python
tracker = get_tracker()

# Get top 5 models for Challenge 2
best_models = tracker.get_best_models(challenge=2, n=5)

for model in best_models:
    print(f"Experiment: {model['experiment_name']}")
    print(f"Val Loss: {model['metrics']['best_val_loss']}")
    print(f"Config: {model['config']}")
    print()
```

### Example 3: Compare Experiments

```python
tracker = get_tracker()

# Compare two experiments
exp_ids = ['671234567890abcdef123456', '671234567890abcdef654321']
comparison = tracker.compare_experiments(exp_ids)

for exp in comparison:
    print(f"{exp['experiment_name']}: {exp['metrics']['best_val_loss']}")
```

### Example 4: Search with Filters

```python
tracker = get_tracker()

# Find all CPU experiments from last week
from datetime import datetime, timedelta

one_week_ago = datetime.utcnow() - timedelta(days=7)

experiments = tracker.search_experiments(
    query={
        'config.device': 'cpu',
        'start_time': {'$gte': one_week_ago},
        'status': 'completed'
    },
    sort=[('metrics.best_val_loss', 1)],  # 1 = ascending
    limit=10
)
```

### Example 5: Get Complete History

```python
tracker = get_tracker()

# Get all data for one experiment
exp_id = '671234567890abcdef123456'
history = tracker.get_experiment_history(exp_id)

experiment = history['experiment']
epochs = history['epochs']
checkpoints = history['checkpoints']

# Plot learning curves
import matplotlib.pyplot as plt

train_losses = [e['metrics']['train_loss'] for e in epochs]
val_losses = [e['metrics']['val_loss'] for e in epochs]

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.legend()
plt.show()
```

---

## Query Patterns

### Common Queries

#### 1. Latest Running Experiments

```python
experiments = tracker.search_experiments(
    query={'status': 'running'},
    sort=[('start_time', -1)]
)
```

#### 2. Best Model Per Challenge

```python
# Using aggregation pipeline
pipeline = [
    {'$match': {'status': 'completed'}},
    {'$sort': {'metrics.best_val_loss': 1}},
    {'$group': {
        '_id': '$challenge',
        'best_experiment': {'$first': '$$ROOT'}
    }}
]

results = list(tracker.experiments.aggregate(pipeline))
```

#### 3. Experiments by Tag

```python
experiments = tracker.search_experiments(
    query={'tags': 'baseline'},
    projection={'experiment_name': 1, 'metrics': 1}
)
```

#### 4. Failed Experiments (for debugging)

```python
failed = tracker.search_experiments(
    query={'status': 'failed'},
    sort=[('start_time', -1)],
    limit=20
)
```

#### 5. Training Time Analysis

```python
# Average training time by model
pipeline = [
    {'$match': {'status': 'completed'}},
    {'$group': {
        '_id': '$model.name',
        'avg_hours': {'$avg': '$metrics.training_time_hours'},
        'count': {'$sum': 1}
    }},
    {'$sort': {'avg_hours': -1}}
]

results = list(tracker.experiments.aggregate(pipeline))
```

---

## Performance Optimization

### Index Strategy

MongoDB automatically uses indices for:
- **Exact matches**: `{'challenge': 2}`
- **Range queries**: `{'start_time': {'$gte': date}}`
- **Sorting**: `.sort([('start_time', -1)])`

### Query Performance Tips

1. **Use projection** to limit returned fields:
   ```python
   tracker.search_experiments(
       query={'challenge': 2},
       projection={'experiment_name': 1, 'metrics': 1, '_id': 0}
   )
   ```

2. **Limit results** for large datasets:
   ```python
   tracker.search_experiments(query={}, limit=100)
   ```

3. **Use indices** in queries:
   ```python
   # Good - uses index
   {'challenge': 2, 'status': 'completed'}
   
   # Bad - no index on nested field
   {'config.learning_rate': 0.001}
   ```

4. **Aggregation pipelines** for complex analysis:
   ```python
   pipeline = [
       {'$match': {'challenge': 2}},
       {'$group': {'_id': '$model.name', 'count': {'$sum': 1}}},
       {'$sort': {'count': -1}}
   ]
   ```

### Monitoring Performance

Check index usage:
```python
# Explain query
tracker.experiments.find({'challenge': 2}).explain()
```

---

## Migration from SQLite

### SQLite Schema Mapping

| SQLite Table | MongoDB Collection | Notes |
|-------------|-------------------|-------|
| `training_runs` | `experiments` | Direct mapping with enriched schema |
| `epoch_history` | `epochs` | Foreign key → ObjectId reference |
| `model_checkpoints` | `checkpoints` | Foreign key → ObjectId reference |
| `subjects` | `subjects` | Optional, for analysis |
| `cache_files` | *(embedded in experiments)* | Part of `dataset` field |

### Migration Script

```python
import sqlite3
from src.data.nosql_backend import MongoExperimentTracker

# Connect to both databases
sqlite_conn = sqlite3.connect('data/metadata.db')
tracker = MongoExperimentTracker()

# Migrate training runs
cursor = sqlite_conn.execute('SELECT * FROM training_runs')
for row in cursor:
    run_id, challenge, model_name, start_time, end_time, status, \
        best_val_loss, best_epoch, total_epochs, config, notes = row
    
    # Create experiment in MongoDB
    exp_id = tracker.create_experiment(
        experiment_name=f"migrated_{run_id}",
        challenge=challenge,
        model={'name': model_name},
        config=json.loads(config) if config else {},
        status=status,
        start_time=datetime.fromisoformat(start_time),
        note=notes
    )
    
    # Update with final metrics
    if status == 'completed':
        tracker.update_experiment_status(
            exp_id,
            status='completed',
            end_time=datetime.fromisoformat(end_time) if end_time else None,
            metrics={
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'total_epochs': total_epochs
            }
        )
    
    # Migrate epochs
    epoch_cursor = sqlite_conn.execute(
        'SELECT * FROM epoch_history WHERE run_id = ?', (run_id,)
    )
    for epoch_row in epoch_cursor:
        _, _, epoch, train_loss, val_loss, lr, duration, timestamp = epoch_row
        tracker.log_epoch(
            exp_id,
            epoch,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': lr
            },
            timing={'duration_seconds': duration}
        )

sqlite_conn.close()
tracker.close()

print("✅ Migration complete!")
```

---

## Summary

### Benefits Over SQLite

| Feature | SQLite | MongoDB |
|---------|--------|---------|
| Concurrent writes | ❌ Single writer | ✅ Unlimited |
| Distributed access | ❌ Local only | ✅ Network access |
| Complex queries | ⚠️ Limited SQL | ✅ Rich aggregation |
| Schema flexibility | ❌ Fixed schema | ✅ Dynamic schema |
| Real-time monitoring | ❌ Polling only | ✅ Change streams |
| Web UI | ❌ None | ✅ Mongo Express |

### Next Steps

1. ✅ **Setup complete** - MongoDB running
2. 🔄 **Integration** - Update training scripts to use MongoDB
3. 📊 **Migrate** - (Optional) Move historical data from SQLite
4. 🎯 **Query** - Start using rich query capabilities
5. 📈 **Monitor** - Build dashboards with real-time data

---

**Last Updated**: October 20, 2025  
**Author**: EEG2025 Team
