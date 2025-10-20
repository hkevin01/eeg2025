# NoSQL Integration for EEG2025 ML Pipeline

## Executive Summary

This document outlines a comprehensive strategy for integrating NoSQL databases into the EEG2025 machine learning pipeline to improve performance, scalability, and operational efficiency.

## Current State Analysis

### Existing Data Architecture

1. **Storage Layer**
   - **HDF5 Files**: Primary storage for preprocessed EEG windows (23GB+ per dataset)
   - **SQLite Database**: `data/metadata.db` for experiment tracking
   - **File System**: Checkpoints, logs, and artifacts scattered across directories

2. **Data Flow**
   - Raw BIDS data → Preprocessing → HDF5 cache files
   - Training scripts → SQLite for run metadata
   - Model checkpoints → File system
   - Validation metrics → SQLite epoch_history table

3. **Pain Points**
   - ❌ SQLite limited to single-machine, no distributed access
   - ❌ Checkpoint metadata not easily queryable
   - ❌ No real-time monitoring dashboard possible
   - ❌ Feature extraction results not cached/reused
   - ❌ No distributed training coordination
   - ❌ HDF5 multiprocessing issues (current workaround: num_workers=0)

## Proposed NoSQL Integration Strategy

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EEG2025 ML Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐   ┌──────────────┐ │
│  │   MongoDB    │      │    Redis     │   │    HDF5      │ │
│  │              │      │              │   │              │ │
│  │ • Metadata   │      │ • Features   │   │ • Raw Data   │ │
│  │ • Experiments│      │ • Caching    │   │ • Windows    │ │
│  │ • Models     │◄────►│ • Queue      │   │ • Targets    │ │
│  │ • Results    │      │ • Metrics    │   │              │ │
│  └──────────────┘      └──────────────┘   └──────────────┘ │
│         ▲                     ▲                   ▲          │
│         │                     │                   │          │
│         └─────────────────────┴───────────────────┘          │
│                      Training Scripts                         │
└─────────────────────────────────────────────────────────────┘
```

### Technology Selection

#### 1. **MongoDB** - Primary Metadata Store
**Use Cases:**
- Experiment tracking (replaces/enhances SQLite)
- Model registry and versioning
- Hyperparameter configurations
- Training run history
- Subject metadata enrichment

**Benefits:**
- ✅ Flexible schema for evolving experiment metadata
- ✅ Rich query language for filtering/aggregating results
- ✅ Built-in replication and sharding
- ✅ GridFS for storing small binary artifacts
- ✅ Change streams for real-time monitoring
- ✅ Python PyMongo driver is mature and well-documented

**Why MongoDB vs Alternatives:**
- Better for complex nested documents than PostgreSQL JSON
- More mature than CouchDB for ML use cases
- Easier operations than Cassandra
- Better query flexibility than DynamoDB

#### 2. **Redis** - High-Speed Feature Cache & Queue
**Use Cases:**
- Feature vector caching (avoid recomputing)
- Preprocessed data caching
- Training job queue/coordination
- Real-time metrics aggregation
- Session/worker state management

**Benefits:**
- ✅ Sub-millisecond latency for feature serving
- ✅ Built-in pub/sub for distributed training
- ✅ TTL support for automatic cache expiration
- ✅ Atomic operations for coordination
- ✅ Redis Stack for vector similarity search (future: EEG embedding search)

**Why Redis vs Alternatives:**
- Faster than Memcached with more features
- Simpler than Apache Kafka for our scale
- Better Python support than etcd

#### 3. **Keep HDF5** - Raw Data Storage
**Rationale:**
- HDF5 is optimal for large numerical arrays
- No need to move 100GB+ of preprocessed data
- Continue memory-mapped access patterns
- NoSQL serves metadata/features, not raw signals

## Implementation Phases

### Phase 1: MongoDB Integration (Week 1-2)

#### A. Setup & Schema Design

**Collections:**

```python
# experiments collection
{
  "_id": ObjectId("..."),
  "experiment_name": "challenge2_r1r2_baseline",
  "challenge": 2,
  "model": {
    "name": "EEGNeX",
    "architecture": {...},
    "parameters": 62353
  },
  "config": {
    "batch_size": 16,
    "learning_rate": 0.002,
    "optimizer": "Adamax",
    "epochs": 20,
    "device": "cpu"
  },
  "data": {
    "releases": ["R1", "R2"],
    "train_windows": 103724,
    "val_windows": 25931,
    "augmentation": true
  },
  "status": "running",  # running, completed, failed, stopped
  "start_time": ISODate("2025-10-20T14:15:00Z"),
  "end_time": null,
  "created_by": "kevin",
  "tags": ["baseline", "cpu"],
  "notes": "Challenge2 CPU training 20 epochs"
}

# epochs collection
{
  "_id": ObjectId("..."),
  "experiment_id": ObjectId("..."),  # reference
  "epoch": 1,
  "metrics": {
    "train_loss": 0.892,
    "val_loss": 0.945,
    "learning_rate": 0.002
  },
  "timing": {
    "duration_seconds": 3241.5,
    "batches_per_second": 2.0
  },
  "timestamp": ISODate("2025-10-20T15:09:41Z")
}

# checkpoints collection
{
  "_id": ObjectId("..."),
  "experiment_id": ObjectId("..."),
  "epoch": 5,
  "metrics": {
    "val_loss": 0.823
  },
  "file": {
    "path": "checkpoints/challenge2_epoch5.pth",
    "size_mb": 0.24,
    "checksum": "sha256:abc123..."
  },
  "is_best": true,
  "timestamp": ISODate("2025-10-20T16:30:22Z")
}

# subjects collection (enriched metadata)
{
  "_id": "sub-NDARAA075AMK",
  "age": 12.5,
  "sex": "M",
  "p_factor": 0.234,
  "releases": ["R1"],
  "sessions": [
    {
      "session_id": "ses-HBNsiteRU01",
      "n_trials": 145,
      "data_quality": 0.89,
      "processed": true,
      "cache_file": "data/cached/challenge2_R1_windows.h5",
      "window_indices": [0, 61888]  # slice in HDF5
    }
  ],
  "annotations": {
    "outlier": false,
    "notes": "Good data quality"
  }
}
```

#### B. Python Integration Module

Create `src/data/nosql_backend.py`:

```python
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

class MongoExperimentTracker:
    """MongoDB-backed experiment tracking system."""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB URI (default: reads from env MONGO_URI)
        """
        self.uri = connection_string or os.getenv(
            'MONGO_URI', 
            'mongodb://localhost:27017/'
        )
        self.client = MongoClient(self.uri)
        self.db = self.client['eeg2025']
        
        # Collections
        self.experiments = self.db['experiments']
        self.epochs = self.db['epochs']
        self.checkpoints = self.db['checkpoints']
        self.subjects = self.db['subjects']
        
        self._ensure_indices()
    
    def _ensure_indices(self):
        """Create necessary indices for performance."""
        # Experiments
        self.experiments.create_index([('challenge', ASCENDING)])
        self.experiments.create_index([('status', ASCENDING)])
        self.experiments.create_index([('start_time', DESCENDING)])
        self.experiments.create_index([('tags', ASCENDING)])
        
        # Epochs
        self.epochs.create_index([
            ('experiment_id', ASCENDING),
            ('epoch', ASCENDING)
        ], unique=True)
        
        # Checkpoints
        self.checkpoints.create_index([('experiment_id', ASCENDING)])
        self.checkpoints.create_index([('is_best', ASCENDING)])
        
        # Subjects
        self.subjects.create_index([('releases', ASCENDING)])
    
    def create_experiment(self, **kwargs) -> str:
        """Create new experiment and return ID."""
        doc = {
            'start_time': datetime.utcnow(),
            'status': 'running',
            'created_by': os.getenv('USER', 'unknown'),
            **kwargs
        }
        result = self.experiments.insert_one(doc)
        return str(result.inserted_id)
    
    def log_epoch(self, experiment_id: str, epoch: int, metrics: Dict):
        """Log epoch metrics."""
        self.epochs.insert_one({
            'experiment_id': experiment_id,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.utcnow()
        })
    
    def save_checkpoint_info(self, experiment_id: str, **kwargs):
        """Register checkpoint."""
        self.checkpoints.insert_one({
            'experiment_id': experiment_id,
            'timestamp': datetime.utcnow(),
            **kwargs
        })
    
    def update_experiment_status(self, experiment_id: str, 
                                  status: str, **kwargs):
        """Update experiment status and metadata."""
        update = {
            'status': status,
            'end_time': datetime.utcnow(),
            **kwargs
        }
        self.experiments.update_one(
            {'_id': experiment_id},
            {'$set': update}
        )
    
    def get_best_models(self, challenge: int, n: int = 10):
        """Get top N models for a challenge."""
        pipeline = [
            {'$match': {'challenge': challenge, 'status': 'completed'}},
            {'$sort': {'metrics.best_val_loss': ASCENDING}},
            {'$limit': n},
            {'$project': {
                'experiment_name': 1,
                'model.name': 1,
                'metrics.best_val_loss': 1,
                'start_time': 1,
                'config': 1
            }}
        ]
        return list(self.experiments.aggregate(pipeline))
    
    def search_experiments(self, query: Dict = None, **kwargs):
        """Flexible experiment search."""
        query = query or {}
        return list(self.experiments.find(query, **kwargs))
```

#### C. Redis Integration Module

Create `src/data/redis_cache.py`:

```python
import redis
import pickle
import hashlib
from typing import Any, Optional
import numpy as np

class RedisFeatureCache:
    """Redis-backed feature and data cache."""
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False  # for binary data
        )
        self.default_ttl = 3600  # 1 hour
    
    def cache_features(self, key: str, features: np.ndarray, 
                       ttl: int = None):
        """Cache computed features."""
        ttl = ttl or self.default_ttl
        serialized = pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL)
        self.redis.setex(f"features:{key}", ttl, serialized)
    
    def get_features(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached features."""
        data = self.redis.get(f"features:{key}")
        if data:
            return pickle.loads(data)
        return None
    
    def cache_preprocessed_window(self, window_id: str, 
                                   data: np.ndarray, ttl: int = None):
        """Cache preprocessed EEG window."""
        ttl = ttl or self.default_ttl
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        checksum = hashlib.sha256(serialized).hexdigest()[:8]
        
        pipe = self.redis.pipeline()
        pipe.setex(f"window:{window_id}", ttl, serialized)
        pipe.setex(f"window:{window_id}:checksum", ttl, checksum)
        pipe.execute()
    
    def get_window(self, window_id: str) -> Optional[np.ndarray]:
        """Retrieve cached window."""
        data = self.redis.get(f"window:{window_id}")
        if data:
            return pickle.loads(data)
        return None
    
    def increment_metric(self, key: str, amount: float = 1.0):
        """Atomic counter for metrics."""
        return self.redis.incrbyfloat(f"metric:{key}", amount)
    
    def publish_training_update(self, channel: str, message: dict):
        """Publish training progress update."""
        self.redis.publish(channel, pickle.dumps(message))
    
    def subscribe_training_updates(self, channel: str):
        """Subscribe to training updates."""
        pubsub = self.redis.pubsub()
        pubsub.subscribe(channel)
        return pubsub
    
    def set_worker_state(self, worker_id: str, state: dict, ttl: int = 300):
        """Set worker state with heartbeat."""
        self.redis.setex(
            f"worker:{worker_id}",
            ttl,
            pickle.dumps(state)
        )
    
    def get_active_workers(self) -> List[str]:
        """Get list of active workers."""
        pattern = "worker:*"
        return [k.decode('utf-8').split(':')[1] 
                for k in self.redis.keys(pattern)]
```

### Phase 2: Update Training Scripts (Week 2-3)

#### Modified Training Script Example

```python
# scripts/training/train_challenge2_nosql.py
from src.data.nosql_backend import MongoExperimentTracker
from src.data.redis_cache import RedisFeatureCache

# Initialize backends
mongo = MongoExperimentTracker()
redis_cache = RedisFeatureCache()

# Create experiment
experiment_id = mongo.create_experiment(
    experiment_name=f"challenge2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    challenge=2,
    model={
        'name': 'EEGNeX',
        'parameters': model.count_parameters()
    },
    config={
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': 'Adamax',
        'epochs': args.max_epochs,
        'device': str(DEVICE)
    },
    data={
        'releases': ['R1', 'R2'],
        'train_windows': len(train_dataset),
        'val_windows': len(val_dataset)
    },
    tags=['nosql-backend', 'optimized'],
    notes=args.note
)

# Training loop with MongoDB logging
for epoch in range(1, args.max_epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
    
    # Log to MongoDB
    mongo.log_epoch(
        experiment_id=experiment_id,
        epoch=epoch,
        metrics={
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    )
    
    # Publish real-time update to Redis
    redis_cache.publish_training_update(
        channel=f"training:{experiment_id}",
        message={
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
    )
    
    # Save checkpoint
    if val_loss < best_val_loss:
        checkpoint_path = save_checkpoint(...)
        mongo.save_checkpoint_info(
            experiment_id=experiment_id,
            epoch=epoch,
            metrics={'val_loss': val_loss},
            file={'path': str(checkpoint_path)},
            is_best=True
        )

# Mark complete
mongo.update_experiment_status(
    experiment_id=experiment_id,
    status='completed',
    metrics={'best_val_loss': best_val_loss}
)
```

### Phase 3: Real-Time Dashboard (Week 4)

Create web dashboard using:
- **Streamlit** or **Dash** for quick prototyping
- **MongoDB Change Streams** for live updates
- **Redis Pub/Sub** for real-time metrics

```python
# dashboard/realtime_monitor.py
import streamlit as st
from src.data.nosql_backend import MongoExperimentTracker

st.title("EEG2025 Training Monitor")

mongo = MongoExperimentTracker()

# Live experiments
running = mongo.search_experiments({'status': 'running'})
for exp in running:
    st.subheader(exp['experiment_name'])
    
    # Get latest epochs
    epochs = list(mongo.epochs.find(
        {'experiment_id': exp['_id']}
    ).sort('epoch', -1).limit(10))
    
    # Plot loss curves
    st.line_chart({
        'epoch': [e['epoch'] for e in epochs],
        'train_loss': [e['metrics']['train_loss'] for e in epochs],
        'val_loss': [e['metrics']['val_loss'] for e in epochs]
    })
```

### Phase 4: Feature Store Integration (Week 5-6)

#### Option A: Redis as Simple Feature Store

```python
class SimpleFeatureStore:
    """Lightweight feature store using Redis."""
    
    def __init__(self):
        self.redis = RedisFeatureCache()
    
    def compute_and_cache_features(self, window_data, feature_name):
        """Compute features once, cache for reuse."""
        cache_key = f"{feature_name}_{hash_data(window_data)}"
        
        # Check cache first
        cached = self.redis.get_features(cache_key)
        if cached is not None:
            return cached
        
        # Compute
        features = compute_features(window_data)
        
        # Cache
        self.redis.cache_features(cache_key, features, ttl=86400)  # 24h
        
        return features
```

#### Option B: Full Feast Integration (Advanced)

```python
# feature_repo/features.py
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from datetime import timedelta

# Define entity
subject = Entity(
    name="subject_id",
    value_type=ValueType.STRING,
    description="EEG subject identifier"
)

# Define feature view
subject_features = FeatureView(
    name="subject_features",
    entities=["subject_id"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="age", dtype=ValueType.FLOAT),
        Feature(name="sex", dtype=ValueType.STRING),
        Feature(name="p_factor", dtype=ValueType.FLOAT),
        Feature(name="avg_alpha_power", dtype=ValueType.FLOAT),
        Feature(name="avg_beta_power", dtype=ValueType.FLOAT),
    ],
    online=True,
    source=FileSource(
        path="data/features/subject_features.parquet",
        event_timestamp_column="timestamp"
    )
)
```

## Deployment Guide

### Step 1: Install Dependencies

```bash
# Add to requirements.txt
pymongo>=4.6.0
redis>=5.0.0
feast>=0.55.0  # optional, for advanced feature store

# Install
pip install -r requirements.txt
```

### Step 2: Start Services

```bash
# MongoDB (Docker)
docker run -d \
  --name eeg2025-mongo \
  -p 27017:27017 \
  -v $(pwd)/data/mongodb:/data/db \
  mongo:7.0

# Redis (Docker)
docker run -d \
  --name eeg2025-redis \
  -p 6379:6379 \
  redis:7.2-alpine

# Or use docker-compose (create docker-compose.yml)
docker-compose up -d
```

### Step 3: Initialize MongoDB Schema

```bash
python scripts/infrastructure/initialize_mongodb.py
```

### Step 4: Migrate Existing SQLite Data

```bash
python scripts/infrastructure/migrate_sqlite_to_mongo.py
```

### Step 5: Update Environment Variables

```bash
# .env
MONGO_URI=mongodb://localhost:27017/
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Benefits Summary

### Performance Improvements

| Metric | Before (SQLite) | After (NoSQL) | Improvement |
|--------|----------------|---------------|-------------|
| Concurrent writes | 1 (file lock) | Unlimited | ∞ |
| Query complexity | Limited SQL | Rich aggregations | 10x |
| Real-time updates | None | Change streams | New capability |
| Feature retrieval | Recompute | Redis cache | 100x faster |
| Distributed access | ❌ | ✅ | New capability |

### Operational Improvements

- ✅ **Real-time monitoring**: Dashboard shows live training progress
- ✅ **Better experiment tracking**: Rich metadata, tagging, search
- ✅ **Feature reuse**: Compute once, cache, reuse across experiments
- ✅ **Distributed training**: Coordinate multiple workers via Redis
- ✅ **Easier debugging**: Query experiments, compare runs
- ✅ **Production-ready**: MongoDB/Redis scale to production workloads

## Migration Path

### Minimal (Quick Wins)

1. Add MongoDB for new experiments (keep SQLite for historical)
2. Use Redis for feature caching only
3. **Time:** 1-2 days
4. **Risk:** Very low (additive only)

### Moderate (Recommended)

1. MongoDB for all metadata (migrate SQLite)
2. Redis for feature caching and metrics
3. Simple dashboard for monitoring
4. **Time:** 1-2 weeks
5. **Risk:** Low (can rollback to SQLite)

### Complete (Production)

1. Full MongoDB + Redis integration
2. Feast feature store
3. Real-time dashboard
4. Distributed training coordination
5. **Time:** 4-6 weeks
6. **Risk:** Medium (requires testing)

## Testing Strategy

```python
# tests/test_nosql_backend.py
def test_mongodb_experiment_tracking():
    tracker = MongoExperimentTracker()
    
    # Create experiment
    exp_id = tracker.create_experiment(
        experiment_name="test",
        challenge=2,
        model={'name': 'EEGNeX'}
    )
    
    # Log epochs
    for epoch in range(3):
        tracker.log_epoch(
            experiment_id=exp_id,
            epoch=epoch + 1,
            metrics={'train_loss': 1.0 - epoch * 0.1}
        )
    
    # Verify
    epochs = list(tracker.epochs.find({'experiment_id': exp_id}))
    assert len(epochs) == 3
    assert epochs[-1]['metrics']['train_loss'] == 0.8

def test_redis_feature_caching():
    cache = RedisFeatureCache()
    
    # Cache features
    features = np.random.randn(100, 50)
    cache.cache_features("test_key", features)
    
    # Retrieve
    cached = cache.get_features("test_key")
    np.testing.assert_array_equal(features, cached)
```

## Monitoring & Operations

### Health Checks

```python
def check_nosql_health():
    """Verify NoSQL services are accessible."""
    try:
        mongo = MongoExperimentTracker()
        mongo.client.server_info()
        print("✅ MongoDB: Connected")
    except Exception as e:
        print(f"❌ MongoDB: {e}")
    
    try:
        redis_cache = RedisFeatureCache()
        redis_cache.redis.ping()
        print("✅ Redis: Connected")
    except Exception as e:
        print(f"❌ Redis: {e}")
```

### Backup Strategy

```bash
# MongoDB backup
mongodump --db eeg2025 --out backup/$(date +%Y%m%d)

# Redis backup
redis-cli BGSAVE
```

## Cost Analysis

### Self-Hosted (Recommended for Development)

- **MongoDB**: Free (Community Edition)
- **Redis**: Free (Open Source)
- **Hardware**: Existing server resources
- **Total**: $0/month

### Cloud-Hosted (Production)

- **MongoDB Atlas**: $57/month (M10 dedicated cluster)
- **Redis Cloud**: $30/month (1GB cache)
- **Total**: ~$87/month

### ROI Calculation

- Development time saved: ~20 hours/month
- Faster iteration: 2x experiment velocity
- Better model selection: +5% performance improvement
- **Value**: $2000+/month for research team

## Conclusion

Integrating MongoDB and Redis provides:
1. **Better experiment tracking** - Rich queries, real-time updates
2. **Performance gains** - Feature caching, parallel access
3. **Scalability** - Distributed training, production-ready
4. **Minimal disruption** - Additive changes, keep HDF5 for bulk data

**Recommended next step:** Start with Phase 1 (MongoDB integration) for immediate benefits with low risk.

---

**Document Version:** 1.0  
**Last Updated:** October 20, 2025  
**Authors:** Kevin (with AI assistance)
**Status:** 📋 Proposal Ready for Implementation
