# MongoDB Integration for EEG2025

**Simple, focused solution for experiment tracking**

## Why MongoDB Only?

✅ **Solves the main problem**: SQLite's single-writer limitation  
✅ **Rich queries**: Aggregations, sorting, filtering experiments  
✅ **Scalable**: Handles concurrent writes, distributed access  
✅ **Simple**: One service, easy to manage  
✅ **Lightweight**: ~1.5GB RAM vs 2-3GB for MongoDB+Redis  

## Quick Start

```bash
# One command setup
./scripts/infrastructure/setup_mongo.sh

# That's it! MongoDB is running.
```

## What You Get

- **MongoDB**: Experiment tracking database (port 27017)
- **Web UI**: Browse data at http://localhost:8082 (admin/pass123)
- **Python API**: Clean `MongoExperimentTracker` class

## Usage Example

```python
from src.data.nosql_backend import MongoExperimentTracker

mongo = MongoExperimentTracker()

# Create experiment
exp_id = mongo.create_experiment(
    experiment_name="challenge2_baseline",
    challenge=2,
    model={'name': 'EEGNeX', 'parameters': 62353},
    config={'batch_size': 16, 'lr': 0.002}
)

# Log training progress
for epoch in range(epochs):
    mongo.log_epoch(exp_id, epoch, {
        'train_loss': train_loss,
        'val_loss': val_loss
    })

# Mark complete
mongo.update_experiment_status(exp_id, 'completed')

# Query best models
best_models = mongo.get_best_models(challenge=2, n=10)
```

## Benefits vs SQLite

| Feature | SQLite | MongoDB |
|---------|--------|---------|
| Concurrent writes | ❌ 1 only | ✅ Unlimited |
| Complex queries | 🟡 Limited | ✅ Rich |
| Real-time updates | ❌ | ✅ |
| Distributed access | ❌ | ✅ |
| Web UI | ❌ | ✅ |

## Management

```bash
# Stop MongoDB
docker-compose -f docker/docker-compose.mongo.yml stop

# Start MongoDB
docker-compose -f docker/docker-compose.mongo.yml start

# View logs
docker-compose -f docker/docker-compose.mongo.yml logs -f

# Remove everything
docker-compose -f docker/docker-compose.mongo.yml down -v
```

## Files

- **Setup**: `scripts/infrastructure/setup_mongo.sh`
- **Docker**: `docker/docker-compose.mongo.yml`
- **Python**: `src/data/nosql_backend.py`
- **Full Docs**: `docs/NOSQL_ML_INTEGRATION.md`

## Cost

- **Development**: $0 (self-hosted)
- **Production**: ~$57/month (MongoDB Atlas M10)

## Decision Rationale

**Why not Redis?**
- Feature caching is nice-to-have, not must-have
- Can add Redis later if needed
- MongoDB alone solves 80% of problems

**Why not both?**
- Extra complexity for minimal gain at this stage
- Double the services to manage
- Can scale up later when needed

**Focus**: Get MongoDB working first, it's the highest impact change.

---

**Status**: ✅ Ready to deploy  
**Effort**: 5 minutes setup  
**Impact**: High (better experiment tracking, concurrent access)
