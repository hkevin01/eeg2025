# Infrastructure Scripts

Database and infrastructure management scripts.

## Scripts

- `create_metadata_database.py` - Create SQLite database for tracking
  - Creates tables: training_runs, epoch_history, model_checkpoints, cache_files
  - Sets up views: best_models, training_summary
  
- `check_infrastructure_status.sh` - Check infrastructure health
  - Verifies database exists
  - Checks cache files
  - Reports disk usage

## Usage

**Create database:**
```bash
python3 scripts/infrastructure/create_metadata_database.py
```

**Check status:**
```bash
bash scripts/infrastructure/check_infrastructure_status.sh
```

---
*Organized: October 19, 2025*
