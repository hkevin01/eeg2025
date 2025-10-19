# Cache Creation Scripts

Scripts for creating HDF5 cache files from the EEG dataset.

## Main Scripts

- `create_challenge2_cache.py` - Create cache for all releases (R1-R5)
  - Creates HDF5 files with preprocessed windows
  - Each release: ~10-15GB cache file
  - Total: ~50GB for full dataset

- `create_challenge2_cache_remaining.py` - **CURRENTLY RUNNING** in tmux
  - Creates R3, R4, R5 cache files
  - Running in tmux session: 'cache_remaining'
  - Log: logs/cache_R3_R4_R5_fixed.log

## Support Scripts

- `continue_cache_creation.sh` - Resume interrupted cache creation
- `continue_cache.py` - Python continuation helper
- `create_remaining_cache.py` - Create missing cache files
- `analyze_cache_warnings.py` - Analyze warnings from cache creation

## Cache Files

Cache files are stored in: `data/cached/`

**Format:** `challenge2_R{N}_windows.h5`
- R1: 11GB (61,889 windows) ✅
- R2: 12GB (62,000+ windows) ✅
- R3: Creating... 🔄
- R4: Pending ⏳
- R5: Pending ⏳

## Usage

**Create all caches:**
```bash
python3 scripts/cache/create_challenge2_cache.py
```

**Resume in tmux:**
```bash
tmux attach -t cache_remaining
```

---
*Organized: October 19, 2025*
