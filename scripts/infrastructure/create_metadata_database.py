#!/usr/bin/env python3
"""
Create SQLite Metadata Database
================================
Fast queryable database for subject metadata, training history, and model performance.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("🗄️  CREATING METADATA DATABASE")
print("="*80)

DB_FILE = Path("data/metadata.db")
DB_FILE.parent.mkdir(exist_ok=True, parents=True)

# Create database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create tables
print("\nCreating tables...")

# Subjects table
cursor.execute('''
CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY,
    age REAL,
    sex TEXT,
    p_factor REAL,
    release TEXT,
    n_sessions INTEGER,
    n_trials INTEGER,
    data_quality REAL,
    notes TEXT
)
''')

# Training runs table  
cursor.execute('''
CREATE TABLE IF NOT EXISTS training_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge INTEGER,
    model_name TEXT,
    start_time TEXT,
    end_time TEXT,
    status TEXT,
    best_val_loss REAL,
    best_epoch INTEGER,
    total_epochs INTEGER,
    config TEXT,
    notes TEXT
)
''')

# Epoch history table
cursor.execute('''
CREATE TABLE IF NOT EXISTS epoch_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    epoch INTEGER,
    train_loss REAL,
    val_loss REAL,
    learning_rate REAL,
    duration_seconds REAL,
    timestamp TEXT,
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
)
''')

# Model checkpoints table
cursor.execute('''
CREATE TABLE IF NOT EXISTS model_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    epoch INTEGER,
    val_loss REAL,
    file_path TEXT,
    file_size_mb REAL,
    timestamp TEXT,
    is_best BOOLEAN,
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
)
''')

# Cache files table
cursor.execute('''
CREATE TABLE IF NOT EXISTS cache_files (
    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge INTEGER,
    release TEXT,
    file_path TEXT,
    file_size_mb REAL,
    n_windows INTEGER,
    n_channels INTEGER,
    n_times INTEGER,
    created_at TEXT,
    checksum TEXT
)
''')

# Create indices
print("Creating indices...")
cursor.execute('CREATE INDEX IF NOT EXISTS idx_subjects_release ON subjects(release)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_challenge ON training_runs(challenge)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_epoch_run ON epoch_history(run_id, epoch)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON model_checkpoints(run_id)')

conn.commit()
print(f"✅ Database created: {DB_FILE}")
print(f"   Size: {DB_FILE.stat().st_size / 1024:.1f} KB")

# Create views for easy querying
print("\nCreating views...")

cursor.execute('''
CREATE VIEW IF NOT EXISTS best_models AS
SELECT 
    tr.run_id,
    tr.challenge,
    tr.model_name,
    tr.best_val_loss,
    tr.best_epoch,
    tr.start_time,
    mc.file_path
FROM training_runs tr
LEFT JOIN model_checkpoints mc ON tr.run_id = mc.run_id AND mc.is_best = 1
ORDER BY tr.challenge, tr.best_val_loss
''')

cursor.execute('''
CREATE VIEW IF NOT EXISTS training_summary AS
SELECT 
    challenge,
    COUNT(*) as n_runs,
    MIN(best_val_loss) as best_loss,
    AVG(best_val_loss) as avg_loss,
    AVG(total_epochs) as avg_epochs
FROM training_runs
WHERE status = 'completed'
GROUP BY challenge
''')

conn.commit()
print("✅ Views created")

# Add some helper functions
print("\nCreating helper functions table...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT,
    description TEXT
)
''')

# Insert default config
config_items = [
    ('db_version', '1.0', 'Database schema version'),
    ('created_at', datetime.now().isoformat(), 'Database creation time'),
    ('cache_dir', 'data/cached', 'HDF5 cache directory'),
    ('checkpoints_dir', 'checkpoints', 'Model checkpoints directory'),
]

cursor.executemany('INSERT OR REPLACE INTO config VALUES (?, ?, ?)', config_items)
conn.commit()

print("\n" + "="*80)
print("🎉 METADATA DATABASE READY!")
print("="*80)
print("\nTables created:")
print("  ✅ subjects - Subject metadata and demographics")
print("  ✅ training_runs - Training session tracking")
print("  ✅ epoch_history - Per-epoch metrics")
print("  ✅ model_checkpoints - Saved model tracking")
print("  ✅ cache_files - HDF5 cache file registry")
print("\nViews created:")
print("  ✅ best_models - Best model for each challenge")
print("  ✅ training_summary - Training statistics")
print("\n📊 Query examples:")
print("  SELECT * FROM best_models;")
print("  SELECT * FROM training_summary;")
print("  SELECT * FROM epoch_history WHERE run_id = 1 ORDER BY epoch;")

conn.close()
