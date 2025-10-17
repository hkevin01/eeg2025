from eegdash import EEGChallengeDataset
from pathlib import Path

dataset = EEGChallengeDataset(
    release='R1',
    mini=True,
    query=dict(task="contrastChangeDetection"),
    cache_dir=Path('data/raw')
)

# Get first valid dataset
ds = dataset.datasets[0]
print("Available event descriptions:")
for desc in sorted(set(ds.raw.annotations.description)):
    count = sum(1 for a in ds.raw.annotations if a['description'] == desc)
    print(f"  {desc}: {count} events")
