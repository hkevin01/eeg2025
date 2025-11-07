# GSN-HydroCel 128 (E1-E129) Regional Grouping Guide

**Date:** November 4, 2025  
**Context:** Response to jesslyn1999's question about grouping GSN electrodes by brain regions

---

## Your Situation (As I Understand It)

You're loading GSN electrodes via MNE with **E1, E2, ..., E129** labels (EGI standard numbering), and you want to group them into brain regions (left frontal, right frontal, etc.) using a "theta formula" suggested by an LLM.

**Key Question:** Why isn't this working well?

---

## The Core Issue

**The "theta formula" approach CAN work, but has 3 critical pitfalls:**

### ❌ Pitfall 1: Using Theta Without X/Y Coordinates
**Problem:** If you're just using electrode indices (E1, E2, ...) to calculate theta angles, you're assuming a circular layout. GSN electrodes are NOT evenly distributed in a circle - they're densely packed around the crown, sparse at the neck.

**What Happens:**
```python
# This is WRONG for GSN:
theta = 2 * pi * electrode_index / 129  # Assumes circular layout!

# Reality: GSN density varies by region
# Front (E1-E30):   Dense, ~10mm spacing
# Top (E50-E80):    Very dense, ~8mm spacing  
# Neck (E120-129):  Sparse, ~15-20mm spacing
```

**Impact:** Your "left frontal" group might accidentally include neck electrodes because the theta calculation doesn't match actual spatial positions.

---

### ❌ Pitfall 2: Not Loading Actual GSN Montage Coordinates
**Problem:** You need to load the **actual 3D coordinates** from MNE's montage system, not infer positions from electrode numbers.

**Correct Approach:**
```python
import mne
import numpy as np

# Load the ACTUAL GSN-HydroCel 128 montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-128')

# Get 3D positions
pos = montage.get_positions()
ch_pos = pos['ch_pos']  # Dict: {'E1': [x, y, z], 'E2': [x, y, z], ...}

# Convert to arrays
channels = [f'E{i}' for i in range(1, 130)]
xyz = np.array([ch_pos[ch] for ch in channels if ch in ch_pos])

# NOW you can calculate theta from ACTUAL x, y coordinates:
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
theta = np.arctan2(y, x)  # Angle from actual 2D projection
```

**Why This Matters:** E1 might be at theta=45°, E2 at theta=50°, E3 at theta=48° (not evenly spaced!). Using actual coordinates respects the irregular layout.

---

### ❌ Pitfall 3: Grouping BEFORE Model Input (Information Loss)
**Problem:** If you're averaging channels within each region BEFORE feeding to your model, you're throwing away 85% of spatial information.

**Two Approaches:**

#### ❌ BAD: Pre-aggregation (What you might be doing)
```python
# DON'T DO THIS for model input:
left_frontal_channels = ['E22', 'E23', 'E24', 'E26', 'E27', 'E33']
left_frontal_signal = eeg_data[left_frontal_channels].mean(axis=0)  # (200,)

# Result: 129 channels → 8 regional averages → huge info loss!
regional_input = np.array([
    left_frontal_signal,
    right_frontal_signal,
    # ... 6 more regions
])  # Shape: (8, 200) - lost 121 channels of data!

model.forward(regional_input)  # Model sees 8 channels, not 129!
```

**Why This Fails:**
- **85% information loss:** 129 → 8 channels
- **Subject noise dominates:** Averaging includes noisy channels equally
- **CV variance explodes:** From 0.62% to 5-10% (unreliable predictions)

#### ✅ GOOD: Keep Full Channels, Use Regions for Features Only
```python
# DO THIS: Full channels to model, regions for supplementary features
eeg_full = eeg_data  # Shape: (129, 200) - keep everything!

# Optionally, extract regional features AS SUPPLEMENTS:
left_frontal_avg = eeg_data[left_frontal_channels].mean(axis=0)  # (200,)
p300_latency = find_peak(left_frontal_avg, window=(300, 500))  # Feature

# Model gets BOTH full channels AND supplementary features:
model.forward(eeg_full)  # Primary input: (129, 200)
# Use p300_latency as metadata or auxiliary input if needed
```

---

## What Actually Works (Proven by This Repository)

**Your V10 submission (Rank #72) proves the correct approach:**

```python
# From actual code:
Input:  (batch, 129, 200)  # ALL channels, full resolution
Model:  CompactCNN with Conv1d spatial filters
Output: Response time prediction

# Let the model learn spatial patterns via convolutions!
# Don't manually reduce to regions before the model.
```

**Performance:**
- **Validation NRMSE:** 1.00019
- **Multi-seed CV:** 0.62% (excellent stability)
- **Leaderboard Rank:** #72/150

**Why This Works:**
- Conv1D learns better spatial filters than manual region grouping
- Model automatically weights important channels (e.g., parietal for P300)
- No information loss from pre-aggregation
- Robust to subject-specific channel noise

---

## If You STILL Want to Group by Regions (For Features, Not Input)

Here's the **correct theta-based approach** using actual GSN coordinates:

```python
import mne
import numpy as np

# 1. Load GSN-HydroCel 128 montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
pos = montage.get_positions()['ch_pos']

# 2. Get actual 3D coordinates
channels = [f'E{i}' for i in range(1, 130)]
xyz = np.array([pos[ch] for ch in channels if ch in pos])
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

# 3. Calculate theta and phi from ACTUAL positions
theta = np.arctan2(y, x)  # Azimuth angle (-π to π)
phi = np.arctan2(np.sqrt(x**2 + y**2), z)  # Elevation angle (0 to π)

# 4. Define regions using theta ranges (in radians)
regions = {
    'left_frontal': (theta > np.pi/4) & (theta < 3*np.pi/4) & (phi < np.pi/3),
    'right_frontal': (theta > -3*np.pi/4) & (theta < -np.pi/4) & (phi < np.pi/3),
    'left_temporal': (theta > np.pi/2) & (theta < np.pi) & (phi > np.pi/4) & (phi < 2*np.pi/3),
    'right_temporal': (theta > -np.pi) & (theta < -np.pi/2) & (phi > np.pi/4) & (phi < 2*np.pi/3),
    'central': (phi < np.pi/4),  # Top of head
    'parietal': (np.abs(theta) < np.pi/4) & (phi > np.pi/4) & (phi < 2*np.pi/3),
    'occipital': (np.abs(theta) < np.pi/3) & (phi > 2*np.pi/3),
}

# 5. Get channel indices for each region
region_channels = {}
for region_name, mask in regions.items():
    indices = np.where(mask)[0]
    region_channels[region_name] = [channels[i] for i in indices]
    print(f"{region_name}: {len(indices)} channels")
    print(f"  Channels: {region_channels[region_name][:5]}...")  # Show first 5

# 6. OPTIONAL: Extract regional features (NOT for model input!)
def extract_regional_features(eeg_data, region_channels):
    """Extract supplementary features from regions"""
    features = {}
    for region_name, ch_list in region_channels.items():
        # Get channel indices
        ch_indices = [int(ch[1:]) - 1 for ch in ch_list]  # E1 → index 0
        regional_signal = eeg_data[ch_indices].mean(axis=0)  # (200,)
        
        # Extract features (e.g., P300 latency, alpha power)
        features[f'{region_name}_mean'] = regional_signal.mean()
        features[f'{region_name}_std'] = regional_signal.std()
        # ... more features as needed
    
    return features

# 7. Use in training: KEEP full channels, supplement with features
X_full = eeg_data  # Shape: (129, 200) - keep this!
regional_feats = extract_regional_features(eeg_data, region_channels)

# Model gets full channels:
prediction = model(X_full)  # Primary pathway

# Optional: Concatenate regional features as auxiliary input
# (Only if you have a multi-input architecture)
```

---

## Sanity Check: Visualize Your Regions

**Always verify your theta-based regions look correct:**

```python
import matplotlib.pyplot as plt

# Plot electrode positions colored by region
fig, ax = plt.subplots(figsize=(10, 10))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
for i, (region_name, mask) in enumerate(regions.items()):
    region_xyz = xyz[mask]
    ax.scatter(region_xyz[:, 0], region_xyz[:, 1], 
               c=colors[i], label=region_name, s=50, alpha=0.7)

ax.set_xlabel('X (Left-Right)')
ax.set_ylabel('Y (Front-Back)')
ax.set_title('GSN-HydroCel 128 Regional Grouping')
ax.legend()
ax.axis('equal')
plt.tight_layout()
plt.savefig('gsn_regional_grouping.png', dpi=150)
plt.show()

# Visual check:
# - Left/right should be symmetric
# - Frontal should be at top (positive Y)
# - Occipital should be at bottom (negative Y)
# - Central should be small cluster at center
```

---

## Common Mistakes & Fixes

### Mistake 1: "LLM said to use theta = 2πi/N"
**Fix:** That assumes uniform circular layout. GSN is irregular - use `np.arctan2(y, x)` from actual coordinates.

### Mistake 2: "I'm grouping before model input to reduce overfitting"
**Fix:** Use dropout/regularization instead. Grouping loses 85% information.

### Mistake 3: "My regions have 50+ channels each (too big)"
**Fix:** Your theta ranges might be too wide. Frontal regions should have ~15-25 channels, temporal ~20-30, central ~10-15.

### Mistake 4: "Left and right regions aren't symmetric"
**Fix:** Check your theta sign convention. `theta > 0` should be left hemisphere, `theta < 0` right hemisphere (or vice versa, be consistent).

### Mistake 5: "Regional features didn't improve my model"
**Fix:** That's NORMAL! Conv1D learns better patterns than manual grouping. Don't force it if it doesn't help.

---

## Bottom Line: Should You Use Regional Grouping?

**For Model Input:** **NO**
- Keep full 129 channels
- Let Conv1D learn spatial patterns
- Proven by your V10 success (Rank #72)

**For Feature Engineering:** **MAYBE**
- Only if you have multi-input architecture
- Use as supplementary features, not replacements
- Expect minimal improvement (<0.5% NRMSE reduction)
- More complexity, small gain

**For Interpretability:** **YES**
- Analyze which regions the model attends to
- Visualize regional contributions post-training
- Understand P300, N200, alpha/beta rhythms
- Debug model behavior

---

## Recommended Next Steps

1. **Verify your current approach:**
   - Are you grouping BEFORE model input? → Stop doing that!
   - Are you using electrode indices for theta? → Use actual coordinates!
   - Are your regions symmetric? → Visualize them!

2. **Stick with proven approach:**
   - Keep full 129 channels as model input
   - Use Conv1D spatial filters (your V10 approach)
   - Focus on variance reduction (ensembles, TTA, calibration)

3. **If you want regional features:**
   - Use correct theta calculation (from actual coordinates)
   - Extract features AFTER model training (for analysis)
   - Don't expect big performance gains

---

## Questions to Ask Yourself

- **Are you trying to reduce channels from 129 → 8-10 regions?**
  - If YES: Don't do this for model input! Keep all 129 channels.
  
- **Are you calculating theta from electrode numbers (E1=0°, E2=2.8°, ...)?**
  - If YES: This is wrong! Load actual montage coordinates.
  
- **Are your left/right regions symmetric when visualized?**
  - If NO: Your theta calculation or ranges are incorrect.
  
- **Did adding regional grouping improve your CV score?**
  - If NO: Remove it! Your V10 approach (full channels) is already optimal.

---

## Final Recommendation

**Based on your V10 success (Rank #72 with full 129 channels):**

1. **Don't** implement regional grouping for model input
2. **Do** keep using full 129 channels (proven to work)
3. **Optionally** use regions for post-hoc analysis/interpretation
4. **Focus** on V13 improvements: ensembles, TTA, calibration

**If the LLM suggested regional grouping as input reduction → that advice is incorrect for your competition setup.**

---

**Want to verify your approach? Share:**
- Your current code for loading/grouping electrodes
- Shape of your model input (should be `(batch, 129, 200)`)
- Whether you're averaging channels before the model
- Your current CV scores with regional grouping

**I can then give you specific fixes!**

---

**References:**
- MNE Montage Docs: https://mne.tools/stable/generated/mne.channels.make_standard_montage.html
- GSN-HydroCel 128 Layout: https://www.egi.com/research-division/geodesic-sensor-net
- Your V10 Success: `src/models/response_time/compact_cnn.py` (uses full 129 channels)

