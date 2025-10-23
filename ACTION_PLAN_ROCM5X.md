# Action Plan: ROCm 5.x GPU Acceleration

**Created**: Oct 22, 2025 10:35  
**Context**: Community suggests ROCm 5.x works better with RX 5600 XT  
**Goal**: Get GPU training working for 10-20x speedup

## 📊 Current Situation

### Fast CPU Training (Running Now)
- **Status**: ✅ Epoch 1, batch 350/811 (~43%)
- **Time per batch**: ~5 seconds
- **Epoch 1 ETA**: ~2-3 hours remaining
- **Total ETA**: ~8 hours for 3 epochs
- **Performance**: Loss dropping (0.956 → 0.105)

### GPU Training (Failed)
- **Issue**: ROCm 6.2 + gfx1030 + EEGNeX = Memory violation
- **Fix**: Try ROCm 5.7 (community recommendation)

## 🎯 Recommended Action: Test ROCm 5.7 PyTorch

### Quick Test (30 minutes)

```bash
# 1. Check if conda is available
which conda || echo "Need to install conda/miniconda first"

# 2. Create test environment
conda create -n eeg_rocm5 python=3.11 -y
conda activate eeg_rocm5

# 3. Install PyTorch with ROCm 5.7
pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Install dependencies
pip3 install braindecode mne scikit-learn h5py numpy pandas

# 5. Test EEGNeX on GPU
cd /home/kevin/Projects/eeg2025
python3 -c "
import torch
from braindecode.models import EEGNeX
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.is_available()}')
model = EEGNeX(n_outputs=1, n_chans=129, n_times=200, sfreq=100).cuda()
x = torch.randn(16, 129, 200).cuda()
y = model(x)
print(f'✅ SUCCESS! Output: {y.shape}')
"
```

### If Test Succeeds → Start GPU Training

```bash
# In the same conda environment
cd /home/kevin/Projects/eeg2025

# Start GPU training with ROCm 5.7
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -u scripts/training/train_challenge2_r1r2.py \
    --batch-size 64 \
    --num-workers 2 \
    --max-epochs 3 \
    --device cuda \
    --no-pin-memory \
    --note "GPU ROCm 5.7 - RX 5600 XT" \
    2>&1 | tee logs/training_c2_gpu_rocm57_$(date +%Y%m%d_%H%M%S).log &
```

**Expected speedup**: 10-20x faster than CPU
- CPU: ~5 sec/batch → GPU: ~0.25-0.5 sec/batch
- Epoch time: 7 hours → 20-40 minutes

## 📋 Decision Matrix

| Scenario | Action | Time Investment | Expected Outcome |
|----------|--------|-----------------|------------------|
| **A: ROCm 5.7 test succeeds** | Start GPU training | 30 min setup + 2 hrs training | Best weights in ~2.5 hrs |
| **B: ROCm 5.7 test fails** | Continue CPU training | 0 (already running) | Good weights in ~8 hrs |
| **C: Don't test, just wait** | Let CPU finish | 0 | Good weights in ~8 hrs |

**Recommendation**: Try Scenario A
- **Risk**: Low (30 minutes to test)
- **Reward**: High (6 hour time savings if it works)
- **Fallback**: CPU training continues regardless

## ⏱️ Timeline Comparison

### With ROCm 5.7 GPU (If it works)
```
Now:        Test ROCm 5.7 setup (30 min)
10:45:      Start GPU training
12:45:      Epoch 1 complete
14:45:      Epoch 2 complete  
16:45:      Epoch 3 complete ✅
```

### CPU Only (Current path)
```
Now:        Epoch 1 in progress (batch 350/811)
13:00:      Epoch 1 complete
20:00:      Epoch 2 complete
03:00 +1:   Epoch 3 complete ✅
```

**Difference**: 11 hours saved if GPU works!

## 🔍 Alternative: Ollama Investigation

If you have Ollama running well:

```bash
# Check Ollama's GPU setup
which ollama
ldd $(which ollama) | grep -i rocm
ps aux | grep ollama

# Might reveal working ROCm configuration
```

## ✅ Immediate Next Step

**Choose one**:

**Option 1** (Recommended): Try ROCm 5.7
```bash
# 30-minute test
conda create -n eeg_rocm5 python=3.11 -y && \
conda activate eeg_rocm5 && \
pip3 install torch==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.7 && \
pip3 install braindecode mne scikit-learn h5py numpy pandas
```

**Option 2**: Just wait for CPU
```bash
# Monitor progress
tail -f logs/training_c2_fast_20251021_233017.log
```

**Option 3**: Upload submission now
```bash
# You already have excellent weights!
ls -lh submission_simple_READY_TO_UPLOAD.zip
# Go upload to https://www.codabench.org/competitions/4287/
```

---

**Your call!** Want me to help set up the ROCm 5.7 test? 🚀
