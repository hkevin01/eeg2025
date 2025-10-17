# 🌙 Overnight Training - Setup Complete

**Started:** October 16, 2025 23:01
**Expected Completion:** October 17, 2025 05:01 (6 hours)
**Goal:** Improve from #47 (2.01) to #25-30 (1.5-1.7)

## ✅ Status

- **Challenge 1:** Running (PID: 1565960, CPU: 88%, MEM: 5.7%)
- **Challenge 2:** Running (PID: 1565961, CPU: 88%, MEM: 5.7%)
- **Disk Space:** 82GB available (was 28GB - cleaned 54GB)
- **Memory:** 31GB total, 23GB available
- **Device:** GPU (CUDA) with CPU fallback if needed

## 📊 Monitor Progress

**Live Monitor:**
```bash
cd /home/kevin/Projects/eeg2025
./monitor_overnight.sh
```

**Check Logs:**
```bash
tail -f logs/train_c1_robust_hybrid.log
tail -f logs/train_c2_robust_hybrid.log
```

**Quick Status:**
```bash
ps aux | grep train_challenge | grep -v grep
```

## 🛠️ What's Running

### Challenge 1: Response Time Prediction
- Training: R1+R2+R3 (80% train, 20% validation)
- Model: CompactResponseTimeCNN
- Loss: Huber loss (robust to outliers)
- Epochs: 50
- Device: GPU with CPU fallback

### Challenge 2: Externalizing Behavior
- Training: R1+R2+R3 (80% train, 20% validation)
- Model: CompactExternalizingCNN
- Loss: Huber loss
- Epochs: 50
- Device: GPU with CPU fallback

## 📝 Timeline

1. **Data Loading:** ~30 min (loading R1, R2, R3)
2. **Training:** ~4-6 hours (50 epochs each)
3. **Total:** ~6 hours

## 🎯 Expected Output

When training completes, you'll find:
- `weights/weights_challenge_1_robust.pt` (~300-400 KB)
- `weights/weights_challenge_2_robust.pt` (~300-400 KB)
- Training logs with validation scores

## 🚨 If Something Goes Wrong

**Check if running:**
```bash
ps aux | grep train_challenge
```

**Restart training:**
```bash
bash run_overnight_training.sh
```

**Stop training:**
```bash
pkill -f train_challenge
```

## 🔍 Key Fixes Applied

1. **Fixed windowing:** Using `add_aux_anchors` and explicit event mapping
2. **Challenge 2 fix:** Using `create_fixed_length_windows` (was using wrong function)
3. **GPU fallback:** Auto-switches to CPU if GPU fails
4. **Disk cleanup:** Freed 54GB by removing old venvs

## ☕ Next Morning

Check completion:
```bash
grep "Training complete" logs/train_c1_robust_hybrid.log
grep "Training complete" logs/train_c2_robust_hybrid.log
ls -lh weights/weights_challenge_*.pt
```

If successful, create submission and upload to Codabench!
