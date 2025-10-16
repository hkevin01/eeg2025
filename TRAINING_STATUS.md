# EEG Training Status - Fresh Start

## Current Status: ✅ ACTIVE

**Started:** 2024-10-16 15:44 UTC  
**Status:** Both challenges training successfully  
**GPU:** AMD Radeon RX 5700 XT (ROCm enabled)

---

## Challenge 1: Response Time Prediction

**Training Strategy:** R1+R2 (44,440 trials) → Validate on R3 (28,758 trials)  
**Model:** CompactResponseTimeCNN (200K parameters)  
**Status:** ✅ Running (PID 1241446)  

**Latest Results:**
- Epoch: 7/50  
- Train NRMSE: 0.9616  
- Val NRMSE: 1.0152  
- Best Val: 1.0034 (Epoch 1)  

**Progress:** 14% complete (~1h 45m remaining)

---

## Challenge 2: Externalizing Prediction

**Training Strategy:** R1+R2 combined, 80/20 split (98,613 train / 24,654 val)  
**Model:** CompactExternalizingCNN (64K parameters)  
**Status:** ✅ Running (PID 1241636)  

**Latest Results:**
- Epoch: 3/50  
- Train NRMSE: 0.7344  
- Val NRMSE: 0.6145  
- Best Val: 0.6145 (Epoch 2)  

**Data Variance:** ✅ FIXED - Range [0.325, 0.620], Mean 0.482, Std 0.147

---

## Monitoring

**Quick Check:**
```bash
tail -20 logs/challenge1_fresh_start.log
tail -20 logs/challenge2_fresh_start.log
```

**Full Monitor:**
```bash
./monitor_training_enhanced.sh
```

---

## Expected Completion

- Challenge 1: ~17:30 UTC
- Challenge 2: ~18:00 UTC
- Submission ready: ~18:15 UTC

---

*Last Updated: 2024-10-16 16:00 UTC*
