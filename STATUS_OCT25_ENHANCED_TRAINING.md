# Enhanced C1 Training - Status Update (Oct 25, 11:37am)

## Problem Summary
Enhanced training with EnhancedEEGNeX (254K params) consistently crashes with:
```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address
```

## What We've Tried

### ✅ Successfully Fixed:
1. **Augmentation disabled** - Commented out temporal_masking, magnitude_warping, mixup (all caused HIP kernel errors)
2. **DataLoader configuration** - Set num_workers=0, pin_memory=False  
3. **MNE configuration** - Added MNE_USE_CUDA=false
4. **Reduced batch size** - From 16 → 4
5. **Reduced subject count** - From 50 → 35 → 20 per dataset
6. **LR scheduler disabled** - Removed ReduceLROnPlateau

### ❌ Still Crashing:
- Error occurs during GPU initialization/data loading
- Not related to training code - happens before training starts
- Fundamental ROCm/HIP memory allocation issue

## Root Cause Analysis  
AMD Radeon RX 5600 XT (gfx1010) + ROCm SDK 6.1.2 has memory allocation issues:
- GPU reports 6.43 GB but HSA memory aperture violations occur
- Even minimal GPU operations trigger the error
- Previous successful training: baseline EEGNeX (62K params) - 4x smaller than enhanced (254K)
- Enhanced model likely exceeds some undocumented memory limit

## Recommendations

### ✅ IMMEDIATE: Upload SAM Baseline Submission
- **File**: submission_sam_fixed.zip (467 KB) - Ready now
- **Performance**: C1 NRMSE 0.3008, C2 NRMSE 0.2042  
- **Expected combined**: 0.25-0.45 (competitive)
- **Advantage**: Comprehensive logging for competition environment debugging
- **Risk**: Low - all local tests passed

### Option 2: Simplify Enhanced Model  
- Remove MultiScaleFeaturesExtractor
- Keep only TemporalAttention + SAM
- Reduce to ~150K params
- Test on GPU again
- **Risk**: Might still crash
- **Time**: 1-2 hours to implement + test

### Option 3: Train Enhanced on CPU
- **Time**: 2-3 days for 30 epochs
- **Viability**: Low - defeats purpose

## Current Status
- ✅ submission_sam_fixed.zip - Ready to upload
- ✅ Baseline weights tested and working
- ⛔ Enhanced training blocked by GPU memory issues
- 🔄 Waiting for decision on next steps

## Todo List
```markdown
- [ ] Upload submission_sam_fixed.zip to Codabench
- [ ] Monitor evaluation results  
- [ ] Decide: Try simplified model OR stick with baseline
- [ ] If simplified: Implement TemporalAttention-only model
- [ ] If baseline: Focus on ensemble/post-processing optimizations
```
