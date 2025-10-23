# Submission Package: CPU-Forced Hotfix

- **Date**: 2025-10-21
- **Status**: ⚠️ Superseded (use for reference only)
- **Description**: First revision that forced CPU execution to avoid ROCm crashes. Still relied on `braindecode` imports without confirming availability on the Codabench image.
- **Source Script**: `archive/submissions/submission_legacy_gpu.py` (earlier root submission variant)
- **Weights**:
  - `weights_challenge_1.pt`
  - `weights_challenge_2.pt`
- **Known Issues**: Challenge 2 may fail if `braindecode` is absent. Retained for historical debugging and diff comparisons.

```text
submission.zip
├── SUBMISSION_INFO.txt
└── submission.py
```
