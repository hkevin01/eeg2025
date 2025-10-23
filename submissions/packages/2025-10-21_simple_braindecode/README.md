# Submission Package: Simple (Braindecode)

- **Date**: 2025-10-21
- **Status**: ✅ Validated locally (recommended)
- **Description**: Uses `braindecode.models.EEGNeX` directly with the cleaned `submission.py` that resides at the repository root and in `submissions/simple/submission.py`.
- **Source Script**: `submissions/simple/submission.py`
- **Weights**:
  - `weights_challenge_1.pt` (TCN, epoch 2)
  - `weights_challenge_2.pt` (EEGNeX, epoch 1)
- **Testing**: `tests` folder smoke tests (`/tmp/test_simple`) confirm forward passes and correct parameter counts.
- **Notes**: Preferred package for Codabench uploads because the competition image already includes `braindecode 1.2.0`. Bundles `/etc/localtime` to satisfy Codabench scoring environment requirements.

```text
submission.zip
├── submission.py
├── weights_challenge_1.pt
├── weights_challenge_2.pt
└── localtime

artifacts/
└── unpacked/               # Full directory extracted during testing
```
