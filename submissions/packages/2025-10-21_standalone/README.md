# Submission Package: Standalone (No External Dependencies)

- **Date**: 2025-10-21
- **Status**: ✅ Validated locally (backup option)
- **Description**: Ships a fully self-contained EEGNeX implementation that avoids any `braindecode` dependency. Mirrors braindecode's layer layout and includes custom weight-mapping logic.
- **Source Script**: `submissions/standalone/submission.py`
- **Weights**:
  - `weights_challenge_1.pt` (TCN, epoch 2)
  - `weights_challenge_2.pt` (EEGNeX, epoch 1)
- **Testing**: `/tmp/test_standalone` smoke test verifies successful weight loading and inference.
- **Notes**: Use when the competition platform cannot import `braindecode` or when a dependency-free package is required.

```text
submission.zip
├── submission.py
├── weights_challenge_1.pt
└── weights_challenge_2.pt
```
