# Submission Package: Competition Failure Snapshot

- **Date**: 2025-10-21
- **Status**: ❌ Failed validation (kept for diagnostics)
- **Description**: Standalone EEGNeX package that used a custom `resolve_path` pointing to `/app/ingestion_program/` and `/app/`. The competition container stores resources under `/app/input/res/`, causing the scorer to miss weight files.
- **Source Script**: Extracted from `submission_competition` folder before fix.
- **Weights**:
  - `weights_challenge_1.pt`
  - `weights_challenge_2.pt`
- **Known Issues**: Incorrect path resolution prevents the Codabench scorer from locating model weights, resulting in "could not find scores file" failures.
- **Action Items**: Use the updated simple or standalone packages produced after 22:30 UTC with the corrected `resolve_path` implementation.

```text
submission.zip
├── submission.py
├── weights_challenge_1.pt
└── weights_challenge_2.pt
```
