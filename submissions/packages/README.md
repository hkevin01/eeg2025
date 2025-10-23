# Submission Packages Index

Each directory under this folder captures a dated competition package along with a short README describing its purpose and validation status.

| Folder | Status | Notes |
|--------|--------|-------|
| `2025-10-21_simple_braindecode/` | ✅ Recommended | Uses platform `braindecode`, mirrors root `submission.py`, bundles `/etc/localtime`. |
| `2025-10-21_standalone/` | ✅ Backup | Dependency-free EEGNeX implementation with custom weight mapping. |
| `2025-10-21_cpu_fixed/` | ⚠️ Superseded | Early CPU-forced build; kept for diff history. |
| `2025-10-21_competition_failure/` | ❌ Failed | Incorrect `resolve_path` (misses `/app/input/res/`). |
| `2025-10-21_info_snapshot/` | 📄 Info only | Contains documentation snapshot, no trained weights. |

New packages should follow the same pattern: `YYYY-MM-DD_description/` containing `submission.zip` plus a `README.md` detailing the build.
