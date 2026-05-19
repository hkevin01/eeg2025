# Submission Packages Index

Each directory under this folder captures a dated competition package along with a short README describing its purpose and validation status.

| <sub>Folder</sub> | <sub>Status</sub> | <sub>Notes</sub> |
|--------|--------|-------|
| <sub>`2025-10-21_simple_braindecode/`</sub> | <sub>✅ Recommended</sub> | <sub>Uses platform `braindecode`, mirrors root `submission.py`, bundles `/etc/localtime`.</sub> |
| <sub>`2025-10-21_standalone/`</sub> | <sub>✅ Backup</sub> | <sub>Dependency-free EEGNeX implementation with custom weight mapping.</sub> |
| <sub>`2025-10-21_cpu_fixed/`</sub> | <sub>⚠️ Superseded</sub> | <sub>Early CPU-forced build; kept for diff history.</sub> |
| <sub>`2025-10-21_competition_failure/`</sub> | <sub>❌ Failed</sub> | <sub>Incorrect `resolve_path` (misses `/app/input/res/`).</sub> |
| <sub>`2025-10-21_info_snapshot/`</sub> | <sub>📄 Info only</sub> | <sub>Contains documentation snapshot, no trained weights.</sub> |

New packages should follow the same pattern: `YYYY-MM-DD_description/` containing `submission.zip` plus a `README.md` detailing the build.