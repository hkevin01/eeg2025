# Codebase Reorganization Plan

## File Categories and Destinations

### 1. Documentation (→ docs/)
- All GPU/ROCm related .md files
- Status reports and summaries
- Training guides

### 2. Test Files (→ tests/)
- test_*.py files from root
- Quick validation scripts

### 3. Scripts (→ scripts/)
- Shell scripts (.sh)
- Training launcher scripts

### 4. Submissions (→ submissions/archives/)
- .zip files
- Old submission packages

### 5. Keep in Root
- README.md (main docs)
- setup.py (package setup)
- submission.py (active submission script)
- activate_sdk.sh (frequently used)
- train_universal.py (main entry point)

## Execution Order
1. Create destination folders
2. Move documentation files
3. Move test files
4. Move scripts
5. Move zip files
6. Update imports/references
