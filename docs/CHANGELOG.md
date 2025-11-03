# Changelog

All notable changes to the EEG2025 Foundation Model project.

## [2025-11-03] - Major Project Cleanup

### Changed
- Reorganized entire project structure
- Moved scripts to organized folders (training/, submission/, infrastructure/)
- Cleaned up root directory (8 essential files, 22 directories)
- Archived 500+ AI session artifacts and excessive documentation

### Removed
- 36 excessive .md files from root (session summaries, analyses, strategies)
- .copilot/ directory
- memory-bank/ (AI session logs)
- 17 temporary/old directories
- 500+ documentation files from docs/ (archived)

### Added
- scripts/validate_submission.py - Pre-upload validation script
- docs/CLEANUP_REPORT.md - Detailed cleanup documentation
- Clean, minimal documentation structure

## [2025-11-02] - V16 Submission

### Fixed
- Fixed `weights_only` parameter bug in submission.py (3rd occurrence)
- Created V16 fixed package ready for upload

### Added
- 5-seed ensemble training for Challenge 1
- Test-Time Augmentation (TTA) implementation
- Pre-submission validation script

## [2025-11-01] - V15 Success

### Changed
- V15 submission achieved rank #77 with C1: 1.00019, C2: 1.00066

## [2025-10-31] - Competition Preparations

### Added
- Challenge 1 improved training pipeline
- Challenge 2 EEGNeX model implementation
- Competition submission infrastructure

## [2025-10-16] - Initial Competition Submission

### Added
- Initial competition submission (V1)
- Basic model architectures
- Data preprocessing pipeline

## [2025-09-05] - Project Initialization

### Added
- Initial project structure
- README and documentation
- Basic dependencies and setup
