#!/bin/bash
echo "🧹 Organizing docs folder..."

# Guides
mv *GUIDE*.md guides/ 2>/dev/null
mv *CHECKLIST*.md guides/ 2>/dev/null  
mv WORKFLOW.md guides/ 2>/dev/null
mv usage_examples.md guides/ 2>/dev/null
mv api_documentation.md guides/ 2>/dev/null
mv pipeline_diagrams.md guides/ 2>/dev/null
mv ablation_studies.md guides/ 2>/dev/null

# Historical
mv SESSION*.md historical/ 2>/dev/null
mv BASELINE*.md historical/ 2>/dev/null
mv TASKS_COMPLETED*.md historical/ 2>/dev/null
mv *RESULTS*.md historical/ 2>/dev/null
mv PHASE1*.md historical/ 2>/dev/null

# Summaries
mv *SUMMARY*.md summaries/ 2>/dev/null

# Analysis  
mv *ANALYSIS*.md analysis/ 2>/dev/null
mv *CRISIS*.md analysis/ 2>/dev/null
mv *DISCOVERY*.md analysis/ 2>/dev/null

# Planning
mv *PLAN*.md planning/ 2>/dev/null
mv *TODO*.md planning/ 2>/dev/null
mv WEEK_BY_WEEK*.md planning/ 2>/dev/null
mv PROJECT_PLAN*.md planning/ 2>/dev/null

# Archive
mv COMPLETION*.md archive/ 2>/dev/null
mv ORGANIZATION*.md archive/ 2>/dev/null
mv CORRECTED*.md archive/ 2>/dev/null
mv TODAY*.md archive/ 2>/dev/null
mv DAILY*.md archive/ 2>/dev/null
mv NEXT_STEPS*.md archive/ 2>/dev/null
mv UNDERSTANDING*.md archive/ 2>/dev/null

# GPU/Hardware specific
mv GPU_*.md status/ 2>/dev/null
mv AMD*.md status/ 2>/dev/null
mv ROCM*.md status/ 2>/dev/null
mv *OPTIMIZATION*.md status/ 2>/dev/null
mv *SAFEGUARDS*.md status/ 2>/dev/null
mv CUFFT*.md status/ 2>/dev/null
mv *HYBRID*.md status/ 2>/dev/null

# Challenge 2 specific  
mv CHALLENGE2*.md analysis/ 2>/dev/null

# Competition specific
mv COMPETITION*.md planning/ 2>/dev/null
mv COMPETITIVE*.md planning/ 2>/dev/null

# Enhancements
mv *ENHANCEMENT*.md status/ 2>/dev/null
mv *IMPROVEMENT*.md planning/ 2>/dev/null

# Pipeline
mv PIPELINE*.md status/ 2>/dev/null
mv CROSS_TASK*.md status/ 2>/dev/null

# Index and structure
mv INDEX.md ./ 2>/dev/null
mv DIRECTORY*.md ./ 2>/dev/null

echo "✅ Done!"
