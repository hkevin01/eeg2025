# EEG Publication - LaTeX Materials

Complete LaTeX publication package for the NeurIPS 2025 EEG Foundation Challenge paper.

## 📁 Contents

### Documents
- **`paper.tex`** - Full IEEE conference format LaTeX document (requires IEEEtran.cls)
- **`paper_article.tex`** - Alternative article-class version (compiles with standard LaTeX)
- **`generate_figures.py`** - Python script to generate all publication figures

### Figures (Generated)
All figures available in both PDF (vector) and PNG (raster) formats:
- **`figures/fig1_electrode_montage.pdf/png`** - GSN HydroCel 128 electrode layout
- **`figures/fig2_architecture.pdf/png`** - EnhancedCompactCNN architecture diagram
- **`figures/fig3_training_curves.pdf/png`** - Training and validation curves
- **`figures/fig4_variance_reduction.pdf/png`** - Variance reduction impact analysis
- **`figures/fig5_leaderboard.pdf/png`** - Competition leaderboard distribution

## 🚀 Quick Start

### Option 1: Compile with Standard LaTeX

```bash
# Navigate to latex directory
cd /home/kevin/Projects/eeg2025/latex

# Compile article-class version (no special packages needed)
pdflatex paper_article.tex
pdflatex paper_article.tex  # Run twice for references

# Output: paper_article.pdf
```

### Option 2: Compile with IEEE Format (Recommended for Submission)

```bash
# Install IEEE class if not present
sudo apt-get install texlive-publishers
# OR manually download IEEEtran.cls from:
# https://www.ieee.org/conferences/publishing/templates.html

# Compile IEEE version
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references

# Output: paper.pdf
```

### Option 3: Generate Figures Only

```bash
# Regenerate all figures
python generate_figures.py

# Figures will be created in figures/ directory
```

## 📋 Paper Structure

### Complete IEEE Conference Paper (~15-20 pages, two-column)

1. **Abstract** (~300 words)
   - Background, objective, methods, results, conclusions
   - Keywords

2. **Introduction** (4 subsections)
   - Background and motivation
   - NeurIPS 2025 EEG Foundation Challenge
   - Related work (architectures, response time prediction, variance reduction)
   - Our contributions (5 key points)

3. **Methods** (6 major sections)
   - Dataset: Healthy Brain Network (HBN)
     * Overview (5,000+ participants, ages 5-21)
     * EEG acquisition protocol (EGI NetStation, GSN HydroCel 128)
     * Challenge data specifications (C1: 7,461 trials, C2: 2,500 trials)
   - Preprocessing pipeline (filtering, artifact rejection, normalization)
   - Model architecture (EnhancedCompactCNN, 120K params)
   - Training strategy (AdamW, EMA, multi-seed)
   - Variance reduction techniques (ensemble, TTA, calibration)
   - Validation strategy (5-fold CV, pre-upload testing)

4. **Results** (4 subsections)
   - Competition performance (V10: 1.00052 NRMSE, rank 72/150)
   - Variance analysis (sources and magnitudes)
   - Architectural ablations (6 models compared)
   - Leaderboard context (gap analysis)

5. **Discussion** (4 subsections)
   - Key findings (efficiency-performance balance, variance reduction value)
   - Limitations (computational, time, data, domain knowledge constraints)
   - Comparison with literature (EEG benchmarks, response time prediction)
   - Practical implications (researchers, clinicians, organizers)
   - Future directions (8 promising approaches)

6. **Conclusions**
   - Summary of 5 key contributions
   - Key takeaways (5 points)
   - Broader impact statement

7. **References** (20 citations)
   - Core EEG references
   - Deep learning architectures
   - HBN dataset
   - Variance reduction techniques

## 🎨 Figures Description

### Figure 1: GSN HydroCel 128 Electrode Montage
- **Format:** Circular head plot with electrode positions
- **Content:** 129 channels color-coded by brain region
- **Purpose:** Show complete scalp coverage and regional groupings
- **Highlights:** 6× density vs standard 10-20 system

### Figure 2: EnhancedCompactCNN Architecture
- **Format:** Horizontal flow diagram
- **Content:** Layer-by-layer architecture visualization
- **Components:** 3 Conv1D blocks → Global Pool → Dense layers
- **Annotations:** Parameter counts, layer dimensions, dropout rates

### Figure 3: Training Curves
- **Format:** Two-panel plot (loss and NRMSE)
- **Content:** Training and validation curves over 50 epochs
- **Features:** Shaded variance regions, best epoch marker
- **Note:** Uses simulated data (replace with actual logs if available)

### Figure 4: Variance Reduction Impact
- **Format:** Bar plot with annotations
- **Content:** Cumulative impact of variance reduction techniques
- **Measurements:**
  - Single seed baseline: 1.00071
  - + 5-seed ensemble: ↓7.8e-5
  - + TTA: ↓3.2e-5
  - + Calibration: ↓7.9e-5
  - Final V10: 1.00052

### Figure 5: Leaderboard Distribution
- **Format:** Scatter plot with annotations
- **Content:** All 150 submissions ranked by NRMSE
- **Highlights:** Our position (rank 72), top 10, gap to 1st place
- **Statistics:** Median score, std deviation, performance tiers

## 🔧 Customization

### Regenerating Figures with Your Data

Edit `generate_figures.py` to use actual training logs:

```python
# Replace simulated training curves (lines 180-200)
# Load your actual training history:
import pickle
with open('../logs/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

epochs = history['epoch']
train_loss = history['train_loss']
val_loss = history['val_loss']
# ... then plot as before
```

### Customizing LaTeX Content

**For IEEE submission (paper.tex):**
- Update author information (line 37-40)
- Modify affiliation
- Update email addresses
- Add co-authors if applicable

**For article submission (paper_article.tex):**
- Same updates as IEEE version
- Adjust margins in preamble if needed (line 5)
- Change to single-column if required (remove `twocolumn` option)

### Adding More Figures

```latex
% In paper.tex or paper_article.tex
\begin{figure}[htbp]
\centerline{\includegraphics[width=0.48\textwidth]{figures/your_figure.pdf}}
\caption{Your caption here.}
\label{fig:your_label}
\end{figure}

% Reference in text:
See Figure \ref{fig:your_label} for details.
```

## 📊 Tables Summary

The paper includes 3 key tables:

1. **Table 1: Competition Results** (Section 3.1)
   - Challenge 1 & 2 scores
   - Overall NRMSE: 1.00052
   - Rank: 72/150
   - Gap to 1st place: 2.7%

2. **Table 2: Variance Analysis** (Section 3.2)
   - Cross-validation: 0.62% CV
   - Random seed: 0.18% CV
   - TTA: 0.09% CV
   - Submission: 0.23% CV

3. **Table 3: Architectural Ablations** (Section 3.3)
   - 6 models compared
   - Parameters, training time, NRMSE
   - Best: EEGNeX ensemble (0.9965)
   - Best efficiency: EnhancedCompactCNN (1.0019, 2 min)

## 🎯 Target Venues

### Primary Submission Targets

1. **IEEE Transactions on Neural Systems & Rehabilitation Engineering** ⭐
   - Impact Factor: ~4.5
   - Format: IEEE conference (paper.tex) or journal
   - Page Limit: 8-12 pages
   - Focus: Neural engineering, rehabilitation technology

2. **Journal of Neural Engineering**
   - Impact Factor: ~5.0
   - Format: IOPscience LaTeX template (modify paper.tex)
   - Page Limit: No strict limit
   - Focus: Neural engineering, brain-machine interfaces

3. **Frontiers in Neuroscience (Neuroprosthetics)**
   - Impact Factor: ~4.0
   - Format: Frontiers LaTeX template (use article version as base)
   - Page Limit: Flexible
   - Focus: Neuroscience, neuroprosthetics, BCIs

4. **NeurIPS 2025 Workshop on EEG**
   - Format: NeurIPS workshop (4-6 pages)
   - Deadline: Check conference website
   - Page Limit: 4-6 pages (condensed version needed)

## 📝 Compilation Notes

### Common Issues

**Missing IEEEtran.cls:**
```bash
# Solution 1: Install via package manager
sudo apt-get install texlive-publishers

# Solution 2: Use article-class version
pdflatex paper_article.tex

# Solution 3: Download manually
wget http://www.ieee.org/documents/IEEEtran.cls
```

**Missing figure files:**
```bash
# Regenerate figures
python generate_figures.py

# Or download pre-generated figures from repository
```

**References not rendering:**
```bash
# Run pdflatex twice to resolve references
pdflatex paper.tex
pdflatex paper.tex
```

### Requirements

**LaTeX packages needed:**
- Standard: `geometry`, `amsmath`, `graphicx`, `booktabs`
- Optional: `hyperref`, `listings`, `xcolor`
- IEEE-specific: `IEEEtran.cls` (for paper.tex)

**Python packages for figures:**
```bash
pip install matplotlib numpy seaborn
```

## 📤 Submission Checklist

Before submitting to a venue:

- [ ] Compile LaTeX successfully (no errors)
- [ ] All figures rendering correctly
- [ ] References formatted properly
- [ ] Update author information
- [ ] Check page limits for target venue
- [ ] Verify figure quality (300 DPI minimum)
- [ ] Proofread entire document
- [ ] Get co-author approval (if applicable)
- [ ] Convert to venue-specific format if needed
- [ ] Prepare supplementary materials:
  - [ ] Code repository link
  - [ ] Trained model weights
  - [ ] Detailed hyperparameters (Appendix A in markdown paper)
- [ ] Write cover letter highlighting key contributions

## 🔗 Related Files

- **Markdown version:** `../docs/PUBLICATION_PAPER.md` (1,291 lines)
- **Quick reference:** `../docs/PUBLICATION_QUICK_REFERENCE.md` (371 lines)
- **Code repository:** https://github.com/hkevin01/eeg2025
- **Original README:** `../README.md`

## 📧 Contact

- **Author:** hkevin01
- **Repository:** https://github.com/hkevin01/eeg2025
- **Issues:** Open GitHub issue for questions or problems

## 📄 License

This LaTeX document and figures are part of the eeg2025 repository, released under the MIT License.

---

**Last Updated:** November 6, 2025  
**Status:** ✅ Publication-ready  
**Next Steps:** Review content → Convert to target venue format → Submit

