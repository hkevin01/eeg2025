# Methods Document - PDF Conversion Instructions

## Current Status
- ✅ Markdown version: `METHODS_DOCUMENT.md`
- ✅ HTML version: `METHODS_DOCUMENT.html`
- ⏳ PDF version: Needs manual conversion

## How to Convert to PDF

### Option 1: Browser Print (Easiest)
1. Open `METHODS_DOCUMENT.html` in a browser
2. Press Ctrl+P (or Cmd+P on Mac)
3. Select "Save as PDF"
4. Save as `METHODS_DOCUMENT.pdf`

### Option 2: Google Docs
1. Copy content from `METHODS_DOCUMENT.md`
2. Paste into Google Docs
3. Format as needed
4. File → Download → PDF

### Option 3: Install LaTeX and Use Pandoc
```bash
sudo apt-get install texlive-xetex
pandoc METHODS_DOCUMENT.md -o METHODS_DOCUMENT.pdf --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt
```

### Option 4: Online Converter
1. Go to https://www.markdowntopdf.com/
2. Upload `METHODS_DOCUMENT.md`
3. Download PDF

## Required for Submission
- File: `METHODS_DOCUMENT.pdf`
- Size: Should be 2 pages
- Format: PDF
- Content: Methods, results, discussion

## Current File Locations
- Markdown: `/home/kevin/Projects/eeg2025/METHODS_DOCUMENT.md`
- HTML: `/home/kevin/Projects/eeg2025/METHODS_DOCUMENT.html`
- PDF: (to be created)
