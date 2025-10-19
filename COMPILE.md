# LaTeX Compilation Instructions

## Prerequisites
Install LaTeX distribution:
- **Windows**: MiKTeX or TeX Live
- **Mac**: MacTeX
- **Linux**: `sudo apt-get install texlive-full`

## File Structure
```
project/
├── main.tex
├── references.bib
└── runs/
    ├── eda/
    │   ├── class_counts.png
    │   ├── quality_hists.png
    │   ├── det_coverage_hists.png
    │   └── intra_inter_cosine.png
    ├── explainability/
    │   ├── lbp_explainability.png
    │   └── deep_attention.png
    ├── fairness_analysis.png
    ├── robustness_analysis.png
    ├── crowd_analysis.png
    ├── ai_detection_distributions.png
    ├── ai_detection_feature_importance.png
    └── real_vs_ai_samples.png
```

## Compilation Steps

### Method 1: Command Line
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Method 2: Overleaf (Recommended for Beginners)
1. Go to https://www.overleaf.com/
2. Create new project → Upload Project
3. Upload `main.tex` and `references.bib`
4. Create folder structure and upload all images
5. Click "Recompile"

### Method 3: TeXstudio
1. Open `main.tex` in TeXstudio
2. Press F5 or click "Build & View"

## Troubleshooting

**Missing images error:**
- Ensure all image paths match exactly
- Check that `runs/` folder exists in same directory as `main.tex`

**Bibliography not appearing:**
- Run: `pdflatex → bibtex → pdflatex → pdflatex` (4 times total)

**Missing packages:**
- On first run, MiKTeX will auto-install missing packages
- For TeX Live, run: `tlmgr install <package-name>`

**Overfull/underfull warnings:**
- These are formatting warnings, PDF will still compile
- Can be safely ignored for draft versions

## Output
Successfully compiled PDF: `main.pdf` (~10-12 pages)
```

---

## **QUICK START**

1. **Save these 3 files:**
   - `main.tex`
   - `references.bib`
   - `COMPILE.md`

2. **Ensure your image files are in correct paths:**
```
   runs/eda/class_counts.png
   runs/eda/quality_hists.png
   runs/eda/det_coverage_hists.png
   runs/eda/intra_inter_cosine.png
   runs/explainability/lbp_explainability.png
   runs/explainability/deep_attention.png
   runs/fairness_analysis.png
   runs/robustness_analysis.png
   runs/crowd_analysis.png
   runs/ai_detection_distributions.png
   runs/ai_detection_feature_importance.png
   runs/real_vs_ai_samples.png