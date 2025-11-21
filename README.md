# Face Recognition System Analysis

**Comprehensive Evaluation of Face Recognition Across Performance, Robustness, Fairness, and Ethics**

---

## 🎯 Project Overview

This project presents a systematic evaluation of face recognition systems, comparing classical machine learning (LBP+SVM) against modern deep learning approaches (ArcFace-based Buffalo_L and AntelopeV2). Beyond just measuring accuracy, this project analyzes:

- **Performance**: Accuracy across different architectures
- **Robustness**: Resilience to real-world conditions (occlusions, noise, blur, compression)
- **Explainability**: Understanding model decision-making processes
- **Fairness**: Demographic bias and performance disparities
- **Security**: Vulnerability to AI-generated synthetic faces

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Best Accuracy** | 95.27% (Buffalo_L) |
| **Classical Baseline** | 24.59% (LBP+SVM) |
| **Improvement** | +71 percentage points |
| **Dataset Size** | 40,709 images, 247 identities |
| **Worst Robustness** | 34.2% (face occlusions) |
| **Bias Gap** | 32.8% (skin tone disparity) |
| **Crowd Performance** | 33.3% (vs 95% single-face) |
| **AI Face Detection** | 100% (current generation) |

---

## 🗂️ Project Structure

```
facial-recognition/
├── README.md                          # This file
├── PRESENTATION_GUIDE.md              # Comprehensive presentation guide
├── QUICK_SUMMARY.md                   # Quick reference card
├── SLIDES_OUTLINE.md                  # Slide-by-slide content
├── PRE_PRESENTATION_CHECKLIST.md      # Pre-demo checklist
├── DEMO_COMMANDS.sh                   # Executable demo script
│
├── main.pdf                           # Full LaTeX report (12 pages)
├── short_report.pdf                   # Condensed version
├── main.tex                           # LaTeX source (666 lines)
├── COMPILE.md                         # LaTeX compilation instructions
│
├── configs/
│   └── default.yaml                   # Model & evaluation config
│
├── data/
│   ├── aligned/                       # Face-aligned images
│   ├── celeb/                         # Celebrity metadata
│   ├── raw/                           # Original images
│   └── raw_selected/                  # Filtered subset
│
├── src/                               # Evaluation scripts
│   ├── eval_arcface_closedset.py      # Buffalo_L evaluation
│   ├── eval_insightface_closedset.py  # AntelopeV2 evaluation
│   ├── eval_lbp_svm.py                # Classical baseline
│   ├── benchmark_models.py            # Robustness testing
│   ├── bias_eval.py                   # Fairness analysis
│   ├── explain_lbp.py                 # LBP explainability
│   ├── explain_occlusion.py           # Attention analysis
│   └── robust_plots.py                # Visualization generation
│
├── scripts/                           # Data preprocessing
│   ├── select_ids.py                  # Identity selection
│   ├── materialize_selected.py        # Dataset creation
│   ├── prepare_and_align.py           # Face detection & alignment
│   └── cache_embeddings.py            # Embedding pre-computation
│
├── runs/                              # Generated results
│   ├── eda/                           # Exploratory data analysis
│   │   ├── class_counts.png
│   │   ├── quality_hists.png
│   │   ├── intra_inter_cosine.png
│   │   └── umap_embeddings.png
│   ├── explainability/
│   │   ├── lbp_explainability.png
│   │   └── deep_attention.png
│   ├── robustness_analysis.png
│   ├── fairness_analysis.png
│   ├── crowd_analysis.png
│   ├── ai_detection_distributions.png
│   └── ai_detection_feature_importance.png
│
├── eda.ipynb                          # Primary analysis notebook (298 KB)
├── remaining.ipynb                    # Additional experiments (3.8 MB)
└── requirements.txt                   # Python dependencies
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project
cd /Users/Agriya/Desktop/monsoon25/AI/facial-recognition

# Activate virtual environment
source .venv/bin/activate

# Verify installation
pip list | grep insightface
```

### 2. Run Evaluations

```bash
# Evaluate Buffalo_L (best model)
python src/eval_arcface_closedset.py

# Run robustness benchmarks
python src/benchmark_models.py

# Analyze fairness/bias
python src/bias_eval.py
```

### 3. View Results

```bash
# Open main report
open main.pdf

# View key visualizations
open runs/robustness_analysis.png
open runs/fairness_analysis.png
open runs/crowd_analysis.png
```

### 4. Run Demo Script

```bash
# Execute pre-made demo
./DEMO_COMMANDS.sh
```

---

## 📈 Key Findings

### ✅ What Works Well

1. **High Accuracy**: 95.27% on clean, single-face images
2. **Lighting Robustness**: Minimal degradation (45.0% under brightness changes)
3. **Compression Resilience**: JPEG artifacts have limited impact (45.3%)
4. **AI Detection**: Current AI-generated faces perfectly separable (100% accuracy)
5. **Biological Plausibility**: Attention maps focus on eyes/nose (human-like)

### ❌ Critical Vulnerabilities

1. **Face Occlusions**: Accuracy drops to 34.2% with masks/covers
   - COVID-19 implications (masks rendered many systems ineffective)
2. **Crowd Scenarios**: 33.3% accuracy in multi-face images (vs 95% single-face)
   - Surveillance/security limitations
3. **Demographic Bias**: 32.8% performance gap across skin tone proxies
   - Ethical concerns: discriminatory outcomes
4. **Heavy Degradation**: Noise (σ=25) and blur (k=11) cause 20-25% drops
5. **Classical Methods**: LBP+SVM completely fails (24.59%)

---

## 🔬 Models Evaluated

### 1. Buffalo_L (Best: 95.27%)
- **Architecture**: ResNet-50 backbone
- **Training**: WebFace600K dataset
- **Loss**: ArcFace (additive angular margin)
- **Output**: 512-dimensional L2-normalized embeddings
- **Strengths**: Balanced accuracy and robustness

### 2. AntelopeV2 (94.59%)
- **Architecture**: ResNet-100 (100M parameters)
- **Training**: Large-scale InsightFace datasets
- **Strengths**: Highest clean accuracy
- **Weakness**: More vulnerable to blur

### 3. LBP+SVM (Baseline: 24.59%)
- **Method**: Local Binary Patterns + SVM classifier
- **Features**: Hand-crafted 16,384D texture descriptors
- **Strengths**: Interpretable
- **Weakness**: Fails to capture complex patterns

---

## 📊 Dataset

- **Source**: Indian celebrity images (Bollywood, South Indian cinema)
- **Scale**: 40,709 images across 247 identities
- **Distribution**: Highly imbalanced (14-620 images/identity, μ=164.8, σ=106.3)
- **Split**: 60% train / 20% validation / 20% test (stratified)
- **Quality**: Varied brightness, blur, aspect ratios (reflects real-world conditions)

---

## 🧪 Experiments Conducted

### 1. Performance Evaluation
- Closed-set recognition (k-NN with cosine similarity)
- Threshold tuning (τ=0.25 for Buffalo_L)
- Macro F1, precision, recall metrics

### 2. Robustness Testing (15 Perturbations)
- **Lighting**: Brightness scaling (0.6×, 0.8×)
- **Noise**: Gaussian σ ∈ {5, 15, 25}
- **Blur**: Gaussian kernel ∈ {3, 7, 11}
- **Compression**: JPEG quality ∈ {20, 50, 90}
- **Occlusion**: Eye bar, mouth mask, 50% full mask

### 3. Explainability Analysis
- LBP pattern visualization
- Occlusion-based attention mapping
- Saliency heatmaps

### 4. Fairness/Bias Testing
- Skin tone proxy via brightness (Dark/Medium/Light)
- Performance disparity measurement
- Statistical significance testing

### 5. Advanced Security Testing
- Crowd image recognition (27 faces across 10 images)
- AI-generated face detection (StyleGAN2, ThisPersonDoesNotExist)
- Feature artifact analysis (FFT, edge detection)

---

## 📝 Documentation

### Primary Report
- **File**: `main.pdf` (12 pages, 666 lines LaTeX)
- **Sections**: Introduction, Methodology, Results, Ethics, Conclusion
- **Figures**: 12 professional visualizations
- **Tables**: 5 result summaries
- **References**: 9 academic citations

### Presentation Materials
1. **PRESENTATION_GUIDE.md**: Comprehensive 20-min presentation script
2. **QUICK_SUMMARY.md**: One-page reference card
3. **SLIDES_OUTLINE.md**: 20 slides with backup slides
4. **PRE_PRESENTATION_CHECKLIST.md**: Pre-demo verification

---

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, InsightFace (ArcFace implementation)
- **Classical ML**: scikit-learn (SVM, metrics)
- **Computer Vision**: OpenCV, Albumentations (augmentation)
- **Visualization**: Matplotlib, Seaborn
- **Analysis**: Pandas, NumPy, SciPy (statistics)
- **Explainability**: pytorch-grad-cam (although custom occlusion used)
- **Documentation**: LaTeX (professional report)

---

## ⚖️ Ethical Considerations

### Privacy & Surveillance
- Mass surveillance enables authoritarian control
- Biometric data is irreversible (can't change your face)
- Consent mechanisms often inadequate

### Bias & Discrimination
- 32.8% performance gap across demographics
- Real-world harms: false arrests (Robert Williams case)
- Gender Shades study: 34% error on dark-skinned women vs 0.8% light-skinned men

### Recommendations
1. **Diverse training data** with balanced representation
2. **Fairness constraints** during optimization
3. **Regular audits** across demographic groups
4. **Human oversight** for high-stakes decisions
5. **Transparent disclosure** of limitations
6. **Liveness detection** for anti-spoofing
7. **Multi-modal authentication** (face + fingerprint)
8. **Context-specific regulation** (ban in schools, allow in passports)

---

## 🔮 Future Work

1. **Adversarial Robustness**: FGSM, PGD, C&W attacks
2. **Temporal Consistency**: Video sequence testing
3. **Cross-Dataset Generalization**: Train on CASIA, test on LFW/AgeDB
4. **Fairness Interventions**: Reweighting, adversarial debiasing
5. **Federated Learning**: Privacy-preserving training
6. **Inherent Interpretability**: Neural-symbolic methods
7. **Next-Gen AI Faces**: Test on DALL-E 3, Midjourney v6 outputs

---

## 🎓 Academic Context

**Course**: CS-4440 Artificial Intelligence  
**Institution**: Ashoka University  
**Instructor**: Prof. Lipika Dey  
**Student**: Agriya Yadav (1020231092)  
**Date**: October 19, 2025  

---

## 📚 Key References

1. Deng et al. (2019): "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
2. Buolamwini & Gebru (2018): "Gender Shades: Intersectional Accuracy Disparities"
3. NIST (2020): "Face recognition with face masks" performance report
4. Peterson & Eckstein (2012): "Looking just below the eyes is optimal across face recognition tasks"

---

## 🎯 For Presentation Tomorrow

### Must-Have Files Open:
1. ✅ `QUICK_SUMMARY.md` - Your cheat sheet
2. ✅ `main.pdf` - Main deliverable
3. ✅ `runs/robustness_analysis.png` - Key visual
4. ✅ `runs/fairness_analysis.png` - Bias results
5. ✅ `PRE_PRESENTATION_CHECKLIST.md` - Last-minute review

### Quick Demo:
```bash
cd /Users/Agriya/Desktop/monsoon25/AI/facial-recognition
source .venv/bin/activate
./DEMO_COMMANDS.sh
# Then open visualizations
```

### Numbers to Memorize:
- 40,709 images, 247 identities
- 95.27% best accuracy (+71 over classical)
- 34.2% with occlusions (worst)
- 32.8% bias gap
- 33.3% crowd accuracy

---

## 💡 Key Takeaway

**Face recognition technology achieves 95% accuracy in ideal conditions, but exhibits critical vulnerabilities in real-world deployment: face masks reduce accuracy to 34%, crowd scenarios drop it to 33%, and demographic bias creates a 33% performance gap. Technical excellence must be coupled with ethical responsibility.**

---

## 📞 Contact

**Agriya Yadav**  
Student ID: 1020231092  
Ashoka University  
agriya.yadav@ashoka.edu.in

---

## 🚀 You've Got This!

You've done comprehensive, rigorous research across 5 critical dimensions. Your project goes beyond most academic work by evaluating not just accuracy, but robustness, fairness, explainability, and ethics. Be confident!

**Good luck with your presentation! 🍀**
