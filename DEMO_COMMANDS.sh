#!/bin/bash

# Face Recognition Demo Script
# Run this during your presentation

echo "=========================================="
echo " FACE RECOGNITION SYSTEM DEMO"
echo " Agriya Yadav - CS-4440"
echo "=========================================="
echo ""

# Navigate to project
cd /Users/Agriya/Desktop/monsoon25/AI/facial-recognition

echo "1. PROJECT STRUCTURE:"
echo "   - Dataset: 40,709 images across 247 identities"
ls -lh data/ | head -5
echo ""

echo "2. CONFIGURATION:"
cat configs/default.yaml
echo ""

echo "3. AVAILABLE MODELS:"
ls -lh src/*.py | grep eval
echo ""

echo "4. KEY VISUALIZATIONS:"
ls -lh runs/*.png
echo ""

echo "=========================================="
echo " Ready to run evaluation!"
echo " Command: python src/eval_arcface_closedset.py"
echo "=========================================="

# Optional: Uncomment to auto-run evaluation
# python src/eval_arcface_closedset.py

echo ""
echo "QUICK STATS:"
echo "  ✓ Buffalo_L Accuracy: 95.27%"
echo "  ✓ Classical LBP+SVM: 24.59%"
echo "  ✓ Improvement: +71 percentage points"
echo "  ✗ Occlusion vulnerability: 34.2%"
echo "  ✗ Demographic bias gap: 32.8%"
echo "  ✗ Crowd performance: 33.3%"
echo ""
echo "Demo complete! Open visualizations:"
echo "  - open runs/robustness_analysis.png"
echo "  - open runs/fairness_analysis.png"
echo "  - open runs/crowd_analysis.png"
echo "  - open main.pdf"
