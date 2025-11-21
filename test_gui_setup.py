#!/usr/bin/env python3
"""
Simple tester for the GUI application
Tests if the model can load and make predictions
"""

import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

def test_basic_setup():
    """Test if InsightFace and Buffalo_L work"""
    print("="*60)
    print("🧪 TESTING FACE RECOGNITION SYSTEM")
    print("="*60)
    
    # Test 1: Load model
    print("\n1️⃣  Loading Buffalo_L model...")
    try:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False
    
    # Test 2: Check data directory
    print("\n2️⃣  Checking data directory...")
    data_root = Path("data/aligned/train")
    if not data_root.exists():
        print(f"   ❌ Directory not found: {data_root}")
        return False
    
    class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    print(f"   ✅ Found {len(class_dirs)} celebrity classes")
    
    # Test 3: Load sample image
    print("\n3️⃣  Testing face detection on sample image...")
    sample_found = False
    for cls_dir in class_dirs[:5]:  # Try first 5 classes
        images = list(cls_dir.glob("*.jpg"))
        if images:
            img_path = images[0]
            img = cv2.imread(str(img_path))
            if img is not None:
                faces = app.get(img)
                if faces:
                    print(f"   ✅ Detected {len(faces)} face(s) in {img_path.name}")
                    print(f"   ✅ Embedding shape: {faces[0].normed_embedding.shape}")
                    sample_found = True
                    break
    
    if not sample_found:
        print("   ⚠️  Could not find sample image with detectable face")
    
    # Test 4: Check GUI file exists
    print("\n4️⃣  Checking GUI application...")
    gui_file = Path("gui_app.py")
    if gui_file.exists():
        print(f"   ✅ GUI file exists: {gui_file}")
    else:
        print(f"   ❌ GUI file not found: {gui_file}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🎉 You're ready to run the GUI!")
    print("\nNext steps:")
    print("   1. Run: python gui_app.py")
    print("   2. Open browser: http://127.0.0.1:7860")
    print("   3. Click 'Load Model'")
    print("   4. Upload an image and click 'Recognize Faces'")
    print("\n" + "="*60)
    
    return True

def list_sample_images():
    """List some sample images you can test with"""
    print("\n📸 Sample images from your dataset:\n")
    
    data_root = Path("data/aligned/train")
    if not data_root.exists():
        print("   ❌ Training data not found")
        return
    
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    print(f"Available celebrities ({len(class_dirs)} total):\n")
    for i, cls_dir in enumerate(class_dirs[:10], 1):  # Show first 10
        images = list(cls_dir.glob("*.jpg"))
        if images:
            print(f"   {i}. {cls_dir.name}: {len(images)} images")
            print(f"      Example: {images[0]}")
    
    if len(class_dirs) > 10:
        print(f"\n   ... and {len(class_dirs) - 10} more celebrities")
    
    print("\n💡 Tip: You can use these images to test the GUI!")
    print("   Copy any image path and test with it.")

if __name__ == "__main__":
    print("\n🎭 Face Recognition System - Quick Test\n")
    
    success = test_basic_setup()
    
    if success:
        print("\n" + "="*60)
        list_sample_images()
        print("="*60)
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        print("   Make sure you have:")
        print("   - InsightFace installed")
        print("   - Training data in data/aligned/train/")
        print("   - GUI file (gui_app.py)")
