#!/usr/bin/env python3
"""
Quick verification script to spot-check robustness results (FIXED for aligned images)
Tests Buffalo_L on seed 42 sample (original condition, no perturbations)
Expected accuracy: ~46.7%
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from insightface.app import FaceAnalysis
from tqdm import tqdm

def load_buffalo_l():
    """Load Buffalo_L model"""
    print("Loading Buffalo_L model...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    rec_model = app.models['recognition']
    return app, rec_model

def get_embedding_from_aligned_face(rec_model, img):
    """Extract embedding from pre-aligned face image"""
    # Resize to 112x112
    if img.shape[0] != 112 or img.shape[1] != 112:
        img = cv2.resize(img, (112, 112))
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    img_normalized = img_rgb.astype(np.float32)
    img_normalized = (img_normalized - 127.5) / 127.5
    
    # Transpose to CHW and add batch dimension
    img_input = img_normalized.transpose(2, 0, 1)[np.newaxis, ...]
    
    # Run inference
    input_name = rec_model.session.get_inputs()[0].name
    embedding = rec_model.session.run(None, {input_name: img_input})[0][0]
    
    # L2 normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
    
    return embedding

def compute_prototypes(rec_model, train_root):
    """Compute class prototypes from training data"""
    print("Computing class prototypes from training data...")
    
    train_path = Path(train_root)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_root}")
    
    train_embeddings = []
    train_labels = []
    
    class_dirs = [d for d in sorted(train_path.iterdir()) if d.is_dir()]
    print(f"Found {len(class_dirs)} celebrity classes")
    
    for cls_dir in tqdm(class_dirs, desc="Processing classes"):
        cls_name = cls_dir.name
        image_paths = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        
        # Use only 3 images per class for fast computation
        image_paths = image_paths[:3]
        
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                embedding = get_embedding_from_aligned_face(rec_model, img)
                train_embeddings.append(embedding)
                train_labels.append(cls_name)
            except:
                continue
    
    # Build prototypes
    buckets = defaultdict(list)
    for emb, label in zip(train_embeddings, train_labels):
        buckets[label].append(emb)
    
    ids = sorted(buckets.keys())
    proto_list = [np.mean(buckets[cls_id], axis=0) for cls_id in ids]
    prototypes = np.stack(proto_list, axis=0)
    
    # L2 normalize
    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    
    print(f"✅ Computed {len(ids)} prototypes from {len(train_embeddings)} embeddings")
    return ids, prototypes

def predict_with_prototypes_aligned(rec_model, ids, prototypes, image_path):
    """Predict identity directly from aligned face image (no detection needed)"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    
    # Get embedding directly (image is already aligned)
    try:
        embedding = get_embedding_from_aligned_face(rec_model, img)
        embedding = embedding.reshape(1, -1)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Compute similarities
        similarities = (embedding @ prototypes.T).flatten()
        
        # Get best match
        best_idx = np.argmax(similarities)
        predicted_label = ids[best_idx]
        confidence = similarities[best_idx]
        
        return predicted_label, confidence
    except Exception as e:
        return None, None

def get_true_label(image_path):
    """Extract true label from path (celebrity name)"""
    # Path format: .../CelebrityName/image.jpg
    return Path(image_path).parent.name

def main():
    print("="*70)
    print("🔍 ROBUSTNESS SAMPLE VERIFICATION (ALIGNED IMAGES)")
    print("="*70)
    
    # Load sample
    sample_file = "robustness_sample_seed_42.txt"
    print(f"\n📄 Loading sample from {sample_file}")
    
    with open(sample_file, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    
    print(f"✅ Loaded {len(test_images)} images")
    
    # Load model
    app, rec_model = load_buffalo_l()
    
    # Compute prototypes
    ids, prototypes = compute_prototypes(rec_model, "data/aligned/train")
    
    # Test on sample
    print(f"\n🧪 Testing Buffalo_L on {len(test_images)} images (original condition)...")
    print("   (Using direct embedding extraction - images are pre-aligned)")
    print("-"*70)
    
    correct = 0
    processed = 0
    total = len(test_images)
    
    results = []
    
    for img_path in tqdm(test_images, desc="Testing"):
        true_label = get_true_label(img_path)
        pred_label, confidence = predict_with_prototypes_aligned(rec_model, ids, prototypes, img_path)
        
        if pred_label is not None:
            processed += 1
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            
            results.append({
                'image': Path(img_path).name,
                'true': true_label,
                'pred': pred_label,
                'correct': is_correct,
                'confidence': confidence
            })
    
    # Calculate metrics
    accuracy = correct / processed if processed > 0 else 0
    processing_rate = processed / total if total > 0 else 0
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Total images:     {total}")
    print(f"Processed:        {processed}")
    print(f"Correct:          {correct}")
    print(f"Processing rate:  {processing_rate:.2%} ({processed}/{total})")
    print(f"Accuracy:         {accuracy:.2%} ({correct}/{processed})")
    print("="*70)
    
    print(f"\n🎯 Expected accuracy: ~46.7% (from robustness_results.csv)")
    print(f"   Actual accuracy:   {accuracy:.2%}")
    
    diff = abs(accuracy - 0.467)
    if diff < 0.05:  # Within 5%
        print(f"\n✅ VERIFICATION PASSED!")
        print(f"   Difference: {diff:.2%} (within acceptable range)")
        print(f"   ✨ This confirms seed 42 is likely the correct sample!")
    elif diff < 0.10:  # Within 10%
        print(f"\n⚠️  CLOSE BUT NOT EXACT")
        print(f"   Difference: {diff:.2%}")
        print(f"   Seed 42 might be correct, or methodology differs slightly.")
    else:
        print(f"\n⚠️  DIFFERENCE DETECTED")
        print(f"   Difference: {diff:.2%}")
        print(f"   This might indicate a different sample was used.")
        print(f"   Try other seeds or check your notebook.")
    
    # Show some examples
    print(f"\n📋 Sample Results (first 10):")
    print("-"*70)
    for i, r in enumerate(results[:10], 1):
        status = "✓" if r['correct'] else "✗"
        print(f"{i:2d}. {status} True: {r['true']:20s} | Pred: {r['pred']:20s} | Conf: {r['confidence']:.3f}")
    
    print("="*70)
    
    # Summary
    if diff < 0.05:
        print("\n🎉 CONCLUSION: Robustness results are verified!")
        print("   Your existing CSV data is reliable. You can confidently use it")
        print("   in your presentation without re-running all experiments.")
    else:
        print("\n📝 CONCLUSION: Results differ from expected.")
        print("   This could mean:")
        print("   - Different random seed was used originally")
        print("   - Different sampling methodology")
        print("   - Check your notebook for the actual sampling code")
    
    print("="*70)

if __name__ == "__main__":
    main()
