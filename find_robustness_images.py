#!/usr/bin/env python3
"""
Script to find/recreate robustness test images
Tries common random seeds to reproduce the 150-image sample
"""
import random
import numpy as np
from pathlib import Path
import sys

def get_test_images(test_root="data/aligned/test"):
    """Get all test images sorted consistently"""
    test_path = Path(test_root)
    if not test_path.exists():
        print(f"❌ Error: {test_root} not found!")
        return []
    
    images = sorted(list(test_path.rglob("*.jpg")) + 
                   list(test_path.rglob("*.png")) + 
                   list(test_path.rglob("*.jpeg")))
    return images

def try_seed(all_images, seed, n_sample=150):
    """Try sampling with a given seed"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if len(all_images) < n_sample:
        print(f"⚠️  Warning: Only {len(all_images)} images available, need {n_sample}")
        n_sample = len(all_images)
    
    sample = random.sample(all_images, n_sample)
    return sample

def main():
    print("="*70)
    print("🔍 ROBUSTNESS TEST IMAGE FINDER")
    print("="*70)
    
    # Get all test images
    all_images = get_test_images()
    print(f"\n📊 Found {len(all_images)} total test images")
    
    if not all_images:
        print("❌ No test images found. Check data directory.")
        return
    
    # Try common seeds
    common_seeds = [42, 0, 123, 2024, 2025, 1, 99, 1337, None]
    
    print(f"\n🎲 Trying {len(common_seeds)} common random seeds...")
    print("="*70)
    
    for i, seed in enumerate(common_seeds, 1):
        sample = try_seed(all_images, seed, 150)
        
        seed_label = f"seed_{seed}" if seed is not None else "no_seed"
        output_file = f"robustness_sample_{seed_label}.txt"
        
        print(f"\n[{i}/{len(common_seeds)}] Seed: {seed if seed is not None else 'None (random)'}")
        print("-" * 70)
        
        # Show first 3 and last 3
        print("📄 First 3 images:")
        for img in sample[:3]:
            print(f"   {img.relative_to('data/aligned/test')}")
        
        print("   ...")
        
        print("📄 Last 3 images:")
        for img in sample[-3:]:
            print(f"   {img.relative_to('data/aligned/test')}")
        
        # Save to file
        with open(output_file, "w") as f:
            for img in sample:
                f.write(str(img) + "\n")
        
        print(f"💾 Saved to: {output_file}")
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print(f"\nGenerated {len(common_seeds)} sample lists in current directory.")
    print("\n📌 Next steps:")
    print("1. Check if any of these match your robustness analysis results")
    print("2. Open your Jupyter notebook to find the actual seed/sampling code")
    print("3. Or use the characterization approach in ROBUSTNESS_IMAGE_FINDER.md")
    print("="*70)

if __name__ == "__main__":
    main()
