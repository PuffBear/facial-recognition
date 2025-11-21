#!/usr/bin/env python3
"""
Face Recognition GUI Application
Upload an image → Detect faces → Classify with confidence scores
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# ==================== MODEL LOADING ====================

class FaceRecognitionSystem:
    def __init__(self, data_root="data/aligned"):
        """Initialize the face recognition system with trained model"""
        self.data_root = Path(data_root)
        self.app = None
        self.ids = None
        self.prototypes = None
        self.loaded = False
        self.loading = False
        self.demo_mode = False  # True if running without training data

    def load_model(self):
        """Load Buffalo_L model and compute class prototypes"""
        if self.loading:
            return "⏳ Model is already loading, please wait..."

        if self.loaded:
            if self.demo_mode:
                return "✅ Model loaded in DEMO MODE (face detection only)."
            return f"✅ Model already loaded! {len(self.ids)} celebrities ready."

        self.loading = True

        try:
            print("Loading Buffalo_L model...")
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            print("✅ Buffalo_L model loaded successfully!")

            # Check if training data exists
            train_root = self.data_root / "train"

            if not train_root.exists():
                print(f"⚠️ Training data not found at {train_root}")
                print("Switching to DEMO MODE (face detection + embedding only)...")

                # Load known celebrity IDs from confusion.txt if available
                confusion_path = Path("runs/arcface_base/confusion.txt")
                if confusion_path.exists():
                    with open(confusion_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2 and lines[0].startswith("IDS:"):
                            self.ids = lines[1].strip().split(',')
                            print(f"📋 Loaded {len(self.ids)} celebrity names from previous run")

                self.demo_mode = True
                self.loaded = True
                self.loading = False

                status_msg = "✅ Model loaded in DEMO MODE!\n\n"
                status_msg += "📌 Face detection and embedding extraction are available.\n"
                status_msg += "⚠️ Classification is limited (no training data).\n"
                if self.ids:
                    status_msg += f"📋 Known celebrities: {len(self.ids)} (from previous run)"
                return status_msg

            print("Computing class prototypes from training data...")

            # Load training images
            train_embeddings = []
            train_labels = []

            class_dirs = [d for d in sorted(train_root.iterdir()) if d.is_dir()]
            print(f"Found {len(class_dirs)} celebrity classes")

            # ENHANCED ERROR TRACKING
            failed_classes = []
            successful_classes = []

            for idx, cls_dir in enumerate(class_dirs, 1):
                cls_name = cls_dir.name
                image_paths = list(cls_dir.glob("*.jpg"))

                if not image_paths:
                    print(f"  ⚠️  No .jpg images found in {cls_name}, trying other formats...")
                    # Try other common image formats
                    image_paths = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))

                # Use only 3 images per class for FAST prototype computation
                image_paths = image_paths[:3]

                class_embeddings_count = 0
                for img_path in image_paths:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"  ⚠️  Could not read image: {img_path}")
                        continue

                    # Check if image is valid
                    if img.size == 0:
                        print(f"  ⚠️  Empty image: {img_path}")
                        continue

                    faces = self.app.get(img)
                    if faces:
                        train_embeddings.append(faces[0].normed_embedding)
                        train_labels.append(cls_name)
                        class_embeddings_count += 1
                    else:
                        print(f"  ⚠️  No face detected in: {img_path.name}")

                if class_embeddings_count == 0:
                    failed_classes.append(cls_name)
                    print(f"  ❌ No embeddings extracted for {cls_name}")
                else:
                    successful_classes.append(cls_name)

                # Progress feedback with running total
                if idx % 5 == 0:
                    print(f"  Processed {idx}/{len(class_dirs)} celebrities... ({len(train_embeddings)} embeddings so far)")

            # Final summary
            print(f"\n📊 Processing Summary:")
            print(f"  Total embeddings: {len(train_embeddings)}")
            print(f"  Successful classes: {len(successful_classes)}/{len(class_dirs)}")
            print(f"  Failed classes: {len(failed_classes)}")

            if failed_classes:
                print(f"  Classes with no embeddings: {', '.join(failed_classes[:10])}" +
                      (f" (and {len(failed_classes)-10} more)" if len(failed_classes) > 10 else ""))

            if not train_embeddings:
                # Fall back to demo mode
                print("⚠️ No embeddings computed, switching to DEMO MODE...")
                self.demo_mode = True
                self.loaded = True
                self.loading = False
                return "✅ Model loaded in DEMO MODE (face detection only - no training embeddings computed)."

            print(f"✅ Successfully computed {len(train_embeddings)} embeddings")

            # Build prototypes (mean embedding per class)
            buckets = defaultdict(list)
            for emb, label in zip(train_embeddings, train_labels):
                buckets[label].append(emb)

            self.ids = sorted(buckets.keys())
            proto_list = [np.mean(buckets[cls_id], axis=0) for cls_id in self.ids]
            self.prototypes = np.stack(proto_list, axis=0)

            # L2 normalize prototypes
            self.prototypes = self.prototypes / (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-9)

            self.loaded = True
            self.loading = False
            print(f"✅ Model loaded successfully! {len(self.ids)} celebrities ready.")
            return f"✅ Model loaded! {len(self.ids)} celebrities in database ({len(train_embeddings)} total embeddings)."

        except Exception as e:
            self.loading = False
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Error loading model:\n{error_detail}")
            return f"❌ Error loading model: {str(e)}"
    
    def recognize_face(self, image):
        """
        Main recognition function
        Args:
            image: numpy array (RGB format from Gradio)
        Returns:
            annotated_image, results_text
        """
        if not self.loaded:
            return None, "❌ Model not loaded! Click 'Load Model' first."

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = self.app.get(img_bgr)

        if not faces:
            return image, "❌ No faces detected in the image."

        # Create annotated image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        results_text = f"**🔍 Detected {len(faces)} face(s)**\n\n"

        # Check if we're in demo mode (no prototypes available)
        can_classify = self.prototypes is not None and len(self.prototypes) > 0

        if self.demo_mode and not can_classify:
            results_text += "⚠️ **DEMO MODE**: Classification unavailable (no training data).\n"
            results_text += "Showing face detection and embedding info only.\n\n"

        # Process each detected face
        for idx, face in enumerate(faces):
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

            # Get embedding
            embedding = face.normed_embedding.reshape(1, -1)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

            if can_classify:
                # Full classification mode
                # Compute similarities to all class prototypes
                similarities = (embedding @ self.prototypes.T).flatten()

                # Get top-5 predictions
                top_k = min(5, len(self.ids))
                top_indices = np.argsort(similarities)[::-1][:top_k]

                # Get the best match
                best_idx = top_indices[0]
                best_name = self.ids[best_idx]
                best_score = similarities[best_idx]

                # Convert cosine similarity to percentage
                confidence = (best_score + 1) / 2 * 100  # Map [-1,1] to [0,100]

                # Add label to image
                label = f"{best_name}: {confidence:.1f}%"
                ax.text(
                    x1, y1 - 10,
                    label,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lime', alpha=0.8),
                    fontsize=12, color='black', weight='bold'
                )

                # Add to results text
                results_text += f"### Face #{idx + 1}\n"
                results_text += f"**📍 Location:** ({x1}, {y1}) to ({x2}, {y2})\n\n"
                results_text += "**🎯 Top Predictions:**\n\n"

                for rank, pred_idx in enumerate(top_indices, 1):
                    pred_name = self.ids[pred_idx]
                    pred_score = similarities[pred_idx]
                    pred_confidence = (pred_score + 1) / 2 * 100

                    # Create confidence bar
                    bar_length = int(pred_confidence / 5)
                    bar = "█" * bar_length + "░" * (20 - bar_length)

                    results_text += f"{rank}. **{pred_name}**\n"
                    results_text += f"   {bar} {pred_confidence:.2f}%\n"
                    results_text += f"   (cosine similarity: {pred_score:.4f})\n\n"

                results_text += "---\n\n"
            else:
                # Demo mode - just show detection info
                det_score = float(getattr(face, 'det_score', 0.0))

                # Add label to image
                label = f"Face #{idx + 1} ({det_score:.2f})"
                ax.text(
                    x1, y1 - 10,
                    label,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.8),
                    fontsize=12, color='black', weight='bold'
                )

                # Add to results text
                results_text += f"### Face #{idx + 1}\n"
                results_text += f"**📍 Location:** ({x1}, {y1}) to ({x2}, {y2})\n"
                results_text += f"**🎯 Detection Score:** {det_score:.4f}\n"
                results_text += f"**📐 Face Size:** {x2-x1}x{y2-y1} pixels\n"
                results_text += f"**🔢 Embedding:** 512-dimensional vector extracted\n\n"
                results_text += "---\n\n"

        # Convert matplotlib figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        annotated_img = Image.open(buf)
        plt.close(fig)

        return annotated_img, results_text

# ==================== GRADIO INTERFACE ====================

# Initialize system
system = FaceRecognitionSystem()

def load_model_wrapper():
    """Wrapper for model loading button"""
    return system.load_model()

def recognize_wrapper(image):
    """Wrapper for recognition function"""
    if image is None:
        return None, "Please upload an image first."
    return system.recognize_face(image)

# Create Gradio interface
with gr.Blocks(title="Face Recognition System") as demo:
    
    # Header
    gr.Markdown("""
    # 🎭 Celebrity Face Recognition System
    **Agriya Yadav** | CS-4440: Artificial Intelligence | Ashoka University  
    Powered by **Buffalo_L (ArcFace)** | 95.27% Accuracy
    """)
    
    # Model loading section
    gr.Markdown("## 🚀 Step 1: Load Model")
    with gr.Row():
        load_btn = gr.Button("🔄 Load Buffalo_L Model", variant="primary")
        load_status = gr.Textbox(
            label="Status",
            value="Click 'Load Model' to initialize the system...",
            interactive=False
        )
    
    gr.Markdown("---")
    
    # Recognition section
    gr.Markdown("## 📸 Step 2: Upload Image for Recognition")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image",
                type="numpy"
            )
            recognize_btn = gr.Button("🎯 Recognize Faces", variant="primary")
            
            gr.Markdown("""
            **📌 Tips:**
            - Upload any image containing faces
            - System detects faces automatically
            - Works with single or multiple faces
            - Best results with clear, frontal faces
            """)
        
        with gr.Column():
            output_image = gr.Image(
                label="Detected Faces",
                type="pil"
            )
    
    # Results section
    gr.Markdown("## 📊 Recognition Results")
    results_text = gr.Markdown(
        value="Results will appear here after recognition..."
    )
    
    # Info section
    gr.Markdown("""
    ---
    ### 📚 About This System
    
    - **Model**: Buffalo_L (ResNet-50 backbone with ArcFace loss)
    - **Dataset**: 40,709 images across 247 Indian celebrities  
    - **Accuracy**: 95.27% on test set
    - **Features**: Real-time face detection, Multi-face recognition, Confidence scores, Top-5 predictions
    
    ### 🔬 How It Works:
    1. **Face Detection**: InsightFace detects faces and computes bounding boxes
    2. **Embedding**: Each face → 512-dimensional vector
    3. **Classification**: Cosine similarity to class prototypes
    4. **Ranking**: Top-5 most similar identities with confidence scores
    
    ### ⚠️ Limitations:
    - Works best with clear, frontal faces
    - Performance degrades with occlusions (masks, sunglasses)
    - Trained on Indian celebrities (Bollywood & South Indian cinema)
    """)
    
    # Connect buttons to functions
    load_btn.click(
        fn=load_model_wrapper,
        inputs=[],
        outputs=[load_status]
    )
    
    recognize_btn.click(
        fn=recognize_wrapper,
        inputs=[input_image],
        outputs=[output_image, results_text]
    )

# ==================== LAUNCH ====================

if __name__ == "__main__":
    print("="*60)
    print("🎭 FACE RECOGNITION SYSTEM - GUI")
    print("="*60)
    print("\nStarting Gradio interface...")
    print("\n📌 Instructions:")
    print("1. Click 'Load Buffalo_L Model' to initialize")
    print("2. Upload an image containing faces")
    print("3. Click 'Recognize Faces' to see results")
    print("="*60)
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )