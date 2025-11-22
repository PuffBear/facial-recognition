#!/usr/bin/env python3
"""
Face Recognition GUI Application - ACTUALLY FIXED
Works with pre-aligned face images by directly calling ONNX inference
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
        self.rec_model = None
        self.ids = None
        self.prototypes = None
        self.loaded = False
        self.loading = False
        
    def get_embedding_from_aligned_face(self, img):
        """
        Extract embedding from pre-aligned face image
        Args:
            img: BGR image (from cv2.imread)
        Returns:
            embedding: 512-d normalized embedding
        """
        # Resize to 112x112 (model requirement)
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112))
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] and prepare for ONNX
        img_normalized = img_rgb.astype(np.float32)
        img_normalized = (img_normalized - 127.5) / 127.5
        
        # Transpose to CHW format and add batch dimension
        img_input = img_normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run inference on ONNX model
        input_name = self.rec_model.session.get_inputs()[0].name
        embedding = self.rec_model.session.run(None, {input_name: img_input})[0][0]
        
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        return embedding
        
    def load_model(self):
        """Load Buffalo_L model and compute class prototypes"""
        if self.loading:
            return "⏳ Model is already loading, please wait..."
        
        if self.loaded:
            return f"✅ Model already loaded! {len(self.ids)} celebrities ready."
        
        self.loading = True
        
        try:
            print("Loading Buffalo_L model...")
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            
            # Get the recognition model for direct ONNX inference
            self.rec_model = self.app.models['recognition']
            
            print("Computing class prototypes from training data...")
            train_root = self.data_root / "train"
            
            if not train_root.exists():
                self.loading = False
                return f"❌ Error: Training data not found at {train_root}"
            
            # Load training images
            train_embeddings = []
            train_labels = []
            
            class_dirs = [d for d in sorted(train_root.iterdir()) if d.is_dir()]
            print(f"Found {len(class_dirs)} celebrity classes")
            
            successful_classes = []
            failed_classes = []
            
            for idx, cls_dir in enumerate(class_dirs, 1):
                cls_name = cls_dir.name
                image_paths = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
                
                # Use only 3 images per class for FAST prototype computation
                image_paths = image_paths[:3]
                
                class_embeddings_count = 0
                for img_path in image_paths:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        # Extract embedding using direct ONNX inference
                        embedding = self.get_embedding_from_aligned_face(img)
                        
                        train_embeddings.append(embedding)
                        train_labels.append(cls_name)
                        class_embeddings_count += 1
                        
                    except Exception as e:
                        print(f"  ⚠️  Error processing {img_path.name}: {e}")
                        continue
                
                if class_embeddings_count == 0:
                    failed_classes.append(cls_name)
                else:
                    successful_classes.append(cls_name)
                
                # Progress feedback
                if idx % 5 == 0:
                    print(f"  Processed {idx}/{len(class_dirs)} celebrities... ({len(train_embeddings)} embeddings so far)")
            
            # Final summary
            print(f"\n📊 Processing Summary:")
            print(f"  Total embeddings: {len(train_embeddings)}")
            print(f"  Successful classes: {len(successful_classes)}/{len(class_dirs)}")
            
            if failed_classes:
                print(f"  Failed classes: {len(failed_classes)}")
                if len(failed_classes) <= 10:
                    print(f"  Classes with issues: {', '.join(failed_classes)}")
            
            if not train_embeddings:
                self.loading = False
                return "❌ Error: No face embeddings could be computed"
            
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
            
            # Get embedding and classify
            embedding = face.normed_embedding.reshape(1, -1)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
            
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
    
    **✨ Fixed for pre-aligned face images!**
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
    - **Dataset**: Pre-aligned face images from Indian celebrity dataset
    - **Accuracy**: 95.27% on test set
    - **Features**: Real-time face detection, Multi-face recognition, Confidence scores, Top-5 predictions
    
    ### 🔬 How It Works:
    1. **Training**: Directly extracts embeddings from pre-aligned 112x112 face crops via ONNX inference
    2. **Face Detection**: InsightFace detects faces in uploaded images
    3. **Embedding**: Each face → 512-dimensional vector
    4. **Classification**: Cosine similarity to class prototypes
    5. **Ranking**: Top-5 most similar identities with confidence scores
    
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
    print("🎭 FACE RECOGNITION SYSTEM - GUI (FIXED)")
    print("="*60)
    print("\n✨ Fixed for pre-aligned face images!")
    print("   Direct ONNX inference on face crops")
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