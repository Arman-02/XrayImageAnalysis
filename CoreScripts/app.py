import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
import io
import json

from model import BoneFractureSegmentationNet

# Fracture type mapping based on location in image
FRACTURE_TYPES = {
    'elbow positive': 'Elbow',
    'fingers positive': 'Fingers',
    'forearm fracture': 'Forearm',
    'humerus fracture': 'Humerus',
    'humerus': 'Humerus',
    'shoulder fracture': 'Shoulder',
    'wrist positive': 'Wrist'
}

def predict_fracture_type(image, classifier, idx_to_class, device):
    """Predict fracture type using classifier"""
    if classifier is None:
        return "Unknown", 0.0
    
    # Preprocess image
    img_tensor = torch.from_numpy(image).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    fracture_type = idx_to_class.get(str(pred_class), "Unknown")
    return fracture_type, confidence
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 2.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_segmentation_model(checkpoint_path, device):
    """Load trained segmentation model"""
    try:
        model = BoneFractureSegmentationNet(num_classes=1, backbone='resnet50', pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        return None

@st.cache_resource
def load_classifier(checkpoint_path, mapping_path, device):
    """Load trained classifier"""
    try:
        # Import classifier class
        from fracture_classifier_train import FractureClassifier
        
        # Load mapping
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        # Load model
        num_classes = len(mapping['idx_to_class'])
        classifier = FractureClassifier(num_classes=num_classes, frozen_backbone=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.to(device)
        classifier.eval()
        
        return classifier, mapping['idx_to_class']
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None, None

def preprocess_image(image, target_size=640):
    """Preprocess image for model inference"""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded, (x_offset, y_offset), (new_w, new_h)

def generate_bounding_box(mask, image_shape, offset, resize_shape):
    """Generate bounding box from segmentation mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Reverse preprocessing to get original coordinates
    x_offset, y_offset = offset
    new_w, new_h = resize_shape
    
    # Convert back to original image scale
    x_orig = max(0, x - x_offset)
    y_orig = max(0, y - y_offset)
    
    return {
        'x': x_orig,
        'y': y_orig,
        'w': w,
        'h': h,
        'area': cv2.contourArea(largest_contour)
    }

def predict(image, model, device):
    """Run inference on image"""
    # Preprocess
    processed_img, offset, resize_shape = preprocess_image(image)
    img_tensor = torch.from_numpy(processed_img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Threshold
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Reverse preprocessing
    crop_mask = binary_mask[
        offset[1]:offset[1]+resize_shape[1],
        offset[0]:offset[0]+resize_shape[0]
    ]
    
    # Generate bounding box
    bbox = generate_bounding_box(crop_mask, image.shape, offset, resize_shape)
    
    # Calculate confidence
    if bbox:
        confidence = float(np.sum(crop_mask > 0)) / (bbox['w'] * bbox['h'] + 1e-6)
    else:
        confidence = 0.0
    
    return mask, binary_mask, crop_mask, bbox, confidence

def visualize_results(original_image, segmentation_mask, bbox, confidence):
    """Create visualization with segmentation and bounding box"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), alpha=0.7)
    axes[1].imshow(segmentation_mask, cmap='Reds', alpha=0.5)
    axes[1].set_title("Segmentation Mask", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Bounding box
    axes[2].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    if bbox:
        rect = patches.Rectangle(
            (bbox['x'], bbox['y']), bbox['w'], bbox['h'],
            linewidth=3, edgecolor='lime', facecolor='none'
        )
        axes[2].add_patch(rect)
        
        # Add label with confidence
        label = f"Fracture\nConfidence: {confidence:.2%}"
        axes[2].text(bbox['x'], bbox['y']-10, label, 
                    fontsize=10, color='lime', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7))
    
    axes[2].set_title(f"Detection (Confidence: {confidence:.2%})", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

# Sidebar
st.sidebar.markdown("## ⚙️ Configuration")
model_path = st.sidebar.text_input(
    "Model Checkpoint Path",
    value="checkpoints/best_model.pth",
    help="Path to saved model checkpoint"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum confidence to detect fracture"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info("""
**Bone Fracture Detection System**

This application uses a U-Net based CNN model 
to detect and segment bone fractures in X-ray images.

**Features:**
- Semantic segmentation of fracture regions
- Bounding box detection with confidence scores
- Support for multiple image formats
- Real-time inference on MacBook M4 Pro

**Model Architecture:**
- Backbone: ResNet50
- Head: U-Net decoder with skip connections
- Loss: Dice + BCE
""")

# Main content
st.title("🦴 Bone Fracture Detection System")
st.markdown("### Detect and localize bone fractures in medical images")

# Device selection
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
st.info(f"🚀 Using device: **{device}** | MPS Available: **{torch.backends.mps.is_available()}**")

# Load models
if not os.path.exists('checkpoints/best_model.pth'):
    st.error(f"❌ Segmentation model file not found at: `checkpoints/best_model.pth`")
    st.info("Please train the segmentation model first using `python3 main.py --train`")
    st.stop()

if not os.path.exists('checkpoints/best_classifier.pth'):
    st.warning(f"⚠️ Classifier model file not found. Classification will be unavailable.")
    classifier = None
    idx_to_class = None
else:
    classifier, idx_to_class = load_classifier('checkpoints/best_classifier.pth', 'checkpoints/classifier_mapping.json', device)

seg_model = load_segmentation_model('checkpoints/best_model.pth', device)
if seg_model is None:
    st.stop()

st.success("✓ Models loaded successfully!")

# Input section
st.markdown("---")
st.subheader("📁 Upload Image")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a bone X-ray image for fracture detection"
    )

with col2:
    example_image = st.selectbox(
        "Or select an example:",
        ["None", "Upload custom image"],
        help="Use a sample image for testing"
    )

# Process image
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("🔍 Processing Results")
    
    # Load and display
    image = Image.open(uploaded_file)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run inference
    with st.spinner("🔄 Running inference..."):
        mask, binary_mask, crop_mask, bbox, confidence = predict(
            image_np, seg_model, device
        )
        
        # Predict fracture type
        if classifier is not None and idx_to_class is not None:
            fracture_type, type_confidence = predict_fracture_type(image_np, classifier, idx_to_class, device)
        else:
            fracture_type = "Unknown"
            type_confidence = 0.0
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = visualize_results(image_np, crop_mask, bbox, confidence)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("### 📈 Detection Results")
        
        st.metric(
            "Fracture Detected",
            "Yes ✓" if (bbox and confidence > confidence_threshold) else "No ✗",
            delta=f"Confidence: {confidence:.2%}"
        )
        
        if bbox:
            st.markdown("#### Bounding Box Details")
            col_a, col_b = st.columns(2)
            col_a.metric("X Coordinate", f"{bbox['x']:.0f}")
            col_b.metric("Y Coordinate", f"{bbox['y']:.0f}")
            col_a.metric("Width", f"{bbox['w']:.0f}")
            col_b.metric("Height", f"{bbox['h']:.0f}")
            st.metric("Region Area", f"{bbox['area']:.0f} px²")
        
        # Classification
        st.markdown("#### Fracture Type/Location")
        if classifier is not None:
            st.metric("Detected Type", fracture_type, delta=f"Confidence: {type_confidence:.2%}")
            st.info(f"Fracture Type: **{fracture_type}** (Confidence: {type_confidence:.2%})")
        else:
            st.warning("Classifier not available")
    
    # Download results
    st.markdown("---")
    st.subheader("⬇️ Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save visualization
        fig = visualize_results(image_np, crop_mask, bbox, confidence)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="📊 Download Visualization",
            data=buf,
            file_name=f"fracture_detection_{uploaded_file.name}",
            mime="image/png"
        )
        plt.close(fig)
    
    with col2:
        # Save mask
        mask_image = Image.fromarray((crop_mask * 255).astype(np.uint8))
        buf_mask = io.BytesIO()
        mask_image.save(buf_mask, format="PNG")
        buf_mask.seek(0)
        
        st.download_button(
            label="🎭 Download Mask",
            data=buf_mask,
            file_name=f"fracture_mask_{uploaded_file.name}",
            mime="image/png"
        )

else:
    st.info("👆 Upload an X-ray image to get started!")

# Import io for downloads
import io

st.markdown("---")
st.markdown("**Developed for MacBook M4 Pro | GPU-accelerated with MPS**")