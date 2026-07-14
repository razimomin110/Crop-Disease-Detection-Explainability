"""
Streamlit Demo Application for Crop Disease Detection.

Upload a leaf image to:
  1. See the predicted disease class and confidence.
  2. View the Grad-CAM heatmap overlay.
  3. Get an estimated disease severity percentage.

Run:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, TARGET_CLASSES
from explainability.gradcam import GradCAM
from explainability.severity import estimate_severity
from models.model import CropDiseaseModel
from torchvision import transforms

# ─────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
}
.severity-mild { color: #f0ad4e; font-weight: bold; }
.severity-moderate { color: #e67e22; font-weight: bold; }
.severity-severe { color: #e74c3c; font-weight: bold; }
.severity-healthy { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, class_names_path: str):
    """Load model and class names (cached across reruns)."""

    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = TARGET_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force EfficientNet (matches trained checkpoint)
    model = CropDiseaseModel(
        num_classes=len(class_names),
        backbone="efficientnet",
        pretrained=False,
    ).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model_loaded = True
    else:
        model_loaded = False

    return model, class_names, device, model_loaded


# ─────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────
_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────
def main():

    st.markdown("<h1 class='main-header'>🌿 Crop Disease Detection</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:gray;'>"
        "Upload a leaf image to detect plant diseases using deep learning with Grad-CAM explainability."
        "</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_path = st.text_input("Model checkpoint", value="models/best_model.pth")
        class_names_path = st.text_input("Class names JSON", value="models/class_names.json")

        st.markdown("---")
        st.markdown("### 📋 Supported Classes")

        for cls in TARGET_CLASSES:
            crop, disease = cls.split("___")
            emoji = "✅" if "healthy" in disease.lower() else "🦠"
            st.markdown(f"{emoji} **{crop}** — {disease.replace('_',' ')}")

    model, class_names, device, model_loaded = load_model(
        model_path,
        class_names_path,
    )

    if not model_loaded:
        st.warning(
            f"⚠️ Model checkpoint not found at `{model_path}`."
            " Train the model first or adjust the path."
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📷 Upload a leaf image",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None and model_loaded:

        pil_image = Image.open(uploaded_file).convert("RGB")
        original_np = np.array(pil_image)

        input_tensor = _inference_transform(pil_image).unsqueeze(0).to(device)

        # GradCAM
        target_layer = model.get_target_layer()
        grad_cam = GradCAM(model, target_layer)

        heatmap = grad_cam.generate_heatmap(input_tensor)
        overlay = GradCAM.overlay_heatmap(original_np, heatmap)

        # Prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)

        predicted_class = class_names[pred_idx.item()]
        confidence_val = confidence.item() * 100

        # Severity
        if "healthy" in predicted_class.lower():
            severity = {"severity_pct": 0.0, "severity_label": "Healthy"}
        else:
            severity = estimate_severity(heatmap)

        grad_cam.remove_hooks()

        # Display images
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Original Image")
            st.image(original_np, width="stretch")

        with col2:
            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(overlay, width="stretch")

        # Results
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        r1, r2, r3 = st.columns(3)

        with r1:
            crop, disease = predicted_class.split("___")
            st.metric("🌱 Disease", f"{crop} — {disease.replace('_',' ')}")

        with r2:
            st.metric("🎯 Confidence", f"{confidence_val:.1f}%")

        with r3:
            sev_color = {
                "Healthy": "severity-healthy",
                "Mild": "severity-mild",
                "Moderate": "severity-moderate",
                "Severe": "severity-severe",
            }.get(severity["severity_label"], "")

            st.metric("🩺 Severity", f"{severity['severity_pct']:.1f}%")

            st.markdown(
                f"<span class='{sev_color}'>{severity['severity_label']}</span>",
                unsafe_allow_html=True,
            )

        # Top-5 predictions
        with st.expander("🔬 Top-5 Predictions"):

            top5_probs, top5_idx = probs.topk(min(5, len(class_names)), dim=1)

            for prob, idx in zip(top5_probs[0], top5_idx[0]):
                name = class_names[idx.item()]
                st.write(f"**{name}** — {prob.item()*100:.2f}%")

    elif uploaded_file is not None and not model_loaded:
        st.error("Cannot run inference: model not loaded.")


if __name__ == "__main__":
    main()