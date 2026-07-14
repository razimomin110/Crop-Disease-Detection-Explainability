# 🌿 Crop Disease Detection via Smartphone Images

A deep learning pipeline that classifies plant leaf diseases from smartphone images using **EfficientNet-B0** with transfer learning, provides **Grad-CAM explainability**, and estimates **disease severity**.

---

## 📁 Project Structure

```
Crop-Disease-Prediction/
├── data/
│   └── dataset.py          # Dataset loader + augmentation pipelines
├── models/
│   └── model.py             # EfficientNet-B0 / ResNet50 architecture
├── training/
│   └── train.py             # Two-phase training loop with early stopping
├── evaluation/
│   └── evaluate.py          # Metrics, classification report, confusion matrix
├── explainability/
│   ├── gradcam.py           # Grad-CAM heatmap generation
│   └── severity.py          # Disease severity estimation from heatmaps
├── inference/
│   └── predict.py           # Single-image prediction CLI
├── app/
│   └── streamlit_app.py     # Interactive Streamlit demo
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd Crop-Disease-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

This project uses the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) — a subset of **10 classes**:

| Crop   | Classes                                    |
|--------|--------------------------------------------|
| Tomato | Early Blight, Late Blight, Healthy         |
| Potato | Early Blight, Late Blight, Healthy         |
| Corn   | Common Rust, Healthy                       |
| Apple  | Apple Scab, Healthy                        |

### Download & Organize

1. Download the PlantVillage dataset (color images).
2. Place the class folders inside `data/PlantVillage/`:

```
data/PlantVillage/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Corn_(maize)___Common_rust_/
├── Corn_(maize)___healthy/
├── Apple___Apple_scab/
└── Apple___healthy/
```

The loader automatically filters to these 10 classes.

---

## 🏋️ Training

```bash
python -m training.train --data_dir data/PlantVillage --epochs 20 --batch_size 32
```

**Two-phase transfer learning:**
- **Phase 1:** Backbone frozen — trains classifier head only.
- **Phase 2:** Last layers unfrozen — fine-tunes the full network.

Early stopping monitors validation loss (patience = 5). Best checkpoint saved to `models/best_model.pth`.

---

## 📊 Evaluation

```bash
python -m evaluation.evaluate --data_dir data/PlantVillage --model models/best_model.pth
```

Outputs: accuracy, precision, recall, F1-score, and a confusion matrix saved as `evaluation/confusion_matrix.png`.

---

## 🔍 Inference

```bash
python -m inference.predict --image path/to/leaf.jpg --model models/best_model.pth
```

Prints predicted disease, confidence, and estimated severity. Saves Grad-CAM overlay to `inference/gradcam_output.png`.

---

## 🌐 Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Upload a leaf image → view prediction, confidence, Grad-CAM heatmap, and severity estimate.

---

## 🧠 Key Techniques

| Feature                  | Implementation                        |
|--------------------------|---------------------------------------|
| Transfer Learning        | EfficientNet-B0 (timm) / ResNet50     |
| Data Augmentation        | Smartphone-realistic transforms       |
| Explainability           | Grad-CAM heatmaps                     |
| Severity Estimation      | Thresholded heatmap area percentage   |
| Early Stopping           | Patience-based on validation loss     |

---

## 📄 License

This project is for educational and academic purposes.
