# 🫁 Pneumonia Detection AI

An AI-powered web application that detects Pneumonia from Chest X-Ray images using Deep Learning.

## 🎯 Project Overview
- **Task:** Binary Classification — Normal vs Pneumonia
- **Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle
- **Training Images:** 5,216 chest X-ray images
- **Best Model:** CNN with 87.34% accuracy

## 🏗️ Project Architecture

User → Gradio UI → FastAPI Backend → CNN Model → Prediction

## 📊 Model Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| ANN   | 38.94%   | ❌ Poor — not suitable for images |
| CNN   | 87.34%   | ✅ Best — designed for image data |

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | TensorFlow, Keras |
| Backend API | FastAPI |
| Frontend UI | Gradio |
| Deployment | HuggingFace Spaces |
| Language | Python 3.12 |

## 📁 Project Structure

PNEUMONIA_AI/
├── model/
│   └── pneumonia_cnn_best.h5   # Trained CNN model
├── main.py                      # FastAPI backend
├── app.py                       # Gradio UI
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation

## 🧠 Model Details

### Why CNN over ANN?
- ANN flattens the image — loses all spatial information
- CNN uses filters to detect edges, textures, and patterns
- CNN improved accuracy from 38.94% to 87.34% — 48% improvement!

### Training Details
- **Architecture:** 3 Conv2D blocks + Dense layers
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 10
- **Image Size:** 224x224
- **Augmentation:** Rotation, Zoom, Horizontal Flip

### Dataset Details
- **Total Images:** 5,216
- **Normal:** 1,341 images
- **Pneumonia:** 3,875 images
- **Class Imbalance:** Handled using class weights (2.9x)

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pneumonia-detection-ai
cd pneumonia-detection-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Backend
```bash
python -m uvicorn main:app --reload
```

### 4. Run Gradio UI
```bash
python app.py
```

### 5. Open in browser

Gradio UI  → http://127.0.0.1:7860
API Docs   → http://127.0.0.1:8000/docs

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.34% |
| Test Confidence | 97%+ |
| Training Images | 5,216 |
| Test Images | 624 |

## 🔗 Links
- 🤗 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/yourusername/pneumonia-ai)
- 📊 **Dataset:** [Kaggle - Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 💻 **GitHub:** [Repository](https://github.com/yourusername/pneumonia-detection-ai)

## ⚠️ Disclaimer
> This AI model is for **educational purposes only.**
> Always consult a qualified doctor for medical diagnosis.
> Do not use this tool as a substitute for professional medical advice.