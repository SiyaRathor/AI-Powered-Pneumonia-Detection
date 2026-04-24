import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model/pneumonia_cnn_best.h5")

def predict_pneumonia(image):
    # Preprocess
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "PNEUMONIA"
        prob  = confidence * 100
    else:
        label = "NORMAL"
        prob  = (1 - confidence) * 100

    if label == "PNEUMONIA":
        return (
            f"⚠️ PNEUMONIA DETECTED\n"
            f"Confidence: {prob:.2f}%\n\n"
            f"Please consult a doctor immediately!"
        )
    else:
        return (
            f"✅ NORMAL\n"
            f"Confidence: {prob:.2f}%\n\n"
            f"No pneumonia detected!"
        )

interface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(
        type="pil",
        label="📤 Upload Chest X-Ray"
    ),
    outputs=gr.Textbox(
        label="🔍 Result",
        lines=4
    ),
    title="🫁 AI-Powered Pneumonia Detection",
    description="""
    ## AI-Powered Chest X-Ray Analysis
    Upload a chest X-ray image to detect Pneumonia.
    - Model: CNN (87.34% accuracy)
    - Dataset: 5,216 chest X-ray images
    """,
)

interface.launch()