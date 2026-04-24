import gradio as gr
import requests
from PIL import Image
import tempfile
import os

def predict_pneumonia(image):
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": f}
        )

    os.unlink(tmp_path)

    result = response.json()
    prediction = result["prediction"]
    confidence = result["confidence"]

    if prediction == "PNEUMONIA":
        return (
            f"⚠️ PNEUMONIA DETECTED\n"
            f"Confidence: {confidence}%\n\n"
            f"Please consult a doctor immediately!"
        )
    else:
        return (
            f"✅ NORMAL\n"
            f"Confidence: {confidence}%\n\n"
            f"No pneumonia detected!"
        )

interface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(
        type="pil",
        label="📤 Upload Chest X-Ray "
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
    theme=gr.themes.Soft(),
    examples=[],
)

interface.launch(share=True)  