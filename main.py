from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Pneumonia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading model...")
model = tf.keras.models.load_model("model/pneumonia_cnn_best.h5")
print("✅ Model loaded!")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running! ✅"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        label = "PNEUMONIA"
        prob  = confidence * 100
    else:
        label = "NORMAL"
        prob  = (1 - confidence) * 100
    
    return {
        "prediction": label,
        "confidence": round(prob, 2),
        "status": "success"
    }