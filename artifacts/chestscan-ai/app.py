import os
import sys
import json
import base64
import io
import logging
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_setup

app = FastAPI(title="ChestScan AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
CLASS_NAMES = None
IMG_SIZE = 224

def load_resources():
    global MODEL, CLASS_NAMES, IMG_SIZE
    try:
        import tensorflow as tf
        logger.info("Loading Keras model...")
        MODEL = tf.keras.models.load_model(model_setup.paths["cnn_model_lung_detection.keras"])
        input_shape = MODEL.input_shape
        logger.info(f"Model input shape: {input_shape}")
        if input_shape[1] is not None:
            IMG_SIZE = input_shape[1]
        else:
            IMG_SIZE = 224
        logger.info(f"Using image size: {IMG_SIZE}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        MODEL = None

    try:
        with open(model_setup.paths["class_names.json"]) as f:
            CLASS_NAMES = json.load(f)
        logger.info(f"Class names: {CLASS_NAMES}")
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        CLASS_NAMES = ["healthy", "pneumonia", "tuberculosis", "covid"]

load_resources()

def preprocess_image(image_bytes: bytes):
    from PIL import Image
    import tensorflow as tf
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>ChestScan AI</h1><p>Loading...</p>")

@app.get("/app/status")
async def status():
    return {
        "model_loaded": MODEL is not None,
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
    }

@app.post("/app/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet, please wait.")
    try:
        image_bytes = await file.read()
        arr = preprocess_image(image_bytes)
        preds = MODEL.predict(arr, verbose=0)[0]
        results = []
        for i, name in enumerate(CLASS_NAMES):
            results.append({
                "label": name,
                "confidence": float(preds[i]),
            })
        results.sort(key=lambda x: x["confidence"], reverse=True)
        top = results[0]
        return {
            "prediction": top["label"],
            "confidence": top["confidence"],
            "all_classes": results,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/app/samples")
async def list_samples():
    sample_dir = Path(model_setup.paths.get("sample_images", "")) if "sample_images" in model_setup.paths else None
    if sample_dir is None or not sample_dir.exists():
        sample_dir = Path(__file__).parent / "sample_images"
    samples = []
    if sample_dir.exists():
        for f in sorted(sample_dir.glob("*.png")):
            label = f.stem.split("_")[0]
            samples.append({"filename": f.name, "label": label})
    return {"samples": samples}

@app.get("/samples/{filename}")
async def get_sample(filename: str):
    sample_dir = Path(__file__).parent / "sample_images"
    file_path = sample_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return FileResponse(str(file_path), media_type="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
