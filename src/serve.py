import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.model import get_model

CHECKPOINT_PATH = Path(
    os.getenv("MODEL_CHECKPOINT", "checkpoints/classifier_v1.pt")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

model = get_model("resnet18", 10)
model_loaded = False
model_error = None

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    model_loaded = True
except Exception as error:
    model_error = str(error)

app = FastAPI(title="CIFAR-10 Model Serving API")

@app.get("/health")
def health():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_error}",
        )

    return {
        "status": "ok",
        "model": "classifier_v1",
        "device": str(DEVICE),
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {error}",
        ) from error

    inputs = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
        class_id = int(probabilities.argmax().item())

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(float(probabilities[class_id]), 4),
        "model": "classifier_v1",
    }

