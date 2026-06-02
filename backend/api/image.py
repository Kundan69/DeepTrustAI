from fastapi import APIRouter, UploadFile, File
import shutil
import os

from backend.detectors.image_detector import predict_image

router = APIRouter()

UPLOAD_DIR = "backend/uploads/images"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)

    return {
        "status": "success",
        "result": result
    }