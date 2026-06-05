from fastapi import APIRouter, UploadFile, File
from fastapi import HTTPException
import shutil
import os
from pathlib import Path
from uuid import uuid4

from backend.detectors.image_detector import predict_image

router = APIRouter()

UPLOAD_DIR = "backend/uploads/images"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Upload JPG, JPEG, PNG, or WEBP."
        )

    safe_filename = f"{uuid4().hex}{extension}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)

    if "error" in result:
        raise HTTPException(
            status_code=422,
            detail=result["error"]
        )

    return {
        "status": "success",
        "filename": safe_filename,
        "original_filename": original_name,
        "result": result
    }
