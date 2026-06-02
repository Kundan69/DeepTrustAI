from fastapi import FastAPI
from backend.api.image import router as image_router

app = FastAPI(title="DeepTrust AI")

app.include_router(image_router)

@app.get("/")
def home():
    return {"message": "DeepTrust AI Running"}