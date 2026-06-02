import torch
import torch.nn as nn
import os

from torchvision import models, transforms
from PIL import Image, ImageFile

# -------------------------------------------------
# Safe image loading (handles slightly corrupted images)
# -------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------------------
# Device (CPU for now)
# -------------------------------------------------
device = torch.device("cpu")

# -------------------------------------------------
# Model Setup (ResNet18)
# -------------------------------------------------
model = models.resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# -------------------------------------------------
# Dynamic model path (NO HARD CODE)
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "image",
    "deepfake_image.pth"
)

# -------------------------------------------------
# Load trained weights
# -------------------------------------------------
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

# -------------------------------------------------
# Image Transform
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# Classes
# -------------------------------------------------
classes = ["fake", "real"]

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_image(image_path: str):

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "error": "Invalid image file",
            "details": str(e)
        }

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }


# -------------------------------------------------
# Manual Test (optional)
# -------------------------------------------------
if __name__ == "__main__":

    test_path = os.path.join(
        BASE_DIR,
        "..",
        "..",
        "datasets",
        "image",
        "fake",
        "easy_80_0001.jpg"
    )

    result = predict_image(test_path)

    print(result)