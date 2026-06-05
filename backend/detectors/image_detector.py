import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image

# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =====================================
# Model Path
# =====================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "image",
    "best_model_v2.pth"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "image",
    "best_model_v2.metadata.json"
)

# =====================================
# Model Architecture
# MUST MATCH TRAINING
# =====================================

model = models.efficientnet_b0(
    weights=None
)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)

# =====================================
# Load Weights
# =====================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================
# Transform
# SAME AS TRAINING
# =====================================

def crop_largest_face(image):
    image_array = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_BGR2GRAY
    )

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return image, None

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    padding = int(0.25 * max(w, h))
    left = max(x - padding, 0)
    top = max(y - padding, 0)
    right = min(x + w + padding, image.width)
    bottom = min(y + h + padding, image.height)

    return image.crop((left, top, right, bottom)), {
        "x": int(left),
        "y": int(top),
        "width": int(right - left),
        "height": int(bottom - top)
    }


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================
# Class Mapping
# =====================================

DEFAULT_CLASSES = {
    0: "FAKE",
    1: "REAL"
}


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}

    with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


metadata = load_metadata()


def load_classes():
    if not metadata:
        return DEFAULT_CLASSES

    idx_to_class = metadata.get("idx_to_class", {})

    if not idx_to_class:
        return DEFAULT_CLASSES

    return {
        int(index): label.upper()
        for index, label in idx_to_class.items()
    }


classes = load_classes()
use_face_crop = (
    metadata.get("preprocessing") ==
    "largest_face_crop_with_full_image_fallback"
)


def get_risk_level(fake_probability):
    if fake_probability >= 80:
        return "HIGH"

    if fake_probability >= 55:
        return "MEDIUM"

    return "LOW"

# =====================================
# Prediction Function
# =====================================

def predict_image(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        face_box = None

        if use_face_crop:
            image, face_box = crop_largest_face(image)

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(device)

        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            print("\nRaw Output:")
            print(outputs)

            print("\nProbabilities:")
            print(probabilities)

            confidence, prediction = torch.max(
                probabilities,
                1
            )

            fake_probability = (
                probabilities[0][0].item()
                * 100
            )

            real_probability = (
                probabilities[0][1].item()
                * 100
            )

            predicted_label = classes[prediction.item()]

            print(
                f"\nPrediction: {predicted_label}"
            )

            print(
                f"Fake: {fake_probability:.2f}% | "
                f"Real: {real_probability:.2f}%"
            )

        return {

            "prediction":
                predicted_label,

            "confidence":
                round(
                    confidence.item() * 100,
                    2
                ),

            "risk_level":
                get_risk_level(fake_probability),

            "face_detected":
                face_box is not None,

            "face_box":
                face_box,

            "fake_probability":
                round(
                    fake_probability,
                    2
                ),

            "real_probability":
                round(
                    real_probability,
                    2
                )
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# =====================================
# Manual Test
# =====================================

if __name__ == "__main__":

    test_image = "sample.jpg"

    result = predict_image(
        test_image
    )

    print(result)
