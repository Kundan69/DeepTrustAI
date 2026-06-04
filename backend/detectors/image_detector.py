import os
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

print(MODEL_PATH)

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

classes = {
    0: "FAKE",
    1: "REAL"
}

# =====================================
# Prediction Function
# =====================================

def predict_image(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

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

            print(
                f"\nPrediction: {classes[prediction.item()]}"
            )

            print(
                f"Fake: {fake_probability:.2f}% | "
                f"Real: {real_probability:.2f}%"
            )

        return {

            "prediction":
                classes[prediction.item()],

            "confidence":
                round(
                    confidence.item() * 100,
                    2
                ),

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