import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "models", "image", "best_model_v2.pth")
)
METADATA_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "models", "image", "best_model_v2.metadata.json")
)


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None

    with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def build_model(device):
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


class FaceCrop:
    def __init__(self, padding_ratio=0.25):
        self.padding_ratio = padding_ratio
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def __call__(self, image):
        image = image.convert("RGB")
        image_array = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            return image

        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        padding = int(self.padding_ratio * max(w, h))
        left = max(x - padding, 0)
        top = max(y - padding, 0)
        right = min(x + w + padding, image.width)
        bottom = min(y + h + padding, image.height)

        return image.crop((left, top, right, bottom))


def build_prediction_mapper(metadata, external_class_to_idx):
    if not metadata:
        return lambda prediction_index: prediction_index

    idx_to_class = metadata.get("idx_to_class", {})

    if not idx_to_class:
        return lambda prediction_index: prediction_index

    normalized_external = {
        class_name.lower(): index
        for class_name, index in external_class_to_idx.items()
    }

    def map_prediction(prediction_index):
        training_class = idx_to_class[str(prediction_index)].lower()

        if training_class not in normalized_external:
            raise ValueError(
                f"Class '{training_class}' from training metadata does not exist "
                "in the external dataset folders."
            )

        return normalized_external[training_class]

    return map_prediction


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the image deepfake model on an external dataset."
    )
    parser.add_argument(
        "dataset_dir",
        help="Folder with class subfolders, for example external_test/fake and external_test/real."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        FaceCrop(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        args.dataset_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    metadata = load_metadata()

    print("\nExternal Dataset Mapping:")
    print(dataset.class_to_idx)

    if metadata:
        print("\nTraining Dataset Mapping:")
        print(metadata.get("class_to_idx"))

    model = build_model(device)
    map_prediction = build_prediction_mapper(
        metadata,
        dataset.class_to_idx
    )

    labels = []
    predictions = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            outputs = model(images)
            batch_predictions = torch.argmax(outputs, dim=1)

            labels.extend(batch_labels.numpy())
            predictions.extend(
                map_prediction(prediction)
                for prediction in batch_predictions.cpu().numpy()
            )

    accuracy = accuracy_score(labels, predictions)

    print("\n===== EXTERNAL EVALUATION =====")
    print(f"Images   : {len(dataset)}")
    print(f"Accuracy : {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=dataset.classes,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions))


if __name__ == "__main__":
    main()
