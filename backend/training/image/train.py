import json
import random
import argparse
import os
from io import BytesIO
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader
from PIL import Image

# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

parser = argparse.ArgumentParser(
    description="Train the image deepfake detector."
)
parser.add_argument(
    "--data-dir",
    default=os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "datasets",
            "image"
        )
    ),
    help="Dataset folder containing train, val, and test subfolders."
)

args = parser.parse_args()

# =====================================
# Dataset Paths
# =====================================

DATA_DIR = os.path.abspath(args.data_dir)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

for dataset_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Missing dataset folder: {dataset_path}\n"
            "Expected structure: data-dir/train, data-dir/val, data-dir/test "
            "with fake and real folders inside each."
        )

# =====================================
# Image Transforms
# =====================================


class RandomJpegCompression:
    def __init__(self, quality_range=(35, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        quality = random.randint(*self.quality_range)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        return Image.open(buffer).convert("RGB")


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


train_transform = transforms.Compose([
    FaceCrop(),
    transforms.RandomResizedCrop(
        224,
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.2,
        hue=0.02
    ),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3)],
        p=0.2
    ),
    RandomJpegCompression(
        quality_range=(35, 95),
        p=0.45
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    FaceCrop(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================
# Load Datasets
# =====================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print("\nClass Mapping:")
print(train_dataset.class_to_idx)

idx_to_class = {
    index: class_name
    for class_name, index in train_dataset.class_to_idx.items()
}

print(f"\nTrain Images : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Test Images : {len(test_dataset)}")

# =====================================
# Data Loaders
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# =====================================
# Model
# =====================================

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)

model = model.to(device)
torch.backends.cudnn.benchmark = True

# =====================================
# Loss Function
# =====================================

criterion = nn.CrossEntropyLoss()

# =====================================
# Optimizer
# =====================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# =====================================
# Scheduler
# =====================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# =====================================
# Training Settings
# =====================================

epochs = 30

best_acc = 0.0

early_stop_patience = 5
counter = 0

MODEL_SAVE_PATH = "../../models/image/best_model_v2.pth"
METADATA_SAVE_PATH = "../../models/image/best_model_v2.metadata.json"

# =====================================
# Training Loop
# =====================================

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()


        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    # =====================================
    # Validation
    # =====================================
    train_acc = 100 * train_correct / train_total
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100 * correct / total

    scheduler.step(val_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Loss: {running_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # =====================================
    # Save Best Model
    # =====================================

    if val_acc > best_acc:

        best_acc = val_acc
        counter = 0

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "class_to_idx": train_dataset.class_to_idx,
                    "idx_to_class": idx_to_class,
                    "architecture": "efficientnet_b0",
                    "image_size": 224,
                    "preprocessing": "largest_face_crop_with_full_image_fallback",
                    "best_validation_accuracy": round(best_acc, 4)
                },
                metadata_file,
                indent=2
            )

        print(
            f"Best Model Saved ({best_acc:.2f}%)"
        )

    else:

        counter += 1

        if counter >= early_stop_patience:

            print(
                f"\nEarly Stopping at Epoch {epoch+1}"
            )

            break

# =====================================
# Load Best Model
# =====================================

print(
    f"\nLoading Best Model ({best_acc:.2f}%)"
)

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        map_location=device
    )
)

# =====================================
# Test Evaluation
# =====================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

test_acc = 100 * correct / total

print(
    f"\nFinal Test Accuracy: {test_acc:.2f}%"
)

print(
    "\nBest Model Saved At:\n",
    MODEL_SAVE_PATH
)
