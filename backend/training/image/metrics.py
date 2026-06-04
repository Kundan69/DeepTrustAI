import torch
import torch.nn as nn

from torchvision import models
from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================
# Device
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================
# Test Dataset
# ==========================

TEST_DIR = "../../../datasets/image/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# ==========================
# Load Model
# ==========================

model = models.efficientnet_b0()

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    torch.load(
        "../../models/image/best_model_v2.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# ==========================
# Evaluation
# ==========================

all_labels = []
all_preds = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_preds.extend(
            predicted.cpu().numpy()
        )

# ==========================
# Metrics
# ==========================

acc = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds
)

recall = recall_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\n===== RESULTS =====")

print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print("\nConfusion Matrix:")
print(cm)