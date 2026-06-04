import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

# =====================================
# Dataset Paths
# =====================================

TRAIN_DIR = "../../../datasets/image/train"
VAL_DIR = "../../../datasets/image/val"
TEST_DIR = "../../../datasets/image/test"

# =====================================
# Image Transforms
# =====================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
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
            "../../models/image/best_model_v2.pth"
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
        "../../models/image/best_model_v2.pth",
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
    "\nBest Model Saved At:\n"
    "../../models/image/best_model_v2.pth"
)