from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "../../../datasets/image",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total Images:", len(dataset))