from torchvision import datasets

val_dataset = datasets.ImageFolder(
    "../../../datasets/image/val"
)

fake = 0
real = 0

for _, label in val_dataset.samples:
    if label == 0:
        fake += 1
    else:
        real += 1

print("Fake:", fake)
print("Real:", real)
print("Total:", fake + real)