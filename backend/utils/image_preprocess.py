import torch

from torchvision import transforms

from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def preprocess_image(image):
    """
    OpenCV image -> tensor
    """

    image = Image.fromarray(
        image[:, :, ::-1]
    )

    image = transform(image)

    image = image.unsqueeze(0)

    return image