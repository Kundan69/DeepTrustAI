from PIL import Image
import os

dataset_path = "../../../datasets/image"

count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img.load()

            count += 1

        except Exception as e:
            print("BAD:", path)
            print(e)
            break

print("Valid Images:", count)