import cv2
import os
from PIL import Image

file_path = "reference_image.jpg"

img = Image.open(file_path)

image = img.convert('L')

image = cv2.imread(file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Reference Image", image)



print(image.dtype)
dataset_dir = os.path.join("dataset")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
image_path = os.path.join(dataset_dir, f"reference_image_2.jpg")
cv2.imwrite(image_path, image)
cv2.waitKey(0)