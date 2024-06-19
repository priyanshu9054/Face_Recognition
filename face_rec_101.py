import face_recognition
import cv2
from PIL import Image
file_path = "dataset//reference_image_2.jpg"
mode = Image.open(file_path).mode
if mode == 'L':
    mode = "GRAY"
print(cv2.imread(file_path).dtype,mode)
image = face_recognition.load_image_file(file_path)
face_recognition.api.batch_face_locations(image, number_of_times_to_upsample=1, batch_size=128)