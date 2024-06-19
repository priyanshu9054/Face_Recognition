import face_recognition
import cv2
import os
import numpy as np

# Function to check if an image is grayscale or RGB
def is_grayscale(image):
    return image.ndim == 2

# Load the reference images
known_face_encodings = []
known_face_names = []

# Load the reference images from the dataset directory
dataset_dir = "dataset"
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Convert to RGB if image is in BGR format
        if image.shape[2] == 3 and image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check if the loaded image is in grayscale or RGB format
        if not is_grayscale(image) and image.shape[2] != 3:
            print(f"Image {image_path} is not in grayscale or RGB format.")
            continue
        
        # Use face_recognition to get face encodings
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(person_name)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    # Capture a frame from the video stream
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        continue

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame
    try:
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:  # Proceed only if faces are detected
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each face in the frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the face encoding with the known face encodings
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match is found, get the name of the matched person
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Draw a box around the face and display the name
                cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(rgb_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(rgb_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    except Exception as e:
        print("Error processing frame:", e)

    # Display the resulting frame
    cv2.imshow('Face Recognition', rgb_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
