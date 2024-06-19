import face_recognition
import cv2
import os

# Load the reference images
known_face_encodings = []
known_face_names = []

# Load the reference images from a directory
script_dir = os.path.dirname(os.path.abspath(__file__))

for image_path in [os.path.join(script_dir, "reference_image.jpg")]:
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(image_path.split('/')[-1].split('.')[0])  # Extract the name from the file path

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    # Capture a frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame from BGR color to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, get the name of the matched person
        if True in matches:
            matched_indices = [i for (i, b) in enumerate(matches) if b]
            names = [known_face_names[i] for i in matched_indices]
            name = ', '.join(names)

        # Draw a box around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()