import cv2
from deepface import DeepFace
import os

# Get the full path of the reference image
script_dir = os.path.dirname(os.path.abspath(__file__))
reference_img_path = os.path.join(script_dir, "reference_image.jpg")

# Load the reference image
reference_img = cv2.imread(reference_img_path)

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Verify the face in the frame against the reference image
    result = DeepFace.verify(reference_img, frame)

    if result["verified"]:
        cv2.putText(frame, "Face Matched", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Face Not Matched", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

# import threading

# import cv2
# from deepface import DeepFace

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# counter = 0

# face_match = False

# reference_img = cv2.imread("Reference_image.jpg")

# def check_face(frame):
#     global face_match
#     try:
#         if DeepFace.verify(frame, reference_img.copy())['verified']:
#             face_match = True
#         else:
#             face_match = False

#     except ValueError:
#         face_match = False
#         pass


# while True:
#     ret, frame = cap.read()

#     if ret:
#         if counter % 30 == 0:
#             try:
#                 threading.Thread(target=check_face, args=(frame.copy(),)).start()
#             except ValueError:
#                 pass

#         counter += 1

#         if face_match:
#             cv2.putText(frame, "MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
#         else:
#             cv2.putText(frame, "NO MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        
#         cv2.imshow("video",frame)

#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cv2.destroyAllWindows()