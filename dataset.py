import cv2
import os
import numpy as np

def capture_images(name):
    # Create a directory for the person if it doesn't exist
    dataset_dir = os.path.join("dataset\\Priyanshu", name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)  # 0 for the default webcam
    image_count = 0
    while image_count < 100:  # Capture 20 images
        ret, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame, dtype=np.uint8)

        if not ret:
            continue

        # Display the frame
        cv2.imshow('Capture Images - Press "q" to quit', frame)

        # Save the frame as an image file
        image_path = os.path.join(dataset_dir, f"{name}_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        image_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    person_name = input("Enter the name of the person: ")
    capture_images(person_name)
