import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
import face_recognition
import os

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_model = load_model("fer2013_mini_XCEPTION.107-0.66.hdf5")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Directory to save screenshots
screenshot_dir = "screenshots"

# Create the directory if it doesn't exist
os.makedirs(screenshot_dir, exist_ok=True)

# Open a connection to the camera (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

            # Extract the region of interest (ROI) and preprocess it
            roi_gray = gray[y:y + h, x:x + w]

            # Ensure that roi_gray has a valid size before resizing
            if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                # Resize the ROI to match the expected input size of the emotion recognition model
                roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
                roi_gray = roi_gray.reshape(1, 64, 64, 1)

                # Predict emotion
                emotion_prediction = emotion_model.predict(roi_gray)
                emotion_label = emotion_labels[np.argmax(emotion_prediction)]

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Check if the detected face matches any face in the screenshots directory
                matches = []
                for filename in os.listdir(screenshot_dir):
                    screenshot_path = os.path.join(screenshot_dir, filename)

                    # Check if a face is detected before attempting to extract encoding
                    try:
                        known_face_encoding = face_recognition.face_encodings(
                            face_recognition.load_image_file(screenshot_path)
                        )[0]
                    except IndexError:
                        continue  # Skip to the next image if no face is found

                    current_face_encoding = face_recognition.face_encodings(
                        frame, [(y, x + w, y + h, x)]
                    )[0]
                    match = face_recognition.compare_faces(
                        [known_face_encoding], current_face_encoding
                    )[0]
                    matches.append(match)

                if any(matches):
                    cv2.putText(frame, "Hello, friend - "+emotion_label,
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # Ask for the name
                    name = input("What's your name? ")

                    # Take a screenshot
                    screenshot_path = os.path.join(
                        screenshot_dir, f"{name}_screenshot.jpg"
                    )
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved as {screenshot_path}")

        # Display the resulting frame
        cv2.imshow("Facial Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
