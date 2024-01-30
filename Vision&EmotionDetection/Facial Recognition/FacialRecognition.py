import cv2
import numpy as np
import dlib
import face_recognition
import os
from datetime import datetime

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Directory to save screenshots
faces_dir = "Faces"

# Create the directory if it doesn't exist
os.makedirs(faces_dir, exist_ok=True)

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

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Check if the detected face matches any face in the screenshots directory
                matches = []
                for filename in os.listdir(faces_dir):
                    faces_path = os.path.join(faces_dir, filename)

                    # Check if a face is detected before attempting to extract encoding
                    try:
                        face_encodings = face_recognition.face_encodings(
                            face_recognition.load_image_file(faces_path)
                        )
                        if not face_encodings:
                            print(f"No face found in {faces_path}. Skipping.")
                            continue  # Skip to the next image if no face is found

                        # Only take the first face encoding if there are multiple faces
                        known_face_encoding = face_encodings[0]

                    except IndexError:
                        print(f"IndexError: No face encoding found in {faces_path}. Skipping.")
                        continue  # Skip to the next image if no face encoding is found

                    except Exception as e:
                        print(f"Error processing {faces_path}: {e}")
                        continue  # Skip to the next image in case of other exceptions

                    current_face_encoding = face_recognition.face_encodings(
                        frame, [(y, x + w, y + h, x)]
                    )[0]
                    match = face_recognition.compare_faces(
                        [known_face_encoding], current_face_encoding
                    )[0]
                    matches.append((match, filename))

                if any(matches):
                    matched_name = [name for match, name in matches if match][0]
                    cv2.putText(
                        frame,
                        f"{os.path.splitext(matched_name.split('_')[0])[0]}",
                        # Extracting name from filename
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                else:
                    # Display 'unknown' with date and time stamp
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    unknown_label = f"Unknown_{current_time}"
                    cv2.putText(
                        frame,
                        f"{unknown_label}",
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    # Save the image with a filename including date and time stamp
                    faces_path = os.path.join(
                        faces_dir, f"{unknown_label}.jpg"
                    )
                    cv2.imwrite(faces_path, frame)
                    print(f"Screenshot saved as {faces_path}")

        # Display the resulting frame
        cv2.imshow("Facial Recognition", frame)
        # Use 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
