import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import face_recognition
import os


def create_and_compile_emotion_model():
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten layer and fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # Assuming 7 emotion classes

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define emotion labels
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Directory to save screenshots
faces_dir = "Faces"

# Create the directory if it doesn't exist
os.makedirs(faces_dir, exist_ok=True)

# Open a connection to the camera (0 for default camera)
cap = cv2.VideoCapture(0)

# Check if the model file exists, and load the model accordingly
model_filename = "emotion_model.keras"
if os.path.exists(model_filename):
    emotion_model = load_model(model_filename)
else:
    # If the model file doesn't exist, create and compile a new model
    emotion_model = create_and_compile_emotion_model()  # You need to implement this function

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
                for filename in os.listdir(faces_dir):
                    screenshot_path = os.path.join(faces_dir, filename)

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
                    matches.append((match, filename))

                if any(matches):
                    matched_name = [name for match, name in matches if match][0]
                    cv2.putText(
                        frame,
                        f"{os.path.splitext(matched_name.split('_')[0])[0]} - " + emotion_label,
                        # Extracting name from filename
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
                    faces_path = os.path.join(
                        faces_dir, f"{name}.jpg"
                    )
                    cv2.imwrite(faces_path, frame)
                    print(f"Screenshot saved as {faces_path}")

        # Display the resulting frame
        cv2.imshow("Facial Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Save the entire model (architecture, optimizer, and weights)
save_model(emotion_model, model_filename) + '.keras'

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
