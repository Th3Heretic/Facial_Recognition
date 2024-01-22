import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
#import face_recognition


# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load pre-trained emotion recognition model
emotion_model = load_model("fer2013_mini_XCEPTION.107-0.66.hdf5")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open a connection to the camera (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

        # Extract the region of interest (ROI) and preprocess it
        roi_gray = gray[y:y + h, x:x + w]
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

        # Display detected emotion
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Facial Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
