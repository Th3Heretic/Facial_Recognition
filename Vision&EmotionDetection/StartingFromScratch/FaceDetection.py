import cv2
import face_recognition
import os
import datetime

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and recognize faces in real-time using the camera
def detect_and_recognize_faces_camera():
    try:
        # Open a connection to the camera (0 represents the default camera)
        cap = cv2.VideoCapture(0)

        # Load images and corresponding names for face recognition
        known_face_encodings = []
        known_face_names = []

        # Load known faces from the 'data/faces' directory
        for filename in os.listdir('data/faces'):
            if filename.endswith(".jpg"):
                name = os.path.splitext(filename)[0]  # Extract the name from the file name
                known_face_names.append(name)

                image = face_recognition.load_image_file(f"data/faces/{filename}")
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame using OpenCV
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Convert the frame to RGB for face recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find face locations and face encodings using face_recognition
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each detected face
            for (x, y, w, h), face_encoding in zip(faces, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"

                # If a match is found, use the name of the first matching known face
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Draw rectangles around the detected faces
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Put the name above the detected face
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Take a screenshot if an unknown face is detected
                if name == "Unknown":
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    screenshot_name = f"unknown_face_{timestamp}.jpg"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"Unknown face detected! Screenshot saved as {screenshot_name}")

            # Display the frame with detected and recognized faces
            cv2.imshow('Detected and Recognized Faces', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

# Call the function for real-time face detection and recognition
detect_and_recognize_faces_camera()
