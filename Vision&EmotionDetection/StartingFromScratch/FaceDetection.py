import cv2
import dlib
import datetime
import os
import face_recognition
import logging

# Set up logging
logging.basicConfig(filename='data/face_detection_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Function to detect and recognize faces in real-time using the camera
def detect_and_recognize():
    try:
        # Open a connection to the camera (0 represents the default camera)
        cap = cv2.VideoCapture(0)

        # Load images and corresponding names for face recognition
        known_face_encodings = []
        known_face_names = []
        face_detected = {}  # Dictionary to track the state of each known face
        unknown_faces_screenshot_taken = set()  # Set to track unknown faces for which a screenshot has been taken
        unknown_faces_video_recording = {}  # Dictionary to track video recording for each unique unknown face

        # Load known faces from the 'data/faces' directory
        for filename in os.listdir('data/faces'):
            if filename.endswith(".jpg"):
                name = os.path.splitext(filename)[0]  # Extract the name from the file name
                known_face_names.append(name)
                face_detected[name] = False  # Initialize as not detected

                image = face_recognition.load_image_file(f"data/faces/{filename}")
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame using dlib
            faces = detector(gray)

            # Loop through each detected face
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                # Extract the face region from the grayscale image
                face_roi = gray[y:y+h, x:x+w]

                # Convert the face region to RGB for face recognition
                rgb_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

                # Find face locations and face encodings using face_recognition
                face_locations = face_recognition.face_locations(rgb_face)
                face_encodings = face_recognition.face_encodings(rgb_face, face_locations)

                # Check if any face encodings are found
                if face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])

                    name = "Unknown"

                    # If a match is found, use the name of the first matching known face
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                        # If the face was not detected in the previous frame, create a log entry
                        if not face_detected.get(name, False):
                            log_info = f"Face detected - Name: {name}, Timestamp: {datetime.datetime.now()}"
                            logging.info(log_info)
                            face_detected[name] = True  # Update state to detected
                            unknown_faces_screenshot_taken.discard(name)  # Reset screenshot status for this face

                            # Take a screenshot for this unique unknown face
                            if name not in unknown_faces_screenshot_taken:
                                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                                screenshot_name = f"data/faces/unknown_face_{timestamp}.jpg"
                                cv2.imwrite(screenshot_name, frame)
                                print(f"Unknown face detected! Screenshot saved as {screenshot_name}")
                                unknown_faces_screenshot_taken.add(name)

                            # Start video recording for this unique unknown face
                            if name not in unknown_faces_video_recording:
                                video_directory = f"data/unknown_face_video/{name}_{timestamp}/"
                                os.makedirs(video_directory, exist_ok=True)
                                video_filename = f"{video_directory}unknown_face_{timestamp}.mp4"
                                video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
                                unknown_faces_video_recording[name] = video_writer
                                print(f"Video recording started for {name}. Saving to {video_filename}")

                    # If the face was detected in the previous frame, create a log entry
                    elif face_detected.get(name, False):
                        log_info = f"Face lost - Name: {name}, Timestamp: {datetime.datetime.now()}"
                        logging.info(log_info)
                        face_detected[name] = False  # Update state to not detected

                        # Stop video recording for this unique unknown face
                        if name in unknown_faces_video_recording:
                            unknown_faces_video_recording[name].release()
                            del unknown_faces_video_recording[name]
                            print(f"Video recording stopped for {name}")

                    # Draw rectangles around the detected faces
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # Put the name above the detected face
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                else:
                    print("No face encodings found for the detected face.")

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
detect_and_recognize()
