import cv2
import dlib
from imutils import face_utils
import numpy as np
import serial
import time

# Initialize Arduino serial communication
arduino = serial.Serial('COM4', 9600)  # Replace 'COM3' with your Arduino port

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define constants
EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold for drowsiness detection
EAR_CONSEC_FRAMES = 90  # Number of consecutive frames eyes must be closed to trigger alarm

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize drowsy frame counter
drowsy_counter = 0

# Loop over frames from the webcam
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Loop over detected faces
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye coordinates
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        
        # Compute the eye aspect ratio for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Compute the average eye aspect ratio
        ear = (left_ear + right_ear) / 2.0
        
        # Check if the eye aspect ratio is below the threshold
        if ear < EAR_THRESHOLD:
            # Increment drowsy frame counter
            drowsy_counter += 1
            # If eyes have been closed for a sufficient number of frames, trigger the alarm
            if drowsy_counter >= EAR_CONSEC_FRAMES:
                # Sound the alarm
                arduino.write(b'1')
                # Display drowsiness warning on frame
                cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw eye borders in red
                cv2.polylines(frame, [left_eye], True, (0, 0, 255), 1)
                cv2.polylines(frame, [right_eye], True, (0, 0, 255), 1)
        else:
            # Reset drowsy frame counter
            drowsy_counter = 0
            arduino.write(b'0')
            cv2.putText(frame, "AWAKE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Draw eye borders in green
            # cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            # cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # Draw a square box around the detected face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
