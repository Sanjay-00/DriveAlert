# Import necessary libraries
from scipy.spatial import distance as dist   # calculating distance specifically for calculating the eye aspect ratio.
from imutils.video import VideoStream       # for capturing video streams for webcam
from imutils import face_utils              #
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3
from playsound import playsound
import smtplib
import ssl
import pygame
# Function to handle text-to-speech alarm
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust the speaking rate
    engine.setProperty('volume', 1)  # Set volume level between 0 and 1

    # Continuously speak the message while the alarm is active
    while alarm_status:
        engine.say(msg)
        engine.runAndWait()

    # Speak the message once if the second alarm status is active
    if alarm_status2:
        saying = True
        engine.say(msg)
        engine.runAndWait()
        saying = False



# def sound_alarm():
#     """The function plays an alarm sound"""
#     # playsound("Data/alarm.wav")
#     # Initialize pygame mixer
#     pygame.mixer.init()
#     # Load and play the alarm sound
#     pygame.mixer.music.load("Data/alarm2.mp3")  # You can also use an absolute path here
#     pygame.mixer.music.play()


def sound_alarm():
    """The function plays an alarm sound twice nonstop"""
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load and play the alarm sound
    pygame.mixer.music.load("Data/alarm2.mp3")  # Ensure the path and file extension are correct

    # Play the sound once
    pygame.mixer.music.play()

    # Wait for the first play to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Play the sound again immediately after the first one finishes
    pygame.mixer.music.play()

    # Wait for the second play to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate distances between eye landmarks
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance

    # Compute the EAR
    ear = (A + B) / (2.0 * C)

    return ear

# Function to calculate the average EAR for both eyes
def final_ear(shape):
    # Get the indices for the left and right eye landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Extract eye landmarks from the shape
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # Calculate EAR for both eyes
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # Calculate the average EAR
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Function to calculate the distance between the top and bottom lips
def lip_distance(shape):
    # Get landmarks for the top lip
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    # Get landmarks for the bottom lip
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    # Calculate the mean positions for both lips
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    # Compute the vertical distance between the lips
    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Argument parsing for webcam index
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# Thresholds and global variables for detection
EYE_AR_THRESH = 0.3                # Threshold for Eye Aspect Ratio
EYE_AR_CONSEC_FRAMES = 50          # Number of frames to trigger alert
YAWN_AR_CONSEC_FRAMES = 50
YAWN_THRESH = 20                  # Threshold for lip distance to detect yawning
NO_FACE_FRAMES_THRESH = 50         # Frames to wait before declaring no driver present
alarm_status = False                # Alarm status for drowsiness
alarm_status2 = False               # Alarm status for yawning
saying = False                      # Flag to indicate if the alarm is speaking
COUNTER = 0  
COUNTER_YAWN = 0                       # Counter for drowsiness frames
no_face_counter = 0                # Counter for no face detected

# Load the face detector and landmark predictor
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("Data/ren/haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('Data/ren/shape_predictor_68_face_landmarks.dat')

# Start the video stream from the webcam
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()

time.sleep(1.0)  # Allow the camera to warm up

# Main loop for processing video frames
while True:
    # Read the current frame from the video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=450)  # Resize frame for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces in the frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Check if no faces are detected
    if len(rects) == 0:
        no_face_counter += 1  # Increment no face counter
        # Check if the counter exceeds the threshold
        if no_face_counter >= NO_FACE_FRAMES_THRESH:
            if not alarm_status:
                alarm_status = True  # Activate alarm if no face detected
                t = Thread(target=alarm, args=('Driver missing!',))
                t.daemon = True  # Run thread as a daemon
                t.start()  # Start the alarm thread
    else:
        no_face_counter = 0  # Reset counter if a face is detected

    # Process each detected face
    for (x, y, w, h) in rects:
        # Create a rectangle around the detected face
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # Predict facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # Convert landmarks to NumPy array

        # Calculate EAR and lip distance
        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw contours for eyes and lips on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # Draw left eye contour
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # Draw right eye contour

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)  # Draw lip contour

        # Check if EAR is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1  # Increment drowsiness counter

            # Trigger drowsiness alert if threshold is met
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=sound_alarm )
                    t.daemon = True
                    t.start()  # Start the drowsiness alarm thread

                # Display drowsiness alert on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0  # Reset drowsiness counter if eyes are open
            alarm_status = False  # Reset alarm status

        # Check if lip distance exceeds yawn threshold
        if distance > YAWN_THRESH:
            COUNTER_YAWN += 1  # Increment drowsiness counter
            if COUNTER_YAWN>= YAWN_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Display yawn alert

                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    t = Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()  # Start the yawn alarm thread
        else:
            alarm_status2 = False  # Reset yawn alarm status
            COUNTER_YAWN = 0 
        # Display EAR and lip distance on the frame
        # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Cleanup and release resources
cv2.destroyAllWindows()
vs.stop()  # Stop the video stream
