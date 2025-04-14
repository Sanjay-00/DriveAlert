from imutils.video import VideoStream
from PIL import ImageTk, Image
from imutils import face_utils
from tkinter import messagebox
from collections import deque
import tkinter as tk
import threading
import datetime
import imutils
import time
import dlib
import cv2
import PIL

# custom scripts
import blink_score
import yawn_score
import drowsiness_alert

# Constants
FRAMES_PER_SECOND = 3
EYE_ASPECT_RATIO_THRESHOLD = 0.27
YAWN_LIPS_DISTANCE = 12
EMAIL_THRESHOLD = 3
BLINK_COUNT_THRESHOLD = 10
YAWN_COUNT_THRESHOLD = 10
NO_FACE_THRESHOLD = 80


class DrowsinessDetector:

    def __init__(self, vs, username, contact_name, contact_email):
        self.username = username
        self.contact_name = contact_name
        self.contact_email = contact_email
        self.vs = vs
        self.thread = None
        self.stop_event = threading.Event()
        self.panel = None

        self.root = tk.Tk()
        self.root.title("DriveAlert")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.message = tk.Label(self.root, fg="#085768", text=f"Hi {self.username}, drive carefully!", font=('Goudy old style', 20, 'bold'))
        self.message.pack(side="top", expand="yes", padx=10, pady=10)

        tk.Button(self.root, text="Stop Driving", command=self.on_close, bg="#ABCAD5", font=("times new roman", 12)).pack(side="bottom", expand="yes", padx=10, pady=10)

    def video_loop(self):
        start_drive_time = last_frame_time = datetime.datetime.now()
        travel_duration = datetime.timedelta(0)
        alarm_on = False
        alarm_counter = 0
        blink_counter = 0
        yawn_counter = 0
        no_face_counter = 0

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("Data/shape_predictor.dat")

        while not self.stop_event.is_set():
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = detector(gray, 0)

            if not faces:
                no_face_counter += 1
                if no_face_counter >= NO_FACE_THRESHOLD:
                    no_face_counter = 0
                    if not alarm_on:
                        threading.Thread(target=drowsiness_alert.sound_alarm2, daemon=True).start()
                        alarm_on = True
                        # alarm_counter += 1
                continue
            else:
                no_face_counter = 0

            shape = predictor(gray, faces[0])
            shape = face_utils.shape_to_np(shape)

            if datetime.datetime.now() - last_frame_time >= datetime.timedelta(seconds=1/FRAMES_PER_SECOND):
                # Blink detection
                if blink_score.compute_average_eye_aspect_ratios(shape) < EYE_ASPECT_RATIO_THRESHOLD:
                    blink_counter += 1
                else:
                    blink_counter = 0

                # Yawn detection
                if yawn_score.compute_lips_distance(shape) > YAWN_LIPS_DISTANCE:
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                current_time = datetime.datetime.now()
                travel_duration = current_time - start_drive_time

                if blink_counter >= BLINK_COUNT_THRESHOLD or yawn_counter >= YAWN_COUNT_THRESHOLD:
                    blink_counter = 0
                    yawn_counter = 0

                    if not alarm_on:
                        threading.Thread(target=drowsiness_alert.sound_alarm, daemon=True).start()
                        alarm_on = True
                        alarm_counter += 1

                        alert_text = " (Get some rest) "
                        if alarm_counter == EMAIL_THRESHOLD:
                            threading.Thread(target=drowsiness_alert.send_email, args=(self.username, self.contact_name, self.contact_email), daemon=True).start()
                            alert_text = " (email was sent)"
                        threading.Thread(target=self.show_alert, args=(alert_text,), daemon=True).start()
                else:
                    alarm_on = False

                last_frame_time = current_time

            # Draw visuals
            cv2.drawContours(rgb, [cv2.convexHull(shape[42:48])], -1, (0, 255, 0), 1)
            cv2.drawContours(rgb, [cv2.convexHull(shape[36:42])], -1, (0, 255, 0), 1)
            cv2.drawContours(rgb, [shape[48:60]], -1, (0, 255, 0), 1)

            cv2.putText(rgb, str(travel_duration)[:-7], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(rgb, f"Blinks: {blink_counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(rgb, f"Yawns: {yawn_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            img = PIL.Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            if self.panel is None:
                self.panel = tk.Label(image=imgtk)
                self.panel.image = imgtk
                self.panel.pack(side="left", padx=10, pady=10)
            else:
                self.panel.configure(image=imgtk)
                self.panel.image = imgtk

    def show_alert(self, additional_text=""):
        old_msg, old_color = self.message['text'], self.message['fg']
        self.message['text'] = "Drowsiness Alert!" + additional_text
        self.message['fg'] = "red"
        time.sleep(4.0)
        self.message['text'], self.message['fg'] = old_msg, old_color

    def on_close(self):
        if self.thread.is_alive():
            self.stop_event.set()
            self.vs.stop()
        self.root.destroy()


def start_driving(username, contact_name, contact_email):
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    dd = DrowsinessDetector(vs, username, contact_name, contact_email)
    dd.root.mainloop()
