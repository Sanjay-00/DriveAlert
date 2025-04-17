# Alert functions: alarm and emergency email

# import packages
import RPi.GPIO as GPIO
import smtplib
import ssl
import pyttsx3
import time

# Set up GPIO for the buzzer
BUZZER_PIN = 18  # Change this to the GPIO pin you are using for the buzzer
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def sound_alarm():
    """The function activates the buzzer twice nonstop"""
    # Activate the buzzer (turn on)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(1)  # Buzzer on for 1 second
    
    # Deactivate the buzzer (turn off)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(1)  # Buzzer off for 1 second

    # Play the alarm again immediately
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(1)  # Buzzer on for 1 second
    
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(1)  # Buzzer off for 1 second

# def sound_alarm2():
#     """The function plays a voice message saying 'No Driver'"""
#     # Initialize the pyttsx3 engine
#     engine = pyttsx3.init()

#     # Set properties (optional)
#     engine.setProperty('rate', 150)  # Speed of speech (words per minute)
#     engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

#     # Speak the message
#     engine.say("Driver missing")
    
#     # Wait for the speech to finish
#     engine.runAndWait()

def send_email(username, contact_name, contact_email):
    sender_email = "drivealert49@gmail.com"
    sender_password = "vkwc otnb zmpj wour"  # Use App Password here

    message = f"Subject: Drowsiness Alert!\n\nHello {contact_name},\n\n{username} is asleep while driving! Please take immediate action."

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, contact_email, message)
    
    print(f"Email sent to {contact_email}!")

# Example usage
# send_email("Sanjay", "Emergency Contact", "emergency@example.com")
