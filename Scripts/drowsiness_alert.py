
# Alert functions: alarm and emergency email


# import packages
from playsound import playsound
import smtplib
import ssl
import pygame
import pyttsx3

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



def sound_alarm2():
    """The function plays a voice message saying 'No Driver'"""
    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech (words per minute)
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

    # Speak the message
    engine.say("Driver missing")
    
    # Wait for the speech to finish
    engine.runAndWait()
    





def send_email(username, contact_name, contact_email):
    sender_email = "drivealert49@gmail.com"
    sender_password = "vkwc otnb zmpj wour"  # Use App Password here

    message = f"Subject: Drowsiness Alert!\n\nHello {contact_name},\n\n{username} is asleep while driving! Please take immediate action."

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, contact_email, message)
    
    print(f"Email sent to {contact_email}!")

# Example usage
send_email("Sanjay", "Emergency Contact", "emergency@example.com")



