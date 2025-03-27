
# # GUI first page
# # Driver and emergency contact details before starting driving


# # import packages
# from tkinter import *
# from tkinter import messagebox
# from PIL import ImageTk
# from PIL import Image
# import re

# # import scripts
# import drowsiness_classification

# # regular expression for validating an email
# REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'


# def is_valid_email(email):
#     """This function checks if the email address is valid"""
#     return re.fullmatch(REGEX, email)


# class InfoPage:

#     def __init__(self, root):
#         """This function initializes the info-page"""

#         # root
#         self.root = root
#         self.root.title("DriveAlert")
#         self.root.geometry("800x420+100+50")
#         self.root.resizable(False, False)  # disable resizing

#         # background image
#         bg = Image.open("Data/drowsy.png")
#         scale_x = 0.85
#         new_width = int(bg.width * scale_x)
#         new_height = int(bg.height * scale_x)
#         bg_resized = bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
#         background = ImageTk.PhotoImage(bg_resized)
#         Label(self.root, image=background).place(x=400, y=5)

#         # info page frame
#         info_page = Frame(root, bg="white")
#         info_page.place(x=0, y=0, height=920, width=400)

#         # logo
#         logo_frame = Frame(info_page, width=100, height=200)  # create a frame to place the logo on
#         logo_frame.place(anchor='se', relx=0.82, rely=0.09)
#         logo = Image.open("Data/logo.png")
#         scale_x = 0.1  # Example: reduce the image size to 50%
#         new_width = int(logo.width * scale_x)
#         new_height = int(logo.height * scale_x)
#         logo_resized = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
#         logo_image = ImageTk.PhotoImage(logo_resized)  # create an ImageTk object
#         Label(logo_frame, image=logo_image, background="white").pack()  # create a label widget to display the image

#         # note label
#         Label(info_page, text="Note that your contact will receive an email \n in case of a repeating drowsiness detection.", font=("Goudy pld style", 13), fg="#619BAF", bg="white").place(x=30, y=90)

#         # driver name label and text box
#         Label(info_page, text="Driver name", font=("Goudy pld style", 12, "bold"), fg="#B80008", bg="white").place(x=30, y=150)
#         self.txt_driver_name = Entry(info_page, font=("times new roman", 15), bg="lightgray")
#         self.txt_driver_name.place(x=35, y=180, width=350, height=25)

#         # contact name label and text box
#         Label(info_page, text="Contact name", font=("Goudy pld style", 12, "bold"), fg="#B80008", bg="white").place(x=30, y=210)
#         self.txt_name_contact = Entry(info_page, font=("times new roman", 15), bg="lightgray")
#         self.txt_name_contact.place(x=35, y=240, width=350, height=25)

#         # contact email label and text box
#         Label(info_page, text="Contact email", font=("Goudy pld style", 12, "bold"), fg="#B80008", bg="white").place(x=30, y=270)
#         self.txt_email_contact = Entry(info_page, font=("times new roman", 15), bg="lightgray")
#         self.txt_email_contact.place(x=35, y=300, width=350, height=25)

#         # start button
#         Button(self.root, command=self.start_function, text="Start Driving", bg="#ABCAD5", font=("times new roman", 12)).place(x=150, y=350, width=100, height=30)

#         # infinite loop waiting for an event to occur and process the event as long as the window is not closed
#         root.mainloop()

#     def start_function(self):
#         """This function checks the correctness of the input and starts the system, in case of missing or incorrect details messagebox will appear"""

#         # if there is an empty field, show an error message
#         if self.txt_email_contact.get() == "" or self.txt_driver_name.get() == "" or self.txt_name_contact.get() == "":
#             messagebox.showerror("Error", "All fields are required.", parent=self.root)

#         # if the email address is invalid, show an error message
#         elif not is_valid_email(self.txt_email_contact.get()):
#             messagebox.showerror("Error", "Invalid email address.", parent=self.root)

#         # otherwise, show a success message, close the window and start driving
#         else:
#             messagebox.showinfo("Start Driving", f"{self.txt_driver_name.get()}, have a pleasant journey! \nYour emergency contact is {self.txt_name_contact.get()}. \n\nPlease wait a few seconds...", parent=self.root)
#             username, contact_name, contact_email = self.txt_driver_name.get(), self.txt_name_contact.get(), self.txt_email_contact.get()  # these will be deleted when the root is destroyed
#             self.root.destroy()
#             drowsiness_classification.start_driving(username, contact_name, contact_email)


# def main():
#     InfoPage(Tk())


# if __name__ == '__main__':
#     main()


# Import drowsiness classification module
import drowsiness_classification

def main():
    """This function starts the drowsiness detection system directly without a GUI"""
    
    # Default values (Modify these as needed)
    username = "Rajesh"
    contact_name = "Sanjay"
    contact_email = "sanjaydutta2830@gmail.com"

    print("Starting Drowsiness Detection System...")
    drowsiness_classification.start_driving(username, contact_name, contact_email)

if __name__ == "__main__":
    main()
