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
