import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_folder = "Data/New Data"  # Path to the original dataset folder
output_folder = "Data/split_dataset"  # Path to store split data

# Class folders
classes = ["Open", "Closed", "Yawn", "No_Yawn"]

# Create directories for train, validation, and test splits
splits = ["train", "val", "test"]
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_folder, split, cls), exist_ok=True)

# Split data for each class
for cls in classes:
    class_folder = os.path.join(dataset_folder, cls)  # Path to class folder
    images = os.listdir(class_folder)  # List all images in this class
    images = [img for img in images if img.endswith(('.jpg', '.png', '.jpeg'))]  # Filter image files
    
    # Split into train (70%), val (15%), and test (15%)
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    # Copy images to respective directories
    for img in train_images:
        shutil.copy(os.path.join(class_folder, img), os.path.join(output_folder, "train", cls))
    for img in val_images:
        shutil.copy(os.path.join(class_folder, img), os.path.join(output_folder, "val", cls))
    for img in test_images:
        shutil.copy(os.path.join(class_folder, img), os.path.join(output_folder, "test", cls))

print("Dataset split completed!")
