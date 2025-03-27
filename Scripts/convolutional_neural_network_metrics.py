# # import packages
# from keras.preprocessing import image
# from keras.models import load_model
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report

# def load_image(img_path):
#     """This function loads and preprocesses an image."""
#     img = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor /= 255.
#     return img_tensor

# # Load model
# model = load_model("Data/Model/yawn_detection.h5")

# # Initialize labels and predictions
# y_true = []
# y_pred = []

# # Loop over 'yawn' class
# yawn_path = "Data/Dataset/test/yawn/"
# for img in os.listdir(yawn_path):
#     predictions = model.predict(load_image(yawn_path + img))
#     y_true.append(1)  # Ground truth: 'yawn'
#     y_pred.append(1 if predictions[0][0] < predictions[0][1] else 0)

# # Loop over 'no_yawn' class
# no_yawn_path = "Data/Dataset/test/no_yawn/"
# for img in os.listdir(no_yawn_path):
#     predictions = model.predict(load_image(no_yawn_path + img))
#     y_true.append(0)  # Ground truth: 'no yawn'
#     y_pred.append(0 if predictions[0][0] > predictions[0][1] else 1)

# # Calculate confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# tn, fp, fn, tp = cm.ravel()

# # Precision, Recall, and F1-Score
# precision = tp / (tp + fp) if (tp + fp) != 0 else 0
# recall = tp / (tp + fn) if (tp + fn) != 0 else 0
# f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# # Print results
# print("Confusion Matrix:")
# print(cm)
# print("\nPrecision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)

# # Save metrics to a file
# with open("Data/Model/metrics.txt", "w") as f:
#     f.write(f"TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}\n\n")
#     f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1_score:.4f}")

# # Plot Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.set(font_scale=1.2)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Yawn", "Yawn"], yticklabels=["No Yawn", "Yawn"])
# plt.title("Confusion Matrix")
# plt.ylabel("Actual")
# plt.xlabel("Predicted")
# plt.savefig("Data/Model/confusion_matrix.png")
# plt.show()

# # Print Detailed Report (Precision, Recall, F1 score for each class)
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=["No Yawn", "Yawn"]))










# Import packages
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_image(img_path):
    """Load and preprocess an image for MobileNetV2."""
    img = image.load_img(img_path, target_size=(224, 224))  # Change to (224, 224) & keep RGB
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.  # Normalize
    return img_tensor

# Load the new model trained with MobileNetV2
model_path = "Data/Model/yawn_detection_mobilenet2.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)

# Initialize labels and predictions
y_true = []
y_pred = []

# Class labels (based on training directory order)
class_labels = {0: "No Yawn", 1: "Yawn"}

# Define test directories
yawn_path = "Data/Datasett/test/yawn/"   # Fixed potential typo in "Datasett"
no_yawn_path = "Data/Datasett/test/no_yawn/"

# Ensure directories exist
if not os.path.exists(yawn_path) or not os.path.exists(no_yawn_path):
    raise FileNotFoundError("One or both test directories do not exist.")

# Process 'yawn' images
if os.listdir(yawn_path):  # Ensure folder is not empty
    for img in os.listdir(yawn_path):
        img_path = os.path.join(yawn_path, img)
        predictions = model.predict(load_image(img_path))
        predicted_label = np.argmax(predictions)  # Get the class index with highest probability
        y_true.append(1)  # Ground truth: 'yawn'
        y_pred.append(predicted_label)
else:
    print("[WARNING] No images found in:", yawn_path)

# Process 'no_yawn' images
if os.listdir(no_yawn_path):  # Ensure folder is not empty
    for img in os.listdir(no_yawn_path):
        img_path = os.path.join(no_yawn_path, img)
        predictions = model.predict(load_image(img_path))
        predicted_label = np.argmax(predictions)  # Get the class index with highest probability
        y_true.append(0)  # Ground truth: 'no yawn'
        y_pred.append(predicted_label)
else:
    print("[WARNING] No images found in:", no_yawn_path)

# Check if y_true and y_pred are populated
if not y_true or not y_pred:
    raise ValueError("No predictions were made. Check if test images exist.")

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Precision, Recall, and F1-Score
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# Print results
print("Confusion Matrix:")
print(cm)
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Save metrics to a file
metrics_path = "Data/Model/metrics_mobilenet.txt"
with open(metrics_path, "w") as f:
    f.write(f"TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}\n\n")
    f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1_score:.4f}")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Yawn", "Yawn"], yticklabels=["No Yawn", "Yawn"])
plt.title("Confusion Matrix - MobileNetV2")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("Data/Model/confusion_matrix_mobilenet.png")
plt.show()

# Print Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["No Yawn", "Yawn"]))
