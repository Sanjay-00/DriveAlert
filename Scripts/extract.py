import cv2
import os

def extract_frames(video_path, output_folder):
    """
    Extracts all frames from a video and saves them as individual image files.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Initialize frame count
    frame_count = 0
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit when no frames are left to read
        
        # Define output image path
        frame_file = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        
        # Save the current frame
        cv2.imwrite(frame_file, frame)
        
        # Print progress
        print(f"Extracted frame {frame_count}")
        
        # Increment the frame count
        frame_count += 1
    
    # Release video capture
    cap.release()
    print(f"All frames have been saved to {output_folder}")

# Example usage
video_path = "sample.mp4"  # Replace with the path to your video
output_folder = "Data/extracted_frames"  # Folder to save frames
extract_frames(video_path, output_folder)
