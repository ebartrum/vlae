import cv2
import os
import numpy as np

def draw_bboxes():
    # Define paths
    video_path = 'data/videos/davis_camel.mp4'
    mask_path = 'data/videos/davis_camel_mask.mp4'
    output_path = 'output/davis_camel_bbox.mp4'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open video capture objects
    cap_video = cv2.VideoCapture(video_path)
    cap_mask = cv2.VideoCapture(mask_path)

    if not cap_video.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    if not cap_mask.isOpened():
        print(f"Error opening mask file: {mask_path}")
        return

    # Get video properties
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames...")

    frame_count = 0
    while True:
        ret_video, frame = cap_video.read()
        ret_mask, mask_frame = cap_mask.read()

        if not ret_video or not ret_mask:
            break

        # Convert mask to grayscale if it's not already
        if len(mask_frame.shape) == 3:
            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_frame

        # Threshold the mask to binary
        _, thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the bounding box for the largest contour (assuming single object)
        if contours:
            # You might want to combine all contours into one bounding box 
            # or just take the largest one. Here finding the bounding box of all white pixels.
            
            # Alternative: simpler way using non-zero points
            points = cv2.findNonZero(thresh)
            if points is not None:
                x, y, w, h = cv2.boundingRect(points)
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Write the frame into the file 'output.mp4'
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    # Release everything if job is finished
    cap_video.release()
    cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Output saved to {output_path}")

if __name__ == "__main__":
    draw_bboxes()
