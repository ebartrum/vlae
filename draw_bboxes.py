import cv2
import os
import numpy as np
import argparse
from glob import glob
from pathlib import Path

def draw_bboxes():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on a video or frame sequence.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file or directory of frames.")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask video file or directory of frames.")
    parser.add_argument("--output", type=str, required=True, help="Path to output video file.")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    mask_path = Path(args.mask)
    output_path = args.output
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    is_image_sequence = video_path.is_dir()

    if is_image_sequence:
        frame_files = sorted(sorted(video_path.glob("*.jpg")) + sorted(video_path.glob("*.png")))
        mask_files = sorted(sorted(mask_path.glob("*.jpg")) + sorted(mask_path.glob("*.png")))
        total_frames = min(len(frame_files), len(mask_files))
        
        if total_frames == 0:
            print("No frames found!")
            return
            
        # Read first frame to get dims
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape
        fps = 24.0 # Default for image sequences
    else:
        cap_video = cv2.VideoCapture(str(video_path))
        cap_mask = cv2.VideoCapture(str(mask_path))

        if not cap_video.isOpened():
            print(f"Error opening video file: {video_path}")
            return
        if not cap_mask.isOpened():
            print(f"Error opening mask file: {mask_path}")
            return

        width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_video.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames...")

    frame_count = 0
    # while True:
    for i in range(total_frames):
        if is_image_sequence:
            frame = cv2.imread(str(frame_files[i]))
            mask_frame = cv2.imread(str(mask_files[i]))
            if frame is None or mask_frame is None:
                break
        else:
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
        # Handle DAVIS/Palette images (foreground > 0)
        # Binarize properly to avoid issues
        _, thresh = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

        # Find the bounding box using non-zero points
        points = cv2.findNonZero(thresh)
        if points is not None:
            x, y, w, h = cv2.boundingRect(points)
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Write the frame into the file
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    # Release everything if job is finished
    if not is_image_sequence:
        cap_video.release()
        cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Output saved to {output_path}")

if __name__ == "__main__":
    draw_bboxes()
