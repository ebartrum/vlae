import cv2
import argparse
import os
import sys

def extract_first_frame(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        sys.exit(1)

    cv2.imwrite(output_path, frame)
    print(f"Successfully saved first frame to {output_path}")
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first frame from video")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", help="Path to output image (default: <video_name>_frame0.png)")
    
    args = parser.parse_args()
    
    if args.output:
        output_path = args.output
    else:
        # Default output path is same directory as video, with _frame0 suffix
        base_name = os.path.splitext(args.video_path)[0]
        output_path = f"{base_name}_frame0.png"
        
    extract_first_frame(args.video_path, output_path)
