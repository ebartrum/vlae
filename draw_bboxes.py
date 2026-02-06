import cv2
import os
import numpy as np
import argparse
from glob import glob
from pathlib import Path

class FrameReader:
    def __init__(self, path):
        self.path = Path(path)
        self.is_dir = self.path.is_dir()
        self.cap = None
        self.files = []
        self.idx = 0
        self.length = 0
        self.width = 0
        self.height = 0
        self.fps = 24.0

        if self.is_dir:
            self.files = sorted(sorted(self.path.glob("*.jpg")) + sorted(self.path.glob("*.png")))
            self.length = len(self.files)
            if self.length > 0:
                img = cv2.imread(str(self.files[0]))
                if img is not None:
                    self.height, self.width, _ = img.shape
        else:
            self.cap = cv2.VideoCapture(str(self.path))
            if self.cap.isOpened():
                self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        if self.is_dir:
            if self.idx >= self.length:
                return False, None
            frame = cv2.imread(str(self.files[self.idx]))
            self.idx += 1
            return (frame is not None), frame
        else:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()

def draw_bboxes():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on a video or frame sequence.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file or directory of frames.")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask video file or directory of frames.")
    parser.add_argument("--output", type=str, required=True, help="Path to output video file.")
    
    args = parser.parse_args()
    
    video_reader = FrameReader(args.video)
    mask_reader = FrameReader(args.mask)
    output_path = args.output
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if video_reader.length == 0:
        print(f"Error: No frames found in video source: {args.video}")
        return
    if mask_reader.length == 0:
        print(f"Error: No frames found in mask source: {args.mask}")
        return

    width = video_reader.width
    height = video_reader.height
    fps = video_reader.fps
    total_frames = min(video_reader.length, mask_reader.length)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames...")

    frame_count = 0
    for i in range(total_frames):
        ret_video, frame = video_reader.read()
        ret_mask, mask_frame = mask_reader.read()

        if not ret_video or not ret_mask:
            break
            
        if frame is None or mask_frame is None:
            break

        # Resize mask to match video frame size if necessary
        if frame.shape[:2] != mask_frame.shape[:2]:
            mask_frame = cv2.resize(mask_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

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
    video_reader.release()
    mask_reader.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Output saved to {output_path}")

if __name__ == "__main__":
    draw_bboxes()
