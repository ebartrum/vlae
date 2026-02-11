import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, Compose
import torch.nn.functional as F

class VideoCaptionDataset(Dataset):
    def __init__(self, dataset_path, max_num_frames=81, frame_interval=1, steps_per_epoch=None):
        self.dataset_path = dataset_path
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        
        self.videos_dir = os.path.join(dataset_path, "videos")
        self.captions_dir = os.path.join(dataset_path, "captions")
        
        self.video_files = []
        if os.path.exists(self.videos_dir):
            valid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
            self.video_files = [
                f for f in os.listdir(self.videos_dir) 
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
        
        self.length = len(self.video_files)
        if steps_per_epoch is not None:
             self.length = steps_per_epoch

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Allow infinite indexing if needed
        real_idx = idx % len(self.video_files)
            
        video_filename = self.video_files[real_idx]
        video_path = os.path.join(self.videos_dir, video_filename)
        
        # Determine caption path
        video_basename = os.path.splitext(video_filename)[0]
        # Try both .txt and same name
        caption_path = os.path.join(self.captions_dir, video_basename + ".txt")
        
        # Load Caption
        text = ""
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                text = f.read().strip()
        else:
            print(f"Warning: No caption found for {video_filename}, using empty string.")
            
        # Load Video
        try:
            import warnings
            # Suppress the specific deprecation warning from torchvision
            warnings.filterwarnings("ignore", category=UserWarning, message="The video decoding and encoding capabilities of torchvision are deprecated")
            
            # Using read_video. Default output is [T, H, W, C]
            video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit='sec')
            # If successful, output is [T, C, H, W]
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            # Return dummy or next
            return self.__getitem__((idx + 1) % len(self.video_files))

        # Handle Frame Count (1 mod 8)
        # Target should be <= max_num_frames
        T = video.shape[0]
        
        # Calculate max we can use
        target_T = min(T, self.max_num_frames)
        
        # Ensure 1 mod 8
        # (target_T - 1) // 8 * 8 + 1
        target_T = (target_T - 1) // 8 * 8 + 1
        
        if target_T < 1: 
             # Too short? loop?
             # For now just repeat to meet at least 9 frames?
             pass 
             
        # If video is shorter than needed for minimal 1 mod 8 (e.g. < 1?? 1 mod 8 is 1, 9, 17...)
        # Minimal is 1. If T >= 1, we are good.
        
        # Sampling
        # Simple random crop
        if T > target_T:
             start_idx = torch.randint(0, T - target_T + 1, (1,)).item()
             video = video[start_idx : start_idx + target_T]
        else:
             # Just use it (if it matches 1 mod 8 logic above)
             video = video[:target_T] # Should be essentially no-op if logic is right
             
        # Handle Resolution
        # [T, C, H, W]
        _, _, H, W = video.shape
        new_H = round(H / 16) * 16
        new_W = round(W / 16) * 16
        
        if (new_H, new_W) != (H, W):
             video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False, antialias=True)
        
        # Normalize to [-1, 1]
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
            
        video = 2.0 * video - 1.0 # [-1, 1]
        
        first_frame = (video[0] + 1.0) / 2.0 * 255.0
        first_frame = first_frame.permute(1, 2, 0).to(torch.uint8) # [H, W, C]
        
        return {
            "video": video.permute(1, 0, 2, 3), # [C, T, H, W]
            "text": text,
            "path": video_path,
            "first_frame": first_frame # [H, W, C]
        }
