import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, Compose
import torch.nn.functional as F

class VideoCaptionDataset(Dataset):
    def __init__(self, dataset_path, num_frames=33, height=512, width=512, frame_interval=1, steps_per_epoch=None):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
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
             
        self.transforms = Compose([
            Resize(size=min(height, width), antialias=True),
            CenterCrop(size=(height, width))
        ])

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
            # Using read_video. Default output is [T, H, W, C]
            video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit='sec')
            # If successful, output is [T, C, H, W]
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            # Return dummy or next
            return self.__getitem__((idx + 1) % len(self.video_files))

        if video.shape[0] < self.num_frames:
             # Loop padding
             repeats = (self.num_frames + video.shape[0] - 1) // video.shape[0]
             video = video.repeat(repeats, 1, 1, 1)
             
        # Simple sampling: random start
        total_frames = video.shape[0]
        max_start = max(0, total_frames - self.num_frames * self.frame_interval)
        start_idx = torch.randint(0, max_start + 1, (1,)).item()
        indices = torch.arange(start_idx, start_idx + self.num_frames * self.frame_interval, self.frame_interval)
        indices = torch.clamp(indices, 0, total_frames - 1)
        
        video = video[indices] # [F, C, H, W]
        
        # Transform
        # Resize expects [C, H, W] or [B, C, H, W]
        # We have [T, C, H, W]. T acts as Batch
        video = self.transforms(video)
        
        # Normalize to [-1, 1] (Assuming read_video returns 0-255 uint8)
        # Verify read_video output. If output_format="TCHW", it returns tensor. 
        # Usually it is uint8 0-255 if directly read? Or float? 
        # Checking docs: read_video returns (video, audio, info). Video is tensor.
        # Dtype depends on backend but usually uint8.
        
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
            
        video = 2.0 * video - 1.0 # [-1, 1]
        
        # First frame for conditioning
        # Training code expects [H, W, C] in 0-255 range for image encoder? 
        # Let's see train_wan_lora.py logic:
        # first_frame_tensor = 2*(batch["first_frame"].float() / 255.) - 1
        # So batch["first_frame"] should be 0-255.
        
        first_frame = (video[0] + 1.0) / 2.0 * 255.0
        first_frame = first_frame.permute(1, 2, 0).to(torch.uint8) # [H, W, C]
        
        return {
            "video": video.permute(1, 0, 2, 3), # [C, T, H, W] - standard for video models often
            # BUT: WanLoRALightningModel expects:
            # latents = self.vae.encode(video, ...) where video is [B, C, T, H, W]
            # So dataset should return [C, T, H, W] or [T, C, H, W]? 
            # batch["video"] will add B dim.
            # train_wan_lora.py: 
            # _, _, num_frames, height, width = video.shape -> [B, C, T, H, W]
            # So we need to return [C, T, H, W]
            
            "text": text,
            "path": video_path,
            "first_frame": first_frame # [H, W, C]
        }
