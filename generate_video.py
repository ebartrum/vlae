import argparse
import torch
import os
import numpy as np
from PIL import Image
from core.utils import save_video
from wan_generator import WanGenerator
from torchvision.transforms.functional import to_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--output_filename", default="inference.mp4", type=str, help="Output video filename")
    parser.add_argument("--caption", default=None, type=str, help="Path to caption file")
    
    # VAE params
    parser.add_argument("--use_full_vae", action="store_true", help="Use full Wan VAE instead of TinyVAE")
    parser.add_argument("--rgba_checkpoint", default="checkpoints/taew2_1.pth", type=str, help="RGBA TAEHV checkpoint path (used if specific checkpoint needed for TinyVAE)")
    
    # Generation params
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--cfg_scale", default=5.0, type=float)
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.input_image):
        print(f"Error: Input image {args.input_image} not found.")
        return

    # Load Image
    try:
        image = Image.open(args.input_image)
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")
            
        # Calculate dimensions preserving aspect ratio
        orig_w, orig_h = image.size
        # Snap to multiples of 16
        width = round(orig_w / 16) * 16
        height = round(orig_h / 16) * 16
        
        print(f"Input image size: {orig_w}x{orig_h}")
        print(f"Target video size: {width}x{height}")
             
        if (width, height) != (orig_w, orig_h):
             image = image.resize((width, height), Image.LANCZOS)
             
        condition_image = to_tensor(image).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Initialize Generator
    wan_gen = WanGenerator(device=device, vae_checkpoint=args.rgba_checkpoint, use_full_vae=args.use_full_vae)
    
    # Update config
    wan_gen.expt_config.height = height
    wan_gen.expt_config.width = width
    wan_gen.expt_config.num_frames = 81 # Default
    
    # Load Prompt
    # Load Prompt
    if args.caption:
        caption_path = args.caption
    else:
        caption_path = os.path.splitext(
            args.input_image.replace("images/","captions/"))[0] + ".txt"

    if os.path.exists(caption_path):
        with open(caption_path, 'r') as f:
            prompt = f.read().strip()
    else:
        raise ValueError(f"Could not read caption file {caption_path}")
            
    print(f"Generating video with prompt: '{prompt}'")
    print(f"Output dimensions: {width}x{height}")
    print(f"Refining with {args.num_inference_steps} steps")
    print(f"Using {'Full VAE' if args.use_full_vae else 'TinyVAE'}")
    
    video = wan_gen.generate_video(
        prompt,
        condition_image,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps
    )
    
    # Save
    video_rgb = video[:, :3, :, :]
    # video is [T, C, H, W]
    video_rgb_hwc = video_rgb.permute(0, 2, 3, 1).cpu().numpy()
    video_rgb_hwc = (video_rgb_hwc * 255).astype(np.uint8)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_video(video_rgb_hwc, os.path.join(output_dir, args.output_filename), fps=20)
    print(f"Successfully saved video to {args.output_filename}")

if __name__ == "__main__":
    main()
