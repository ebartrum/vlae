import os
import torch
from lightning.pytorch.callbacks import Callback
from .utils import save_video, CHINESE_NEGATIVE_PROMPT
from PIL import Image

class GenerateSampleCallback(Callback):
    def __init__(self, every_n_train_steps = -1):
        self._every_n_train_steps = every_n_train_steps

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx):
            skip_batch = self._every_n_train_steps < 1 or\
                (trainer.global_step % self._every_n_train_steps != 0) or\
                (not pl_module.logger)
            if skip_batch:
                return

            expt_config = pl_module.cfg
            image_path = expt_config.validation_image
            
            if not os.path.exists(image_path):
                print(f"Warning: Validation image {image_path} not found. Skipping validation generation.")
                return

            ref_frame = Image.open(image_path).convert("RGB")
            
            # Default prompt logic
            caption = "A video of a camel" # Fallback
            caption_path = image_path.replace("images", "captions").replace(".png", ".txt")
            if os.path.exists(caption_path):
                 with open(caption_path) as f:
                    caption = f.read().strip()
            
            print(f"Generating validation sample for step {trainer.global_step} with prompt: {caption[:50]}...")

            # Save training state to restore later? 
            # pl_module.pipe is used for generation. 
            # We need to make sure we don't mess up training state (e.g. gradients)
            # torch.no_grad() is used.
            
            with torch.no_grad():
                # Ensure models are on device
                # pl_module.pipe.to(pl_module.device) # Should be already
                
                output_video = pl_module.pipe(
                    prompt=caption,
                    negative_prompt=CHINESE_NEGATIVE_PROMPT,
                    input_image=ref_frame,
                    num_frames=expt_config.num_frames,
                    num_inference_steps=50,
                    cfg_scale=5.0,
                    seed=42,
                    height=expt_config.height,
                    width=expt_config.width,
                    output_type = "tensor" # Returns [B, C, F, H, W] or [C, F, H, W]?
                )

            # Check output shape
            # diffsynth pipe usually returns [1, C, F, H, W] or [C, F, H, W]
            if len(output_video.shape) == 5:
                output_video = output_video[0] # [C, F, H, W]
                
            output_video = output_video.permute(1, 2, 3, 0) # [F, H, W, C]
            
            # Save
            output_dir = os.path.join(pl_module.logger.log_dir, "validation")
            os.makedirs(output_dir, exist_ok=True)
            output_name = f"step{trainer.global_step:06d}.mp4"
            output_path = os.path.join(output_dir, output_name)

            # save_video expects [T, H, W, C] in 0-1 range? or 0-255?
            # From core/utils.py: "T, H, W, C = frames.shape ... out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))"
            # It expects numpy array, likely uint8 0-255 if using cv2.
            # But earlier code used save_video(video_rgb_hwc, ...) where video_rgb_hwc was uint8.
            # Convert to uint8
            output_video_uint8 = (output_video.cpu().numpy() * 255).astype("uint8")
            
            save_video(output_video_uint8, output_path, fps=15)
            print(f"Validation video saved to {output_path}")
