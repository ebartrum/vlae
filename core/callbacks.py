import os
import torch
from lightning.pytorch.callbacks import Callback
from .utils import save_video, CHINESE_NEGATIVE_PROMPT
from PIL import Image

class GenerateSampleCallback(Callback):
    def __init__(self, every_n_train_steps = -1):
        self._every_n_train_steps = every_n_train_steps

    def on_train_start(self, trainer, pl_module):
        if pl_module.global_rank == 0 and pl_module.logger:
             self._generate(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0) or (not pl_module.logger):
            return
        
        # Avoid running twice if on_train_start ran at step 0?
        # trainer.global_step is usually 0 *after* first batch? Or during?
        # If running before first batch updates weights, global_step is 0.
        # on_train_batch_end runs after batch. 
        # If we run at on_train_start, we probably don't need to run at step 0 again immediately.
        if trainer.global_step == 0:
            return

        self._generate(trainer, pl_module)
            
    def _generate(self, trainer, pl_module):
        expt_config = pl_module.cfg
        image_path = expt_config.validation_image
        
        if not os.path.exists(image_path):
            print(f"Warning: Validation image {image_path} not found. Skipping validation generation.")
            return

        ref_frame = Image.open(image_path).convert("RGB")
        w, h = ref_frame.size
        new_w = round(w / 16) * 16
        new_h = round(h / 16) * 16
        if (new_w, new_h) != (w, h):
             ref_frame = ref_frame.resize((new_w, new_h), Image.LANCZOS)
        
        # Default prompt logic
        caption_path = image_path.replace("images", "captions").replace(".png", ".txt")
        if os.path.exists(caption_path):
             with open(caption_path) as f:
                caption = f.read().strip()
        else:
            raise FileNotFoundError(f"Caption file not found at {caption_path} for validation image {image_path}")
        
        print(f"Generating validation sample for step {trainer.global_step} with prompt: {caption[:50]}...")
        
        with torch.no_grad():
            output_video = pl_module.pipe(
                prompt=caption,
                negative_prompt=CHINESE_NEGATIVE_PROMPT,
                input_image=ref_frame,
                num_frames=expt_config.validation_num_frames,
                num_inference_steps=expt_config.validation_num_inference_steps,
                cfg_scale=5.0,
                seed=42,
                height=new_h,
                width=new_w,
                output_type = "tensor"
            )

        if len(output_video.shape) == 5:
            output_video = output_video[0] # [C, F, H, W]
            
        output_video = output_video.permute(1, 2, 3, 0) # [F, H, W, C]
        
        # Check range
        vmin, vmax = output_video.min(), output_video.max()
        print(f"Validation video range: [{vmin:.3f}, {vmax:.3f}]")

        if vmax > 2.0:
            print(f"Validation video range: [{vmin:.3f}, {vmax:.3f}] - likely [0, 255]. Dividing by 255.")
            output_video = output_video / 255.0

        vmin, vmax = output_video.min(), output_video.max()
        if vmin < -0.1:
            print(f"Validation video range: [{vmin:.3f}, {vmax:.3f}] - likely [-1, 1]. Shifting to [0, 1].")
            output_video = (output_video + 1.0) / 2.0
            
        output_video = output_video.clamp(0, 1)

        output_dir = os.path.join(pl_module.logger.log_dir, "validation")
        os.makedirs(output_dir, exist_ok=True)
        # Use simple naming or include epoch?
        output_name = f"step{trainer.global_step:06d}.mp4"
        output_path = os.path.join(output_dir, output_name)

        output_video_uint8 = (output_video.cpu().numpy() * 255).astype("uint8")
        
        save_video(output_video_uint8, output_path, fps=15)
        print(f"Validation video saved to {output_path}")
