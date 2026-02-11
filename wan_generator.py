import torch
import os
import glob
from omegaconf import OmegaConf
from core.utils import load_wan_lora_pipe, CHINESE_NEGATIVE_PROMPT

from core.wan_utils import model_fn_wan_video, TeaCache
from diffsynth.utils import preprocess_images
from torchvision.transforms.functional import to_pil_image
import numpy as np
from tqdm import tqdm

from core.config import parse_args

class VPredStrategy:
    """Abstract base class for velocity prediction strategies."""
    def encode_conditions(self, generator, prompt, condition_image):
        """Encode prompts and images. Return context dict."""
        raise NotImplementedError

    def predict_velocity(self, generator, latents, timestep, context, extra_input, cfg_scale):
        """Predict velocity (v_pred) given current state."""
        raise NotImplementedError

class DefaultVPredStrategy(VPredStrategy):
    """Standard Wan velocity prediction."""
    def encode_conditions(self, generator, prompt, condition_image):
         # Load all models needed for conditioning (Text + Image + VAE)
        generator.pipe.load_models_to_device(["text_encoder", "image_encoder", "vae"])
        
        # Text
        prompt_emb_posi = generator.pipe.encode_prompt(prompt, positive=True)
        prompt_emb_nega = generator.pipe.encode_prompt(CHINESE_NEGATIVE_PROMPT, positive=False)
        
        # Ensure prompt embeddings are on device
        for k, v in prompt_emb_posi.items():
            if isinstance(v, torch.Tensor):
                prompt_emb_posi[k] = v.to(generator.device)
        for k, v in prompt_emb_nega.items():
            if isinstance(v, torch.Tensor):
                prompt_emb_nega[k] = v.to(generator.device)
                
        # Image
        # generator.pipe.load_models_to_device(["image_encoder", "vae"]) # Loaded by caller or here?
        # Caller (generate_trajectory) loads it.
        
        if generator.pipe.image_encoder is not None and condition_image is not None:
             first_frame_pil = to_pil_image(condition_image.cpu())
             image_emb = generator.pipe.image_conditions(
                first_frame_pil, 
                generator.expt_config.num_frames, 
                generator.expt_config.height, 
                generator.expt_config.width
             )
             # Ensure image_emb on device
             for k, v in image_emb.items():
                if isinstance(v, torch.Tensor):
                    image_emb[k] = v.to(generator.device)
                elif isinstance(v, list):
                    image_emb[k] = [item.to(generator.device) if isinstance(item, torch.Tensor) else item for item in v]
        else:
             image_emb = {}
             
        return {
            "prompt_emb_posi": prompt_emb_posi,
            "prompt_emb_nega": prompt_emb_nega,
            "image_emb": image_emb
        }

    def predict_velocity(self, generator, latents, timestep, context, extra_input, cfg_scale):
        prompt_emb_posi = context["prompt_emb_posi"]
        prompt_emb_nega = context["prompt_emb_nega"]
        image_emb = context["image_emb"]
        
        v_pred_posi = model_fn_wan_video(generator.pipe.dit, latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)
        v_pred_nega = model_fn_wan_video(generator.pipe.dit, latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input)
        
        return v_pred_nega + cfg_scale * (v_pred_posi - v_pred_nega)



class WanGenerator:
    def __init__(self, device="cuda:0", vae_checkpoint="checkpoints/taew2_1.pth", vae_type="wan_vae", checkpoint_path=None):
        self.device = torch.device(device)
        self.vae_type = vae_type
        
        # Load Wan Pipeline
        # Load default config
        self.expt_config = parse_args([])
            
        # Pass device to load_wan_lora_pipe
        self.pipe = load_wan_lora_pipe(self.expt_config, checkpoint_path, device=device)
        # pipe.device is now set correctly by load_wan_lora_pipe
        
        if self.vae_type == "tiny_vae":
            print(f"Loading TinyVAE (WanCompatibleTAEHV) from {vae_checkpoint}")
            from taehv import WanCompatibleTAEHV
            self.vae = WanCompatibleTAEHV(checkpoint_path=vae_checkpoint).to(self.device).to(self.pipe.torch_dtype)
            self.vae.eval()
            # Also update pipe.vae for consistency if used elsewhere
            self.pipe.vae = self.vae
        else:
            print("Using Standard Wan VAE.")
            self.vae = self.pipe.vae
            # Ensure it's on device if not loaded yet (though load_wan_lora_pipe might load it lazily)
            # We can load it explicitly or trust the pipeline checks
            pass

        
    @torch.no_grad()
    def compute_video(self, rendered_video, prompt, t_min=0.02, t_max=0.98, seed=None, condition_image=None, timestep_ratio=None, cfg_scale=5.0):
        """
        Compute a target video by adding noise to the rendered video and denoising it with Wan.
        
        Args:
            rendered_video: [T, C, H, W] tensor in [0, 1] (RGB)
            prompt: Text prompt
            t_min, t_max: Range for random timestep sampling
            condition_image: Optional [C, H, W] tensor in [0, 1] (RGB) to use as condition.
                             If None, uses the first frame of rendered_video.
            timestep_ratio: Optional float in [0, 1]. If provided, selects a deterministic timestep
                            based on this ratio (0=start, 1=end of schedule). Overrides t_min/t_max.
            
        Returns:
            target_video: [T, 4, H, W] tensor in [0, 1] (RGBA)
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # 1. Prepare Input Latents
        # Wan requires num_frames % 4 == 1
        T_orig = rendered_video.shape[0]
        H, W = rendered_video.shape[2], rendered_video.shape[3]
        
        if T_orig % 4 != 1:
            T_pad = (T_orig + 2) // 4 * 4 + 1
            # Pad by repeating last frame
            last_frame = rendered_video[-1:]
            padding = last_frame.repeat(T_pad - T_orig, 1, 1, 1)
            video_for_wan = torch.cat([rendered_video, padding], dim=0)
        else:
            video_for_wan = rendered_video
            T_pad = T_orig

        # Preprocess images expects list of PIL or tensor in [0, 255]
        # rendered_video is [T, C, H, W] in [0, 1]
        # We assume standard normalization [-1, 1]
        video_normalized = (video_for_wan * 2.0) - 1.0
        video_normalized = video_normalized.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device) # [1, T, C, H, W]
        
        # VAE encode expects [B, C, T, H, W]
        video_normalized = video_normalized.permute(0, 2, 1, 3, 4) # [1, C, T, H, W]
        
        # Encode
        # Load VAE if needed (though self.vae should be ready)
        self.pipe.load_models_to_device(['vae'])
        
        latents = self.vae.encode(video_normalized, device=self.device)
        
        # 2. Sample Timestep
        # Wan uses sigma, but let's use the scheduler's timesteps
        self.pipe.scheduler.set_timesteps(50) # Standard steps
        
        if timestep_ratio is not None:
            # Deterministic timestep selection
            # timestep_ratio should be in [0, 1] (or close to it)
            # Map to index [0, 49]
            # Invert ratio because index 0 is High Noise, but ratio 1.0 is High Noise
            idx = int(50 * (1 - timestep_ratio))
            idx = max(0, min(idx, 49))
        else:
            # Sample a random index
            idx = torch.randint(int(50 * t_min), int(50 * t_max), (1,)).item()
            
        timestep = self.pipe.scheduler.timesteps[idx]
        timestep = timestep.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
        
        # 3. Add Noise
        noise = torch.randn_like(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        # 4. Prepare Conditions
        self.pipe.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.pipe.encode_prompt(prompt, positive=True)
        prompt_emb_nega = self.pipe.encode_prompt(CHINESE_NEGATIVE_PROMPT, positive=False)

        # Ensure prompt embeddings are on device
        for k, v in prompt_emb_posi.items():
            if isinstance(v, torch.Tensor):
                prompt_emb_posi[k] = v.to(self.device)
        for k, v in prompt_emb_nega.items():
            if isinstance(v, torch.Tensor):
                prompt_emb_nega[k] = v.to(self.device)
            
        # 5. Prepare Image Conditioning (for I2V models)
        if self.pipe.image_encoder is not None:
            self.pipe.load_models_to_device(["image_encoder", "vae"])
            
            if condition_image is not None:
                first_frame = condition_image
            else:
                # rendered_video is [T, C, H, W] in [0, 1]
                first_frame = rendered_video[0] # [C, H, W]
                
            # Convert to PIL for pipe.image_conditions
            first_frame_pil = to_pil_image(first_frame.cpu())
            
            image_emb = self.pipe.image_conditions(
                first_frame_pil, 
                T_pad, 
                H, 
                W
            )
            
            # Ensure all tensors in image_emb are on the correct device
            for k, v in image_emb.items():
                if isinstance(v, torch.Tensor):
                    image_emb[k] = v.to(self.device)
                elif isinstance(v, list):
                    image_emb[k] = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in v]
        else:
            image_emb = {}
            
        # Extra input (freqs etc)
        extra_input = self.pipe.prepare_extra_input(noisy_latents)
        
        # 6. Run Wan DiT
        self.pipe.load_models_to_device(["dit"])
        
        # CFG
        v_pred_posi = model_fn_wan_video(self.pipe.dit, noisy_latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)
        v_pred_nega = model_fn_wan_video(self.pipe.dit, noisy_latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input)
        v_pred = v_pred_nega + cfg_scale * (v_pred_posi - v_pred_nega)
        
        # 7. Compute x1 Estimate (x1_hat)
        sigma = self.pipe.scheduler.get_sigma(timestep)
        latent_x1_hat = noisy_latents - sigma * v_pred
        
        return self._decode_latents(latent_x1_hat, T_orig=T_orig)
        
    def _decode_latents(self, latents, T_orig=None):
        """
        Helper to decode latents using unified self.vae interface.
        """
        self.pipe.load_models_to_device(['vae'])

        # Decode
        # self.vae.decode expects [B, C, T, H, W]
        # returns [B, C, T, H, W] in [-1, 1]
        decoded_video_batch = self.vae.decode(latents, device=self.device)
        
        decoded_video = decoded_video_batch.squeeze(0) # [C, T, H, W]
        decoded_video = decoded_video.permute(1, 0, 2, 3) # [T, C, H, W]
        
        # Denormalize [-1, 1] -> [0, 1]
        decoded_video = (decoded_video + 1.0) / 2.0
        decoded_video = decoded_video.clamp(0, 1)
        
        # Crop back to original length if needed
        if T_orig is not None:
             decoded_video = decoded_video[:T_orig]

        # Handle RGB only output (pad alpha)
        if decoded_video.shape[1] == 3:
            alpha = torch.ones_like(decoded_video[:, :1])
            decoded_video = torch.cat([decoded_video, alpha], dim=1)
        
        return decoded_video.float()

    @torch.no_grad()
    def generate_trajectory(self, prompt, condition_image, seed=None, cfg_scale=5.0, num_inference_steps=50, n_trajectory_samples=1, show_progress=True, v_pred_strategy=None):
        """
        Generate a trajectory of video samples from noise using Wan.
        
        Args:
            prompt: Text prompt
            condition_image: [C, H, W] tensor in [0, 1] (RGB)
            seed: Random seed
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            n_trajectory_samples: Number of samples to return from the trajectory.
                                  1 = Only final result
                                  2 = Start (Noise) and End (Result)
                                  >2 = Evenly spaced samples including Start and End.
            show_progress: Whether to show progress bar
            v_pred_strategy: Optional VPredStrategy instance. If None, uses DefaultVPredStrategy.
            
        Returns:
            dict: {
                "trajectory": [
                    {"timestep_index": int, "sigma": float, "video": [T, 4, H, W] Tensor, "is_final": bool},
                    ...
                ]
            }
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        if v_pred_strategy is None:
            v_pred_strategy = DefaultVPredStrategy()
            
        # 1. Prepare Conditions
        print("Encoding conditions (text/image)...")
        self.pipe.load_models_to_device(["text_encoder", "image_encoder", "vae"])
        
        context = v_pred_strategy.encode_conditions(self, prompt, condition_image)
                
        # 2. Initialize Noise
        num_frames = self.expt_config.num_frames
        h_latent = self.expt_config.height // 8
        w_latent = self.expt_config.width // 8
        t_latent = (num_frames - 1) // 4 + 1
        
        latents = torch.randn(1, 16, t_latent, h_latent, w_latent, device=self.device, dtype=self.pipe.torch_dtype)
        
        # 3. Scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        
        # 4. Trajectory Sampling Indices
        if n_trajectory_samples == 1:
             # Only final result
             target_indices = {num_inference_steps} 
        else:
             target_indices = set(np.linspace(0, num_inference_steps, n_trajectory_samples, dtype=int))
        
        trajectory_results = []
        
        # 5. Denoising Loop
        self.pipe.load_models_to_device(["dit"])
        
        extra_input = self.pipe.prepare_extra_input(latents)
        
        timesteps = self.pipe.scheduler.timesteps
        if show_progress:
            timesteps = tqdm(timesteps, desc="Generating Video")
        
        for i, t in enumerate(timesteps):
            # Check if we need to save THIS step (state i)
            # We do this AFTER prediction so we can get Tweedie estimate
            
            timestep = t.unsqueeze(0).to(dtype=self.pipe.torch_dtype, device=self.device)
            
            # Predict v using Strategy
            v_pred = v_pred_strategy.predict_velocity(self, latents, timestep, context, extra_input, cfg_scale)
            
            # Current step index is i (0 to N-1)
            if i in target_indices:
                # Calculate x1 Estimate (x1 hat)
                sigma = self.pipe.scheduler.get_sigma(t).item()
                latent_x1_hat = latents - sigma * v_pred
                
                # Decode both
                video_noisy = self._decode_latents(latents)
                video_x1_hat = self._decode_latents(latent_x1_hat)
                
                trajectory_results.append({
                    "timestep_index": i,
                    "sigma": sigma,
                    "video": video_noisy.cpu(),
                    "video_x1_hat": video_x1_hat.cpu(),
                    "is_final": False
                })
                
                # Reload DiT if we are continuing and it might have been unloaded by decoding logic
                if i < len(timesteps) - 1:
                    self.pipe.load_models_to_device(["dit"])
            
            # Step
            latents = self.pipe.scheduler.step(v_pred, t, latents)
            
        # Handle Final Step (N)
        # Latents are now z_N (Clean result)
        if num_inference_steps in target_indices:
             video_final = self._decode_latents(latents)
             trajectory_results.append({
                    "timestep_index": num_inference_steps,
                    "sigma": 0.0,
                    "video": video_final.cpu(),
                    "video_x1_hat": video_final.cpu(), # x1_hat of clean is clean
                    "is_final": True
                })
            
        return {"trajectory": trajectory_results}

    def generate_video(self, prompt, condition_image, seed=None, cfg_scale=5.0, num_inference_steps=50, show_progress=True):
        """
        Generate a full target video from noise using Wan.
        Legacy wrapper around generate_trajectory.
        """
        result = self.generate_trajectory(
            prompt, 
            condition_image, 
            seed=seed, 
            cfg_scale=cfg_scale, 
            num_inference_steps=num_inference_steps, 
            n_trajectory_samples=1, 
            show_progress=show_progress
        )
        
        # Result should contain one final video
        # We find the one marked is_final, or just the last one
        traj = result["trajectory"]
        final_video = traj[-1]["video"]
        
        return final_video.to(self.device) # Ensure returned on device as expected by legacy code
