import sys
import os
import pickle
from omegaconf import OmegaConf
import torch
import torchvision
from torchvision.io import write_video
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import cv2
from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.models.lora import GeneralLoRAFromPeft
from diffsynth.models.utils import load_state_dict
import subprocess
from PIL import Image
from io import BytesIO
from peft import LoraConfig, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "false" #needed to avoid a warning when saving video

CHINESE_NEGATIVE_PROMPT="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"

def save_video(frames, output_path, fps=20):
    if torch.is_tensor(frames):
        frames = frames.numpy()
    T, H, W, C = frames.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) # switch frames from RGB to BGR to write
    out.release()

def load_video_tracks_pipe(expt_config, checkpoint_path):
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [expt_config.image_encoder_path],
        torch_dtype=torch.float32, # Image Encoder is loaded with float32
    )
    model_manager.load_models(
        [
            expt_config.dit_path if os.path.isfile(expt_config.dit_path) else expt_config.dit_path.split(","),
            expt_config.text_encoder_path,
            expt_config.vae_path,
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )

    if expt_config.reload_checkpoint:
        model_manager.load_lora_v2(expt_config.reload_checkpoint, lora_alpha=1.0)
        print("Warning: experimental feature!!!")
        print("Loading a lora checkpoint which was reloaded before starting\
              training. This may not work as expected...")
    model_manager.load_lora_v2(checkpoint_path, lora_alpha=1.0)

    pipe = VideoTracksPipeline(
        device="cuda:0", torch_dtype=torch.bfloat16, config=expt_config)
    pipe.fetch_models(expt_config, model_manager)

    #Add LoRA to tracks model, if found
    state_dict = load_state_dict(checkpoint_path)
    state_dict_tracks_model_lora = {}
    for key in state_dict.keys():
        if 'lora' in key and 'tracks_model' in key:
            key_new = key.split("pipe.tracks_model.")[1]
            state_dict_tracks_model_lora[key_new] = state_dict[key]

    lora = GeneralLoRAFromPeft()
    match_results = lora.match(
        pipe.video_tracks_model.tracks_model,
        state_dict_tracks_model_lora)
    if match_results is not None:
        print("Adding LoRA to tracks_model.")
        lora_prefix, model_resource = match_results
        lora.load(pipe.video_tracks_model.tracks_model,
              state_dict_tracks_model_lora,
              lora_prefix, alpha=1.0,
                  model_resource=model_resource)

    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    pipe.cuda()
    return pipe

def load_wan_lora_pipe(expt_config, checkpoint_path, device="cuda:0"):
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [expt_config.image_encoder_path],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_models(
        [
            expt_config.dit_path if os.path.isfile(expt_config.dit_path) else expt_config.dit_path.split(","),
            expt_config.text_encoder_path,
            expt_config.vae_path,
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )

    if expt_config.reload_checkpoint:
        model_manager.load_lora_v2(expt_config.reload_checkpoint, lora_alpha=1.0)
        print("Warning: experimental feature!!!")
        print("Loading a lora checkpoint which was reloaded before starting\
              training. This may not work as expected...")
    
    if checkpoint_path is not None:
        model_manager.load_lora_v2(checkpoint_path, lora_alpha=1.0)

    pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)
    pipe.fetch_models(model_manager)

    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    # pipe.cuda() is handled by device arg now
    return pipe


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def log_training_info(args, logger):
    if logger:
        log_dir = logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        training_command = "python " + " ".join(sys.argv)
        expt_logfile = os.path.join(
            log_dir, "expt_logs.txt")
        expt_options_file = os.path.join(
            log_dir, "options_used.yaml")
        with open(expt_logfile, "w") as f:
            f.write("training_command:\n")
            f.write(training_command)
        with open(expt_logfile, "a") as f:
            f.write("\ngit commit:\n")
            f.write(get_git_revision_short_hash() + "\n")
        omega_config = OmegaConf.create(vars(args))
        OmegaConf.save(config=omega_config, f=expt_options_file)

def log_data(data_item):
    os.makedirs('outputs/log_data', exist_ok=True)
    torchvision.io.write_video(
        "outputs/log_data/data_video.mp4", 255*(data_item["video"]*0.5 + 0.5).permute(1,2,3,0), fps=20)
    torchvision.utils.save_image(
        data_item['first_frame'].permute(2,0,1)/255.,"outputs/log_data/first_frame.png")
    with open("outputs/log_data/caption.txt", "w") as text_file:
        text_file.write(data_item['text'])
    exit()

def add_lora_to_model(model, lora_rank=4, lora_alpha=4, 
                         lora_target_modules="q,k,v,o,ffn.0,ffn.2", 
                         init_lora_weights="kaiming", 
                         pretrained_lora_path=None):

    if init_lora_weights == "kaiming":
        init_lora_weights = True
        
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = get_peft_model(model, lora_config)
    
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32) # Train LoRA in fp32
            
    if pretrained_lora_path:
        print(f"Loading pretrained LoRA from {pretrained_lora_path}")
        
        if pretrained_lora_path.endswith(".ckpt"):
             state_dict = torch.load(pretrained_lora_path, map_location="cpu")
             if "state_dict" in state_dict:
                 state_dict = state_dict["state_dict"]
        else:
             state_dict = load_state_dict(pretrained_lora_path)
             
        # Filter for DiT keys if needed
        keys_to_load = {}
        for k, v in state_dict.items():
            if "pipe.dit." in k:
                keys_to_load[k.split("pipe.dit.")[1]] = v
            elif k.startswith("dit."):
                keys_to_load[k.split("dit.", 1)[1]] = v
            else:
                keys_to_load[k] = v

        # Handle PeftModel prefix
        new_keys_to_load = {}
        model_keys = set(model.state_dict().keys())
        prefix = "base_model.model."
        for k, v in keys_to_load.items():
             if f"{prefix}{k}" in model_keys:
                 new_keys_to_load[f"{prefix}{k}"] = v
             else:
                 new_keys_to_load[k] = v
                
        missing, unexpected = model.load_state_dict(new_keys_to_load, strict=False)
        print(f"Loaded LoRA: {len(missing)} missing, {len(unexpected)} unexpected keys.")
        
    return model
