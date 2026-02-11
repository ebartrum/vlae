import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="images/hi3d_example_images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors," +\
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors," +\
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors," +\
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors," +\
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors," +\
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors," +\
                "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        default=False,
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--validation_num_frames",
        type=int,
        default=81,
        help="Number of frames for validation/inference.",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=81,
        help="Maximum number of frames for training.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=2000,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=64.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=True,
        type=str2bool,
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=True,
        type=str2bool,
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--save_checkpoint_every_n",
        type=int,
        default=200,
        help="How often to save model checkpoints",
    )
    parser.add_argument(
        "--validation_every_n",
        type=int,
        default=200,
        help="How often to save validation samples",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for validation.",
    )
    parser.add_argument(
        "--training_speed_every_n",
        type=int,
        default=100,
        help="How often to log training speed",
    )
    parser.add_argument(
        "--loss_tracker_every_n",
        type=int,
        default=100,
        help="How often to plot losses",
    )
    parser.add_argument(
        "--diagnostic_every_n",
        type=int,
        default=-1,
        help="How often to log diagnostic outputs",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default="data/images/tigerD8H_Jump0.png",
        help="Image prompt used for validation",
    )
    parser.add_argument(
        "--log_data",
        action="store_true",
        help="Option to log data from dataset, and exit (for debugging purposes)",
    )
    parser.add_argument(
        "--no_logging",
        action="store_false",
        help="Option to disable filesystem logging",
        dest="use_logging"
    )
    parser.add_argument(
        "--use_low_precision",
        action="store_true",
        help="Option to use nf4 precision",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="Cache the first batch's latents and embeddings for overfitting testing.",
    )
    parser.add_argument(
        "--reload_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--disable_progress_bar",
        dest="enable_progress_bar",
        action="store_false",
        default=True,
        help="Disable the training progress bar.",
    )
    args = parser.parse_args(args)
    return args
