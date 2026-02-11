import time
import torch, os
import lightning as pl
from lightning.pytorch import loggers
from diffsynth import ModelManager, load_state_dict, load_state_dict_from_folder
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.prompters import WanPrompter
from peft import LoraConfig, inject_adapter_in_model
import random
import torch.nn.functional as F
from core.dataset import VideoCaptionDataset
from core.config import parse_args
from core.utils import log_training_info, log_data
from core.callbacks import GenerateSampleCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanLoRALightningModel(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        num_denoising_steps = 1000,
        lora_rank=4, lora_alpha=4, train_architecture="lora",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        vae=None,
        image_encoder=None,
        prompter=None,
        cfg=None,
        variant="i2v",
        model_type="lora",
    ):
        super().__init__()

        self.cfg= cfg
        self.variant = variant
        self.model_type = model_type

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if not os.path.isfile(dit_path):
            dit_path = dit_path.split(",")

        # Load models using manager (except image encoder as per ref, but we pass it anyway?)
        model_manager.load_models(
            [dit_path, cfg.text_encoder_path]) 

        self.pipe = WanVideoPipeline(device="cuda:0", torch_dtype=torch.bfloat16)

        self.scheduler = FlowMatchScheduler(
            shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_denoising_steps, training=True)

        if cfg.reload_checkpoint:
            model_manager.load_lora_v2(cfg.reload_checkpoint, lora_alpha=1.0)
            print("LoRA loaded from checkpoint")
            
        self.pipe.fetch_models(model_manager)


        self.vae = vae
        self.image_encoder = image_encoder
        self.prompter = prompter
        self.dit = self.pipe.dit
        self.pipe.vae = self.vae

        self.tiler_kwargs = {"tiled": cfg.tiled,
             "tile_size": (cfg.tile_size_height, cfg.tile_size_width),
             "tile_stride": (cfg.tile_stride_height, cfg.tile_stride_width)}

        self.pipe.image_encoder = self.image_encoder

        self.set_trainable_parameters()
        
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.dit,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.dit.requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def set_trainable_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4,
          lora_target_modules="q,k,v,o,ffn.0,ffn.2",
          init_lora_weights="kaiming",
          pretrained_lora_path=None,
          state_dict_converter=None):
        
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32) # Train LoRA in fp32
                
        if pretrained_lora_path is not None:
            print(f"Loading pretrained LoRA from {pretrained_lora_path}")
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            
            # Helper to filter keys. 
            # Assuming standard Peft naming or matching keys from reference. 
            # Reference logic for filtering specific keys is complex. 
            # Let's trust peft load or do simple matching if needed.
            # But here we implement manual loading to be safe as per reference.
            
            state_dict_new = {}
            for key in state_dict.keys():
                if 'pipe.dit.' in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                # Add other mappings if needed
            
            if not state_dict_new:
                 # If empty, maybe keys are already correct?
                 state_dict_new = state_dict

            missing, unexpected = model.load_state_dict(state_dict_new, strict=False)
            print(f"Loaded LoRA: {len(missing)} missing, {len(unexpected)} unexpected keys.")
    
    def encode_video(self, batch):
        text = batch["text"][0]
        video = batch["video"]
        path = batch["path"][0]

        with torch.no_grad():
            if video is not None:
                prompt_emb = {'context': self.prompter.encode_prompt(text)}
                
                video = video.to(dtype=torch.bfloat16, device=self.device) # [B, C, T, H, W]
                latents = self.vae.encode(video, device=self.device, **self.tiler_kwargs)[0] # [16, T_latent, H_latent, W_latent]
                
                # Image Conditioning (First Frame)
                if "first_frame" in batch:
                    # batch["first_frame"] is [B, H, W, C] (0-255 uint8)
                    first_frame_tensor = batch["first_frame"].float() / 255.0
                    first_frame_tensor = 2.0 * first_frame_tensor - 1.0
                    first_frame_tensor = first_frame_tensor.permute(0, 3, 1, 2) # [B, C, H, W]
                    
                    # Ensure correct device and dtype
                    first_frame_tensor = first_frame_tensor.to(device=self.device, dtype=torch.bfloat16)

                    clip_context = self.image_encoder.encode_image(
                        [first_frame_tensor]).to(
                            dtype=torch.bfloat16,
                            device=self.device)
                    
                    # VAE condition
                    num_frames = video.shape[2] 
                    latent_condition = self.vae.latent_image_condition(
                        first_frame_tensor, num_frames).to(
                        dtype=torch.bfloat16, device=self.device)

                    image_emb = {
                        "clip_feature": clip_context, "y": latent_condition}
                else:
                    image_emb = {}
                
                return {"latents": latents.unsqueeze(0),
                    "prompt_emb": prompt_emb, "image_emb": image_emb}
        return None

    def training_step(self, batch, batch_idx):
        diagnostics = {}
        
        batch_train = None
        # Check for disk cache
        video_path = batch["path"][0]
        dirname = os.path.dirname(video_path)
        basename = os.path.basename(video_path)
        cache_dir = os.path.join(dirname, ".latents_cache")
        cache_filename = basename + ".pt"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            try:
                batch_train = torch.load(cache_path, map_location=self.device)
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}")

        if batch_train is None:
            batch_train = self.encode_video(batch)

        p = random.random()
        latents = batch_train["latents"].to(self.device).to(torch.bfloat16)
        prompt_emb = batch_train["prompt_emb"]
        
        prompt_emb["context"] = prompt_emb["context"].to(self.device).to(torch.bfloat16)
        image_emb = batch_train["image_emb"]
        
        # Dropout
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device).to(torch.bfloat16)
            if p < 0.1:
                image_emb["clip_feature"] = torch.zeros_like(image_emb["clip_feature"])
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"].to(self.device).to(torch.bfloat16)
            if p < 0.1:
                image_emb["y"] = torch.zeros_like(image_emb["y"])
    
        # Loss
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=torch.bfloat16, device=self.device)
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
        training_target = self.scheduler.training_target(latents, noise, timestep)
        
        noise_pred = self.dit(
            noisy_latents,
            timestep=timestep,
            is_train=True,
            **prompt_emb, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            add_condition = None)
    
        loss = F.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
    
        self.log("train_loss", loss, prog_bar=True)
        diagnostics["timestep"] = int(timestep.item())
    
        return {"loss": loss, "diagnostics": diagnostics}

    def configure_optimizers(self):
        trainable_modules = [
            {'params': filter(lambda p: p.requires_grad, self.dit.parameters())},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Save only LoRA weights
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters())) 
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)

def train(args):
    dataset = VideoCaptionDataset(
        args.dataset_path,
        max_num_frames=args.max_num_frames,
        steps_per_epoch=args.steps_per_epoch,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True, # Random sampling
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([args.image_encoder_path])

    prompter = WanPrompter()
    prompter.text_encoder = WanTextEncoder()
    prompter.text_encoder.load_state_dict(
        torch.load(args.text_encoder_path, map_location="cpu"))
    prompter.text_encoder = prompter.text_encoder.cuda()
    prompter.fetch_tokenizer(
        os.path.join(
            os.path.dirname(args.text_encoder_path), "google/umt5-xxl"))

    vae = WanVideoVAE()
    vae.model.load_state_dict(torch.load(args.vae_path))
    vae = vae.to(torch.device("cuda")).to(torch.bfloat16)
    
    image_encoder = model_manager.fetch_model("wan_video_image_encoder").to(torch.device("cuda")).to(torch.bfloat16)

    model = WanLoRALightningModel(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        vae = vae,
        image_encoder = image_encoder,
        prompter = prompter,
        cfg = args,
        variant=args.variant,
        model_type=args.model,
    )

    # Always check/generate cache
    # Move model to GPU for encoding (if generation needed)
    cache_dir = os.path.join(dataset.videos_dir, ".latents_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    num_files = len(dataset.video_files)
    from torch.utils.data import default_collate
    
    model_on_gpu = False

    print("Checking latent cache...")
    for i in range(num_files):
        sample = dataset[i] 
        path = sample["path"]
        filename = os.path.basename(path)
        cache_filename = filename + ".pt"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        if args.recompute_latents or not os.path.exists(cache_path):
            if not model_on_gpu:
                model.cuda()
                model_on_gpu = True

            print(f"Generating cache for {filename}...")
            batch = default_collate([sample])
            
            # Encode
            encoded = model.encode_video(batch)

            # Move to CPU for saving
            encoded_cpu = {}
            for k, v in encoded.items():
                if isinstance(v, torch.Tensor):
                    encoded_cpu[k] = v.cpu()
                elif isinstance(v, dict):
                    encoded_cpu[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, torch.Tensor):
                            encoded_cpu[k][k2] = v2.cpu()
                        else:
                            encoded_cpu[k][k2] = v2
                else:
                    encoded_cpu[k] = v
            
            torch.save(encoded_cpu, cache_path)
    
    print(f"Latent cache verified/generated at {cache_dir}")


    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        logger=loggers.CSVLogger(save_dir=args.output_path) if args.use_logging else False,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                save_top_k=1, every_n_train_steps=args.save_checkpoint_every_n),
            GenerateSampleCallback(args.validation_every_n),
        ]
    )
    
    log_training_info(args, trainer.logger)
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    args = parse_args()
    train(args)
