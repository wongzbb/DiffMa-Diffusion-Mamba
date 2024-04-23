"""
A minimal training script for DiM using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
from loguru import logger
import os
import wandb
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from load_data_for_brain import train_dataset, sampler
from open_clip import create_model_from_pretrained
from torch.cuda.amp import GradScaler, autocast
from model import DiM_models
from block.CT_encoder import CT_Encoder

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def find_model(model_name):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint: 
        checkpoint = checkpoint["ema"]
    return checkpoint

def find_model_model(model_name):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    checkpoint = checkpoint["model"]
    return checkpoint

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logger.add(f"{logging_dir}/log"+f"_{dist.get_rank()}.txt", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiM model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiM-XL/2 --> DiM-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        if args.wandb:
            wandb.init(project=args.model.replace('/','_'))
            # wandb.init(project=args.model.replace('/','_'), id='ylhfep72', resume='must')   # load the previous run
            wandb.config = {"learning_rate": 0.0001, 
                            "epochs": args.epochs, 
                            "batch_size": args.global_batch_size,
                            "dt-rank": args.dt_rank,
                            "d-state": args.d_state,
                            "save-path": experiment_dir,
                            "autocast": args.autocast,
                            }
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiM_models[args.model](
        input_size=latent_size,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
    )

    # load model from pretrained model
    # state_dict = find_model_model('./results/002-DiM-L-2/checkpoints/0360000.pt')
    # model.load_state_dict(state_dict)

    # Note that parameter initialization is done within the DiM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # load ema from pretrained model
    # ema = DiM_models[args.model](
    #     input_size=latent_size,
    #     dt_rank=args.dt_rank,
    #     d_state=args.d_state,
    # ).to(device)
    # state_dict_ema = find_model('./results/002-DiM-L-2/checkpoints/0360000.pt')
    # ema.load_state_dict(state_dict_ema)
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule, see ./diffusion/__init__.py
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # load CT encoder
    ct_encoder = CT_Encoder(
        img_size=args.image_size // 8, 
        patch_size=int(args.model[-1]), #Note that it corresponds to the patch size of DiM
        in_channels=4, 
        embed_dim=512, #Corresponding to the output dimension of CLIP's image encoder, the dimension is 384 (for ViT-L/14), 512 (for ViT-B/16 or BiomedCLIP), or 1024 (for Rn50)
        contain_mask_token=True,
        ).to(device)
    ct_ckpt_path = args.ct_ckpt
    ct_state_dict = find_model(ct_ckpt_path)
    ct_encoder.load_state_dict(ct_state_dict)
    ct_encoder.eval()  # important!

    if rank == 0:
        logger.info(f"DiM Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Use half-precision training? {args.autocast}")

    #load CLIP image encoder
    clip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')  #here we use BiomedCLIP
    image_encoder = clip_model.visual.to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    train_loader = DataLoader(train_dataset, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False, sampler=sampler, num_workers=args.num_workers, drop_last=True) # When using a DistributedSampler, you should set shuffle to False.

    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset)}.")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    image_encoder.eval()  # image encoder should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    # train_steps = 360000
    log_steps = 0
    running_loss = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        # print(epoch)
        item = 0
        for x_ct, _, z_mri in train_loader:
            item += 1

            # If your input dimensions are [B,C,W,H], delete it
            x_ct = torch.cat([x_ct] * 3, dim=1) 
            z_mri = torch.cat([z_mri] * 3, dim=1)

            x_ct = x_ct.to(device)
            z_mri = z_mri.to(device)

            with torch.no_grad():
                if not torch.all((z_mri >= -1) & (z_mri <= 1)):
                    z_mri = ((z_mri - z_mri.min()) * 1.0 / (z_mri.max() - z_mri.min())) * 2.0 - 1.0  #4.21改

                z_mri = vae.encode(z_mri).latent_dist.sample().mul_(0.18215)

                x_ = vae.encode(x_ct).latent_dist.sample().mul_(0.18215)
                weight, x_ct_2 = ct_encoder(x_)

                x_ct = image_encoder(x_ct)

            t = torch.randint(0, diffusion.num_timesteps, (z_mri.shape[0],), device=device)

            model_kwargs = dict(y=x_ct, y2=x_ct_2, w=weight)

            with autocast(enabled=args.autocast):
                loss_dict = diffusion.training_losses(model, z_mri, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            if rank == 0 and args.wandb:
                wandb.log({"loss": loss.item()})

            if torch.isnan(loss).any():  #important
                logger.info(f"nan......      ignore losses......")
                continue

            with autocast(enabled=args.autocast):
                scaler.scale(loss).backward()

            if train_steps % args.accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                update_ema(ema, model.module)
                opt.zero_grad()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                epoch_isfinish = int(args.global_batch_size // dist.get_world_size()) * item / len(train_dataset) * 100
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if rank == 0:
                    logger.info(f"({epoch_isfinish:.1f}%) (step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiM checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if rank == 0:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0 and args.wandb:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiM-L/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiM_models.keys()), default="DiM-L/2")
    parser.add_argument("--image-size", type=int, choices=[224, 256, 512], default=224)  #If it is not 224, need to modify the size in load_data.py
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.")
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.")
    parser.add_argument("--ct-ckpt", type=str, default=None, help="Optional path to a ct-encoder checkpoint.")
    parser.add_argument("--dt-rank", type=int, default=32, help="Mamba block parameters.")
    parser.add_argument("--d-state", type=int, default=32, help="Mamba block parameters.")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation.")

    args = parser.parse_args()
    main(args)
