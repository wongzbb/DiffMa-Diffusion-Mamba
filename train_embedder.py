import torch
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from load_data import get_sampler, transform_train, NpyDataset
from torch.cuda.amp import GradScaler, autocast
import wandb
from glob import glob
from loguru import logger
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers.models import AutoencoderKL
from time import time
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from block.CT_encoder import CT_Encoder
from omegaconf import OmegaConf

def create_logger(logging_dir):
    if dist.get_rank() == 0:  # real logger
        logger.add(f"{logging_dir}/log"+f"_{dist.get_rank()}.txt", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    return logger

def infoNCE_loss_b(input_tensor, tau=0.07):
    batch_size, seq_len, feat_dim = input_tensor.shape
    reshaped_tensor = input_tensor.reshape(batch_size, seq_len*feat_dim )
    reshaped_tensor = F.normalize(reshaped_tensor, p=2, dim=1)
    sim_matrix = torch.matmul(reshaped_tensor, reshaped_tensor.T) / tau
    labels = torch.arange(reshaped_tensor.size(0), dtype=torch.long, device=reshaped_tensor.device)
    loss = F.cross_entropy(sim_matrix, labels) 
    return loss

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    seed = args.embedder_global_seed
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        if args.wandb:
            wandb.init(project="CT_encoder")
            wandb.config = {"learning_rate": 0.0001, "epochs": args.embedder_epoch, "batch_size": args.embedder_global_batch_size}

        os.makedirs(args.embedder_results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.embedder_results_dir}/*"))
        model_string_name = "vision_encoder" 
        experiment_dir = f"{args.embedder_results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    model = CT_Encoder(img_size=28, patch_size=args.embedder_patch_size, in_channels=4, embed_dim=args.embedder_embed_dim, contain_mask_token=True)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[rank], find_unused_parameters=True)
    vae_path = "./models/stabilityai/sd-vae-ft-ema"  # 本地路径
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    if rank == 0:
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    train_dataset = NpyDataset(args.ct_image_folder_train, args.mask_image_folder_train, args.mir_image_folder_train, transform=transform_train)
    sampler=get_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=int(args.embedder_global_batch_size // dist.get_world_size()), shuffle=False, sampler=sampler, num_workers=args.embedder_num_workers, drop_last=True)
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval() 
    model.train()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    criterion_main = nn.CrossEntropyLoss()

    if rank == 0:
        logger.info(f"Training for {args.embedder_epoch} epochs...")
    for epoch in range(args.embedder_epoch):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        # print(epoch)
        item = 0
        for x_ct, _, _ in train_loader:
            item+=1
            x_ct = x_ct.to(device)
            x_ct = torch.cat([x_ct] * 3, dim=1)
            with torch.no_grad():
                x_ct = vae.encode(x_ct).latent_dist.sample().mul_(0.18215)

            opt.zero_grad()
            with autocast(enabled=args.autocast):
                weight, x = model(x_ct)

            loss = infoNCE_loss_b(x)

            if rank == 0 and args.wandb:
                wandb.log({"loss": loss.item()})

            if torch.isnan(loss).any():
                logger.info(f"nan...      ignore losses....")
                continue

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                epoch_isfinish = int(args.embedder_global_batch_size // dist.get_world_size()) * item / len(train_dataset) * 100
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"({epoch_isfinish:.1f}%) (step={train_steps:07d}) Train Loss: {avg_loss:.8f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.embedder_ckpt_every == 0 and train_steps > 0:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.")
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.")
    args = parser.parse_args()

    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)