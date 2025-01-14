import torch
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
from load_data import get_sampler, transform_train, NpyDataset
from open_clip import create_model_from_pretrained
from torch.cuda.amp import GradScaler, autocast
from model import DiffMa_models
from block.CT_encoder import CT_Encoder
from omegaconf import OmegaConf

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
    Trains a new DiffMa model.
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
        model_string_name = args.model.replace("/", "-")  
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
    model = DiffMa_models[args.model](
        input_size=latent_size,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
        use_mamba2 = args.use_mamba2,
    )

    if args.init_from_pretrain_ckpt:
        #load model
        model_state_dict_ = find_model_model(args.pretrain_ckpt_path)
        model.load_state_dict(model_state_dict_)
        #load ema
        ema = deepcopy(model).to(device)
        ema_state_dict_ = find_model(args.pretrain_ckpt_path)
        ema.load_state_dict(ema_state_dict_)
        # log
        logger.info(f"Loaded pretrain model from {args.pretrain_ckpt_path}")
    else:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training


    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule, see ./diffusion/__init__.py
    vae_path = "./models/stabilityai/sd-vae-ft-ema"  # 本地路径
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    # load CT encoder
    ct_encoder = CT_Encoder(
        img_size=args.image_size // 8, 
        patch_size=int(args.model[-1]), #Note that it corresponds to the patch size of DiffMa
        in_channels=4, 
        embed_dim=512, #Corresponding to the output dimension of CLIP's image encoder, the dimension is 384 (for ViT-L/14), 512 (for ViT-B/16 or BiomedCLIP), or 1024 (for Rn50)
        contain_mask_token=True,
        ).to(device)
    ct_ckpt_path = args.ct_ckpt
    ct_state_dict = find_model(ct_ckpt_path)
    ct_encoder.load_state_dict(ct_state_dict)
    ct_encoder.eval()  # important!

    if rank == 0:
        logger.info(f"DiffMa Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Use half-precision training? {args.autocast}")

    #load CLIP image encoder
    # 使用 ViT-B-16
    # 指定模型路径
    model_path = "./models/ViT-B-16"  # 这里的路径应该包含模型的权重文件,我下载在同一根目录下了
    # 加载 CLIP 模型
    print(f"Using model from: {model_path}")
    clip_model, preprocess = create_model_and_transforms(
        model_name="ViT-B-16",  # 指定模型架构
        pretrained="openai"  # 提供本地预训练模型路径
    )
    # 提取图像编码器
    image_encoder = clip_model.visual.to(device)
    image_encoder = clip_model.visual.to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.init_from_pretrain_ckpt:
        lr = args.lr_
    else:
        lr = args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    train_dataset = NpyDataset(args.ct_image_folder_train, args.mask_image_folder_train, args.mir_image_folder_train, transform=transform_train)
    sampler=get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(args.global_batch_size // dist.get_world_size()), 
        shuffle=False, 
        sampler=sampler, 
        num_workers=args.num_workers, 
        drop_last=True
        ) # When using a DistributedSampler, you should set shuffle to False.

    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset)}.")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    image_encoder.eval()  # image encoder should always be in eval mode

    # Variables for monitoring/logging purposes:
    if args.init_from_pretrain_ckpt:
        train_steps = args.init_train_steps
    else:
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
                    z_mri = ((z_mri - z_mri.min()) * 1.0 / (z_mri.max() - z_mri.min())) * 2.0 - 1.0  #6.03改
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

            # Save DiffMa checkpoint:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.")
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.")
    parser.add_argument("--use-mamba2", action="store_true", help="if you want use mamba2.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})

    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)
