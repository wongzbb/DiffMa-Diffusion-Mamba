import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from model import DiffMa_models
import argparse
from load_data import NpyDataset, transform_test, get_sampler
import logging
from open_clip import create_model_from_pretrained
from block.CT_encoder import CT_Encoder
from omegaconf import OmegaConf


def find_model(model_name):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)

    # Print the keys and shapes of the checkpoint for debugging
    print(f"Loaded checkpoint from {model_name}:")
    for key, value in checkpoint.items():
        if hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: No shape (likely not a tensor)")

    if args.load_ckpt_type in checkpoint:
        checkpoint = checkpoint[args.load_ckpt_type]
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.ckpt is None:
        assert args.model == "DiffMa-S"
        assert args.image_size in [224, 256, 512]

    # Load model:
    latent_size = args.image_size // 8
    model = DiffMa_models[args.model](
        input_size=latent_size,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
    ).to(device)

    # Load model weights
    ckpt_path = args.ckpt
    try:
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model from {ckpt_path}: {e}")
        return

    model.eval()  # important!

    diffusion = create_diffusion(str(args.sample_num_steps))
    vae = AutoencoderKL.from_pretrained(f"./models/stabilityai/sd-vae-ft-{args.vae}").to(device)

    clip_model, _ = create_model_from_pretrained(
        model_name="ViT-B-16"
        pretrained="openai"
    )
    image_encoder = clip_model.visual.to(device)
    image_encoder.eval()

    ct_encoder = CT_Encoder(img_size=args.image_size // 8,
                            patch_size=int(args.model[-1]),
                            in_channels=4,
                            embed_dim=512,
                            contain_mask_token=True,
                            ).to(device)

    ct_ckpt_path = args.ct_ckpt or f"./pretrain_ct_encoder/patch_size_2.pt"
    try:
        ct_state_dict = find_model(ct_ckpt_path)
        ct_encoder.load_state_dict(ct_state_dict)
    except Exception as e:
        print(f"Error loading CT encoder from {ct_ckpt_path}: {e}")
        return

    ct_encoder.eval()  # important!

    val_dataset = NpyDataset(args.ct_image_folder_val, args.mask_image_folder_val, args.mir_image_folder_val,
                             transform=transform_test)
    sampler = get_sampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.sample_global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.sample_num_workers,
        drop_last=False,
    )  # CT, MASK, MRI
    print(f"Dataset contains {len(val_dataset)} samples.")

    item = 0
    for x_ct, _, z_mri in val_loader:
        item += 1
        n = x_ct.shape[0]
        z = torch.randn(n, 4, latent_size, latent_size, device=device)  # Random noise

        x_ct = x_ct.to(device)
        x_ct = torch.cat([x_ct] * 3, dim=1)
        x_ct_ = x_ct

        z_mri = z_mri.to(device)
        z_mri = torch.cat([z_mri] * 3, dim=1)

        with torch.no_grad():
            if not torch.all((z_mri >= -1) & (z_mri <= 1)):
                z_mri = ((z_mri - z_mri.min()) * 1.0 / (z_mri.max() - z_mri.min())) * 2.0 - 1.0  # 4.21改
            x_ = vae.encode(x_ct).latent_dist.sample().mul_(0.18215)
            x_ct = image_encoder(x_ct)
            ct_weight, x_ct_2 = ct_encoder(x_)

        model_kwargs = dict(y=x_ct, y2=x_ct_2, w=ct_weight)

        # Sample images:
        samples = diffusion.p_sample_loop(model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
                                          progress=True, device=device)
        samples = vae.decode(samples / 0.18215).sample

        os.makedirs('./' + args.save_dir, exist_ok=True)
        save_image(samples, args.save_dir + '/' + str(item) + '_sample_gen.png', nrow=4, normalize=True,
                   value_range=(-1, 1))
        save_image(z_mri, args.save_dir + '/' + str(item) + '_sample_ori.png', nrow=4, normalize=True,
                   value_range=(-1, 1))
        save_image(x_ct_, args.save_dir + '/' + str(item) + '_sample_ct.png', nrow=4, normalize=True,
                   value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)
