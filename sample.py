"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
# from train import find_model
from torch.utils.data import DataLoader
from model import DiM_models
import argparse
from load_data import val_dataset, sampler_test, train_dataset, sampler
import logging
from open_clip import create_model_from_pretrained
from block.CT_encoder import CT_Encoder

def find_model(model_name):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if args.load_ckpt_type in checkpoint: 
        checkpoint = checkpoint[args.load_ckpt_type]
    return checkpoint

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    if args.ckpt is None:
        assert args.model == "DiM-L/2"
        assert args.image_size in [224, 256, 512]

    # Load model:
    latent_size = args.image_size // 8
    model = DiM_models[args.model](
        input_size=latent_size,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
    ).to(device)

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    clip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    image_encoder = clip_model.visual.to(device)
    image_encoder.eval()

    ct_encoder = CT_Encoder(img_size=args.image_size // 8, 
                            patch_size=int(args.model[-1]), 
                            in_channels=4, 
                            embed_dim=512, 
                            contain_mask_token=True,
                            ).to(device)
    ct_ckpt_path = args.ct_ckpt or f"./pretrain_ct_encoder/patch_size_2.pt"
    ct_state_dict = find_model(ct_ckpt_path)
    ct_encoder.load_state_dict(ct_state_dict)
    ct_encoder.eval()  # important!

    val_loader = DataLoader(val_dataset, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False, sampler=sampler_test, num_workers=args.num_workers, drop_last=False) #CT, MASK, MRI
    print((f"Dataset contains {len(val_dataset)}."))
    
    item = 0
    for x_ct, _, z_mri in val_loader:
        item+=1

        n = x_ct.shape[0]
        z = torch.randn(n, 4, latent_size, latent_size, device=device)  #Random noise

        x_ct = x_ct.to(device)
        x_ct = torch.cat([x_ct] * 3, dim=1)
        x_ct_ = x_ct
        save_image(x_ct, "sample_ct.png", nrow=4, normalize=True, value_range=(-1, 1))

        z_mri = z_mri.to(device)
        z_mri = torch.cat([z_mri] * 3, dim=1)

        with torch.no_grad():
            x_ = vae.encode(x_ct).latent_dist.sample().mul_(0.18215)
            x_ct = image_encoder(x_ct)
            ct_weight, x_ct_2 = ct_encoder(x_)

        model_kwargs = dict(y=x_ct, y2=x_ct_2, w=ct_weight)

        # Sample images:
        samples = diffusion.p_sample_loop(model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
        samples = vae.decode(samples / 0.18215).sample
        
        os.makedirs('./' + args.save_dir, exist_ok=True)
        save_image(samples, args.save_dir + '/' + str(item) + '_sample_gen.png', nrow=4, normalize=True, value_range=(-1, 1))
        save_image(z_mri, args.save_dir + '/' + str(item) + '_sample_ori.png', nrow=4, normalize=True, value_range=(-1, 1))
        save_image(x_ct_, args.save_dir + '/' + str(item) + '_sample_ct.png', nrow=4, normalize=True, value_range=(-1, 1))
        print(samples.shape)
        save_image(samples[:,0,:,:].unsqueeze(1), args.save_dir + '/' + str(item) + '_sample_gen_1.png', nrow=4, normalize=True, value_range=(-1, 1))
        save_image(samples[:,1,:,:].unsqueeze(1), args.save_dir + '/' + str(item) + '_sample_gen_2.png', nrow=4, normalize=True, value_range=(-1, 1))
        save_image(samples[:,2,:,:].unsqueeze(1), args.save_dir + '/' + str(item) + '_sample_gen_3.png', nrow=4, normalize=True, value_range=(-1, 1))


        if item == 4:
            exit()
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiM_models.keys()), default="DiM-L/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[224, 256, 512], default=224)
    # parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a DiM checkpoint.")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ct-ckpt", type=str, default=None, help="Optional path to a ct-encoder checkpoint.")
    parser.add_argument("--dt-rank", type=int, default=32, help="Mamba block parameters.")
    parser.add_argument("--d-state", type=int, default=32, help="Mamba block parameters.")
    parser.add_argument("--save-dir", type=str, default='result_sample', help="Save test sampler.")
    parser.add_argument("--load-ckpt-type", type=str, choices=['ema','model','opt'], default='ema')
    args = parser.parse_args()
    main(args)
