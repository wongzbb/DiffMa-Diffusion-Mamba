# DiM-Diffusion-Mamba
Soft Masked Mamba Diffusion Model for CT to MRI Conversion (Official PyTorch Implementation)
###  [ArXiv Paper](https://arxiv.org) 

## 🛠Setup

```bash
git clone https://github.com/wongzbb/DiffMa-Diffusion-Mamba.git
cd DiffMa-Diffusion-Mamba
conda create -n DiffMa python=3.10.0
conda activate DiffMa

conda install cudatoolkit==11.7 -c nvidia
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc

pip install open_clip_torch loguru wandb diffusers einops omegaconf torchmetrics decord accelerate pytest fvcore chardet yacs termcolor submitit tensorboardX seaborn
```
## 📚Data Preparation
**pelvis**:  You can directly use the [processed images data](https://huggingface.co/datasets/ZhenbinWang/pelvis/tree/main) by ours without further data processing.
```
huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/pelvis --local-dir ./datasets/pelvis/
```
**brain**:   You can directly use the [processed images data](https://huggingface.co/datasets/ZhenbinWang/brain/tree/main) by ours without further data processing.
```
huggingface-cli download --repo-type dataset --resume-download ZhenbinWang/brain --local-dir ./datasets/brain/
```


## 🎇Sampling
You can directly sample the MRI from the checkpoint model. Here is an example for quick usage for using our **pre-trained models**:
1. Download the pre-trained weights from [here]().
2. Run [`sample.py`](sample.py) by the following scripts to customize the various arguments.
```
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=1 sample.py \
  --model DiM-L/2 \
  --image-size 224 \
  --global-batch-size 16 \
  --ct-ckpt your_pretrain_CT-encoder_path \
  --ckpt your_DiM_checkpoints_path \
  --dt-rank 32 \
  --d-state 32 \
  --save-dir result_sample \
  --seed 0 \
  --num-sampling-steps 250 \
  --load-ckpt-type ema
```
- `load-ckpt-type`: ema or model.

## ⏳Training
The weight of pretrained DiM can be found [here](https://github.com), and in our implementation we use DiM-L/2 during training DiM.
Train DiM with the resolution of 224x224 with `2` GPUs.
```
CUDA_VISIBLE_DEVICES=3,2 torchrun --master_port=123456 --nnodes=1 --nproc_per_node=2 train.py \
  --model DiM-L/2 \
  --image-size 224 \
  --global-batch-size 2 \
  --ckpt-every 20_000 \
  --ct-ckpt your_pretrain_ct_encoder_path \
  --num-workers 16 \
  --results-dir results \
  --dt-rank 32 \
  --d-state 32 \
  --global-seed 0 \
  --accumulation-steps 10 \
  --wandb \
  --autocast
```
- `--autocast`: This option enables half-precision training for the model. We recommend disabling it, as it is prone to causing NaN errors. To do so, simply remove this option from the command line.
- `--model`: Below are some optional models that use different scanning methods. The first one is ours, and we're using the rest for comparison.
```
DiM_models = {
    #---------------------------------------Ours------------------------------------------#
    'DiM-XL/2': DiM_XL_2,  'DiM-XL/4': DiM_XL_4,  'DiM-XL/7': DiM_XL_7,
    'DiM-L/2' : DiM_L_2,   'DiM-L/4' : DiM_L_4,   'DiM-L/7' : DiM_L_7,
    'DiM-B/2' : DiM_B_2,   'DiM-B/4' : DiM_B_4,   'DiM-B/7' : DiM_B_7,
    'DiM-S/2' : DiM_S_2,   'DiM-S/4' : DiM_S_4,   'DiM-S/7' : DiM_S_7,
    #-----------------------------code reproduction of zigma------------------------------#
    'ZigMa-XL/2': ZigMa_XL_2,  'ZigMa-XL/4': ZigMa_XL_4,  'ZigMa-XL/7': ZigMa_XL_7,
    'ZigMa-L/2' : ZigMa_L_2,   'ZigMa-L/4' : ZigMa_L_4,   'ZigMa-L/7' : ZigMa_L_7,
    'ZigMa-B/2' : ZigMa_B_2,   'ZigMa-B/4' : ZigMa_B_4,   'ZigMa-B/7' : ZigMa_B_7,
    'ZigMa-S/2' : ZigMa_S_2,   'ZigMa-S/4' : ZigMa_S_4,   'ZigMa-S/7' : ZigMa_S_7,
    #--------------------------code reproduction of Vision Mamba--------------------------#
    'ViM-XL/2': ViM_XL_2,  'ViM-XL/4': ViM_XL_4,  'ViM-XL/7': ViM_XL_7,
    'ViM-L/2' : ViM_L_2,   'ViM-L/4' : ViM_L_4,   'ViM-L/7' : ViM_L_7,
    'ViM-B/2' : ViM_B_2,   'ViM-B/4' : ViM_B_4,   'ViM-B/7' : ViM_B_7,
    'ViM-S/2' : ViM_S_2,   'ViM-S/4' : ViM_S_4,   'ViM-S/7' : ViM_S_7,
    #---------------------------code reproduction of VMamba-------------------------------#
    'VMamba-XL/2': VMamba_XL_2,  'VMamba-XL/4': VMamba_XL_4,  'VMamba-XL/7': VMamba_XL_7,
    'VMamba-L/2' : VMamba_L_2,   'VMamba-L/4' : VMamba_L_4,   'VMamba-L/7' : VMamba_L_7,
    'VMamba-B/2' : VMamba_B_2,   'VMamba-B/4' : VMamba_B_4,   'VMamba-B/7' : VMamba_B_7,
    'VMamba-S/2' : VMamba_S_2,   'VMamba-S/4' : VMamba_S_4,   'VMamba-S/7' : VMamba_S_7,
    #----------------------code reproduction of EfficientVMamba---------------------------#
    'EMamba-XL/2': EMamba_XL_2,  'EMamba-XL/4': EMamba_XL_4,  'EMamba-XL/7': EMamba_XL_7,
    'EMamba-L/2' : EMamba_L_2,   'EMamba-L/4' : EMamba_L_4,   'EMamba-L/7' : EMamba_L_7,
    'EMamba-B/2' : EMamba_B_2,   'EMamba-B/4' : EMamba_B_4,   'EMamba-B/7' : EMamba_B_7,
    'EMamba-S/2' : EMamba_S_2,   'EMamba-S/4' : EMamba_S_4,   'EMamba-S/7' : EMamba_S_7,
}
```
