<div id="top" align="center">

# Soft Masked Mamba Diffusion Model for CT to MRI Conversion
  
  [Zhenbin Wang](https://github.com/wongzbb), Lei Zhang<sup>✉</sup>, [Lituan Wang](https://github.com/LTWangSCU), [Zhenwei Zhang](https://github.com/Zhangzw-99) </br>
  
  [![arXiv](https://img.shields.io/badge/arXiv-2406.15910-b31b1b.svg)](https://arxiv.org/abs/2406.15910)
   [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/ZhenbinWang/DiffMa/tree/main)  </br>


</div>

## News🚀
(2024.06.25) ***The first edition of our paper has been uploaded to arXiv*** 🔥🔥

(2024.06.23) ***We made the code publicly accessible*** 🔥🔥

(2024.06.03) ***Our code integrate Mamba2, use `--use-mamba2` to enjoy it***

(2024.06.10) ***Model weights have been uploaded to HuggingFace for download***

(2024.04.14) ***The project code has been uploaded to Github (set private)*** 🔥🔥

(2024.04.11) ***The processed datasets has been uploaded to HuggingFace***


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

conda install packaging

mkdir whl && cd whl
wget https://github.com/state-spaces/mamba/releases/download/v2.0.4/mamba_ssm-2.0.4+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.2.post1/causal_conv1d-1.2.2.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.2.2.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.0.4+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd ..

pip install --upgrade triton
which ptxas  # will output your_ptxas_path

# for Chinese
export HF_ENDPOINT=https://hf-mirror.com
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
1. Download the pre-trained weights from [here](https://huggingface.co/ZhenbinWang/DiffMa/tree/main). 
2. Run [`sample.py`](sample.py) by the following scripts to customize the various arguments.
```
#for mamba1
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=1 sample.py --config ./config/brain.yaml

#for mamba2
which ptxas  # will output your_ptxas_path
CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=your_ptxas_path torchrun --master_port=12345 --nnodes=1 --nproc_per_node=1 sample.py --config ./config/brain.yaml
```

## ⏳Training
The weight of pretrained DiffMa can be found [here](https://huggingface.co/ZhenbinWang/DiffMa/tree/main).
Train DiffMa with the resolution of 224x224 with `2` GPUs.
```
# use mamba1
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=2 train.py --config ./config/brain.yaml --wandb

# use mamba2
which ptxas  # will output your_ptxas_path
CUDA_VISIBLE_DEVICES=0,1 TRITON_PTXAS_PATH=your_ptxas_path torchrun --master_port=12345 --nnodes=1 --nproc_per_node=2 train.py --config ./config/brain.yaml --use-mamba2 --wandb
```
- `--autocast`: This option enables half-precision training for the model. 


## ⏳Train Vision Embedder
The weight of pretrained Vision Embedder can be found at [`pretrain_ct_embedder`](pretrain_ct_vision_embedder).
Train CT Vision Embedder by the following scripts to customize the various arguments.
```
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=1 train_embedder.py --config ./config/pelvis.yaml
```

Configure the models you wish to train in [`config`](config).
```
DiffMa_models = {
    #---------------------------------------Ours------------------------------------------#
    'DiffMa-XXL/2': DiffMa_XXL_2,  'DiffMa-XXL/4': DiffMa_XXL_4,  'DiffMa-XXL/7': DiffMa_XXL_7,
    'DiffMa-XL/2': DiffMa_XL_2,  'DiffMa-XL/4': DiffMa_XL_4,  'DiffMa-XL/7': DiffMa_XL_7,
    'DiffMa-L/2' : DiffMa_L_2,   'DiffMa-L/4' : DiffMa_L_4,   'DiffMa-L/7' : DiffMa_L_7,
    'DiffMa-B/2' : DiffMa_B_2,   'DiffMa-B/4' : DiffMa_B_4,   'DiffMa-B/7' : DiffMa_B_7,
    'DiffMa-S/2' : DiffMa_S_2,   'DiffMa-S/4' : DiffMa_S_4,   'DiffMa-S/7' : DiffMa_S_7,
    #----------------------code reproduction of zigma-------------------------------------#
    'ZigMa-XL/2': ZigMa_XL_2,  'ZigMa-XL/4': ZigMa_XL_4,  'ZigMa-XL/7': ZigMa_XL_7,
    'ZigMa-L/2' : ZigMa_L_2,   'ZigMa-L/4' : ZigMa_L_4,   'ZigMa-L/7' : ZigMa_L_7,
    'ZigMa-B/2' : ZigMa_B_2,   'ZigMa-B/4' : ZigMa_B_4,   'ZigMa-B/7' : ZigMa_B_7,
    'ZigMa-S/2' : ZigMa_S_2,   'ZigMa-S/4' : ZigMa_S_4,   'ZigMa-S/7' : ZigMa_S_7,
    #----------------------code reproduction of Vision Mamba------------------------------#
    'ViM-XL/2': ViM_XL_2,  'ViM-XL/4': ViM_XL_4,  'ViM-XL/7': ViM_XL_7,
    'ViM-L/2' : ViM_L_2,   'ViM-L/4' : ViM_L_4,   'ViM-L/7' : ViM_L_7,
    'ViM-B/2' : ViM_B_2,   'ViM-B/4' : ViM_B_4,   'ViM-B/7' : ViM_B_7,
    'ViM-S/2' : ViM_S_2,   'ViM-S/4' : ViM_S_4,   'ViM-S/7' : ViM_S_7,
    #----------------------code reproduction of VMamba------------------------------------#
    'VMamba-XL/2': VMamba_XL_2,  'VMamba-XL/4': VMamba_XL_4,  'VMamba-XL/7': VMamba_XL_7,
    'VMamba-L/2' : VMamba_L_2,   'VMamba-L/4' : VMamba_L_4,   'VMamba-L/7' : VMamba_L_7,
    'VMamba-B/2' : VMamba_B_2,   'VMamba-B/4' : VMamba_B_4,   'VMamba-B/7' : VMamba_B_7,
    'VMamba-S/2' : VMamba_S_2,   'VMamba-S/4' : VMamba_S_4,   'VMamba-S/7' : VMamba_S_7,
    #----------------------code reproduction of EfficientVMamba---------------------------#
    'EMamba-XL/2': EMamba_XL_2,  'EMamba-XL/4': EMamba_XL_4,  'EMamba-XL/7': EMamba_XL_7,
    'EMamba-L/2' : EMamba_L_2,   'EMamba-L/4' : EMamba_L_4,   'EMamba-L/7' : EMamba_L_7,
    'EMamba-B/2' : EMamba_B_2,   'EMamba-B/4' : EMamba_B_4,   'EMamba-B/7' : EMamba_B_7,
    'EMamba-S/2' : EMamba_S_2,   'EMamba-S/4' : EMamba_S_4,   'EMamba-S/7' : EMamba_S_7,
    #----------------------code reproduction of DiT---------------------------------------#
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/7': DiT_XL_7,
    'DiT-L/2' : DiT_L_2,   'DiT-L/4' : DiT_L_4,   'DiT-L/7' : DiT_L_7,
    'DiT-B/2' : DiT_B_2,   'DiT-B/4' : DiT_B_4,   'DiT-B/7' : DiT_B_7,
    'DiT-S/2' : DiT_S_2,   'DiT-S/4' : DiT_S_4,   'DiT-S/7' : DiT_S_7,
}
```

## 📜Citation
If you find this work helpful for your project, please consider citing the following paper:
```
@article{wang2024soft,
  title={Soft Masked Mamba Diffusion Model for CT to MRI Conversion},
  author={Wang, Zhenbin and Zhang, Lei and Wang, Lituan and Zhang, Zhenwei},
  journal={arXiv preprint arXiv:2406.15910},
  year={2024}
}
```