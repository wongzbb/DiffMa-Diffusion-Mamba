from PIL import Image
import os
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import random
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

n_cpu = os.cpu_count()
global_seed = 0

class NpyDataset(Dataset):
    def __init__(self, image_folder, mask_folder, mri_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.mri_folder = mri_folder
        self.transform = transform
        self.images = os.listdir(image_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_folder, self.images[index])
        mask_path = os.path.join(self.mask_folder, self.images[index])
        mri_path = os.path.join(self.mri_folder, self.images[index])

        image = np.load(image_path)
        mask = np.load(mask_path)
        mri = np.load(mri_path)

        if self.transform:
            image, mask, mri = self.transform(image, mask, mri)
        mask = (mask + 1) / 2
        return image, mask, mri


def transform_train(image, mask, mri, size=(224,224)):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    mri = Image.fromarray(mri)

    image = TF.resize(image, size)
    mask = TF.resize(mask, size, interpolation=transforms.InterpolationMode.NEAREST)
    mri = TF.resize(mri, size, interpolation=transforms.InterpolationMode.NEAREST)

    # # random spin
    # if random.random() > 0.5:
    #     angle = random.choice([90, 180, 270])
    #     image = TF.rotate(image, angle)
    #     mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
    #     mri = TF.rotate(mri, angle, interpolation=transforms.InterpolationMode.NEAREST)

    # # random horizontal flip
    # if random.random() > 0.5:
    #     image = TF.hflip(image)
    #     mask = TF.hflip(mask)
    #     mri = TF.hflip(mri)

    # to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    mri = TF.to_tensor(mri)
    
    return image, mask, mri


def transform_test(image, mask, mir, size=(224,224)):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    mir = Image.fromarray(mir)

    image = TF.resize(image, size)
    mask = TF.resize(mask, size, interpolation=transforms.InterpolationMode.NEAREST)
    mir = TF.resize(mir, size, interpolation=transforms.InterpolationMode.NEAREST)
    
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    mir = TF.to_tensor(mir)

    return image, mask, mir

dist.init_process_group("nccl")
rank = dist.get_rank()

def get_sampler(dataset_):
    sampler = DistributedSampler(dataset_, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=global_seed)
    return sampler
