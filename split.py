import os
import shutil
import random

def split_dataset(a_dir, b_dir, c_dir, a_train, a_test, b_train, b_test, c_train, c_test, train_ratio=0.8):
    
    assert os.path.exists(a_dir), f"Directory {a_dir} does not exist."
    assert os.path.exists(b_dir), f"Directory {b_dir} does not exist."
    assert os.path.exists(c_dir), f"Directory {c_dir} does not exist."

    
    os.makedirs(a_train, exist_ok=True)
    os.makedirs(a_test, exist_ok=True)
    os.makedirs(b_train, exist_ok=True)
    os.makedirs(b_test, exist_ok=True)
    os.makedirs(c_train, exist_ok=True)
    os.makedirs(c_test, exist_ok=True)

    
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    random.shuffle(files)  

    
    split_idx = int(len(files) * train_ratio)

    
    for f in files[:split_idx]:
        
        shutil.move(os.path.join(a_dir, f), os.path.join(a_train, f))
        shutil.move(os.path.join(b_dir, f), os.path.join(b_train, f))
        shutil.move(os.path.join(c_dir, f), os.path.join(c_train, f))

    for f in files[split_idx:]:
        shutil.move(os.path.join(a_dir, f), os.path.join(a_test, f))
        shutil.move(os.path.join(b_dir, f), os.path.join(b_test, f))
        shutil.move(os.path.join(c_dir, f), os.path.join(c_test, f))

a_dir = '/root/code/CTtoMRI/datasets/SynthRAD2023/train/brain/train2D/A_norm'
b_dir = '/root/code/CTtoMRI/datasets/SynthRAD2023/train/brain/train2D/B_norm'
c_dir = '/root/code/CTtoMRI/datasets/SynthRAD2023/train/brain/train2D/C_norm'
a_train = './datasets/brain/A_train_norm'
a_test = './datasets/brain/A_test_norm'
b_train = './datasets/brain/B_train_norm'
b_test = './datasets/brain/B_test_norm'
c_train = './datasets/brain/C_train_norm'
c_test = './datasets/brain/C_test_norm'

split_dataset(a_dir, b_dir, c_dir, a_train, a_test, b_train, b_test, c_train, c_test)
