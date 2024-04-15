import os
import shutil
import random

def split_dataset(a_dir, b_dir, c_dir, a_train, a_test, b_train, b_test, c_train, c_test, train_ratio=0.8):
    # 确保输入的目录存在
    assert os.path.exists(a_dir), f"Directory {a_dir} does not exist."
    assert os.path.exists(b_dir), f"Directory {b_dir} does not exist."
    assert os.path.exists(c_dir), f"Directory {c_dir} does not exist."

    # 创建训练集和测试集目录
    os.makedirs(a_train, exist_ok=True)
    os.makedirs(a_test, exist_ok=True)
    os.makedirs(b_train, exist_ok=True)
    os.makedirs(b_test, exist_ok=True)
    os.makedirs(c_train, exist_ok=True)
    os.makedirs(c_test, exist_ok=True)

    # 获取A文件夹中的所有图片
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    random.shuffle(files)  # 打乱文件顺序

    # 分割点
    split_idx = int(len(files) * train_ratio)

    # 分配文件到训练集和测试集
    for f in files[:split_idx]:
        # 移动A中的文件到A_train
        shutil.move(os.path.join(a_dir, f), os.path.join(a_train, f))
        # 移动B中的对应文件到B_train
        shutil.move(os.path.join(b_dir, f), os.path.join(b_train, f))
        shutil.move(os.path.join(c_dir, f), os.path.join(c_train, f))

    for f in files[split_idx:]:
        # 移动A中的文件到A_test
        shutil.move(os.path.join(a_dir, f), os.path.join(a_test, f))
        # 移动B中的对应文件到B_test
        shutil.move(os.path.join(b_dir, f), os.path.join(b_test, f))
        shutil.move(os.path.join(c_dir, f), os.path.join(c_test, f))

# 指定目录路径
a_dir = './A_norm'
b_dir = './B_norm'
c_dir = './C_norm'
a_train = './A_train'
a_test = './A_test'
b_train = './B_train'
b_test = './B_test'
c_train = './C_train'
c_test = './C_test'

# 调用函数执行分割
split_dataset(a_dir, b_dir, c_dir, a_train, a_test, b_train, b_test, c_train, c_test)
