import os
import random

import SimpleITK as sitk
import numpy as np
import torch


def copy_image_info(image1, image2):
    if not isinstance(image1, sitk.Image) or not isinstance(image2, sitk.Image):
        raise ValueError("Both inputs must be SimpleITK Image objects.")

    image2.CopyInformation(image1)

    return image2


def set_seed(seed: int):
    """
    设置平台无关的随机种子，确保实验结果的可重复性。
    :param seed: 随机种子
    """
    # 设置 Python 的哈希种子，确保字典等结构的可重复性
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设置 PyTorch GPU 随机种子（如果使用CUDA）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，这行代码也可以保证一致性

    # 设置 CuDNN 使用确定性算法，这对于 GPU 上的操作是必要的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动调优，以确保每次运行结果相同
    print(f'Seed set to {seed} for all libraries (Python, NumPy, PyTorch)')
