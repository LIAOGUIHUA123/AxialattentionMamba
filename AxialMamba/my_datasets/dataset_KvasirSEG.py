import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class KvasirDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集根目录（train/ validation/ test/）
            transform (callable, optional): 可选的变换函数
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取图像和掩码路径
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

        # 获取所有图像文件并排序
        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取文件名（保持原始命名格式）
        base_name = self.image_files[idx]

        # 加载图像和掩码
        img_path = os.path.join(self.image_dir, base_name)
        mask_path = os.path.join(self.mask_dir, base_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 假设掩码是单通道

        # 应用变换（支持同时处理图像和掩码）
        if self.transform:
            image, mask = self.transform(image=image, mask=mask)

        return image, mask