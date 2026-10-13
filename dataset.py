# -*- coding: utf-8 -*-
"""
自定义 Dataset 类 + 数据增强
6类水果新鲜度分类：新鲜苹果/香蕉/橘子 + 腐烂苹果/香蕉/橘子
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# 类别映射（按文件夹名排序保证一致性）
CLASS_NAMES = ['freshapples', 'freshbanana', 'freshoranges',
               'rottenapples', 'rottenbanana', 'rottenoranges']

# 中文显示名
CLASS_NAMES_CN = ['新鲜苹果', '新鲜香蕉', '新鲜橘子',
                  '腐烂苹果', '腐烂香蕉', '腐烂橘子']

# ImageNet 均值与标准差（MobileNetV2 预训练用）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 图片输入尺寸
IMG_SIZE = 224


class FruitDataset(Dataset):
    """水果新鲜度分类数据集
    从文件夹结构读取：root_dir/class_name/*.png
    """
#扫描文件夹，把文件路径和标签扫描至2个列表
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []#存所有图片路径
        self.labels = []#存标签（0-5）

        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):#扫描每个图片
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(idx)#编码转换，把文字转换成数字
    #一共多少张图片    
    def __len__(self):
        return len(self.samples)
    #取第idx张图片
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


#训练集，增强版
def get_train_transform():
    """训练集数据增强"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),#统一尺寸224*224
        transforms.RandomHorizontalFlip(p=0.5),#50%概率翻转
        transforms.RandomRotation(degrees=15),#旋转25度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),#随机调颜色
        transforms.GaussianBlur(kernel_size=3),#轻微去噪
        transforms.ToTensor(),#图片转张量
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),#归一化，这样v2才能看懂我的图片
        #预处理
    ])


def get_val_test_transform():
    """验证/测试集预处理（不做增强）"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),#没有数据增强那3步
    ])


def create_dataloaders(data_dir, batch_size=32, val_ratio=0.1, num_workers=0):
    """
    创建训练/验证/测试 DataLoader
    参数:
        data_dir: 数据集根目录（包含 train/ 和 test/ 子目录）
        batch_size: 一批多少张图片
        val_ratio: 从训练集中划分的验证集比例
        num_workers: 数据加载线程数（Windows 建议设 0 避免多进程问题）
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 完整训练集（含增强）
    full_train_dataset = FruitDataset(train_dir, transform=get_train_transform())

    # 划分 train / val
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # 验证集用无增强的 transform（需替换）
    # random_split 返回的是 Subset，内部仍用原 transform
    # 解决方案：创建一个无增强版本的 dataset 然后用同样的 indices
    val_dataset_full = FruitDataset(train_dir, transform=get_val_test_transform())
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_dataset.indices)

    # 测试集
    test_dataset = FruitDataset(test_dir, transform=get_val_test_transform())

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=False)#只在训练是打乱
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=False)

    return train_loader, val_loader, test_loader