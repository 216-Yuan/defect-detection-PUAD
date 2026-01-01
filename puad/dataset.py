import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PIL import Image
import numpy as np
from puad.common import build_imagenet_normalization
import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms

TransformType = Callable[[Union[Image.Image, np.ndarray, torch.tensor]], Union[Image.Image, np.ndarray, torch.tensor]]


class RandomAugment:
    def __call__(self, img: Image.Image) -> Image.Image:
        i_aug = torch.randint(1, 4, (1,))
        lamda = torch.rand(1) * 0.4 + 0.8
        if i_aug == 1:
            return transforms.functional.adjust_brightness(img, lamda)
        elif i_aug == 2:
            return transforms.functional.adjust_contrast(img, lamda)
        return transforms.functional.adjust_saturation(img, lamda)


class NormalDataset(Dataset):
    def __init__(self, normal_image_dir: str, transform: Optional[TransformType] = None) -> None:
        super().__init__()
        self.img_paths = self._get_img_paths(normal_image_dir)
        self.transform = transform

    def _get_img_paths(self, img_dir: str) -> List[Path]:
        img_extension = ".png"
        img_paths = [p for p in sorted(Path(img_dir).iterdir()) if p.suffix == img_extension]
        return img_paths

    def __getitem__(self, index: int) -> Image.Image:
        path = self.img_paths[index]
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.img_paths)


def split_dataset(
    img_dir: str,
    split_ratio: float,
    transform_1: Optional[TransformType] = None,
    transform_2: Optional[TransformType] = None,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    dataset_1 = NormalDataset(img_dir, transform=transform_1)
    dataset_2 = NormalDataset(img_dir, transform=transform_2)

    num_split_data = len(dataset_1) - int(len(dataset_1) * split_ratio)

    generator = torch.Generator()
    generator.manual_seed(42)

    indices = torch.randperm(len(dataset_1), generator=generator).tolist()
    indices_1, indices_2 = indices[:num_split_data], indices[num_split_data:]

    subset_1 = Subset(dataset_1, indices_1)
    subset_2 = Subset(dataset_2, indices_2)

    return subset_1, subset_2


def build_dataset(
    dataset_path: str,
    img_size: int = 256,
) -> Tuple[
    Union[NormalDataset, torch.utils.data.Subset],
    Union[NormalDataset, torch.utils.data.Subset],
    torchvision.datasets.ImageFolder,
]:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            build_imagenet_normalization(),
        ]
    )

    train_img_dir = os.path.join(dataset_path, "train", "good")
    valid_img_dir = os.path.join(dataset_path, "validation", "good")
    test_img_dir = os.path.join(dataset_path, "test")

    if os.path.exists(valid_img_dir):
        train_dataset = NormalDataset(train_img_dir, transform=transform)
        valid_dataset = NormalDataset(valid_img_dir, transform=transform)
    else:
        train_dataset, valid_dataset = split_dataset(
            train_img_dir, split_ratio=0.15, transform_1=transform, transform_2=transform
        )
    test_dataset = torchvision.datasets.ImageFolder(root=test_img_dir, transform=transform)

    return train_dataset, valid_dataset, test_dataset


def load_ground_truth_masks(dataset_path: str, test_dataset: torchvision.datasets.ImageFolder, img_size: int = 256) -> dict:
    """加载 MVTec LOCO AD 数据集的 Ground Truth Masks（支持多文件合并）
    
    科研动机:
        - PRO 指标需要像素级的 Ground Truth 来评估定位能力
        - MVTec LOCO AD 的 mask 路径结构特殊，需要专门处理
        - ⚠️ 关键修正：部分样本有多张 mask 文件（如 000.png, 001.png...），
          分别标注不同的异常区域，必须合并后才能得到完整标注
    
    参数:
        dataset_path: 数据集根目录 (例如: .../breakfast_box)
        test_dataset: torchvision.datasets.ImageFolder 对象
        img_size: mask 需要 resize 到的尺寸
    
    返回:
        字典 {样本索引: numpy array mask (H, W)}
        只包含异常样本（good 类别无 mask）
    
    MVTec LOCO AD 特殊路径规则:
        测试图像: .../test/logical_anomalies/015.png
        对应 Mask 目录: .../ground_truth/logical_anomalies/015/
        目录内文件: 000.png, 001.png, 002.png... (可能有多个)
        
        ⚠️ 多文件合并原因:
        MVTec LOCO AD 将同一图像中的不同异常区域分别保存为独立的 mask 文件。
        例如：
        - 000.png: 标注第一个螺丝缺失
        - 001.png: 标注第二个螺丝缺失
        必须通过逻辑或操作合并所有 mask，才能得到完整的异常标注。
        否则会导致 PRO 评估和可视化时出现 False Positive（误判）。
    """
    from PIL import Image
    from glob import glob
    
    ground_truth_dir = os.path.join(dataset_path, "ground_truth")
    masks = {}
    
    # 获取 class_to_idx 的反向映射
    idx_to_class = {i: c for c, i in test_dataset.class_to_idx.items()}
    
    for sample_idx, (img_path, label) in enumerate(test_dataset.samples):
        class_name = idx_to_class[label]
        
        # good 类别没有 ground truth mask
        if class_name == "good":
            continue
        
        # 解析图像文件名（不含扩展名）
        # 例如: .../test/logical_anomalies/015.png -> "015"
        img_filename = os.path.basename(img_path)
        img_name_without_ext = os.path.splitext(img_filename)[0]
        
        # 构建 MVTec LOCO AD 的 mask 目录路径
        # 例如: ground_truth/logical_anomalies/015/
        mask_dir = os.path.join(
            ground_truth_dir,
            class_name,
            img_name_without_ext  # 使用测试图像序号作为文件夹名
        )
        
        # 检查 mask 目录是否存在
        if not os.path.exists(mask_dir):
            print(f"⚠️  警告: 未找到 mask 目录 {mask_dir}")
            continue
        
        # 扫描目录下所有 .png 文件
        # 科研说明: 使用 glob 扫描而非硬编码文件名，支持多文件场景
        mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))
        
        if len(mask_files) == 0:
            print(f"⚠️  警告: mask 目录为空 {mask_dir}")
            continue
        
        # 初始化合并后的 mask（全黑背景）
        merged_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # 遍历并合并所有 mask 文件
        # 科研说明: 使用逻辑或操作合并，确保任意一张 mask 标记的异常区域都被保留
        for mask_file in mask_files:
            # 加载单个 mask 文件
            mask = Image.open(mask_file).convert("L")  # 转为灰度图
            mask = mask.resize((img_size, img_size), Image.NEAREST)  # 最近邻插值保持二值性
            mask_array = np.array(mask)
            
            # 二值化: MVTec mask 通常是 0 或 255
            mask_binary = (mask_array > 127).astype(np.uint8)
            
            # 合并到最终 mask（逻辑或操作：有任何一张标记为异常则为异常）
            # np.maximum 等价于逻辑或，因为 mask 只有 0 和 1 两个值
            merged_mask = np.maximum(merged_mask, mask_binary)
        
        # 保存合并后的 mask
        masks[sample_idx] = merged_mask
        
        # 如果检测到多文件，输出提示信息（便于验证修复效果）
        if len(mask_files) > 1:
            print(f"  ✓ 合并 {len(mask_files)} 个 mask 文件: {class_name}/{img_name_without_ext}")
    
    return masks
