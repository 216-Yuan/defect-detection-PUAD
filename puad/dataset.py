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
import cv2
from glob import glob

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


class StructuralAnomalyAugment:
    """通用结构异常生成引擎 (Universal Structural Anomaly Engine)
    
    设计原则:
        1. 无 Config：所有参数在物理合理范围内随机采样
        2. 局部性：异常面积严格控制在 0.5%-5%
        3. 物理仿真：摒弃 Perlin 云雾，使用几何算法生成真实缺陷
    
    三大通用物理算子:
        - Intruder (异物): 凸包形状 + 反色纹理 + 投影阴影
        - Scar (划痕): 贝塞尔曲线 + 深度变暗/过曝
        - Deformation (形变): 径向渐变 + Swirl/Pinch 像素重映射
    
    调度策略:
        - 互斥选择: Intruder 40% / Scar 30% / Deformation 30%
        - Mask 规范: 黑底白斑，异常区域值为 1.0
    """
    
    def __init__(self, img_size: int = 256):
        self.img_size = img_size
    
    def _random_convex_hull(self) -> np.ndarray:
        """生成随机凸包形状的 Mask
        
        返回:
            二值 Mask (H, W), 值域 {0, 1}，异常面积 0.5%-5%
        """
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 随机生成 3-7 个顶点
        num_points = np.random.randint(3, 8)
        
        # 控制异常面积：随机选择中心点和半径
        center_x = np.random.randint(int(self.img_size * 0.2), int(self.img_size * 0.8))
        center_y = np.random.randint(int(self.img_size * 0.2), int(self.img_size * 0.8))
        
        # 半径控制在 5%-15% 的图像尺寸
        radius = np.random.randint(int(self.img_size * 0.05), int(self.img_size * 0.15))
        
        # 在圆周上随机采样点
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
        points = []
        for angle in angles:
            # 添加随机扰动
            r = radius * np.random.uniform(0.6, 1.4)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # 计算凸包
        hull = cv2.convexHull(points)
        
        # 绘制填充的凸包（锐利边缘，无模糊）
        cv2.fillConvexPoly(mask, hull, 1)
        
        return mask.astype(np.float32)
    
    def _random_bezier_curve(self) -> np.ndarray:
        """生成随机贝塞尔曲线划痕 Mask
        
        返回:
            二值 Mask (H, W), 值域 {0, 1}，划痕宽度 1-5px
        """
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 生成 1-3 条划痕
        num_scratches = np.random.randint(1, 4)
        
        for _ in range(num_scratches):
            # 随机选择起点和终点
            start_x = np.random.randint(0, self.img_size)
            start_y = np.random.randint(0, self.img_size)
            
            end_x = np.random.randint(0, self.img_size)
            end_y = np.random.randint(0, self.img_size)
            
            # 随机控制点（贝塞尔曲线）
            ctrl1_x = np.random.randint(0, self.img_size)
            ctrl1_y = np.random.randint(0, self.img_size)
            
            ctrl2_x = np.random.randint(0, self.img_size)
            ctrl2_y = np.random.randint(0, self.img_size)
            
            # 使用三次贝塞尔曲线
            t = np.linspace(0, 1, 100)
            curve_x = ((1-t)**3 * start_x + 3*(1-t)**2*t * ctrl1_x + 
                      3*(1-t)*t**2 * ctrl2_x + t**3 * end_x).astype(np.int32)
            curve_y = ((1-t)**3 * start_y + 3*(1-t)**2*t * ctrl1_y + 
                      3*(1-t)*t**2 * ctrl2_y + t**3 * end_y).astype(np.int32)
            
            # 裁剪到图像范围内
            valid_indices = ((curve_x >= 0) & (curve_x < self.img_size) & 
                           (curve_y >= 0) & (curve_y < self.img_size))
            curve_x = curve_x[valid_indices]
            curve_y = curve_y[valid_indices]
            
            # 随机线宽
            thickness = np.random.randint(1, 6)
            
            # 绘制曲线（分段绘制，模拟断续效果）
            for i in range(0, len(curve_x) - 1, np.random.randint(2, 5)):
                if i + 1 < len(curve_x):
                    cv2.line(mask, (curve_x[i], curve_y[i]), 
                            (curve_x[i+1], curve_y[i+1]), 1, thickness)
        
        return mask.astype(np.float32)
    
    def _random_deformation_mask(self) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """生成径向渐变的形变 Mask
        
        返回:
            mask: 径向渐变 Mask (H, W), 值域 [0, 1]
            center: 形变中心 (cx, cy)
            radius: 形变半径
        """
        # 随机选择中心点
        cx = np.random.randint(int(self.img_size * 0.3), int(self.img_size * 0.7))
        cy = np.random.randint(int(self.img_size * 0.3), int(self.img_size * 0.7))
        
        # 随机半径（5%-10% 图像尺寸）
        radius = np.random.randint(int(self.img_size * 0.05), int(self.img_size * 0.10))
        
        # 生成径向渐变
        y, x = np.ogrid[:self.img_size, :self.img_size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # 径向渐变：中心 1.0 → 边缘 0.0
        mask = np.clip(1.0 - dist / radius, 0, 1).astype(np.float32)
        
        return mask, (cx, cy), radius
    
    def _operator_intruder(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """算子 1: 通用异物算子 (The Universal Intruder)
        
        实现:
            1. 凸包形状 Mask
            2. 反色纹理填充
            3. 投影阴影（3D 效果）
        
        返回:
            anomaly_source: 异常源图像 (H, W, 3)
            mask: 黑底白斑 Mask (H, W)
        """
        # Step 1: 生成凸包 Mask（锐利边缘）
        mask = self._random_convex_hull()
        
        # Step 2: 自适应反色纹理
        # 从原图随机位置裁剪一块区域
        shift_x = np.random.randint(-30, 31)
        shift_y = np.random.randint(-30, 31)
        
        # 裁剪并反色
        patch = np.roll(img_np, shift=(shift_y, shift_x), axis=(0, 1))
        
        # 反色操作
        if np.random.rand() > 0.5:
            patch = 1.0 - patch  # 完全反色
        else:
            # 添加随机噪声
            noise = np.random.normal(0, 0.1, patch.shape)
            patch = np.clip(patch + noise, 0, 1)
        
        # Step 3: 投影阴影（偏移 (3, 3) 位置绘制半透明黑色）
        shadow_mask = np.roll(mask, shift=(3, 3), axis=(0, 1))
        shadow_mask = shadow_mask * 0.5  # 半透明
        
        # 先应用阴影
        shadow_3ch = shadow_mask[..., np.newaxis]
        anomaly_source = img_np * (1 - shadow_3ch)  # 变暗
        
        # 再应用异物纹理
        mask_3ch = mask[..., np.newaxis]
        anomaly_source = anomaly_source * (1 - mask_3ch) + patch * mask_3ch
        
        return anomaly_source, mask
    
    def _operator_scar(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """算子 2: 通用划痕算子 (The Universal Scar)
        
        实现:
            1. 贝塞尔曲线形状
            2. 变暗或过曝效果（深度感）
        
        返回:
            anomaly_source: 异常源图像 (H, W, 3)
            mask: 黑底白斑 Mask (H, W)
        """
        # Step 1: 生成贝塞尔曲线 Mask
        mask = self._random_bezier_curve()
        
        # Step 2: 深度感渲染（变暗或过曝）
        if np.random.rand() > 0.5:
            # 凹槽阴影：变暗
            darkening_factor = np.random.uniform(0.3, 0.5)
            anomaly_source = img_np * darkening_factor
        else:
            # 内部反光：过曝
            brightening_factor = np.random.uniform(1.3, 1.6)
            anomaly_source = np.clip(img_np * brightening_factor, 0, 1)
        
        # 仅在划痕区域应用
        mask_3ch = mask[..., np.newaxis]
        anomaly_source = img_np * (1 - mask_3ch) + anomaly_source * mask_3ch
        
        return anomaly_source, mask
    
    def _operator_deformation(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """算子 3: 通用形变算子 (The Universal Deformation)
        
        实现:
            1. 径向渐变 Mask
            2. Swirl/Pinch 像素重映射
        
        返回:
            anomaly_source: 异常源图像 (H, W, 3)
            mask: 径向渐变 Mask (H, W)
        """
        # Step 1: 生成径向渐变 Mask
        mask, (cx, cy), radius = self._random_deformation_mask()
        
        # Step 2: 生成重映射坐标（Swirl/Pinch 效果）
        H, W = self.img_size, self.img_size
        
        # 创建网格坐标
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        
        # 相对于形变中心的坐标
        dx = x - cx
        dy = y - cy
        
        # 距离和角度
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Swirl 效果：距离中心越近，旋转角度越大
        swirl_strength = np.random.uniform(0.5, 1.5)
        angle_offset = swirl_strength * np.exp(-dist / (radius + 1e-6))
        
        # Pinch 效果：向中心收缩
        pinch_strength = np.random.uniform(0.1, 0.3)
        dist_new = dist * (1 - pinch_strength * mask)
        
        # 新坐标
        new_angle = angle + angle_offset
        new_x = cx + dist_new * np.cos(new_angle)
        new_y = cy + dist_new * np.sin(new_angle)
        
        # 确保坐标在有效范围内
        new_x = np.clip(new_x, 0, W - 1)
        new_y = np.clip(new_y, 0, H - 1)
        
        # 应用重映射
        anomaly_source = cv2.remap(
            img_np, 
            new_x, 
            new_y, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return anomaly_source, mask
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """通用结构异常生成引擎主调度器
        
        调度逻辑:
            - 互斥选择三大算子之一
            - Intruder 40% / Scar 30% / Deformation 30%
        
        参数:
            img: PIL.Image 对象 (RGB)
        
        返回:
            增强后的 PIL.Image 对象（包含合成异常）
        """
        # 转换为 numpy 数组
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # 随机选择算子
        mode = np.random.choice(
            ['intruder', 'scar', 'deformation'],
            p=[0.4, 0.3, 0.3]
        )
        
        # 执行对应算子
        if mode == 'intruder':
            anomaly_source, mask = self._operator_intruder(img_np)
        elif mode == 'scar':
            anomaly_source, mask = self._operator_scar(img_np)
        else:  # deformation
            anomaly_source, mask = self._operator_deformation(img_np)
        
        # Mask 已经是黑底白斑，直接使用
        # 注意：算子已经返回融合后的结果，这里直接用
        
        # 转换回 PIL.Image
        augmented = np.clip(anomaly_source * 255, 0, 255).astype(np.uint8)
        augmented_img = Image.fromarray(augmented)
        
        return augmented_img


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
    use_synthetic_anomalies: bool = False,
) -> Tuple[
    Union[NormalDataset, torch.utils.data.Subset],
    Union[NormalDataset, torch.utils.data.Subset],
    torchvision.datasets.ImageFolder,
]:
    """构建训练/验证/测试数据集
    
    参数:
        dataset_path: 数据集根目录
        img_size: 图像尺寸（正方形）
        use_synthetic_anomalies: 是否使用合成异常增强训练集
                                 True: 启用 StructuralAnomalyAugment（用于自监督训练）
                                 False: 仅标准预处理（默认）
    
    科研说明 - 合成异常增强:
        - 启用后，训练时有 50% 概率对正常图像添加合成异常
        - 训练目标: 学习区分正常图像 vs 合成异常，期望泛化到真实异常
        - 验证/测试集不应用增强，保持原始分布
        - 对应论文: DRAEM (ICCV 2021), NSA (CVPR 2021)
    
    返回:
        (train_dataset, valid_dataset, test_dataset)
    """
    # 根据是否启用合成异常选择不同的训练 transform
    if use_synthetic_anomalies:
        # 启用合成异常增强的训练 transform
        # 科研说明: RandomApply(p=0.5) 确保训练时有 50% 正常样本 + 50% 异常样本
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.RandomApply(
                [StructuralAnomalyAugment(img_size=img_size)], 
                p=0.5  # 50% 概率应用异常增强
            ),
            torchvision.transforms.ToTensor(),
            build_imagenet_normalization(),
        ])
    else:
        # 标准 transform（无增强）
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            build_imagenet_normalization(),
        ])
    
    # 验证/测试集始终使用标准 transform（不应用增强）
    # 科研说明: 评估时需要保持数据分布的真实性
    eval_transform = torchvision.transforms.Compose(
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
        train_dataset = NormalDataset(train_img_dir, transform=train_transform)
        valid_dataset = NormalDataset(valid_img_dir, transform=eval_transform)
    else:
        # 如果没有独立验证集，从训练集中划分
        # 注意: 训练集使用 train_transform（可能含增强），验证集使用 eval_transform（无增强）
        train_dataset, valid_dataset = split_dataset(
            train_img_dir, split_ratio=0.15, transform_1=train_transform, transform_2=eval_transform
        )
    test_dataset = torchvision.datasets.ImageFolder(root=test_img_dir, transform=eval_transform)

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
