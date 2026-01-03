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
    """PhysicallyGuidedCutPaste - 物理引导的异常合成引擎
    
    核心理念:
        不创造新像素，而是利用原图纹理，通过"物理规则重组"来制造异常。
        灵感来源: CutPaste (CVPR 2021) + 物理光学原理
    
    设计原则:
        1. 无 Config：所有参数在物理合理范围内随机采样
        2. 局部性：异常面积严格控制在 0.5%-5%
        3. 纹理真实：废弃几何绘图，使用原图像素重组
    
    三大物理算子:
        - Scar by Misalignment (错位裂缝): 像素平移 + 深灰填补空隙
        - Intruder by Mutation (变异异物): Gamma变换 + 翻转 + 投影阴影
        - Density Collapse (密度塌陷): Downsample + Upsample = 局部模糊
    
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
            
            # 随机线宽（极细：1-3 像素）
            thickness = np.random.randint(1, 4)
            
            # 绘制连续实线（关键修正：移除断续逻辑）
            if len(curve_x) > 1:
                points = np.column_stack([curve_x, curve_y]).astype(np.int32)
                cv2.polylines(mask, [points], isClosed=False, color=1, thickness=thickness)
        
        return mask.astype(np.float32)
    
    def _operator_intruder(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """算子 B: Solid Material Intruder (实体异物)
        
        物理原理: 模拟塑料片/金属碎片等实体异物
        
        实现逻辑:
            1. Mask 生成: 随机凸包
            2. 材质生成: 纯色块 + 高斯噪声 (sigma=20)
            3. 融合: 不透明覆盖
            4. 投影: Drop Shadow 增加立体感
        
        效果: 类似 MVTec LOCO 中的蓝色塑料片异物
        
        返回:
            anomaly_source: 异常源图像 (H, W, 3)
            mask: 黑底白斑 Mask (H, W)
        """
        H, W = self.img_size, self.img_size
        
        # Step 1: 生成凸包 Mask
        mask = self._random_convex_hull()
        
        # Step 2: 材质生成（完全生成，不使用原图）
        # 随机选择纯色（蓝/黑/白/灰/褐）
        color_palette = [
            [0.2, 0.3, 0.8],    # 蓝色塑料
            [0.1, 0.1, 0.1],    # 黑色
            [0.9, 0.9, 0.9],    # 白色
            [0.5, 0.5, 0.5],    # 灰色
            [0.4, 0.3, 0.2],    # 褐色
        ]
        base_color = color_palette[np.random.randint(0, len(color_palette))]
        
        # 创建纯色块
        material = np.ones((H, W, 3), dtype=np.float32)
        material[:, :, 0] = base_color[0]
        material[:, :, 1] = base_color[1]
        material[:, :, 2] = base_color[2]
        
        # 叠加高斯噪声（模拟表面粗糙度）
        noise = np.random.normal(0, 20, (H, W, 3)).astype(np.float32) / 255.0
        material = np.clip(material + noise, 0, 1)
        
        # Step 3: 融合（不透明覆盖）
        mask_3ch = mask[..., np.newaxis]
        anomaly_source = img_np * (1 - mask_3ch) + material * mask_3ch
        
        # Step 4: 投影 (Drop Shadow)
        # 创建偏移的阴影
        shadow_mask = mask.copy()
        shadow_mask_blurred = cv2.GaussianBlur(shadow_mask, (11, 11), 0)
        
        # 投影偏移 (向右下偏移 3px)
        shadow_offset = 3
        shadow_layer = np.zeros((H, W), dtype=np.float32)
        
        src_y_end = H - shadow_offset
        src_x_end = W - shadow_offset
        dst_y_start = shadow_offset
        dst_x_start = shadow_offset
        
        shadow_layer[dst_y_start:, dst_x_start:] = \
            shadow_mask_blurred[:src_y_end, :src_x_end]
        
        # 应用投影 (变暗 50%)
        shadow_3ch = shadow_layer[..., np.newaxis]
        anomaly_source = anomaly_source * (1 - 0.5 * shadow_3ch)
        
        return anomaly_source, mask
    
    def _operator_scar(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """算子 A: Elastic Distortion (弹性畸变)
        
        物理原理: 裂缝是空间的挤压，不是"贴图"，而是"推像素"
        
        实现逻辑:
            1. 生成路径: 绘制贝塞尔曲线
            2. 生成位移场: 在曲线位置赋予随机位移向量，高斯模糊使其平滑衰减
            3. 应用畸变: cv2.remap 实现像素重映射
            4. 深度渲染: 裂缝中心叠加深色
        
        效果: 纹理像果冻一样被挤歪，完全没有"贴纸感"
        
        返回:
            anomaly_source: 异常源图像 (H, W, 3)
            mask: 黑底白斑 Mask (H, W)
        """
        H, W = self.img_size, self.img_size
        
        # Step 1: 生成贝塞尔曲线 Mask
        mask = self._random_bezier_curve()
        
        # 膨胀宽度
        line_width = np.random.randint(3, 7)
        kernel = np.ones((line_width, line_width), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Step 2: 生成位移场 (Displacement Field)
        # 初始化空白位移场
        dx = np.zeros((H, W), dtype=np.float32)
        dy = np.zeros((H, W), dtype=np.float32)
        
        # 在曲线位置赋予随机位移向量
        ys, xs = np.where(mask_dilated > 0.5)
        
        if len(ys) == 0:
            return img_np, mask_dilated
        
        # 随机位移方向和强度
        displacement_strength = np.random.uniform(4, 8)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # 在裂缝位置设置位移
        dx[ys, xs] = displacement_strength * np.cos(angle)
        dy[ys, xs] = displacement_strength * np.sin(angle)
        
        # 高斯模糊使位移场平滑衰减（模拟应力传导）
        blur_sigma = line_width * 2 + 1
        if blur_sigma % 2 == 0:
            blur_sigma += 1
        dx = cv2.GaussianBlur(dx, (blur_sigma, blur_sigma), 0)
        dy = cv2.GaussianBlur(dy, (blur_sigma, blur_sigma), 0)
        
        # Step 3: 应用畸变 (cv2.remap)
        # 创建基础网格
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)
        
        # 计算重映射坐标
        map_x = grid_x + dx
        map_y = grid_y + dy
        
        # 确保坐标在有效范围内
        map_x = np.clip(map_x, 0, W - 1)
        map_y = np.clip(map_y, 0, H - 1)
        
        # 应用像素重映射（关键修复：使用 BORDER_REFLECT_101 消除黑边）
        anomaly_source = cv2.remap(
            img_np, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101  # 关键：镜像填充
        )
        
        # Step 4: 深度渲染（裂缝中心变暗）
        # 在裂缝中心叠加深色
        darken_mask = cv2.GaussianBlur(mask_dilated, (5, 5), 0)
        darken_factor = np.random.uniform(0.75, 0.9)
        darken_3ch = darken_mask[..., np.newaxis]
        anomaly_source = anomaly_source * (1 - darken_3ch * (1 - darken_factor))
        
        return anomaly_source, mask_dilated
    
    def _operator_deformation(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """算子 C: Shadow Blur (暗影模糊)
        
        物理原理: 凹陷 = 模糊 (Blur) + 阴影 (Shadow)
        
        实现逻辑:
            1. Mask 生成: 随机生成一个圆形 Mask
            2. 强力模糊: cv2.GaussianBlur (kernel=7~11)
            3. 叠加阴影: roi_dark = roi_blurred * 0.6~0.8
            4. 融合: 贴回原位
        
        效果: 模拟凹陷处的光学失焦 + 光照不足
        
        返回:
            anomaly_source: 异常源图像 (H, W, 3)
            mask: 径向渐变 Mask (H, W)
        """
        H, W = self.img_size, self.img_size
        
        # Step 1: 生成圆形 Mask
        cx = np.random.randint(int(W * 0.25), int(W * 0.75))
        cy = np.random.randint(int(H * 0.25), int(H * 0.75))
        
        # 半径范围：12% ~ 20% 的图像尺寸（更大更明显）
        radius = np.random.randint(int(self.img_size * 0.12), int(self.img_size * 0.20))
        
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # 圆形二值 Mask
        circle_mask = (dist <= radius).astype(np.float32)
        
        # 径向渐变（中心 1.0 -> 边缘 0.0）
        grad_mask = np.clip(1.0 - dist / radius, 0, 1).astype(np.float32)
        grad_mask = grad_mask * circle_mask
        
        # Step 2: 提取 ROI
        y_min = max(0, cy - radius)
        y_max = min(H, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(W, cx + radius + 1)
        
        roi = img_np[y_min:y_max, x_min:x_max].copy()
        roi_h, roi_w = roi.shape[:2]
        
        if roi_h > 4 and roi_w > 4:
            # Step 3: 强力高斯模糊 (kernel=7~11)
            blur_kernel = np.random.choice([7, 9, 11])
            roi_blurred = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)
            
            # Step 4: 叠加阴影（变暗）
            shadow_factor = np.random.uniform(0.6, 0.8)
            roi_dark = roi_blurred * shadow_factor
            
            # Step 5: 融合（使用径向渐变实现软边缘）
            roi_mask = grad_mask[y_min:y_max, x_min:x_max]
            roi_mask_3ch = roi_mask[..., np.newaxis]
            
            blended_roi = roi * (1 - roi_mask_3ch) + roi_dark * roi_mask_3ch
            
            # 贴回原图
            anomaly_source = img_np.copy()
            anomaly_source[y_min:y_max, x_min:x_max] = blended_roi
        else:
            anomaly_source = img_np.copy()
        
        return anomaly_source, grad_mask
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """PhysicallyGuidedCutPaste 主调度器
        
        调度逻辑:
            - 互斥选择三大创新算子之一
            - Intruder(变异异物) 40% / Scar(错位裂缝) 30% / Deformation(密度塌陷) 30%
        
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
