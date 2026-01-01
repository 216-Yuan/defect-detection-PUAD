"""
EfficientAD 异常检测可视化脚本（批量全类别模式）

功能:
    1. 自动扫描数据集根目录，发现所有类别
    2. 批量加载 EfficientAD 模型并对测试集进行推理
    3. 生成包含原图、GT Mask、Heatmap、Overlay 的可视化对比图
    4. 保存可视化结果到分类别的目录结构

科研动机:
    - 批量可视化所有类别，全面评估模型在不同场景下的表现
    - 对比 Ground Truth 和预测热力图，发现模型的优势和不足
    - 为论文撰写提供全量的可视化素材
    - 支持模型缺失时的容错处理，便于逐步训练和测试
"""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from puad.dataset import build_dataset, load_ground_truth_masks
from puad.efficientad.inference import load_efficient_ad
from puad.common import build_imagenet_normalization


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """反归一化图像，从标准化张量转换为 RGB 图像数组
    
    参数:
        img_tensor: 归一化后的图像张量 (C, H, W)
    
    返回:
        RGB 图像数组 (H, W, 3), 值范围 [0, 255]
    """
    # ImageNet 归一化参数
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 转换为 numpy 并反归一化
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img = img * std + mean  # 反归一化
    img = np.clip(img * 255, 0, 255).astype(np.uint8)  # 转换到 [0, 255]
    
    return img


def apply_colormap_on_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """将异常热力图应用 Jet colormap
    
    参数:
        heatmap: 异常热力图 (H, W), 值范围 [0, 1] 或任意
    
    返回:
        RGB 彩色热力图 (H, W, 3), 值范围 [0, 255]
    """
    # 归一化到 [0, 255]
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
    
    # 应用 Jet colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return heatmap_colored


def create_overlay(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将热力图叠加到原图上
    
    参数:
        original_img: 原始 RGB 图像 (H, W, 3)
        heatmap: 异常热力图 (H, W)
        alpha: 热力图透明度，0=完全透明，1=完全不透明
    
    返回:
        叠加后的 RGB 图像 (H, W, 3)
    """
    # 应用 colormap
    heatmap_colored = apply_colormap_on_heatmap(heatmap)
    
    # 叠加
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_sample(
    img_tensor: torch.Tensor,
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray,
    anomaly_score: float,
    save_path: str,
    sample_name: str,
    class_name: str
):
    """生成并保存单个样本的可视化对比图
    
    参数:
        img_tensor: 归一化后的图像张量 (C, H, W)
        anomaly_map: 预测的异常热力图 (H, W)
        gt_mask: Ground Truth 掩码 (H, W), 二值图 {0, 1}
        anomaly_score: 异常分数
        save_path: 保存路径
        sample_name: 样本名称（如 "000.png"）
        class_name: 类别名称（如 "logical_anomalies"）
    """
    # 反归一化原图
    original_img = denormalize_image(img_tensor)
    
    # 创建叠加图
    overlay_img = create_overlay(original_img, anomaly_map, alpha=0.5)
    
    # 应用 colormap 到热力图
    heatmap_colored = apply_colormap_on_heatmap(anomaly_map)
    
    # 创建 matplotlib 图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'{class_name} - {sample_name}\nAnomaly Score: {anomaly_score:.4f}', 
                 fontsize=16, fontweight='bold')
    
    # (1) 原图
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('(1) Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # (2) GT Mask
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('(2) Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # (3) Heatmap
    axes[1, 0].imshow(heatmap_colored)
    axes[1, 0].set_title('(3) Predicted Heatmap (Jet)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # (4) Overlay
    axes[1, 1].imshow(overlay_img)
    axes[1, 1].set_title('(4) Overlay (Heatmap on Original)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {save_path}")


def discover_categories(dataset_root: str) -> list:
    """自动发现数据集根目录下的所有类别
    
    参数:
        dataset_root: 数据集根目录路径
    
    返回:
        类别名称列表
    
    科研动机:
        - 自动化处理流程，避免硬编码类别名称
        - 支持数据集扩展，新增类别无需修改代码
    """
    if not os.path.exists(dataset_root):
        raise ValueError(f"数据集根目录不存在: {dataset_root}")
    
    categories = []
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        
        # 过滤条件：必须是目录，且包含 test 子目录
        if os.path.isdir(item_path) and not item.startswith('.'):
            test_dir = os.path.join(item_path, "test")
            if os.path.exists(test_dir):
                categories.append(item)
    
    return sorted(categories)


def process_category(
    category: str,
    dataset_root: str,
    model_dir_path: str,
    output_root: str,
    dataset_name: str = "mvtec_loco_anomaly_detection",
    size: str = "s",
    img_size: int = 256,
    max_samples_per_class: int = 10,
    device: str = "cuda"
) -> dict:
    """处理单个类别的可视化
    
    参数:
        category: 类别名称（如 "breakfast_box"）
        dataset_root: 数据集根目录
        model_dir_path: 模型根目录
        output_root: 输出根目录
        dataset_name: 数据集名称
        size: 模型尺寸 ("s" or "m")
        img_size: 图像尺寸
        max_samples_per_class: 每个异常子类别最多采样数量
        device: 计算设备
    
    返回:
        统计信息字典 {"success": bool, "visualized": int, "message": str}
    
    科研动机:
        - 将单类别处理逻辑封装，便于批量调用和错误处理
        - 支持独立运行和容错，某个类别失败不影响其他类别
    """
    print(f"\n{'='*60}")
    print(f"🔥 正在处理类别: {category}")
    print(f"{'='*60}")
    
    dataset_path = os.path.join(dataset_root, category)
    category_output_dir = os.path.join(output_root, category)
    
    try:
        # ========== 加载模型 ==========
        print(f"🔧 正在加载 EfficientAD 模型...")
        efficient_ad = load_efficient_ad(
            model_dir_path=model_dir_path,
            size=size,
            dataset_name=dataset_name,
            category=category,
            img_size=img_size,
            device=device
        )
        print(f"✅ 模型加载成功！")
        
    except FileNotFoundError as e:
        error_msg = f"⚠️  模型文件不存在，跳过类别 {category}"
        print(error_msg)
        print(f"   错误详情: {e}")
        return {"success": False, "visualized": 0, "message": error_msg}
    
    except Exception as e:
        error_msg = f"❌ 模型加载失败，跳过类别 {category}"
        print(error_msg)
        print(f"   错误详情: {e}")
        return {"success": False, "visualized": 0, "message": error_msg}
    
    try:
        # ========== 加载数据集 ==========
        print(f"📂 正在加载测试数据集...")
        _, _, test_dataset = build_dataset(dataset_path, img_size=img_size)
        print(f"✅ 测试集加载成功！总样本数: {len(test_dataset)}")
        
        # ========== 加载 Ground Truth Masks ==========
        print(f"🗺️  正在加载 Ground Truth Masks...")
        gt_masks_dict = load_ground_truth_masks(dataset_path, test_dataset, img_size=img_size)
        print(f"✅ 成功加载 {len(gt_masks_dict)} 个 GT masks")
        
        # ========== 按类别组织样本 ==========
        idx_to_class = {i: c for c, i in test_dataset.class_to_idx.items()}
        class_samples = {}  # {class_name: [(sample_idx, img_path, label), ...]}
        
        for sample_idx, (img_path, label) in enumerate(test_dataset.samples):
            class_name = idx_to_class[label]
            if class_name == "good":
                continue  # 跳过正常样本
            
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append((sample_idx, img_path, label))
        
        # ========== 可视化每个子类别的前 N 个样本 ==========
        print(f"🎨 开始可视化...")
        total_visualized = 0
        
        for class_name, samples in class_samples.items():
            print(f"\n📊 处理子类别: {class_name}")
            
            # 创建子类别输出目录
            class_output_dir = os.path.join(category_output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            # 抽取前 N 个样本
            selected_samples = samples[:max_samples_per_class]
            
            for sample_idx, img_path, label in selected_samples:
                # 获取样本名称
                sample_name = os.path.basename(img_path)
                
                # 检查是否有 GT mask
                if sample_idx not in gt_masks_dict:
                    print(f"  ⚠️  跳过 {sample_name}: 无 GT mask")
                    continue
                
                # 加载图像
                img, _ = test_dataset[sample_idx]
                
                # 推理获取异常图
                anomaly_score, anomaly_map = efficient_ad.run(img, return_map=True)
                
                # 获取 GT mask
                gt_mask = gt_masks_dict[sample_idx]
                
                # 生成保存路径
                save_filename = os.path.splitext(sample_name)[0] + "_vis.png"
                save_path = os.path.join(class_output_dir, save_filename)
                
                # 可视化并保存
                visualize_sample(
                    img_tensor=img,
                    anomaly_map=anomaly_map,
                    gt_mask=gt_mask,
                    anomaly_score=anomaly_score,
                    save_path=save_path,
                    sample_name=sample_name,
                    class_name=class_name
                )
                
                total_visualized += 1
            
            print(f"  ✅ 子类别 {class_name} 完成，可视化 {len(selected_samples)} 个样本")
        
        success_msg = f"类别 {category} 完成，共生成 {total_visualized} 张图片"
        print(f"\n✅ {success_msg}")
        return {"success": True, "visualized": total_visualized, "message": success_msg}
        
    except Exception as e:
        error_msg = f"❌ 处理类别 {category} 时出错"
        print(error_msg)
        print(f"   错误详情: {e}")
        return {"success": False, "visualized": 0, "message": error_msg}


def main():
    """主函数：批量可视化所有类别的 EfficientAD 异常检测结果"""
    
    # ========== 全局配置参数 ==========
    DATASET_ROOT = r"E:\Dataset\mvtec_loco_anomaly_detection"  # 数据集根目录
    MODEL_DIR_PATH = r"E:\Dataset\mvtec_loco_ad_models"  # 模型根目录
    OUTPUT_ROOT = "vis_results_all"  # 输出根目录
    DATASET_NAME = "mvtec_loco_anomaly_detection"
    SIZE = "s"
    IMG_SIZE = 256
    MAX_IMAGES_PER_TYPE = 10  # 每个异常子类别最多采样数量
    
    print("="*60)
    print("🎨 EfficientAD 异常检测批量可视化（全类别模式）")
    print("="*60)
    print(f"📁 数据集根目录: {DATASET_ROOT}")
    print(f"🤖 模型根目录: {MODEL_DIR_PATH}")
    print(f"💾 输出根目录: {OUTPUT_ROOT}")
    print(f"🔢 每类采样上限: {MAX_IMAGES_PER_TYPE}")
    print("="*60)
    
    # ========== 自动发现所有类别 ==========
    print("\n🔎 正在扫描数据集类别...")
    try:
        categories = discover_categories(DATASET_ROOT)
        print(f"✅ 发现 {len(categories)} 个类别: {categories}")
    except Exception as e:
        print(f"❌ 扫描数据集失败: {e}")
        return
    
    if len(categories) == 0:
        print("⚠️  未发现任何有效类别，退出程序")
        return
    
    # ========== 创建输出根目录 ==========
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # ========== 确定计算设备 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 使用计算设备: {device}")
    
    # ========== 批量处理所有类别 ==========
    print("\n🚀 开始批量处理...")
    results = []
    total_visualized = 0
    
    for idx, category in enumerate(categories, 1):
        print(f"\n[{idx}/{len(categories)}]")
        
        result = process_category(
            category=category,
            dataset_root=DATASET_ROOT,
            model_dir_path=MODEL_DIR_PATH,
            output_root=OUTPUT_ROOT,
            dataset_name=DATASET_NAME,
            size=SIZE,
            img_size=IMG_SIZE,
            max_samples_per_class=MAX_IMAGES_PER_TYPE,
            device=device
        )
        
        results.append({"category": category, **result})
        if result["success"]:
            total_visualized += result["visualized"]
    
    # ========== 输出汇总统计 ==========
    print("\n" + "="*60)
    print("📊 批量处理完成！汇总统计：")
    print("="*60)
    
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count
    
    print(f"✅ 成功处理: {success_count}/{len(results)} 个类别")
    print(f"❌ 失败/跳过: {failed_count}/{len(results)} 个类别")
    print(f"🎨 总可视化图片: {total_visualized} 张")
    print(f"📁 结果保存在: {os.path.abspath(OUTPUT_ROOT)}")
    
    # 详细结果列表
    print(f"\n{'='*60}")
    print("详细结果：")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['category']}: {r['message']}")
    
    print("="*60)


if __name__ == "__main__":
    main()
