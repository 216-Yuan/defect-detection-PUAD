"""
视觉图灵测试 (Visual Sanity Check) - 四联图学术级验证脚本

科研目标:
    通过 2×2 布局的对比图验证合成异常的质量，确保满足以下标准：
    1. 软边缘羽化效果（Mask 热力图显示平滑渐变）
    2. 纹理不连续感（旋转模式产生方向冲突）
    3. 异常源与原图的显著差异（物理破损感）
    4. 样本多样性（10 次生成位置/形状各异）

输出结构:
    - 单样本四联图（2×2）：原图 | Mask | 异常源 | 合成图
    - 汇总图（3×4）：原图 + 10 个合成结果
"""

# 修复 OpenMP 库冲突（必须在导入 numpy/torch/cv2 之前）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下使用
import cv2

from puad.dataset import StructuralAnomalyAugment

# ========================================
# 配置区 (Configuration)
# ========================================

# ⚠️ 重要: 请修改为您本地的正常训练图片路径
# Windows 路径示例: Path(r"E:\Dataset\mvtec_loco_anomaly_detection\breakfast_box\train\good\000.png")
INPUT_IMAGE_PATH = Path(r"E:\Dataset\mvtec_loco_anomaly_detection\breakfast_box\train\good\000.png")

# 输出目录
OUTPUT_DIR = Path("debug_results")

# 生成增强样本数量
NUM_AUGMENTATIONS = 10

# 强制模式分布（调试用）
# True = 前10张强制覆盖所有模式，False = 随机模式
FORCE_MODE_DISTRIBUTION = True

# 强制模式调度表（仅在 FORCE_MODE_DISTRIBUTION=True 时生效）
MODE_SCHEDULE = [
    'intruder', 'intruder', 'intruder', 'intruder',  # 前4张：异物
    'scar', 'scar', 'scar',                          # 中3张：划痕
    'deformation', 'deformation', 'deformation'      # 后3张：形变
]

# 随机种子 (默认 None = 每次运行结果不同)
# 科研说明: 设置为固定值 (如 42) 可用于复现实验结果，便于论文图表生成
RANDOM_SEED = None  # 或设置为 42 以启用可复现模式

# 图像尺寸
IMG_SIZE = 256

# ========================================
# 辅助函数 (Helper Functions)
# ========================================

def set_random_seed(seed):
    """设置全局随机种子"""
    if seed is not None:
        np.random.seed(seed)
        print(f"🔒 随机种子: {seed} (可复现模式)")
    else:
        print("🎲 随机种子: None (每次不同)")


def apply_augmentation_with_decomposition(augmentor, img, forced_mode=None):
    """手动调用增强流程并分解中间结果
    
    参数:
        augmentor: StructuralAnomalyAugment 实例
        img: PIL.Image
        forced_mode: str | None, 强制指定模式（'intruder'/'scar'/'deformation'）
    
    返回:
        augmented_img: PIL.Image, 合成结果
        mask: np.ndarray (H, W), Mask
        anomaly_source: np.ndarray (H, W, 3), 异常源内容
        mode: str, 使用的模式名称
    """
    # 转换为 numpy 数组
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 随机或强制选择模式
    if forced_mode is not None:
        mode = forced_mode
    else:
        mode = np.random.choice(
            ['intruder', 'scar', 'deformation'], 
            p=[0.4, 0.3, 0.3]
        )
    
    # 生成异常源和 Mask
    if mode == 'intruder':
        anomaly_source, mask = augmentor._operator_intruder(img_np)
    elif mode == 'scar':
        anomaly_source, mask = augmentor._operator_scar(img_np)
    else:  # deformation
        anomaly_source, mask = augmentor._operator_deformation(img_np)
    
    # 转换回 PIL.Image（anomaly_source 已经是融合后的结果）
    augmented = np.clip(anomaly_source * 255, 0, 255).astype(np.uint8)
    augmented_img = Image.fromarray(augmented)
    
    return augmented_img, mask, anomaly_source, mode


def save_quadrant_comparison(original_img, mask, anomaly_source, augmented_img, save_path, mode, sample_idx):
    """保存 2×2 四联图对比
    
    布局:
        [0,0] 原始图像     | [0,1] 软边缘 Mask (Jet)
        [1,0] 异常源内容   | [1,1] 最终合成图
    
    参数:
        original_img: PIL.Image
        mask: np.ndarray (H, W), [0, 1]
        anomaly_source: np.ndarray (H, W, 3), [0, 1]
        augmented_img: PIL.Image
        save_path: Path
        mode: str, 模式名称
        sample_idx: int, 样本序号
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # [0, 0] 原始图像
    axes[0, 0].imshow(np.array(original_img))
    axes[0, 0].set_title('(1) Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # [0, 1] 黑底白斑 Mask（灰度图）+ 显示最大值和覆盖率
    mask_max = mask.max()
    mask_coverage = (mask > 0.1).sum() / (mask.shape[0] * mask.shape[1]) * 100
    im = axes[0, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'(2) Mask (Max: {mask_max:.2f}, Area: {mask_coverage:.1f}%)', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # [1, 0] 异常源内容（已融合）
    anomaly_source_uint8 = (anomaly_source * 255).astype(np.uint8)
    axes[1, 0].imshow(anomaly_source_uint8)
    axes[1, 0].set_title(f'(3) Anomaly Source [{mode}]', fontsize=14, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # [1, 1] 最终合成图
    axes[1, 1].imshow(np.array(augmented_img))
    axes[1, 1].set_title('(4) Synthesized Anomaly', fontsize=14, fontweight='bold', color='green')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Sample #{sample_idx} - Mode: {mode}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_sheet(original_img, augmented_images, save_path):
    """生成 3×4 汇总图（原图 + 10 个合成结果）
    
    布局:
        [Original] [Aug 1] [Aug 2] [Aug 3]
        [Aug 4]    [Aug 5] [Aug 6] [Aug 7]
        [Aug 8]    [Aug 9] [Aug 10] [Empty]
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # 第一格：原始图像
    axes[0].imshow(np.array(original_img))
    axes[0].set_title('Original', fontsize=12, fontweight='bold', color='blue')
    axes[0].axis('off')
    
    # 后续格子：增强图像
    for i, aug_img in enumerate(augmented_images, start=1):
        if i < len(axes):
            axes[i].imshow(np.array(aug_img))
            axes[i].set_title(f'Aug #{i}', fontsize=10)
            axes[i].axis('off')
    
    # 隐藏多余格子
    for i in range(len(augmented_images) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Augmentation Diversity Summary Sheet', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ========================================
# 主流程 (Main Pipeline)
# ========================================

def main():
    print("=" * 60)
    print("🔬 Visual Sanity Check - 四联图学术级验证")
    print("=" * 60)
    
    # 1. 设置随机种子
    set_random_seed(RANDOM_SEED)
    
    # 2. 检查输入路径
    if not INPUT_IMAGE_PATH.exists():
        print(f"❌ 错误: 输入图像不存在!")
        print(f"   路径: {INPUT_IMAGE_PATH}")
        print(f"   请修改脚本中的 INPUT_IMAGE_PATH")
        return
    
    print(f"📂 输入图像: {INPUT_IMAGE_PATH}")
    
    # 3. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {OUTPUT_DIR.resolve()}")
    
    # 4. 加载原始图像
    print(f"\n⏳ 加载原始图像...")
    original_img = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    original_img = original_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    original_img.save(OUTPUT_DIR / "0_original.png")
    print(f"   ✅ 原始图像: {original_img.size}")
    
    # 5. 实例化增强器
    print(f"\n🔧 初始化 StructuralAnomalyAugment...")
    augmentor = StructuralAnomalyAugment(img_size=IMG_SIZE)
    print(f"   通用结构异常生成引擎（无 Config）")
    print(f"   三大算子: Intruder 40% / Scar 30% / Deformation 30%")
    
    if FORCE_MODE_DISTRIBUTION:
        print(f"   ⚠️ 强制模式分布: 启用（调试模式）")
    else:
        print(f"   随机模式分布: 启用（标准模式）")
    
    # 6. 循环生成增强样本
    print(f"\n🎨 生成 {NUM_AUGMENTATIONS} 个增强样本...")
    augmented_images = []
    mode_statistics = {'intruder': 0, 'scar': 0, 'deformation': 0}
    
    for i in range(1, NUM_AUGMENTATIONS + 1):
        # 确定使用的模式
        if FORCE_MODE_DISTRIBUTION and i <= len(MODE_SCHEDULE):
            forced_mode = MODE_SCHEDULE[i - 1]
        else:
            forced_mode = None
        
        # 应用增强并分解中间结果
        augmented_img, mask, anomaly_source, mode = apply_augmentation_with_decomposition(
            augmentor, original_img, forced_mode=forced_mode
        )
        augmented_images.append(augmented_img)
        
        # 统计模式
        mode_key = mode.split()[0]  # 处理 "colorjitter (fallback)"
        if mode_key in mode_statistics:
            mode_statistics[mode_key] += 1
        
        # 保存四联图
        quad_path = OUTPUT_DIR / f"{i}_quadrant.png"
        save_quadrant_comparison(
            original_img, mask, anomaly_source, augmented_img, 
            quad_path, mode, i
        )
        
        # 保存纯合成图
        aug_only_path = OUTPUT_DIR / f"{i}_augmented_only.png"
        augmented_img.save(aug_only_path)
        
        print(f"   [{i:2d}/{NUM_AUGMENTATIONS}] ✅ Mode: {mode:20s} | {i}_quadrant.png")
    
    # 7. 生成汇总图
    print(f"\n📊 生成汇总图...")
    summary_path = OUTPUT_DIR / "summary_sheet.png"
    generate_summary_sheet(original_img, augmented_images, summary_path)
    print(f"   ✅ 汇总图: summary_sheet.png")
    
    # 8. 输出统计信息
    print("\n" + "=" * 60)
    print("📈 模式分布统计:")
    print("=" * 60)
    for mode, count in mode_statistics.items():
        percentage = (count / NUM_AUGMENTATIONS) * 100
        print(f"   {mode:15s}: {count:2d} / {NUM_AUGMENTATIONS} ({percentage:5.1f}%)")
    
    # 9. 验证清单
    print("\n" + "=" * 60)
    print("✅ 生成完成！请进行人工视觉检查:")
    print("=" * 60)
    print("📋 验证清单 (Visual Checklist):")
    print("   1. [ ] Mask 黑底白斑清晰可见（占比 < 5%）?")
    print("   2. [ ] Intruder: 凸包形状 + 反色纹理 + 投影阴影?")
    print("   3. [ ] Scar: 贝塞尔曲线 + 深度变暗/过曝效果?")
    print("   4. [ ] Deformation: 局部凹陷/扭曲变形可见?")
    print("   5. [ ] 异常区域边缘锐利（无高斯模糊云雾）?")
    print("   6. [ ] 合成图有突兀的物理缺陷感?")
    print("   7. [ ] 10 个样本的位置/形状/类型各不相同?")
    print(f"\n📂 检查文件:")
    print(f"   - 四联图: {OUTPUT_DIR.resolve()}\\*_quadrant.png")
    print(f"   - 汇总图: {OUTPUT_DIR.resolve()}\\summary_sheet.png")
    print("\n💡 提示: 设置 RANDOM_SEED = 42 可启用复现模式")
    print("=" * 60)


if __name__ == "__main__":
    main()
