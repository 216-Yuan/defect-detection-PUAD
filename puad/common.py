import numpy as np
from skimage import measure
import torchvision


def build_imagenet_normalization() -> torchvision.transforms.Normalize:
    # mu and sigma are based on the values used in pytorch official for Wide_ResNet101_2:
    # https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet101_2.html
    imagenet_mu = [0.485, 0.456, 0.406]
    imagenet_sigma = [0.229, 0.224, 0.225]
    return torchvision.transforms.Normalize(mean=imagenet_mu, std=imagenet_sigma)


def compute_pro(anomaly_maps: np.ndarray, ground_truth_masks: np.ndarray, num_thresholds: int = 200) -> float:
    """计算 PRO (Per-Region Overlap) 指标
    
    PRO 指标用于评估异常检测模型的定位能力（Localization Performance）。
    与像素级 AUROC 不同，PRO 考虑了 Ground Truth 中每个连通域（异常区域）的检测情况。
    
    科研动机:
        - 像素级 AUROC 对大面积异常敏感，可能忽略小区域异常的检测性能
        - PRO 通过对每个独立异常区域计算覆盖率，更公平地评估定位能力
        - 适用于评估结构性异常检测（Picturable Anomaly Detection）
    
    参数:
        anomaly_maps: shape (N, H, W)，预测的异常图，值越大表示越异常
        ground_truth_masks: shape (N, H, W)，二值化的 GT mask，1 表示异常区域
        num_thresholds: 阈值采样数量，用于计算积分
    
    返回:
        PRO 分数 (0~1)，越高表示定位能力越好
    
    算法流程:
        1. 在 [0, 1] 之间均匀采样多个阈值
        2. 对每个阈值，将预测图二值化
        3. 提取 GT 中的每个连通域（独立异常区域）
        4. 计算该连通域与预测的重叠比例（Overlap Ratio）
        5. 对所有连通域求平均，得到该阈值下的 PRO
        6. 对所有阈值积分，得到最终 PRO 分数
    """
    assert anomaly_maps.shape == ground_truth_masks.shape, "预测图和GT mask的形状必须一致"
    
    # 归一化异常图到 [0, 1]
    # 科研注意: 使用全局 min/max 归一化，确保不同样本之间的阈值具有可比性
    anomaly_maps_normalized = (anomaly_maps - anomaly_maps.min()) / (anomaly_maps.max() - anomaly_maps.min() + 1e-8)
    
    # 在 [0, 1] 之间均匀采样阈值
    thresholds = np.linspace(0, 1, num_thresholds)
    pros = []
    
    for threshold in thresholds:
        # 对预测图进行二值化
        binary_predictions = (anomaly_maps_normalized >= threshold).astype(np.uint8)
        
        # 对每个样本计算 PRO
        overlaps_per_sample = []
        for i in range(len(ground_truth_masks)):
            gt_mask = ground_truth_masks[i]
            pred_mask = binary_predictions[i]
            
            # 提取 GT 中的连通域（每个独立的异常区域）
            # connectivity=2 表示 8-连通
            labeled_gt, num_components = measure.label(gt_mask, connectivity=2, return_num=True)
            
            if num_components == 0:
                # 该样本无异常区域，跳过
                continue
            
            # 对每个连通域计算与预测的重叠率
            overlaps = []
            for component_id in range(1, num_components + 1):
                # 提取当前连通域
                component_mask = (labeled_gt == component_id)
                component_area = component_mask.sum()
                
                if component_area == 0:
                    continue
                
                # 计算该连通域与预测的交集
                intersection = (component_mask & pred_mask).sum()
                overlap_ratio = intersection / component_area
                overlaps.append(overlap_ratio)
            
            if len(overlaps) > 0:
                # 对该样本的所有连通域求平均
                overlaps_per_sample.append(np.mean(overlaps))
        
        if len(overlaps_per_sample) > 0:
            # 对所有样本求平均，得到该阈值下的 PRO
            pros.append(np.mean(overlaps_per_sample))
        else:
            pros.append(0.0)
    
    # 使用梯形法则计算曲线下面积（积分）
    # 这给出了 0~1 阈值范围内的平均 PRO
    pro_score = np.trapz(pros, thresholds)
    
    return pro_score
