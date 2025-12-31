import argparse
import os

import numpy as np
from puad.common import compute_pro
from puad.dataset import build_dataset, load_ground_truth_masks
from puad.efficientad.inference import load_efficient_ad
from puad.puad import PUAD
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PUAD")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory containing `train` and `test` (and `validation` in MVTec LOCO AD Dataset)",
    )
    parser.add_argument(
        "model_dir_path",
        type=str,
        help="Path to directory containing pretrained models",
    )
    parser.add_argument(
        "--size",
        choices=["s", "m"],
        type=str,
        default="s",
        help=(
            "Specify the size of EfficientAD used for Picturable anomaly detection "
            "and feature extraction for Unpicturable anomaly detection in either `s` or `m`"
        ),
    )
    parser.add_argument(
        "--feature_extractor",
        choices=["student", "teacher"],
        type=str,
        default="student",
        help=(
            "Specify the network in EfficientAD used for feature extraction for Unpicturable anomaly detection "
            "in either `teacher` or `student`"
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir, category = os.path.split(os.path.abspath(args.dataset_path))
    dataset_name = os.path.split(dataset_dir)[1]
    if not (
        os.path.exists(os.path.join(args.dataset_path, "train"))
        and os.path.exists(os.path.join(args.dataset_path, "test"))
    ):
        raise ValueError("The dataset specified in `dataset_path` must contain `train` and `test` directories.")
    print(f"dataset name : {dataset_name}")
    print(f"category : {category}")
    print(f"size : {args.size}")
    print(f"feature extractor : {args.feature_extractor}")

    # load EfficientAD
    efficient_ad_inference = load_efficient_ad(args.model_dir_path, args.size, dataset_name, category)

    # build dataset
    train_dataset, valid_dataset, test_dataset = build_dataset(args.dataset_path)

    # EfficientAD
    efficient_ad_auroc = efficient_ad_inference.auroc(test_dataset)
    print(f"efficient_ad auroc : {efficient_ad_auroc}")

    # PUAD
    puad = PUAD(feature_extractor=args.feature_extractor)
    puad.load_efficient_ad(efficient_ad_inference)
    puad.train(train_dataset)
    puad.valid(valid_dataset)
    puad_auroc, puad_auroc_for_anomalies = puad.auroc_for_anomalies(test_dataset)

    print(f"puad auroc : {puad_auroc}")
    for anomaly_class, auroc_for_anomaly in puad_auroc_for_anomalies.items():
        print(f"puad auroc for {anomaly_class}: {auroc_for_anomaly}")

    # ============================================================
    # PRO (Per-Region Overlap) 评估 - 评估像素级定位能力
    # ============================================================
    print("\n" + "="*60)
    print("开始计算 PRO 指标（Per-Region Overlap）...")
    print("="*60)
    
    try:
        # 加载 Ground Truth Masks（仅异常样本）
        # 科研说明: PRO 指标需要像素级 GT 来评估模型对异常区域的定位能力
        gt_masks_dict = load_ground_truth_masks(args.dataset_path, test_dataset)
        
        if len(gt_masks_dict) == 0:
            print("⚠️  未找到任何 Ground Truth Masks，跳过 PRO 计算")
        else:
            print(f"✓ 成功加载 {len(gt_masks_dict)} 个异常样本的 Ground Truth Masks")
            
            # 收集异常样本的预测异常图
            # 科研说明: 这里使用 EfficientAD 的异常图作为定位基础
            #          PUAD 的马氏距离是全局特征，无法直接映射到像素位置
            anomaly_maps_list = []
            gt_masks_list = []
            
            idx_to_class = {i: c for c, i in test_dataset.class_to_idx.items()}
            
            for sample_idx, (img, label) in enumerate(test_dataset):
                class_name = idx_to_class[label]
                
                # 只处理异常样本
                if class_name == "good" or sample_idx not in gt_masks_dict:
                    continue
                
                # 获取预测的异常图（使用 EfficientAD）
                # 注意: 这里暂时使用 EfficientAD 的输出，因为 PUAD 的全局特征无法生成像素级热图
                _, anomaly_map = efficient_ad_inference.run(img, return_map=True)
                
                anomaly_maps_list.append(anomaly_map)
                gt_masks_list.append(gt_masks_dict[sample_idx])
            
            if len(anomaly_maps_list) > 0:
                # 转换为 numpy 数组
                anomaly_maps_array = np.array(anomaly_maps_list)  # shape: (N, H, W)
                gt_masks_array = np.array(gt_masks_list)          # shape: (N, H, W)
                
                # 计算 PRO 分数
                pro_score = compute_pro(anomaly_maps_array, gt_masks_array)
                
                print(f"\n📊 PRO Score (EfficientAD): {pro_score:.4f}")
                print("   (注: PRO 分数越高表示异常区域定位能力越好)")
            else:
                print("⚠️  没有可用的异常样本进行 PRO 计算")
                
    except Exception as e:
        print(f"⚠️  PRO 计算过程中出现错误: {e}")
        print("   跳过 PRO 评估，继续执行...")
    
    print("="*60)
