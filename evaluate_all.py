import os
import subprocess
import sys

# ================= 配置区域 =================
# 注意：这里使用了 r 前缀来表示原始字符串，避免 Windows 路径的反斜杠问题
DATASET_ROOT = r"E:\Dataset\mvtec_loco_anomaly_detection"
MODEL_ROOT = r"E:\Dataset\mvtec_loco_ad_models"
PYTHON_EXEC = sys.executable  # 获取当前环境的 python 解释器路径
# ===========================================

def run_all():
    # 1. 检查路径是否存在
    if not os.path.exists(DATASET_ROOT):
        print(f"错误: 找不到数据集根目录: {DATASET_ROOT}")
        return

    # 2. 获取根目录下所有的子文件夹（即各个类别，如 breakfast_box, juice_bottle 等）
    categories = [
        d for d in os.listdir(DATASET_ROOT) 
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ]

    if not categories:
        print("未发现任何类别文件夹，请检查路径。")
        return

    print(f"🔎 共发现 {len(categories)} 个类别，准备开始批量评估...\n")

    # 3. 循环遍历每个类别并运行命令
    for i, category in enumerate(categories, 1):
        category_path = os.path.join(DATASET_ROOT, category)
        
        print(f"[{i}/{len(categories)}] 🚀 正在评估类别: {category} ...")
        print(f"{'-'*60}")
        
        # 构造命令：python main.py [类别路径] [模型根路径]
        cmd = [PYTHON_EXEC, "main.py", category_path, MODEL_ROOT]
        
        # 调用系统命令执行
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 类别 {category} 运行出错！")
        
        print(f"{'-'*60}\n")

    print("✅ 所有类别评估已完成！")

if __name__ == "__main__":
    run_all()