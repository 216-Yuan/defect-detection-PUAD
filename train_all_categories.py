import os
import subprocess
import sys
import time

# ================= 配置区域 (根据你的实际路径确认) =================
# Python解释器路径 (如果你用了虚拟环境，这里通常不需要动，默认用当前的)
PYTHON_EXEC = sys.executable

# 1. ImageNet 路径 (你刚才下载并验证好的)
IMAGENET_PATH = r"E:\Dataset\imagenet_val"

# 2. MVTec LOCO 数据集根目录 (包含 breakfast_box, juice_bottle 等子文件夹的目录)
DATASET_ROOT = r"E:\Dataset\mvtec_loco_anomaly_detection"

# 3. 模型保存目录
MODEL_DIR = r"E:\Dataset\mvtec_loco_ad_models"

# 训练脚本的位置
TRAIN_SCRIPT = os.path.join("puad", "efficientad", "train.py")

# =================================================================

def main():
    # 检查路径是否存在
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 错误: 数据集路径不存在: {DATASET_ROOT}")
        return

    # 获取所有子文件夹作为类别名称 (过滤掉非文件夹的文件)
    categories = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    
    # 排序，保证顺序一致
    categories.sort()

    print(f"🚀 扫描到 {len(categories)} 个类别: {categories}")
    print("准备开始批量训练...")
    time.sleep(2)

    for index, category in enumerate(categories):
        print("\n" + "="*60)
        print(f"🔥 [{index+1}/{len(categories)}] 正在训练类别: {category}")
        print("="*60)

        # 构造当前类别的完整路径
        dataset_path = os.path.join(DATASET_ROOT, category)
        
        # 构造训练命令
        # 对应命令: python puad/efficientad/train.py student [ImageNet] [ModelDir] --dataset_path [SubDir]
        cmd = [
            PYTHON_EXEC, 
            TRAIN_SCRIPT, 
            "student", 
            IMAGENET_PATH, 
            MODEL_DIR, 
            "--dataset_path", dataset_path
        ]

        # 打印命令方便调试
        print(f"执行命令: {' '.join(cmd)}")

        try:
            # 开始执行训练，check=True 表示如果出错会抛出异常
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = (time.time() - start_time) / 60
            print(f"✅ 类别 {category} 训练完成！耗时: {duration:.2f} 分钟")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 类别 {category} 训练失败！")
            print(f"错误信息: {e}")
            # 如果想出错继续跑下一个，这里可以写 pass，否则建议 break
            response = input("是否继续训练下一个类别？(y/n): ")
            if response.lower() != 'y':
                break
        except KeyboardInterrupt:
            print("\n🛑 用户手动中断训练。")
            break

    print("\n🎉 所有任务处理完毕！")

if __name__ == "__main__":
    main()