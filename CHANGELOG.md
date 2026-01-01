# PUAD Project Modification Log
# 记录所有的代码修改、实验尝试及背后的原因

## 2024-01-01 - 项目初始化
- **修改文件**: N/A
- **修改内容**: 导入原始 PUAD 项目代码。
- **修改原因**: 建立基准代码库 (Baseline)。

---

## 2025-12-31 - 添加批量评估脚本与全量基准测试
- **修改文件**: `evaluate_all.py` (新增)
- **修改内容**: 
    1. 创建了 `evaluate_all.py` 脚本，实现自动遍历 `mvtec_loco_anomaly_detection` 数据集下的所有类别。
    2. 脚本自动调用 `main.py` 并加载对应的预训练模型进行推理。
- **科研动机**: 
    1. **环境验证**: 确保代码能够处理所有类别的数据，排除路径或缺包错误。
    2. **建立基线 (Baseline)**: 获取原始 PUAD 模型在所有类别上的 AUROC 分数，作为后续改进算法（如引入 GMM 或 SPP）时的对比"锚点"。

### 📊 基准测试结果 (Baseline Results)
**测试配置**: EfficientAD (S) + student feature extractor
**测试日期**: 2025-12-31

| 类别 (Category) | EfficientAD AUROC | PUAD AUROC | PUAD (Logical) | PUAD (Structural) |
|:---:|:---:|:---:|:---:|:---:|
| breakfast_box | 0.8375 | 0.8707 | 0.9108 | 0.8338 |
| juice_bottle | 0.9790 | 0.9968 | 0.9979 | 0.9952 |
| pushpins | 0.9684 | 0.9802 | 0.9744 | 0.9868 |
| screw_bag | 0.7128 | 0.8436 | 0.7944 | 0.9256 |
| splicing_connectors | 0.9633 | 0.9676 | 0.9577 | 0.9802 |
| **平均 (Mean)** | **0.8922** | **0.9318** | **0.9270** | **0.9443** |

**关键发现**:
1. PUAD 在所有类别上均优于单独的 EfficientAD
2. screw_bag 类别的提升最为显著（+13.08%）
3. PUAD 对结构异常的检测能力（0.9443）优于逻辑异常（0.9270）

---

## 2025-12-31 - 实现 PRO (Per-Region Overlap) 评估指标

- **修改文件**: 
  - `requirements.txt` (修改)
  - `puad/efficientad/inference.py` (修改)
  - `puad/puad.py` (修改)
  - `puad/common.py` (修改)
  - `puad/dataset.py` (修改)
  - `main.py` (修改)

- **修改内容**:
  
  1. **添加依赖** (`requirements.txt`):
     - 新增 `scikit-image==0.19.3`，用于连通域分析（Connected Component Analysis）
  
  2. **修改 EfficientAD 推理模块** (`puad/efficientad/inference.py`):
     - 修改 `EfficientADInference.run()` 方法，新增 `return_map` 参数
     - 当 `return_map=True` 时，返回 `(anomaly_score, anomaly_map)`
     - 保持向后兼容性，默认行为不变
     - 添加详细的中文注释说明参数和返回值
  
  3. **修改 PUAD 推理模块** (`puad/puad.py`):
     - 修改 `PUAD.test()` 方法，新增 `return_map` 参数
     - 当需要异常图时，调用 `efficient_ad_inference.run(return_map=True)` 获取像素级预测
     - 注释说明：当前异常图仅包含 EfficientAD 的输出，马氏距离为全局特征无法映射到空间位置
  
  4. **实现 PRO 计算函数** (`puad/common.py`):
     - 新增 `compute_pro()` 函数，实现标准 PRO 指标计算
     - 算法流程：
       * 在 [0, 1] 范围内均匀采样 200 个阈值
       * 对每个阈值，将预测异常图二值化
       * 使用 `skimage.measure.label()` 提取 GT 中的连通域（独立异常区域）
       * 计算每个连通域与预测的重叠率（Overlap Ratio）
       * 对所有连通域和阈值求积分，得到 PRO 分数
     - 包含完整的科研动机注释（为什么需要 PRO，与像素级 AUROC 的区别）
  
  5. **实现 Ground Truth 加载函数** (`puad/dataset.py`):
     - 新增 `load_ground_truth_masks()` 函数
     - **关键适配**: 支持 MVTec LOCO AD 特殊的 mask 路径结构
       * 测试图像路径: `.../test/logical_anomalies/000.png`
       * 对应 Mask 路径: `.../ground_truth/logical_anomalies/000/000.png`
       * 处理逻辑: 解析文件名，构建多层文件夹路径
     - 返回字典 `{样本索引: mask 数组}`，仅包含异常样本
  
  6. **集成 PRO 评估** (`main.py`):
     - 导入 `compute_pro` 和 `load_ground_truth_masks`
     - 在 PUAD 评估完成后，加载所有异常样本的 GT masks
     - 遍历测试集，收集异常样本的预测异常图（使用 EfficientAD）
     - 调用 `compute_pro()` 计算 PRO 分数
     - 输出格式化的 PRO 结果，包含异常处理逻辑

- **科研动机**:
  
  1. **评估定位能力**: AUROC 只能评估分类性能，无法衡量模型是否准确定位到异常区域
  
  2. **公平评估小异常**: 像素级 AUROC 对大面积异常敏感，可能掩盖小区域异常的检测失败。PRO 通过对每个连通域（独立异常区域）计算覆盖率，更公平地评估定位能力
  
  3. **建立定位基线**: 为后续改进（如引入注意力机制、特征金字塔）提供可量化的定位性能对比基准
  
  4. **当前局限**: PRO 仅基于 EfficientAD 的异常图，PUAD 的马氏距离是全局特征无法生成像素级热图。未来可探索将全局特征作为空间加权因子

- **技术要点**:
  - 使用 8-连通 (`connectivity=2`) 提取连通域
  - 对异常图进行全局归一化以确保阈值可比性
  - 使用梯形法则 (`np.trapz`) 计算曲线下面积（积分）
  - 处理边界情况（无异常区域、mask 缺失等）

---

## 2025-12-31 - 修复 Ground Truth 路径解析错误 (MVTec LOCO AD 专用)

- **修改文件**: `puad/dataset.py`

- **问题描述**: 
  - 运行 PRO 评估时，出现大量 "未找到 mask 文件" 警告
  - 164 个异常样本中只成功加载了 2 个 mask
  - 根本原因：对 MVTec LOCO AD 数据集的 ground truth 路径结构理解错误

- **修改内容**:
  
  修正 `load_ground_truth_masks()` 函数中的路径拼接逻辑：
  
  **错误实现**（旧版本）：
  ```python
  # 错误: 使用测试图像的文件名作为 mask 文件名
  mask_path = .../ground_truth/logical_anomalies/015/015.png  ❌
  ```
  
  **正确实现**（修复后）：
  ```python
  # 正确: 文件夹用序号，但内部文件永远是 000.png
  mask_path = .../ground_truth/logical_anomalies/015/000.png  ✅
  ```
  
  **MVTec LOCO AD 真实路径规则**：
  1. 测试图像: `.../test/logical_anomalies/015.png`
  2. 对应 Mask: `.../ground_truth/logical_anomalies/015/000.png`
  3. 关键点：
     - 文件夹名使用测试图像的序号（如 `015`）
     - 但文件夹内的 mask 文件永远叫 `000.png`（固定不变）
     - 序号从 `000` 开始

- **修改细节**:
  ```python
  # 修改前
  mask_path = os.path.join(
      ground_truth_dir,
      class_name,
      img_name_without_ext,
      img_filename  # ❌ 错误：使用原始文件名
  )
  
  # 修改后
  mask_path = os.path.join(
      ground_truth_dir,
      class_name,
      img_name_without_ext,  # 文件夹用序号
      "000.png"              # ✅ 固定文件名
  )
  ```

- **科研影响**:
  - 修复后可正确加载所有结构异常样本的 Ground Truth
  - 使 PRO 指标计算能够覆盖完整测试集
  - 为后续定位能力分析提供准确的评估基础

- **技术备注**:
  - 此路径规则仅适用于 MVTec LOCO AD 数据集
  - 标准 MVTec AD 数据集的路径规则可能不同
  - 已在代码注释中标注此特殊处理

### 📊 PRO 指标评估结果 (修复后)
**测试配置**: EfficientAD (S) + student feature extractor
**测试日期**: 2025-12-31
**评估指标**: PRO (Per-Region Overlap) - 像素级异常定位能力

| 类别 (Category) | EfficientAD AUROC | PUAD AUROC | PUAD (Logical) | PUAD (Structural) | PRO Score | GT Masks 数量 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| breakfast_box | 0.8375 | 0.8707 | 0.9108 | 0.8338 | **0.1217** | 173 |
| juice_bottle | 0.9790 | 0.9968 | 0.9979 | 0.9952 | **0.1355** | 236 |
| pushpins | 0.9684 | 0.9802 | 0.9744 | 0.9868 | **0.0385** | 172 |
| screw_bag | 0.7128 | 0.8436 | 0.7944 | 0.9256 | **0.0431** | 219 |
| splicing_connectors | 0.9633 | 0.9676 | 0.9577 | 0.9802 | **0.0831** | 193 |
| **平均 (Mean)** | **0.8922** | **0.9318** | **0.9270** | **0.9443** | **0.0844** | **993** |

**PRO 评估关键发现**:
1. ✅ **路径修复成功**: 成功加载 993 个异常样本的 Ground Truth Masks（修复前仅 2 个）
2. 📊 **定位能力分析**: 
   - juice_bottle 的 PRO 最高 (0.1355)，说明其异常区域定位最准确
   - pushpins 的 PRO 最低 (0.0385)，尽管其 AUROC 很高，但定位精度有待提升
3. 🔍 **AUROC vs PRO 对比**:
   - screw_bag: AUROC 最低但 PRO 不是最差，说明其分类困难但定位相对准确
   - pushpins: AUROC 高但 PRO 低，揭示了高分类性能不等于精确定位
4. 💡 **科研价值**: PRO 指标揭示了 AUROC 无法反映的定位能力差异，为后续改进提供方向

---

## 2026-01-01 - 批量重新训练 EfficientAD 模型（全类别）

- **修改文件**: `train_all_categories.py` (新增)

- **修改内容**:
  1. 创建批量训练脚本 `train_all_categories.py`
  2. 自动遍历 MVTec LOCO AD 数据集的所有 5 个类别
  3. 对每个类别调用 `puad/efficientad/train.py` 重新训练 student network 和 autoencoder
  4. 使用 ImageNet Validation Set (50k images) 作为负样本训练辅助数据
  5. 每个类别训练 70,000 iterations

- **科研动机**:
  1. **模型重新训练**: 原预训练模型可能不是最优配置，重新训练以建立更可靠的基线
  2. **超参数验证**: 验证训练参数设置对不同类别的影响
  3. **性能基准更新**: 为后续改进算法提供最新的性能对比基准

### 📊 重新训练后的 EfficientAD 性能（2026-01-01）
**训练配置**: EfficientAD (S), 70K iterations
**负样本数据**: ImageNet Validation Set (50k images)
**训练日期**: 2026-01-01

| 类别 (Category) | 训练耗时 (分钟) | EfficientAD AUROC | Logical AUROC | Structural AUROC |
|:---:|:---:|:---:|:---:|:---:|
| breakfast_box | 196.75 | 0.8402 | 0.8336 | 0.8464 |
| juice_bottle | 160.42 | 0.9794 | 0.9664 | 0.9990 |
| pushpins | 173.05 | 0.9267 | 0.8958 | 0.9614 |
| screw_bag | 174.44 | 0.6870 | 0.5624 | 0.8952 |
| splicing_connectors | 171.55 | 0.9635 | 0.9480 | 0.9832 |
| **平均 (Mean)** | **175.24** | **0.8794** | **0.8412** | **0.9370** |

### 📈 性能对比：重新训练 vs 原预训练模型

| 类别 | 原模型 AUROC | 重训练 AUROC | 差异 |
|:---:|:---:|:---:|:---:|
| breakfast_box | 0.8375 | 0.8402 | +0.0027 |
| juice_bottle | 0.9790 | 0.9794 | +0.0004 |
| pushpins | 0.9684 | 0.9267 | **-0.0417** ⚠️ |
| screw_bag | 0.7128 | 0.6870 | **-0.0258** ⚠️ |
| splicing_connectors | 0.9633 | 0.9635 | +0.0002 |
| **平均** | **0.8922** | **0.8794** | **-0.0128** |

**关键发现**:
1. 📉 **整体性能略有下降**: 重训练后平均 AUROC 从 0.8922 降至 0.8794（-1.28%）
2. ⚠️ **pushpins 和 screw_bag 明显下降**: 
   - pushpins 下降 4.17%
   - screw_bag 下降 2.58%
3. ✅ **其他类别保持稳定**: breakfast_box, juice_bottle, splicing_connectors 基本持平
4. 💡 **科研启示**: 
   - 原预训练模型可能采用了更优的训练策略或更长的训练周期
   - 需要进一步调查训练超参数（learning rate, augmentation, iterations）
   - 建议后续实验继续使用原预训练模型作为基线

**技术细节**:
- 训练环境：PyTorch 1.13.1, CUDA enabled
- 平均每个类别耗时：175.24 分钟（约 2.9 小时）
- 总训练时间：876 分钟（约 14.6 小时）

---

## 2026-01-01 - 重新训练模型的完整评估（PUAD + PRO）

- **修改文件**: N/A (使用 `evaluate_all.py` 进行评估)

- **评估内容**:
  - 对重新训练的 EfficientAD 模型进行 PUAD 算法评估
  - 计算所有 5 个类别的 AUROC 和 PRO 指标
  - 对比重训练模型与原模型在 PUAD 框架下的性能

- **科研动机**:
  - **验证 PUAD 鲁棒性**: 测试 PUAD 算法在不同 EfficientAD 基线模型上的表现
  - **完整性能画像**: 同时评估分类能力（AUROC）和定位能力（PRO）
  - **建立新基线**: 为后续算法改进提供基于重训练模型的对比基准

### 📊 重训练模型 + PUAD 完整评估结果（2026-01-01）
**测试配置**: 重训练 EfficientAD (S) + PUAD (student feature extractor)
**测试日期**: 2026-01-01

| 类别 | EfficientAD AUROC | PUAD AUROC | PUAD (Logical) | PUAD (Structural) | PRO Score | GT Masks |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| breakfast_box | 0.8402 | 0.8654 | 0.9096 | 0.8246 | **0.1240** | 173 |
| juice_bottle | 0.9794 | 0.9980 | 0.9987 | 0.9971 | **0.1399** | 236 |
| pushpins | 0.9267 | 0.9368 | 0.8939 | 0.9851 | **0.0357** | 172 |
| screw_bag | 0.6870 | 0.8117 | 0.7381 | 0.9346 | **0.0427** | 219 |
| splicing_connectors | 0.9635 | 0.9672 | 0.9572 | 0.9798 | **0.0850** | 193 |
| **平均 (Mean)** | **0.8794** | **0.9158** | **0.8995** | **0.9442** | **0.0855** | **993** |

### 📈 三模型性能全面对比

**PUAD AUROC 对比**：

| 类别 | 原模型+PUAD | 重训练+PUAD | 差异 |
|:---:|:---:|:---:|:---:|
| breakfast_box | 0.8707 | 0.8654 | -0.0053 |
| juice_bottle | 0.9968 | 0.9980 | +0.0012 |
| pushpins | 0.9802 | 0.9368 | **-0.0434** ⚠️ |
| screw_bag | 0.8436 | 0.8117 | **-0.0319** ⚠️ |
| splicing_connectors | 0.9676 | 0.9672 | -0.0004 |
| **平均** | **0.9318** | **0.9158** | **-0.0160** |

**PRO Score 对比**：

| 类别 | 原模型 PRO | 重训练 PRO | 差异 |
|:---:|:---:|:---:|:---:|
| breakfast_box | 0.1217 | 0.1240 | +0.0023 |
| juice_bottle | 0.1355 | 0.1399 | +0.0044 |
| pushpins | 0.0385 | 0.0357 | -0.0028 |
| screw_bag | 0.0431 | 0.0427 | -0.0004 |
| splicing_connectors | 0.0831 | 0.0850 | +0.0019 |
| **平均** | **0.0844** | **0.0855** | **+0.0011** |

**关键发现**:
1. 📉 **PUAD AUROC 下降**: 重训练模型的 PUAD AUROC 从 0.9318 降至 0.9158（-1.60%）
   - 与 EfficientAD 的下降趋势一致（-1.28%）
   - pushpins 和 screw_bag 依然是下降最明显的类别
2. ✅ **PRO 基本持平**: 重训练模型的 PRO 从 0.0844 微升至 0.0855（+0.11%）
   - 说明定位能力未受重训练影响
   - juice_bottle 定位能力略有提升
3. 🔍 **PUAD 增益保持**: 
   - 原模型: PUAD 相比 EfficientAD 提升 +4.44% (0.8922→0.9318)
   - 重训练: PUAD 相比 EfficientAD 提升 +4.14% (0.8794→0.9158)
   - PUAD 的性能增益在不同基线模型上保持稳定
4. 💡 **科研结论**: 
   - PUAD 算法对基线模型变化具有鲁棒性
   - 重训练导致的性能下降主要来自 EfficientAD 基线，非 PUAD 算法本身
   - 建议后续实验使用原预训练模型以获得最佳性能

---