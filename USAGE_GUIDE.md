# 3rd EBDSC 代码使用指南

本文档详细说明了代码的运行顺序、训练速度优化方案，以及如何进行仅测试评估。

## 🚀 快速开始

### 训练速度问题解决方案

如果你遇到训练时间过长的问题（例如在A6000上显示70+小时），请使用优化后的配置：

```bash
# A6000 GPU 快速训练配置 (预计 8-12 小时)
python tcn_3rd_optimized.py --config a6000_fast

# RTX4090 平衡配置 (预计 10-15 小时)  
python tcn_3rd_optimized.py --config rtx4090_balanced

# 快速测试配置 (预计 1-2 小时，用于验证代码)
python tcn_3rd_optimized.py --config quick_test
```

### 查看所有可用配置

```bash
# 查看所有预定义配置
python training_configs.py --list

# 查看详细优化建议
python training_configs.py --tips
```

## 📁 文件结构说明

### 核心文件

- **`tcn_3rd.py`** - 原始训练脚本（可能很慢）
- **`tcn_3rd_optimized.py`** - 优化后的训练脚本（推荐使用）
- **`tcn_eval_only.py`** - 独立评估脚本（仅测试，不训练）
- **`training_configs.py`** - 训练配置和优化建议
- **`model_upload/my_eval.py`** - 比赛提交用的评估脚本

### 模型文件

- **`ModernTCN.py`** - 主要的TCN模型
- **`ModernTCN_FreTS.py`** - FreTS变体
- **`ModernTCN_FTDW.py`** - FTDW变体

### 数据和工具

- **`ebdsc3rd_datatools.py`** - 数据加载和预处理
- **`my_tools.py`** - 工具函数
- **`data.ipynb`** - 数据探索和分析（Jupyter笔记本）

## 🔄 代码运行顺序

### 1. 完整训练流程

```bash
# 方法 1: 使用优化配置（推荐）
python tcn_3rd_optimized.py --config a6000_fast

# 方法 2: 使用原始脚本（可能很慢）
python tcn_3rd.py
```

**说明：** 运行 `tcn_3rd.py` 或 `tcn_3rd_optimized.py` 会自动完成：
- 数据加载和预处理
- 模型训练
- 验证评估
- 模型保存

### 2. 仅测试评估

如果你已有训练好的模型，只想进行测试：

```bash
# 使用独立评估脚本
python tcn_eval_only.py \
    --model_path ./saved_models/TCN_best.pth \
    --test_data_path ../train_data/ \
    --batch_size 100

# 使用比赛提交脚本（需要修改路径）
cd model_upload
python my_eval.py
```

### 3. 数据探索

```bash
# 启动 Jupyter 查看数据分析
jupyter notebook data.ipynb
```

## ⚡ 训练速度优化详解

### 问题分析

原始配置导致训练缓慢的主要原因：

1. **模型过大**: 24层 + 128维度
2. **批量大小小**: batch_size=50（A6000可支持更大）
3. **训练轮数多**: 64个epoch
4. **数据集大**: 18万个样本

### 优化方案

#### 1. 硬件配置建议

| GPU型号 | 推荐配置 | 预计时间 | 命令 |
|---------|----------|----------|------|
| A6000 (48GB) | a6000_fast | 8-12小时 | `--config a6000_fast` |
| A6000 (48GB) | a6000_accuracy | 20-30小时 | `--config a6000_accuracy` |
| RTX4090 (24GB) | rtx4090_balanced | 10-15小时 | `--config rtx4090_balanced` |
| RTX3090 (24GB) | rtx3090_light | 6-10小时 | `--config rtx3090_light` |
| 任意GPU | quick_test | 1-2小时 | `--config quick_test` |

#### 2. 详细参数对比

| 参数 | 原始配置 | A6000_FAST | 说明 |
|------|----------|------------|------|
| batch_size | 50 | 128 | 充分利用GPU内存 |
| num_layers | 24 | 12 | 减少模型复杂度 |
| d_model | 128 | 96 | 降低维度 |
| max_epoch | 64 | 32 | 减少训练轮数 |
| lr | 0.001 | 0.002 | 提高学习率 |

#### 3. 自定义优化

如果预定义配置不满足需求，可以手动调整：

```bash
python tcn_3rd_optimized.py \
    --batch_size 96 \
    --num_layers 16 \
    --d_model 112 \
    --max_epoch 40 \
    --lr 0.0015 \
    --num_workers 6
```

## 🧪 评估说明

### 评估指标

比赛使用三个指标的加权组合：

- **MT (调制识别)**: 20% 权重，准确率
- **SW (符号宽度)**: 30% 权重，回归误差
- **CQ (码序列)**: 50% 权重，余弦相似度

### 使用独立评估脚本

```bash
python tcn_eval_only.py \
    --model_path ./saved_models/your_model.pth \
    --test_data_path ./test_data/ \
    --output_csv results.csv \
    --batch_size 100
```

**参数说明：**
- `--model_path`: 训练好的模型路径
- `--test_data_path`: 测试数据路径
- `--output_csv`: 输出结果文件名
- `--batch_size`: 推理批量大小

### 比赛提交评估

比赛提交使用 `model_upload/my_eval.py`，该脚本：
- 加载测试数据
- 计算三个指标
- 生成提交格式的CSV文件

## 🔧 常见问题

### Q1: 训练速度仍然很慢怎么办？

1. 检查GPU利用率：`nvidia-smi`
2. 使用更轻量的配置：`--config quick_test`
3. 增大批量大小：`--batch_size 128`
4. 减少层数：`--num_layers 8`

### Q2: 显存不足怎么办？

```bash
# 减小批量大小
python tcn_3rd_optimized.py --config rtx3090_light --batch_size 32

# 或使用梯度累积（需要修改代码）
```

### Q3: 如何继续训练？

```bash
python tcn_3rd_optimized.py \
    --config a6000_fast \
    --best_continue _continued
```

### Q4: data.ipynb 是官方文件吗？

`data.ipynb` 是项目作者创建的数据分析文件，用于：
- 探索数据集结构
- 可视化信号特征
- 理解数据预处理流程

不是比赛官方提供的文件，但对理解数据很有帮助。

## 🎯 推荐工作流程

### 新手入门

1. **快速验证**: `python tcn_3rd_optimized.py --config quick_test`
2. **查看结果**: 检查 `./saved_models/` 中的模型文件
3. **评估测试**: `python tcn_eval_only.py --model_path ./saved_models/TCN_best.pth --test_data_path ../train_data/`

### 正式训练

1. **选择配置**: 根据GPU型号选择合适配置
2. **开始训练**: `python tcn_3rd_optimized.py --config a6000_fast`
3. **监控进度**: 观察训练日志和GPU利用率
4. **评估结果**: 使用评估脚本验证性能

### 参数调优

1. **基线实验**: 先用预定义配置建立基线
2. **逐步调整**: 在基线基础上微调参数
3. **消融实验**: 测试不同组件的影响

## 📊 性能监控

### 训练监控

```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控训练日志
tail -f nohup.out
```

### 使用 Weights & Biases

```bash
python tcn_3rd_optimized.py \
    --config a6000_fast \
    --wandb \
    --name "my_experiment" \
    --tags "optimized" "a6000"
```

## 🤝 贡献和反馈

如果你发现更好的优化方案或遇到问题，欢迎：

1. 提交 Issue 报告问题
2. 提交 Pull Request 改进代码
3. 分享你的训练配置和结果

---

**作者**: Framist & KylinGR - 「QiiQ」战队
**比赛**: 第三届"火眼金睛"电磁大数据非凡挑战赛
**成绩**: 铜奖（初赛第三名，决赛第四名）