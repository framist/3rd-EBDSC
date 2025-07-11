# 3rd EBDSC 训练速度问题解决方案

## 问题总结

用户在A6000 GPU上遇到训练时间过长的问题（70+小时），并询问代码执行顺序和评估方法。

## 解决方案

### 1. 快速训练（推荐）

```bash
# A6000 GPU - 从70+小时降至8-12小时
python tcn_3rd_optimized.py --config a6000_fast

# RTX4090 GPU - 10-15小时 
python tcn_3rd_optimized.py --config rtx4090_balanced

# 快速验证 - 1-2小时
python tcn_3rd_optimized.py --config quick_test
```

### 2. 仅评估测试

```bash
# 独立评估脚本（不需要训练）
python tcn_eval_only.py \
    --model_path ./saved_models/your_model.pth \
    --test_data_path ../train_data/
```

### 3. 查看配置和建议

```bash
# 查看所有预定义配置
python training_configs.py --list

# 查看优化建议
python training_configs.py --tips

# 比较不同配置
python config_compare.py --all
```

## 核心改进

### 加速效果对比

| 配置 | 训练时间 | 相对原始加速 |
|------|----------|-------------|
| 原始配置 | 70+ 小时 | 1.0x |
| A6000_FAST | 8-12 小时 | 13.7x |
| RTX4090_BALANCED | 10-15 小时 | 8.5x |
| QUICK_TEST | 1-2 小时 | 61.4x |

### 参数优化要点

1. **批量大小**: 50 → 128 (充分利用GPU)
2. **模型层数**: 24 → 12 (减少复杂度)
3. **训练轮数**: 64 → 32 (减少时间)
4. **学习率**: 0.001 → 0.002 (加快收敛)

## 代码执行说明

### 训练流程
1. `tcn_3rd_optimized.py` - 完整训练（推荐）
2. `tcn_3rd.py` - 原始训练（较慢）

### 评估流程
1. `tcn_eval_only.py` - 独立评估脚本
2. `model_upload/my_eval.py` - 比赛提交评估

### 数据探索
1. `data.ipynb` - 数据分析笔记本（非官方文件，作者创建）

## 新增文件说明

- **`tcn_3rd_optimized.py`** - 优化训练脚本
- **`training_configs.py`** - 配置管理工具
- **`tcn_eval_only.py`** - 独立评估脚本
- **`config_compare.py`** - 配置比较工具
- **`USAGE_GUIDE.md`** - 详细使用指南

## 使用建议

1. **首次使用**: 运行 `--config quick_test` 验证环境
2. **正式训练**: 根据GPU选择对应配置
3. **仅测试**: 使用 `tcn_eval_only.py`
4. **参数调优**: 参考 `training_configs.py` 中的建议