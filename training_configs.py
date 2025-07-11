#!/usr/bin/env python3
"""
训练配置和优化建议

本文件包含针对不同GPU的训练参数优化建议，帮助解决训练速度过慢的问题。
"""

import argparse

class TrainingConfigs:
    """训练配置类，包含针对不同场景的优化参数"""
    
    # A6000 GPU (48GB VRAM) - 平衡性能和速度
    A6000_FAST = {
        'batch_size': 128,      # 增大批量大小，充分利用GPU
        'num_layers': 12,       # 减少层数，降低计算复杂度  
        'd_model': 96,          # 稍微减小模型维度
        'max_epoch': 32,        # 减少训练轮数
        'lr': 0.002,           # 稍微提高学习率
        'lr_step_size': 8,     # 更频繁的学习率衰减
        'description': 'A6000快速训练配置 - 预计训练时间: 8-12小时'
    }
    
    # A6000 GPU - 高精度配置 
    A6000_ACCURACY = {
        'batch_size': 96,       # 适中的批量大小
        'num_layers': 18,       # 中等层数
        'd_model': 128,         # 标准模型维度
        'max_epoch': 48,        # 适中的训练轮数
        'lr': 0.001,           # 标准学习率
        'lr_step_size': 12,    # 标准学习率衰减
        'description': 'A6000精度优先配置 - 预计训练时间: 20-30小时'
    }
    
    # RTX 4090 (24GB VRAM) - 平衡配置
    RTX4090_BALANCED = {
        'batch_size': 80,       # 适合24GB显存的批量大小
        'num_layers': 12,       # 减少层数
        'd_model': 96,          # 较小的模型维度
        'max_epoch': 32,        # 减少训练轮数
        'lr': 0.0015,          # 适中的学习率
        'lr_step_size': 8,     # 更频繁的学习率衰减
        'description': 'RTX4090平衡配置 - 预计训练时间: 10-15小时'
    }
    
    # RTX 3090/4080 (24GB VRAM) - 轻量配置
    RTX3090_LIGHT = {
        'batch_size': 64,       # 保守的批量大小
        'num_layers': 8,        # 更少的层数
        'd_model': 80,          # 更小的模型维度
        'max_epoch': 24,        # 较少的训练轮数
        'lr': 0.002,           # 较高的学习率
        'lr_step_size': 6,     # 快速学习率衰减
        'description': 'RTX3090轻量配置 - 预计训练时间: 6-10小时'
    }
    
    # 快速测试配置 - 用于验证代码和快速实验
    QUICK_TEST = {
        'batch_size': 32,       # 小批量大小，快速测试
        'num_layers': 4,        # 最少层数
        'd_model': 64,          # 最小模型维度
        'max_epoch': 8,         # 最少训练轮数
        'lr': 0.003,           # 高学习率
        'lr_step_size': 2,     # 快速学习率衰减
        'description': '快速测试配置 - 预计训练时间: 1-2小时'
    }
    
    # 原始配置（可能很慢）
    ORIGINAL = {
        'batch_size': 50,
        'num_layers': 24,
        'd_model': 128,
        'max_epoch': 64,
        'lr': 0.001,
        'lr_step_size': 16,
        'description': '原始配置 - 预计训练时间: 70+ 小时（不推荐）'
    }

def get_config_names():
    """获取所有可用的配置名称"""
    configs = []
    for attr in dir(TrainingConfigs):
        if not attr.startswith('_') and attr.isupper():
            configs.append(attr.lower())
    return configs

def get_config(config_name):
    """根据名称获取配置"""
    config_name = config_name.upper()
    if hasattr(TrainingConfigs, config_name):
        return getattr(TrainingConfigs, config_name)
    else:
        available = get_config_names()
        raise ValueError(f"未知配置: {config_name}. 可用配置: {available}")

def apply_config_to_parser(parser, config_name):
    """将配置应用到参数解析器的默认值"""
    config = get_config(config_name)
    
    # 更新默认值
    for action in parser._actions:
        if action.dest in config:
            action.default = config[action.dest]
    
    return config

def print_all_configs():
    """打印所有可用配置的详细信息"""
    print("=== 可用的训练配置 ===\n")
    
    config_names = get_config_names()
    for name in config_names:
        config = get_config(name)
        print(f"配置名称: {name.upper()}")
        print(f"描述: {config['description']}")
        print("参数:")
        for key, value in config.items():
            if key != 'description':
                print(f"  --{key}: {value}")
        print()

def add_config_arguments(parser):
    """为参数解析器添加配置相关的参数"""
    config_names = get_config_names()
    parser.add_argument('--config', type=str, choices=config_names,
                       help=f'使用预定义的训练配置。可选: {config_names}')
    parser.add_argument('--list_configs', action='store_true',
                       help='列出所有可用的配置详情')
    
    return parser

# 训练速度优化建议
OPTIMIZATION_TIPS = """
=== 训练速度优化建议 ===

1. **硬件优化**:
   - 确保使用了正确的CUDA版本和PyTorch版本
   - 检查GPU利用率: nvidia-smi 
   - 确保没有其他程序占用GPU内存

2. **批量大小优化**:
   - A6000 (48GB): 推荐 batch_size=128-256
   - RTX4090 (24GB): 推荐 batch_size=64-128  
   - RTX3090 (24GB): 推荐 batch_size=32-80
   - 如果显存不足，可以使用梯度累积

3. **模型架构优化**:
   - 减少层数: num_layers=8-16 而不是 24
   - 减小模型维度: d_model=64-96 而不是 128
   - 考虑使用更轻量的模型变体

4. **数据加载优化**:
   - 增加 num_workers (建议 4-8)
   - 启用 pin_memory=True
   - 减少数据增强的复杂度

5. **训练策略优化**:
   - 减少最大训练轮数: max_epoch=16-32
   - 使用更高的初始学习率: lr=0.002-0.003
   - 更频繁的学习率衰减: lr_step_size=4-8
   - 考虑使用混合精度训练 (已启用)

6. **多任务学习优化**:
   - 调整任务权重: --mutitask_weights 0.3 0.3 0.4
   - 考虑先训练单任务，再进行多任务fine-tuning

7. **验证频率优化**:
   - 减少验证频率，例如每2-4个epoch验证一次
   - 在验证时使用更大的batch_size

8. **其他建议**:
   - 使用更少的频率Top-K: --freq_topk 5-7
   - 禁用不必要的数据增强: --dont_data_aug
   - 减小最大码元长度: --max_code_len 200

示例命令:
python tcn_3rd.py --config a6000_fast
python tcn_3rd.py --config rtx4090_balanced  
python tcn_3rd.py --config quick_test
"""

def print_optimization_tips():
    """打印优化建议"""
    print(OPTIMIZATION_TIPS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练配置查看工具')
    parser.add_argument('--list', action='store_true', help='列出所有配置')
    parser.add_argument('--tips', action='store_true', help='显示优化建议')
    parser.add_argument('--config', type=str, help='查看特定配置详情')
    
    args = parser.parse_args()
    
    if args.list:
        print_all_configs()
    elif args.tips:
        print_optimization_tips()
    elif args.config:
        try:
            config = get_config(args.config)
            print(f"配置 {args.config.upper()}:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        except ValueError as e:
            print(f"错误: {e}")
    else:
        print("使用 --list 查看所有配置，--tips 查看优化建议")