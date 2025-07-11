#!/usr/bin/env python3
"""
配置比较工具 - 帮助用户选择最适合的训练配置
"""

import argparse
from training_configs import TrainingConfigs, get_config_names, get_config

def compare_configs():
    """比较不同配置的参数差异"""
    print("="*80)
    print("训练配置对比表")
    print("="*80)
    
    config_names = ['ORIGINAL', 'A6000_FAST', 'A6000_ACCURACY', 'RTX4090_BALANCED', 'RTX3090_LIGHT', 'QUICK_TEST']
    params = ['batch_size', 'num_layers', 'd_model', 'max_epoch', 'lr', 'lr_step_size']
    
    # 打印表头
    print(f"{'参数':<15}", end="")
    for name in config_names:
        print(f"{name:<15}", end="")
    print()
    print("-" * (15 + 15 * len(config_names)))
    
    # 打印每个参数的对比
    for param in params:
        print(f"{param:<15}", end="")
        for name in config_names:
            config = get_config(name.lower())
            value = config.get(param, "N/A")
            print(f"{str(value):<15}", end="")
        print()
    
    print("\n" + "="*80)
    print("训练时间预估")
    print("="*80)
    
    for name in config_names:
        config = get_config(name.lower())
        desc = config.get('description', 'No description')
        print(f"{name:<20}: {desc}")

def calculate_speedup():
    """计算相对于原始配置的加速比"""
    print("\n" + "="*60)
    print("相对原始配置的加速比估算")
    print("="*60)
    
    original = get_config('original')
    
    # 简单的复杂度估算: layers * d_model * epochs * (1/batch_size)
    original_complexity = (
        original['num_layers'] * 
        original['d_model'] * 
        original['max_epoch'] * 
        (50 / original['batch_size'])  # 基准batch_size=50
    )
    
    config_names = ['A6000_FAST', 'A6000_ACCURACY', 'RTX4090_BALANCED', 'RTX3090_LIGHT', 'QUICK_TEST']
    
    for name in config_names:
        config = get_config(name.lower())
        complexity = (
            config['num_layers'] * 
            config['d_model'] * 
            config['max_epoch'] * 
            (50 / config['batch_size'])
        )
        
        speedup = original_complexity / complexity
        print(f"{name:<20}: {speedup:.1f}x 加速")

def recommend_config():
    """根据用户需求推荐配置"""
    print("\n" + "="*60)
    print("配置推荐指南")
    print("="*60)
    
    recommendations = [
        ("初次使用，验证代码", "QUICK_TEST", "1-2小时快速验证"),
        ("A6000 GPU，追求速度", "A6000_FAST", "8-12小时，平衡性能"),
        ("A6000 GPU，追求精度", "A6000_ACCURACY", "20-30小时，高精度"),
        ("RTX4090，平衡性能", "RTX4090_BALANCED", "10-15小时"),
        ("RTX3090/较小GPU", "RTX3090_LIGHT", "6-10小时，轻量级"),
        ("资源充足，不在乎时间", "ORIGINAL", "70+小时，原始配置")
    ]
    
    for scenario, config, note in recommendations:
        print(f"📊 {scenario:<25}: --config {config.lower():<15} ({note})")

def gpu_memory_estimate():
    """估算不同配置的GPU内存需求"""
    print("\n" + "="*60)
    print("GPU内存需求估算")
    print("="*60)
    
    configs = ['QUICK_TEST', 'RTX3090_LIGHT', 'RTX4090_BALANCED', 'A6000_FAST', 'A6000_ACCURACY', 'ORIGINAL']
    
    for name in configs:
        config = get_config(name.lower())
        
        # 简单的内存估算 (不完全准确，仅供参考)
        # 基于: batch_size * seq_len * d_model * layers * 数据类型字节数
        estimated_memory = (
            config['batch_size'] * 
            2000 *  # 假设序列长度
            config['d_model'] * 
            config['num_layers'] * 
            4 * 3  # float32 + 梯度 + 优化器状态
        ) / (1024**3)  # 转换为GB
        
        gpu_recommendation = ""
        if estimated_memory < 8:
            gpu_recommendation = "GTX 1080Ti+ 可用"
        elif estimated_memory < 16:
            gpu_recommendation = "RTX 3080+ 推荐"
        elif estimated_memory < 24:
            gpu_recommendation = "RTX 3090/4090 推荐"
        else:
            gpu_recommendation = "A6000/V100 推荐"
        
        print(f"{name:<20}: ~{estimated_memory:.1f}GB ({gpu_recommendation})")

def main():
    parser = argparse.ArgumentParser(description='训练配置比较和推荐工具')
    parser.add_argument('--compare', action='store_true', help='比较所有配置参数')
    parser.add_argument('--speedup', action='store_true', help='计算加速比')
    parser.add_argument('--recommend', action='store_true', help='显示配置推荐')
    parser.add_argument('--memory', action='store_true', help='估算GPU内存需求')
    parser.add_argument('--all', action='store_true', help='显示所有信息')
    
    args = parser.parse_args()
    
    if args.all or not any([args.compare, args.speedup, args.recommend, args.memory]):
        compare_configs()
        calculate_speedup()
        recommend_config()
        gpu_memory_estimate()
    else:
        if args.compare:
            compare_configs()
        if args.speedup:
            calculate_speedup()
        if args.recommend:
            recommend_config()
        if args.memory:
            gpu_memory_estimate()

if __name__ == "__main__":
    main()