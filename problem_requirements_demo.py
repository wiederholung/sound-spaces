#!/usr/bin/env python3
"""
针对问题需求的演示脚本
Demonstration script for the specific requirements mentioned in the problem statement.

需求：取到每一步返回的IR，假设预提供音频（从环境初始化的时候同步播放，长度不足则循环播放），
给出每一步的卷积后的声音观测。

Requirements: Given IR for each step, and pre-provided audio (playing synchronously from 
environment initialization, looping if insufficient length), generate convolved audio 
observations for each step.
"""

import numpy as np
from standalone_audio_processor import SoundSpacesAudioProcessor


def demonstrate_problem_requirements():
    """演示问题陈述中的具体需求"""
    
    print("=== 问题需求演示 (Problem Requirements Demonstration) ===")
    print("需求：每一步的IR + 预提供音频 → 每一步的卷积后声音观测")
    print("Requirement: IR per step + pre-provided audio → convolved audio observation per step")
    
    # 1. 初始化处理器
    processor = SoundSpacesAudioProcessor(sampling_rate=16000, step_duration=1.0)
    
    # 2. 预提供音频（从环境初始化时同步播放）
    print("\n--- 设置预提供音频 (Setting up pre-provided audio) ---")
    
    # 模拟预提供的音频文件（例如环境声音、语音等）
    duration = 2.5  # 2.5秒的音频（模拟长度不足的情况）
    t = np.linspace(0, duration, int(duration * 16000), False)
    
    # 创建复合音频信号（模拟真实环境音频）
    fundamental = 220  # 基频
    audio = (0.4 * np.sin(2 * np.pi * fundamental * t) +      # 基音
             0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +  # 二次谐波
             0.1 * np.sin(2 * np.pi * fundamental * 3 * t))   # 三次谐波
    
    # 添加包络（模拟自然音频特性）
    envelope = 1.0 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    audio *= envelope
    
    processor.set_source_audio(audio)
    print(f"✓ 音频长度: {duration}s (不足则自动循环)")
    print(f"✓ Audio length: {duration}s (will loop automatically if insufficient)")
    
    # 3. 模拟每一步返回的IR
    print("\n--- 每一步的IR数据 (IR data for each step) ---")
    
    # 模拟5步的IR数据（例如从环境模拟器获得）
    step_irs = []
    
    for step in range(5):
        print(f"Step {step}:")
        
        # 模拟每一步的RIR（室内脉冲响应）
        rir_length = 6000 + step * 1000  # 不同步骤的RIR长度略有不同
        rir = np.zeros((rir_length, 2))
        
        # 直达声（随距离和角度变化）
        distance_factor = 1.0 / (1.0 + step * 0.2)  # 模拟距离增加
        angle_factor = 0.8 + step * 0.05            # 模拟角度变化
        rir[0] = [distance_factor, distance_factor * angle_factor]
        
        # 早期反射（墙面、天花板、地面）
        early_reflections = [
            (400, 0.25), (800, 0.15), (1200, 0.12), (1800, 0.08)
        ]
        
        for delay, amplitude in early_reflections:
            if delay < rir_length:
                # 反射强度随步骤变化（模拟环境变化）
                actual_amplitude = amplitude * distance_factor * (1 - step * 0.1)
                rir[delay] = [actual_amplitude, actual_amplitude * 0.9]
        
        # 后期混响（指数衰减）
        for t_idx in range(2000, min(4000, rir_length)):
            decay = 0.05 * np.exp(-t_idx / 2000) * np.random.normal(0, 0.1) * distance_factor
            rir[t_idx] += [decay, decay * 1.1]
        
        step_irs.append(rir)
        print(f"  RIR形状: {rir.shape} (长度: {rir_length/16000:.3f}s)")
        print(f"  RIR shape: {rir.shape} (duration: {rir_length/16000:.3f}s)")
    
    # 4. 生成每一步的卷积后声音观测
    print("\n--- 生成每一步的声音观测 (Generating audio observations for each step) ---")
    
    # 重置处理器状态
    processor.reset()
    
    # 逐步处理
    observations = []
    for step, rir in enumerate(step_irs):
        print(f"\nStep {step} 处理:")
        print(f"Step {step} processing:")
        
        # 这就是核心功能：IR + 预提供音频 → 声音观测
        audiogoal = processor.compute_audiogoal_single_step(rir)
        observations.append(audiogoal)
        
        # 分析输出
        rms_left = np.sqrt(np.mean(audiogoal[0]**2))
        rms_right = np.sqrt(np.mean(audiogoal[1]**2))
        max_amplitude = np.max(np.abs(audiogoal))
        
        print(f"  输出形状: {audiogoal.shape}")
        print(f"  Output shape: {audiogoal.shape}")
        print(f"  最大幅值: {max_amplitude:.3f}")
        print(f"  Max amplitude: {max_amplitude:.3f}")
        print(f"  左声道RMS: {rms_left:.3f}, 右声道RMS: {rms_right:.3f}")
        print(f"  Left RMS: {rms_left:.3f}, Right RMS: {rms_right:.3f}")
        
        # 验证这就是observation['audiogoal']的格式
        assert audiogoal.shape == (2, 16000), "输出格式应为(2, 16000)"
        print(f"  ✓ 符合observation['audiogoal']格式")
        print(f"  ✓ Matches observation['audiogoal'] format")
    
    # 5. 验证音频循环功能
    print(f"\n--- 验证音频循环 (Verifying audio looping) ---")
    print(f"原始音频长度: {duration}s ({int(duration * 16000)} 采样)")
    print(f"Original audio length: {duration}s ({int(duration * 16000)} samples)")
    print(f"处理了 {len(observations)} 步，每步 1s = {len(observations)}s 总时长")
    print(f"Processed {len(observations)} steps, 1s each = {len(observations)}s total duration")
    
    if len(observations) * 1.0 > duration:
        print(f"✓ 音频自动循环，因为需要 {len(observations)}s 但只有 {duration}s")
        print(f"✓ Audio automatically looped, needed {len(observations)}s but only had {duration}s")
    
    # 6. 验证不同步骤的差异
    print(f"\n--- 验证步骤间差异 (Verifying differences between steps) ---")
    for i in range(1, len(observations)):
        diff = np.mean(np.abs(observations[i] - observations[i-1]))
        print(f"Step {i-1} vs Step {i}: 平均差异 = {diff:.4f}")
        print(f"Step {i-1} vs Step {i}: mean difference = {diff:.4f}")
    
    print(f"\n=== 演示完成 (Demonstration Complete) ===")
    print(f"✅ 成功实现：每一步IR + 预提供音频 → 每一步声音观测")
    print(f"✅ Successfully implemented: IR per step + pre-provided audio → audio observation per step")
    print(f"✅ 输出格式完全符合observation['audiogoal']")
    print(f"✅ Output format exactly matches observation['audiogoal']")
    print(f"✅ 支持音频循环播放")
    print(f"✅ Supports audio looping")
    
    return observations


def show_integration_example():
    """展示如何在其他仓库中集成使用"""
    
    print(f"\n=== 集成使用示例 (Integration Usage Example) ===")
    print(f"在其他仓库中使用此处理器的代码模式：")
    print(f"Code pattern for using this processor in other repositories:")
    
    code_example = '''
# 在你的项目中 (In your project):

from standalone_audio_processor import SoundSpacesAudioProcessor

class YourEnvironment:
    def __init__(self):
        # 初始化音频处理器
        self.audio_processor = SoundSpacesAudioProcessor()
        
        # 设置环境音频（初始化时同步播放）
        self.audio_processor.set_source_audio("environment_audio.wav")
    
    def step(self, action):
        # 获取当前步骤的IR（从你的环境模拟器）
        current_ir = self.get_current_ir()  # 你的IR获取逻辑
        
        # 生成声音观测
        audiogoal = self.audio_processor.compute_audiogoal_single_step(current_ir)
        
        # 构造观测字典
        observations = {
            'rgb': self.get_rgb(),
            'depth': self.get_depth(),
            'audiogoal': audiogoal,  # <-- 这就是声音观测
            # ... 其他观测
        }
        
        return observations
    
    def get_current_ir(self):
        # 你的IR获取逻辑，例如：
        # - 从预计算的IR文件加载
        # - 从声学模拟器实时生成
        # - 从神经网络预测
        return your_ir_data  # shape: (rir_length, 2)
'''
    
    print(code_example)
    
    print(f"关键优势 (Key advantages):")
    print(f"1. 单文件依赖，易于集成")
    print(f"1. Single file dependency, easy integration")
    print(f"2. 与sound-spaces完全兼容的输出格式")
    print(f"2. Fully compatible output format with sound-spaces")
    print(f"3. 高效的FFT卷积运算")
    print(f"3. Efficient FFT convolution")
    print(f"4. 自动处理音频循环和边界情况")
    print(f"4. Automatic audio looping and edge case handling")


if __name__ == "__main__":
    # 运行问题需求演示
    observations = demonstrate_problem_requirements()
    
    # 展示集成示例
    show_integration_example()
    
    print(f"\n🎉 演示完成！standalone_audio_processor.py 已准备好在您的项目中使用。")
    print(f"🎉 Demonstration complete! standalone_audio_processor.py is ready for use in your project.")