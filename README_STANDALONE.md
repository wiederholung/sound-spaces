# 独立音频处理器 (Standalone Audio Processor)

## 概述 (Overview)

本项目成功提取并整合了 sound-spaces 仓库中的音频处理流水线，创建了一个独立的 Python 文件，可以在其他仓库中独立使用。该处理器严格遵循原始仓库的处理流程，实现了从脉冲响应 (IR) 和源音频生成声音观测的完整功能。

## 核心文件 (Core Files)

### 1. `standalone_audio_processor.py`
主要的独立音频处理器实现，包含：
- `SoundSpacesAudioProcessor` 类：核心处理器
- 音频文件加载和预处理
- 双耳脉冲响应 (RIR) 处理
- 卷积运算和混响处理
- 循环音频支持
- 交叉淡化功能

### 2. `audio_processing_pipeline_analysis.md`
详细的音频处理流水线分析文档，包含：
- 原始代码的系统性梳理
- 每个处理步骤的详细说明
- 卷积逻辑的完整分析
- 时序处理和混响传递的机制

### 3. `test_standalone_processor.py`
全面的验证测试套件，包含：
- 基本功能测试
- 文件加载测试
- 混响传递逻辑测试
- 序列处理测试
- 边界情况测试
- 与原始逻辑的验证对比

### 4. `usage_example.py`
完整的使用示例，演示：
- 逐步处理方法
- 序列批处理方法
- 外部 RIR 数据处理
- 高级功能使用

## 主要功能 (Key Features)

### 🎯 核心功能
- **脉冲响应处理**：支持双耳 RIR 文件加载和处理
- **音频卷积**：使用 FFT 卷积实现高效处理
- **时序处理**：正确处理混响传递和时间索引
- **音频循环**：自动处理短音频文件的循环播放
- **格式兼容**：输出格式与原始 `observation['audiogoal']` 完全兼容

### 🔧 高级功能
- **灵活步长**：支持不同的步长时间（默认1秒）
- **批处理**：可以处理 RIR 序列
- **交叉淡化**：支持连续模拟器的交叉淡化功能
- **边界处理**：处理短音频、单声道 RIR 等边界情况
- **缓存优化**：音频文件加载缓存

## 使用方法 (Usage)

### 基本使用
```python
from standalone_audio_processor import SoundSpacesAudioProcessor

# 初始化处理器
processor = SoundSpacesAudioProcessor(sampling_rate=16000, step_duration=1.0)

# 设置源音频
processor.set_source_audio('path/to/audio.wav')

# 加载 RIR 并处理单步
rir = processor.load_rir_from_file('path/to/rir.wav')
audiogoal = processor.compute_audiogoal_single_step(rir)

# audiogoal 形状: (2, 16000) - 双声道，1秒长度
```

### 序列处理
```python
# 处理 RIR 序列
rir_files = ['rir1.wav', 'rir2.wav', 'rir3.wav']
rir_sequence = [processor.load_rir_from_file(f) for f in rir_files]
audiogoal_sequence = processor.compute_audiogoal_sequence(rir_sequence)

# 每个 audiogoal 都是 (2, 16000) 形状
```

### 使用数组数据
```python
# 直接使用 RIR 数组（例如从其他模拟器获得）
import numpy as np

rir_array = np.zeros((8000, 2))  # 0.5秒的双耳 RIR
rir_array[0] = [1.0, 0.8]        # 直达声
rir_array[800] = [0.3, 0.25]     # 早期反射

audiogoal = processor.compute_audiogoal_single_step(rir_array)
```

## 技术细节 (Technical Details)

### 音频处理流水线
1. **音频加载**：使用 librosa 加载音频，重采样到指定采样率
2. **RIR 处理**：从 WAV 文件加载双耳脉冲响应
3. **卷积运算**：
   - 早期步骤：无混响传递，直接卷积
   - 后期步骤：包含前一步的混响传递，使用 'valid' 模式
4. **时序管理**：正确处理音频索引和循环
5. **输出格式化**：确保输出长度一致，形状为 (2, samples_per_step)

### 与原始代码的对应关系
- **SoundSpacesSim** (`soundspaces/simulator.py`) → 离散模拟器逻辑
- **ContinuousSoundSpacesSim** (`soundspaces/continuous_simulator.py`) → 连续模拟器逻辑  
- **AudioGoalDataset** (`ss_baselines/savi/pretraining/audiogoal_dataset.py`) → 数据集处理逻辑

### 关键参数
- `sampling_rate`: 音频采样率（默认 16000 Hz）
- `step_duration`: 每步的时间长度（默认 1.0 秒）
- `samples_per_step`: 每步的采样数（sampling_rate × step_duration）

## 验证结果 (Validation Results)

所有测试均已通过：
- ✅ 基本功能测试
- ✅ 文件加载测试  
- ✅ 混响传递逻辑测试
- ✅ 序列处理测试
- ✅ 边界情况测试
- ✅ 交叉淡化功能测试
- ✅ 原始逻辑验证测试

## 性能特性 (Performance Characteristics)

- **内存效率**：音频文件缓存，避免重复加载
- **计算效率**：使用 FFT 卷积，比时域卷积更快
- **模块化设计**：可以独立使用各个功能组件
- **错误处理**：优雅处理文件错误和边界情况

## 依赖项 (Dependencies)

```bash
pip install numpy scipy librosa
```

- `numpy`: 数组运算
- `scipy`: 信号处理和文件 I/O
- `librosa`: 音频文件加载

## 示例输出 (Example Output)

```
Step 0: RIR shape (8000, 2), Output shape (2, 16000)
         Max amplitude: 0.300, RMS: 0.138
Step 1: RIR shape (8000, 2), Output shape (2, 16000)  
         Max amplitude: 0.210, RMS: 0.134
```

每步输出都是形状为 `(2, 16000)` 的双声道音频，对应1秒长度的 `observation['audiogoal']`。

## 总结 (Summary)

这个独立音频处理器成功实现了：

1. **完整功能复制**：严格遵循 sound-spaces 的音频处理流程
2. **独立部署**：单文件包含所有必要功能，无需原始仓库依赖
3. **灵活使用**：支持文件输入、数组输入、序列处理等多种使用模式
4. **充分验证**：通过了全面的测试，确保与原始实现的一致性
5. **完善文档**：提供了详细的使用示例和技术文档

现在您可以在自己的项目中使用这个处理器，通过提供每一步的 RIR 数据和预设的音频文件，生成每一步卷积后的声音观测结果。