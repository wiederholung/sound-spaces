# Sound-Spaces Audio Processing Pipeline Analysis

## Overview
This document systematically analyzes the acoustic processing pipeline in the sound-spaces repository for generating audio observations from impulse responses (IR) and source audio.

## Core Components

### 1. Impulse Response (RIR) Handling
- **File Format**: WAV files read using `scipy.io.wavfile.read()`
- **Shape**: `(rir_length, 2)` for binaural (left/right channels)
- **Sampling Rate**: 16000 Hz (configurable via `config.AUDIO.RIR_SAMPLING_RATE`)
- **Fallback**: Zero arrays if RIR file is missing or corrupted

### 2. Source Audio Loading  
- **Loading**: `librosa.load()` with target sampling rate
- **Duration**: Can be any length, segmented during processing
- **Preprocessing**: If audio < 1 second, duplicate to be longer than longest RIR
- **Storage**: Cached in `_source_sound_dict` or `source_sound_dict`

### 3. Convolution Processing

#### Key Implementation Locations:
1. **SoundSpacesSim** (`soundspaces/simulator.py:609-666`): Discrete simulator
2. **ContinuousSoundSpacesSim** (`soundspaces/continuous_simulator.py:428-456`): Continuous simulator  
3. **AudioGoalDataset** (`ss_baselines/savi/pretraining/audiogoal_dataset.py:114-140`): Dataset version

#### Convolution Logic:

##### Case 1: Single-step audio (source length == sampling_rate)
```python
binaural_convolved = np.array([fftconvolve(self.current_source_sound, binaural_rir[:, channel]) 
                               for channel in range(binaural_rir.shape[-1])])
audiogoal = binaural_convolved[:, :sampling_rate]
```

##### Case 2: Multi-step audio with temporal indexing
Two sub-cases based on reverb carryover:

**No reverb carryover** (early steps):
```python
if index * sampling_rate - binaural_rir.shape[0] < 0:
    source_sound = current_source_sound[: (index + 1) * sampling_rate]
    binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel])
                                   for channel in range(binaural_rir.shape[-1])])
    audiogoal = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
```

**With reverb carryover** (later steps):
```python
else:
    # include reverb from previous time step
    source_sound = current_source_sound[index * sampling_rate - binaural_rir.shape[0] + 1
                                        : (index + 1) * sampling_rate]
    binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel], mode='valid')
                                   for channel in range(binaural_rir.shape[-1])])
    audiogoal = binaural_convolved  # or binaural_convolved[:, :-1] in dataset version
```

### 4. Temporal Processing Details

#### Audio Indexing:
- **Discrete Simulator**: `self._audio_index` increments each step, wraps with modulo
- **Continuous Simulator**: `self._current_sample_index` tracks exact sample position
- **Dataset**: Random index selection for training diversity

#### Step Duration:
- **Discrete**: 1 second per step (sampling_rate samples)
- **Continuous**: Configurable via `config.STEP_TIME` (e.g., 0.1 seconds)

#### Reverb Carryover:
The system includes acoustic reverb from previous time steps by:
1. Starting audio segment earlier by RIR length
2. Using 'valid' convolution mode to avoid edge artifacts
3. Maintaining temporal continuity in acoustic simulation

### 5. Additional Features

#### Distractor Sounds (SoundSpacesSim):
```python
if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
    distractor_convolved = np.array([fftconvolve(distractor_sound, distractor_rir[:, channel])
                                     for channel in range(distractor_rir.shape[-1])])
    audiogoal += distractor_convolved[:, :sampling_rate]
```

#### Crossfading (ContinuousSoundSpacesSim):
```python
if self.config.AUDIO.CROSSFADE and self._last_rir is not None:
    audiogoal_from_last_rir = self._convolve_with_rir(self._last_rir)
    audiogoal = crossfade(audiogoal_from_last_rir, audiogoal, sampling_rate)
```

## Output Format
- **Shape**: `(2, sampling_rate)` for stereo output
- **Duration**: Fixed length per step (typically 1 second @ 16kHz = 16000 samples)
- **Channels**: Left and right binaural audio
- **Usage**: This becomes `observation['audiogoal']` in the environment

## Key Parameters
- `sampling_rate`: 16000 Hz (standard)
- `rir_length`: Variable based on RIR file (typically ~8000-16000 samples)
- `step_time`: 1.0 seconds (discrete) or configurable (continuous)
- `audio_index`: Current position in source audio timeline