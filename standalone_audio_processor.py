#!/usr/bin/env python3
"""
Standalone Audio Processor for Sound Spaces
============================================

This module provides a standalone implementation of the sound processing pipeline 
from the sound-spaces repository. It can process impulse responses (IRs) with 
source audio to generate spatialized audio observations.

Key functionality:
1. Load source audio files
2. Convolve audio with impulse responses for each step
3. Apply crossfading between consecutive steps
4. Generate complete concatenated audio from all steps

Based on the original sound-spaces implementation:
- soundspaces/continuous_simulator.py
- soundspaces/simulator.py  
- PanoIR/render_panoIR.py

Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
"""

import numpy as np
import librosa
from scipy.signal import fftconvolve
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StandaloneAudioProcessor:
    """
    Standalone audio processor that replicates the sound-spaces audio processing pipeline.
    
    This class can process a sequence of impulse responses with a source audio file
    to generate spatialized audio observations following the exact same workflow
    as the original sound-spaces implementation.
    """
    
    def __init__(self, 
                 sampling_rate: int = 16000,
                 step_time: float = 0.1,
                 crossfade_enabled: bool = True,
                 crossfade_duration: float = 0.05):
        """
        Initialize the audio processor.
        
        Args:
            sampling_rate: Audio sampling rate in Hz (default: 16000)
            step_time: Duration of each step in seconds (default: 0.1)
            crossfade_enabled: Whether to enable crossfading between steps (default: True)
            crossfade_duration: Duration of crossfade in seconds (default: 0.05)
        """
        self.sampling_rate = sampling_rate
        self.step_time = step_time
        self.crossfade_enabled = crossfade_enabled
        self.crossfade_duration = crossfade_duration
        self.num_samples_per_step = int(sampling_rate * step_time)
        
    def load_source_audio(self, audio_path: str) -> np.ndarray:
        """
        Load source audio file and prepare it for processing.
        
        Args:
            audio_path: Path to the source audio file
            
        Returns:
            Audio data as numpy array with shape (num_samples,)
        """
        logger.info(f"Loading source audio from: {audio_path}")
        
        # Load audio with librosa at the specified sampling rate
        audio_data, sr = librosa.load(audio_path, sr=self.sampling_rate)
        
        # If audio is too short (less than 1 second), duplicate it to be longer
        # This follows the same logic as in continuous_simulator.py
        if audio_data.shape[0] // self.sampling_rate < 1:
            audio_data = np.concatenate([audio_data] * 3, axis=0)
            
        logger.info(f"Loaded audio with {len(audio_data)} samples ({len(audio_data)/self.sampling_rate:.2f}s)")
        return audio_data
        
    def _convolve_with_rir(self, 
                          source_audio: np.ndarray, 
                          rir: np.ndarray, 
                          step_index: int) -> np.ndarray:
        """
        Convolve source audio with impulse response for a specific step.
        
        This follows the exact same logic as _convolve_with_rir in continuous_simulator.py
        
        Args:
            source_audio: Source audio data with shape (num_samples,)
            rir: Impulse response with shape (rir_length, num_channels) 
            step_index: Current step index for temporal positioning
            
        Returns:
            Convolved audio with shape (num_channels, num_samples_per_step)
        """
        # Calculate the current sample index based on step
        current_sample_index = int(step_index * self.num_samples_per_step)
        
        # Handle wraparound for looping audio
        current_sample_index = current_sample_index % source_audio.shape[0]
        
        if current_sample_index - rir.shape[0] < 0:
            # Case 1: Not enough previous samples for full reverb
            sound_segment = source_audio[: current_sample_index + self.num_samples_per_step]
            binaural_convolved = np.array([
                fftconvolve(sound_segment, rir[:, channel]) 
                for channel in range(rir.shape[-1])
            ])
            audiogoal = binaural_convolved[:, current_sample_index: current_sample_index + self.num_samples_per_step]
        else:
            # Case 2: Include reverb from previous time step
            if current_sample_index + self.num_samples_per_step < source_audio.shape[0]:
                sound_segment = source_audio[
                    current_sample_index - rir.shape[0] + 1: 
                    current_sample_index + self.num_samples_per_step
                ]
            else:
                # Handle wraparound at end of audio
                wraparound_sample = current_sample_index + self.num_samples_per_step - source_audio.shape[0]
                sound_segment = np.concatenate([
                    source_audio[current_sample_index - rir.shape[0] + 1:],
                    source_audio[: wraparound_sample]
                ])
                
            binaural_convolved = np.array([
                fftconvolve(sound_segment, rir[:, channel], mode='valid')
                for channel in range(rir.shape[-1])
            ])
            audiogoal = binaural_convolved
            
        # Pad to ensure consistent output size
        if audiogoal.shape[1] < self.num_samples_per_step:
            padding = self.num_samples_per_step - audiogoal.shape[1]
            audiogoal = np.pad(audiogoal, [(0, 0), (0, padding)])
        elif audiogoal.shape[1] > self.num_samples_per_step:
            audiogoal = audiogoal[:, :self.num_samples_per_step]
            
        return audiogoal
        
    def _crossfade(self, audio1: np.ndarray, audio2: np.ndarray) -> np.ndarray:
        """
        Apply crossfading between two audio segments.
        
        This replicates the crossfade function from continuous_simulator.py
        
        Args:
            audio1: First audio segment with shape (num_channels, num_samples)
            audio2: Second audio segment with shape (num_channels, num_samples)
            
        Returns:
            Crossfaded audio with shape (num_channels, num_samples)
        """
        crossfade_samples = int(self.crossfade_duration * self.sampling_rate)
        crossfade_samples = min(crossfade_samples, audio1.shape[1], audio2.shape[1])
        
        if crossfade_samples == 0:
            return audio2
            
        # Create crossfade weights
        audio2_weight = np.arange(crossfade_samples + 1) / crossfade_samples
        audio1_weight = np.flip(audio2_weight)
        
        # Apply crossfade
        crossfaded_start = (
            audio1[:, :crossfade_samples + 1] * audio1_weight + 
            audio2[:, :crossfade_samples + 1] * audio2_weight
        )
        
        # Combine crossfaded start with rest of audio2
        result = np.concatenate([crossfaded_start, audio2[:, crossfade_samples + 1:]], axis=1)
        
        return result
        
    def process_impulse_responses(self, 
                                source_audio: np.ndarray, 
                                impulse_responses: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Process a sequence of impulse responses with source audio.
        
        Args:
            source_audio: Source audio data with shape (num_samples,)
            impulse_responses: List of IRs, each with shape (rir_length, num_channels)
            
        Returns:
            Tuple of:
            - List of convolved audio for each step, each with shape (num_channels, num_samples_per_step)
            - Complete concatenated audio with shape (num_channels, total_samples)
        """
        logger.info(f"Processing {len(impulse_responses)} impulse responses")
        
        step_audio_observations = []
        previous_audio = None
        
        for step_idx, rir in enumerate(impulse_responses):
            logger.debug(f"Processing step {step_idx + 1}/{len(impulse_responses)}")
            
            # Ensure IR has correct shape (samples, channels)
            if rir.ndim == 1:
                rir = rir.reshape(-1, 1)  # Mono case
            elif rir.shape[1] > rir.shape[0]:
                rir = rir.T  # Transpose if needed
                
            # Convolve audio with current IR
            current_audio = self._convolve_with_rir(source_audio, rir, step_idx)
            
            # Apply crossfading with previous step if enabled
            if self.crossfade_enabled and previous_audio is not None:
                current_audio = self._crossfade(previous_audio, current_audio)
                
            step_audio_observations.append(current_audio)
            previous_audio = current_audio
            
        # Concatenate all step observations into complete audio
        if step_audio_observations:
            complete_audio = np.concatenate(step_audio_observations, axis=1)
        else:
            complete_audio = np.zeros((2, 0))  # Empty stereo audio
            
        logger.info(f"Generated complete audio with {complete_audio.shape[1]} samples "
                   f"({complete_audio.shape[1]/self.sampling_rate:.2f}s)")
        
        return step_audio_observations, complete_audio
        
    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio to file.
        
        Args:
            audio: Audio data with shape (num_channels, num_samples)
            output_path: Output file path
        """
        # Convert to format expected by audio libraries
        if audio.ndim == 2:
            # Transpose to (num_samples, num_channels) for stereo
            audio_out = audio.T
        else:
            audio_out = audio
            
        # Normalize to prevent clipping
        audio_out = audio_out / (np.max(np.abs(audio_out)) + 1e-8)
        
        # Save using librosa
        import soundfile as sf
        sf.write(output_path, audio_out, self.sampling_rate)
        logger.info(f"Saved audio to: {output_path}")


def demo_usage():
    """
    Demonstrate usage of the StandaloneAudioProcessor.
    
    This function shows how to use the processor with dummy data.
    """
    print("StandaloneAudioProcessor Demo")
    print("=" * 40)
    
    # Initialize processor
    processor = StandaloneAudioProcessor(
        sampling_rate=16000,
        step_time=0.1,
        crossfade_enabled=True,
        crossfade_duration=0.05
    )
    
    # Create dummy source audio (1 second of sine wave)
    duration = 1.0
    t = np.linspace(0, duration, int(processor.sampling_rate * duration))
    source_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Create dummy impulse responses (5 steps, stereo)
    num_steps = 5
    rir_length = 1024
    impulse_responses = []
    
    for i in range(num_steps):
        # Create a simple decaying impulse response
        decay = np.exp(-np.arange(rir_length) / 100)
        rir_left = decay * np.random.randn(rir_length) * 0.1
        rir_right = decay * np.random.randn(rir_length) * 0.1
        rir = np.column_stack([rir_left, rir_right])
        impulse_responses.append(rir)
    
    # Process the audio
    step_observations, complete_audio = processor.process_impulse_responses(
        source_audio, impulse_responses
    )
    
    print(f"Generated {len(step_observations)} step observations")
    print(f"Each step has shape: {step_observations[0].shape}")
    print(f"Complete audio shape: {complete_audio.shape}")
    print(f"Total duration: {complete_audio.shape[1] / processor.sampling_rate:.2f} seconds")
    
    # Optionally save the audio (requires soundfile package)
    try:
        import soundfile
        processor.save_audio(complete_audio, "/tmp/demo_complete_audio.wav")
        print("Saved demo audio to /tmp/demo_complete_audio.wav")
    except ImportError:
        print("soundfile package not available, skipping audio save")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    demo_usage()