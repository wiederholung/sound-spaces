#!/usr/bin/env python3
"""
Standalone Audio Processor for Sound-Spaces

This module replicates the exact acoustic processing pipeline from the sound-spaces
repository, providing a self-contained solution for generating audio observations
from impulse responses and source audio.

Based on the processing logic from:
- soundspaces/simulator.py
- soundspaces/continuous_simulator.py  
- ss_baselines/savi/pretraining/audiogoal_dataset.py
"""

import os
import logging
from typing import List, Optional, Union, Tuple
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import librosa


class SoundSpacesAudioProcessor:
    """
    Standalone audio processor that replicates sound-spaces acoustic processing pipeline.
    
    This class handles:
    - Loading and preprocessing source audio files
    - Processing binaural impulse responses (RIRs)
    - Step-by-step convolution with proper reverb carryover
    - Audio looping for shorter source files
    """
    
    def __init__(self, sampling_rate: int = 16000, step_duration: float = 1.0):
        """
        Initialize the audio processor.
        
        Args:
            sampling_rate: Audio sampling rate in Hz (default: 16000)
            step_duration: Duration of each step in seconds (default: 1.0)
        """
        self.sampling_rate = sampling_rate
        self.step_duration = step_duration
        self.samples_per_step = int(sampling_rate * step_duration)
        
        # Internal state
        self._source_audio = None
        self._audio_length_seconds = 0
        self._current_step = 0
        self._current_sample_index = 0
        
        # Cache for loaded audio
        self._audio_cache = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_source_audio(self, audio_path: str, loop_if_short: bool = True) -> np.ndarray:
        """
        Load source audio file with preprocessing.
        
        Args:
            audio_path: Path to audio file
            loop_if_short: If True, duplicate short audio to be longer than typical RIR
            
        Returns:
            Loaded audio as 1D numpy array
        """
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
            
        try:
            audio_data, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Replicate the short audio handling from continuous_simulator.py
            if loop_if_short and audio_data.shape[0] // self.sampling_rate <= 1:
                audio_data = np.concatenate([audio_data] * 3, axis=0)
                self.logger.info(f"Short audio detected, tripled length: {audio_data.shape[0] / self.sampling_rate:.2f}s")
            
            self._audio_cache[audio_path] = audio_data
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
    
    def set_source_audio(self, audio: Union[str, np.ndarray]) -> None:
        """
        Set the source audio for processing.
        
        Args:
            audio: Audio file path or numpy array
        """
        if isinstance(audio, str):
            self._source_audio = self.load_source_audio(audio)
        else:
            self._source_audio = audio.copy()
            
        self._audio_length_seconds = max(1, self._source_audio.shape[0] / self.sampling_rate)  # Avoid zero
        self._current_step = 0
        self._current_sample_index = 0
        
        self.logger.info(f"Set source audio: {self._audio_length_seconds:.2f}s, {self._source_audio.shape[0]} samples")
    
    def load_rir_from_file(self, rir_path: str) -> np.ndarray:
        """
        Load binaural RIR from WAV file.
        
        Args:
            rir_path: Path to RIR WAV file
            
        Returns:
            RIR array with shape (rir_length, 2)
        """
        try:
            sampling_freq, binaural_rir = wavfile.read(rir_path)
            
            if len(binaural_rir) == 0:
                self.logger.warning(f"Empty RIR file: {rir_path}")
                binaural_rir = np.zeros((self.sampling_rate, 2)).astype(np.float32)
            
            return binaural_rir.astype(np.float32)
            
        except (ValueError, FileNotFoundError) as e:
            self.logger.warning(f"Failed to read RIR {rir_path}: {e}")
            return np.zeros((self.sampling_rate, 2)).astype(np.float32)
    

    def compute_audiogoal_single_step(self, rir: np.ndarray, step_index: int = None) -> np.ndarray:
        """
        Compute audio observation for a single step using provided RIR.
        
        Args:
            rir: Binaural RIR array with shape (rir_length, 2)
            step_index: Step index (if None, uses internal counter)
            
        Returns:
            Audio observation with shape (2, samples_per_step)
        """
        if self._source_audio is None:
            raise ValueError("Source audio not set. Call set_source_audio() first.")
        
        # Ensure RIR has correct shape
        if rir.ndim == 1:
            rir = rir.reshape(-1, 1)
        if rir.shape[1] == 1:
            rir = np.tile(rir, (1, 2))  # Convert mono to stereo
        
        rir_length = rir.shape[0]
        
        # Special case: single-step audio (source length == samples_per_step)
        if self._source_audio.shape[0] == self.samples_per_step:
            binaural_convolved = np.array([
                fftconvolve(self._source_audio, rir[:, channel])
                for channel in range(rir.shape[1])
            ])
            audiogoal = binaural_convolved[:, :self.samples_per_step]
        else:
            # Multi-step audio with temporal processing - match original exactly
            if step_index is not None:
                # Use provided step index
                index = step_index
            else:
                # Use internal audio index and increment it (like original)
                index = self._current_step
                self._current_step = (self._current_step + 1) % int(self._audio_length_seconds)
            
            start_sample = index * self.samples_per_step
            end_sample = (index + 1) * self.samples_per_step
            
            if start_sample - rir_length < 0:
                # Early steps: no reverb carryover (matches original condition)
                audio_segment = self._source_audio[:end_sample]
                binaural_convolved = np.array([
                    fftconvolve(audio_segment, rir[:, channel])
                    for channel in range(rir.shape[1])
                ])
                audiogoal = binaural_convolved[:, start_sample:end_sample]
            else:
                # Later steps: include reverb from previous time step
                segment_start = start_sample - rir_length + 1
                segment_end = end_sample
                
                # Handle audio looping - match original modulo behavior
                audio_length = self._source_audio.shape[0]
                if segment_end <= audio_length:
                    audio_segment = self._source_audio[segment_start:segment_end]
                else:
                    # Wraparound case
                    first_part = self._source_audio[segment_start:]
                    remaining_samples = segment_end - audio_length
                    second_part = self._source_audio[:remaining_samples]
                    audio_segment = np.concatenate([first_part, second_part])
                
                binaural_convolved = np.array([
                    fftconvolve(audio_segment, rir[:, channel], mode='valid')
                    for channel in range(rir.shape[1])
                ])
                
                # Use the full result (matches original line 647)
                audiogoal = binaural_convolved
        
        # Ensure output has correct shape
        if audiogoal.shape[1] < self.samples_per_step:
            padding = self.samples_per_step - audiogoal.shape[1]
            audiogoal = np.pad(audiogoal, [(0, 0), (0, padding)])
        elif audiogoal.shape[1] > self.samples_per_step:
            audiogoal = audiogoal[:, :self.samples_per_step]
        
        return audiogoal
    
    def compute_audiogoal_sequence(self, rir_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute audio observations for a sequence of RIRs.
        
        Args:
            rir_sequence: List of RIR arrays, one for each step
            
        Returns:
            List of audio observations, each with shape (2, samples_per_step)
        """
        if self._source_audio is None:
            raise ValueError("Source audio not set. Call set_source_audio() first.")
        
        audiogoal_sequence = []
        self._current_step = 0  # Reset step counter
        
        for step_idx, rir in enumerate(rir_sequence):
            audiogoal = self.compute_audiogoal_single_step(rir, step_idx)
            audiogoal_sequence.append(audiogoal)
        
        self.logger.info(f"Processed {len(rir_sequence)} steps")
        return audiogoal_sequence
    
    def reset(self) -> None:
        """Reset internal state for new episode."""
        self._current_step = 0
        self._current_sample_index = 0
    
    @staticmethod
    def crossfade(x1: np.ndarray, x2: np.ndarray, crossfade_duration: float = 0.05) -> np.ndarray:
        """
        Apply crossfade between two audio segments (from continuous_simulator.py).
        
        Args:
            x1: First audio segment
            x2: Second audio segment  
            crossfade_duration: Crossfade duration in seconds
            
        Returns:
            Crossfaded audio
        """
        sr = 16000  # Assume standard sampling rate
        crossfade_samples = int(crossfade_duration * sr)
        
        if x1.shape[1] < crossfade_samples or x2.shape[1] < crossfade_samples:
            return x2  # Skip crossfade if segments too short
        
        x2_weight = np.arange(crossfade_samples + 1) / crossfade_samples
        x1_weight = np.flip(x2_weight)
        
        crossfaded_part = (x1[:, :crossfade_samples+1] * x1_weight + 
                          x2[:, :crossfade_samples+1] * x2_weight)
        remaining_part = x2[:, crossfade_samples+1:]
        
        return np.concatenate([crossfaded_part, remaining_part], axis=1)


def demo_usage():
    """Demonstrate usage of the standalone audio processor."""
    
    # Initialize processor
    processor = SoundSpacesAudioProcessor(sampling_rate=16000, step_duration=1.0)
    
    # Example 1: Using synthetic data
    print("=== Demo with synthetic data ===")
    
    # Create synthetic source audio (3 seconds of sine wave)
    duration = 3.0
    freq = 440  # A4 note
    t = np.linspace(0, duration, int(duration * 16000), False)
    source_audio = 0.3 * np.sin(2 * np.pi * freq * t)
    
    processor.set_source_audio(source_audio)
    
    # Create synthetic RIRs for 5 steps
    rir_length = 8000  # 0.5 seconds
    rir_sequence = []
    
    for step in range(5):
        # Create simple decaying impulse response
        impulse = np.zeros((rir_length, 2))
        impulse[0] = [1.0, 0.8]  # Direct sound, slightly quieter on right
        
        # Add some delayed reflections
        for delay in [800, 1600, 3200]:
            if delay < rir_length:
                decay = 0.3 * np.exp(-delay / 4000)  # Exponential decay
                impulse[delay] += [decay, decay * 0.9]
        
        rir_sequence.append(impulse)
    
    # Process the sequence
    audiogoal_sequence = processor.compute_audiogoal_sequence(rir_sequence)
    
    print(f"Generated {len(audiogoal_sequence)} audio observations")
    for i, audiogoal in enumerate(audiogoal_sequence):
        print(f"Step {i}: shape {audiogoal.shape}, max amplitude: {np.max(np.abs(audiogoal)):.3f}")
    
    # Example 2: From files (if available)
    print("\n=== Demo with file loading (if files exist) ===")
    
    # This would work with actual files:
    # processor.set_source_audio("path/to/audio.wav") 
    # rir = processor.load_rir_from_file("path/to/rir.wav")
    # audiogoal = processor.compute_audiogoal_single_step(rir)
    
    print("To use with actual files:")
    print("1. processor.set_source_audio('audio_file.wav')")
    print("2. rir = processor.load_rir_from_file('rir_file.wav')")  
    print("3. audiogoal = processor.compute_audiogoal_single_step(rir)")


if __name__ == "__main__":
    demo_usage()