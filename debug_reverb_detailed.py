#!/usr/bin/env python3
"""Debug the reverb carryover issue with detailed internal logging."""

import numpy as np
from scipy.signal import fftconvolve
from standalone_audio_processor import SoundSpacesAudioProcessor

class DebugAudioProcessor(SoundSpacesAudioProcessor):
    def compute_audiogoal_single_step(self, rir, step_index=None):
        print(f"  Internal step before: {self._current_step}")
        
        if self._source_audio is None:
            raise ValueError("Source audio not set. Call set_source_audio() first.")
        
        # Ensure RIR has correct shape
        if rir.ndim == 1:
            rir = rir.reshape(-1, 1)
        if rir.shape[1] == 1:
            rir = np.tile(rir, (1, 2))
        
        rir_length = rir.shape[0]
        
        if self._source_audio.shape[0] == self.samples_per_step:
            # Single step case
            binaural_convolved = np.array([
                fftconvolve(self._source_audio, rir[:, channel])
                for channel in range(rir.shape[1])
            ])
            audiogoal = binaural_convolved[:, :self.samples_per_step]
        else:
            # Multi-step case
            if step_index is not None:
                index = step_index
                print(f"  Using provided step_index: {index}")
            else:
                index = self._current_step
                print(f"  Using internal index: {index}")
                self._current_step = (self._current_step + 1) % int(self._audio_length_seconds)
                print(f"  Internal step after increment: {self._current_step}")
            
            start_sample = index * self.samples_per_step
            end_sample = (index + 1) * self.samples_per_step
            print(f"  Actual start_sample: {start_sample}, end_sample: {end_sample}")
            print(f"  Audio length: {self._source_audio.shape[0]}")
            print(f"  Condition (start - rir_length < 0): {start_sample - rir_length < 0}")
            
            if start_sample - rir_length < 0:
                print(f"  Taking early path")
                audio_segment = self._source_audio[:end_sample]
                print(f"  Audio segment shape: {audio_segment.shape}")
                binaural_convolved = np.array([
                    fftconvolve(audio_segment, rir[:, channel])
                    for channel in range(rir.shape[1])
                ])
                audiogoal = binaural_convolved[:, start_sample:end_sample]
            else:
                print(f"  Taking later path with reverb carryover")
                segment_start = start_sample - rir_length + 1
                segment_end = end_sample
                print(f"  Segment: {segment_start}:{segment_end}")
                
                audio_length = self._source_audio.shape[0]
                if segment_end <= audio_length:
                    audio_segment = self._source_audio[segment_start:segment_end]
                    print(f"  Simple segment shape: {audio_segment.shape}")
                else:
                    first_part = self._source_audio[segment_start:]
                    remaining_samples = segment_end - audio_length
                    second_part = self._source_audio[:remaining_samples]
                    audio_segment = np.concatenate([first_part, second_part])
                    print(f"  Wraparound segment shape: {audio_segment.shape}")
                
                binaural_convolved = np.array([
                    fftconvolve(audio_segment, rir[:, channel], mode='valid')
                    for channel in range(rir.shape[1])
                ])
                print(f"  Convolved shape: {binaural_convolved.shape}")
                audiogoal = binaural_convolved
        
        # Ensure correct shape
        if audiogoal.shape[1] < self.samples_per_step:
            padding = self.samples_per_step - audiogoal.shape[1]
            audiogoal = np.pad(audiogoal, [(0, 0), (0, padding)])
            print(f"  Added padding: {padding}")
        elif audiogoal.shape[1] > self.samples_per_step:
            audiogoal = audiogoal[:, :self.samples_per_step]
            print(f"  Trimmed to {self.samples_per_step}")
        
        print(f"  Final shape: {audiogoal.shape}")
        return audiogoal

def debug_reverb_issue():
    processor = DebugAudioProcessor()
    
    # Create longer audio (5 seconds)
    duration = 5.0
    t = np.linspace(0, duration, int(duration * 16000), False)
    source_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    processor.set_source_audio(source_audio)
    
    # Create RIR
    rir_length = 8000  # 0.5 seconds
    rir = np.zeros((rir_length, 2))
    rir[0] = [1.0, 0.8]
    rir[1000] = [0.3, 0.25]
    
    # Test different steps with debugging
    step_results = []
    processor.reset()
    
    for step in range(3):
        print(f"\n=== Step {step} ===")
        
        audiogoal = processor.compute_audiogoal_single_step(rir)
        step_results.append(audiogoal)
        
        print(f"Max amplitude: {np.max(np.abs(audiogoal)):.6f}")
        
        if step > 0:
            diff = np.mean(np.abs(step_results[step] - step_results[step-1]))
            print(f"Mean absolute difference from step {step-1}: {diff:.6f}")

if __name__ == "__main__":
    debug_reverb_issue()