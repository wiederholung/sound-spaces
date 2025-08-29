#!/usr/bin/env python3
"""Debug the reverb carryover issue."""

import numpy as np
from standalone_audio_processor import SoundSpacesAudioProcessor

def debug_reverb_issue():
    processor = SoundSpacesAudioProcessor()
    
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
        start_sample = step * 16000
        print(f"Start sample: {start_sample}")
        print(f"RIR length: {rir_length}")
        print(f"Condition (start - rir_length < 0): {start_sample - rir_length < 0}")
        
        audiogoal = processor.compute_audiogoal_single_step(rir)  # Don't provide step index
        step_results.append(audiogoal)
        
        print(f"Output shape: {audiogoal.shape}")
        print(f"Max amplitude: {np.max(np.abs(audiogoal)):.6f}")
        print(f"Mean amplitude: {np.mean(np.abs(audiogoal)):.6f}")
        print(f"RMS: {np.sqrt(np.mean(audiogoal**2)):.6f}")
        
        # Check different regions of the audio
        first_quarter = audiogoal[:, :4000]
        last_quarter = audiogoal[:, -4000:]
        print(f"First quarter RMS: {np.sqrt(np.mean(first_quarter**2)):.6f}")
        print(f"Last quarter RMS: {np.sqrt(np.mean(last_quarter**2)):.6f}")
        
        if step > 0:
            diff = np.mean(np.abs(step_results[step] - step_results[step-1]))
            print(f"Mean absolute difference from step {step-1}: {diff:.6f}")
            correlation = np.corrcoef(step_results[step].flatten(), step_results[step-1].flatten())[0,1]
            print(f"Correlation with step {step-1}: {correlation:.6f}")

if __name__ == "__main__":
    debug_reverb_issue()