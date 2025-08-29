#!/usr/bin/env python3
"""
Simple integration test to verify the standalone processor works correctly
with realistic data and produces expected outputs.
"""

import numpy as np
import sys
import os
sys.path.append('/home/runner/work/sound-spaces/sound-spaces')

from standalone_audio_processor import StandaloneAudioProcessor

def main():
    print("Running integration test...")
    
    # Create processor
    processor = StandaloneAudioProcessor(
        sampling_rate=16000,
        step_time=0.1,
        crossfade_enabled=True
    )
    
    # Create simple test audio (1 second sine wave)
    t = np.linspace(0, 1, 16000)
    source_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Create 5 simple impulse responses
    ir_length = 512
    impulse_responses = []
    
    for i in range(5):
        # Simple impulse at different delays
        ir = np.zeros((ir_length, 2))
        delay = i * 10
        if delay < ir_length:
            ir[delay, 0] = 0.8  # Left
            ir[delay, 1] = 0.6  # Right
        impulse_responses.append(ir)
    
    # Process
    step_obs, complete = processor.process_impulse_responses(source_audio, impulse_responses)
    
    # Verify results
    assert len(step_obs) == 5, f"Expected 5 steps, got {len(step_obs)}"
    assert complete.shape[0] == 2, f"Expected stereo output, got {complete.shape[0]} channels"
    assert complete.shape[1] == 5 * 1600, f"Expected 8000 samples, got {complete.shape[1]}"
    
    # Check that audio is not all zeros
    assert np.max(np.abs(complete)) > 0, "Output audio is silent"
    
    # Check that steps have reasonable amplitudes
    for i, step in enumerate(step_obs):
        rms = np.sqrt(np.mean(step ** 2))
        assert rms > 0.01, f"Step {i} has very low amplitude: {rms}"
    
    # Save test output
    output_path = "/tmp/integration_test_output.wav"
    processor.save_audio(complete, output_path)
    
    print("✅ Integration test passed!")
    print(f"Generated {complete.shape[1]/16000:.2f}s of audio with {len(step_obs)} steps")
    print(f"Output saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print("🎉 All integration tests successful!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)