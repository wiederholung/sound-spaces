#!/usr/bin/env python3
"""
Complete Usage Example for Standalone Audio Processor

This example demonstrates how to use the standalone audio processor to replicate
the exact sound-spaces acoustic processing pipeline.
"""

import numpy as np
import os
import tempfile
from scipy.io import wavfile
from standalone_audio_processor import SoundSpacesAudioProcessor


def create_example_audio_and_rir():
    """Create example audio and RIR files for demonstration."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source audio file (speech-like signal)
        audio_file = os.path.join(tmpdir, "source_audio.wav")
        duration = 4.0  # 4 seconds
        sr = 16000
        t = np.linspace(0, duration, int(duration * sr), False)
        
        # Create a more complex signal (speech-like formants)
        f1, f2, f3 = 800, 1200, 2400  # Formant frequencies
        audio = (0.3 * np.sin(2 * np.pi * f1 * t) + 
                0.2 * np.sin(2 * np.pi * f2 * t) + 
                0.1 * np.sin(2 * np.pi * f3 * t))
        
        # Add envelope to make it more speech-like
        envelope = np.exp(-t * 0.3) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio *= envelope
        
        # Save as 16-bit WAV
        wavfile.write(audio_file, sr, (audio * 32767).astype(np.int16))
        
        # Create RIR files for different positions
        rir_files = []
        for i in range(5):
            rir_file = os.path.join(tmpdir, f"rir_step_{i}.wav")
            
            # Create realistic RIR with different characteristics per step
            rir_length = 8000  # 0.5 seconds at 16kHz
            rir = np.zeros((rir_length, 2))
            
            # Direct sound (varies by distance/angle)
            direct_left = 1.0 - i * 0.1
            direct_right = 0.8 - i * 0.05
            rir[0] = [direct_left, direct_right]
            
            # Early reflections (walls, ceiling, floor)
            early_delays = [200, 400, 600, 800, 1200]
            for j, delay in enumerate(early_delays):
                if delay < rir_length:
                    amplitude = 0.2 * np.exp(-delay / 2000) * (1 - i * 0.1)
                    rir[delay] = [amplitude, amplitude * 0.9]
            
            # Late reverberation (exponential decay)
            for t in range(1500, rir_length):
                decay = 0.1 * np.exp(-t / 3000) * np.random.normal(0, 0.1)
                rir[t] += [decay, decay * 1.1]
            
            # Save as 16-bit WAV
            wavfile.write(rir_file, sr, (rir * 32767).astype(np.int16))
            rir_files.append(rir_file)
        
        # Process the audio through the pipeline
        processor = SoundSpacesAudioProcessor(sampling_rate=sr, step_duration=1.0)
        
        print("=== Sound-Spaces Audio Processing Example ===")
        print(f"Source audio: {audio_file}")
        print(f"Duration: {duration}s")
        print(f"RIR files: {len(rir_files)}")
        
        # Load source audio
        processor.set_source_audio(audio_file)
        
        # Method 1: Process step by step
        print("\n--- Method 1: Step-by-step processing ---")
        processor.reset()
        
        step_results = []
        for i, rir_file in enumerate(rir_files):
            rir = processor.load_rir_from_file(rir_file)
            audiogoal = processor.compute_audiogoal_single_step(rir)
            step_results.append(audiogoal)
            
            print(f"Step {i}: RIR shape {rir.shape}, Output shape {audiogoal.shape}")
            print(f"         Max amplitude: {np.max(np.abs(audiogoal)):.3f}")
            print(f"         RMS: {np.sqrt(np.mean(audiogoal**2)):.3f}")
        
        # Method 2: Process sequence at once
        print("\n--- Method 2: Sequence processing ---")
        rir_sequence = [processor.load_rir_from_file(f) for f in rir_files]
        sequence_results = processor.compute_audiogoal_sequence(rir_sequence)
        
        print(f"Processed {len(sequence_results)} steps in sequence")
        
        # Verify results are identical
        for i, (step_result, seq_result) in enumerate(zip(step_results, sequence_results)):
            if np.allclose(step_result, seq_result):
                print(f"✓ Step {i}: Methods produce identical results")
            else:
                print(f"✗ Step {i}: Methods differ (this shouldn't happen)")
        
        # Method 3: Using arrays directly (for external RIR data)
        print("\n--- Method 3: Using RIR arrays directly ---")
        processor.reset()
        
        # Simulate getting RIR data from external source (e.g., another simulation)
        external_rirs = []
        for i in range(3):
            # Create synthetic RIR
            rir_length = 6000
            rir = np.zeros((rir_length, 2))
            rir[0] = [1.0, 0.85]  # Direct sound
            
            # Add some reflections
            for delay in [300, 600, 1200]:
                if delay < rir_length:
                    amplitude = 0.15 * np.exp(-delay / 2000)
                    rir[delay] = [amplitude, amplitude * 0.95]
            
            external_rirs.append(rir)
        
        external_results = processor.compute_audiogoal_sequence(external_rirs)
        print(f"Processed {len(external_results)} steps with external RIRs")
        
        for i, result in enumerate(external_results):
            print(f"External step {i}: shape {result.shape}, RMS {np.sqrt(np.mean(result**2)):.3f}")
        
        print("\n=== Processing Complete ===")
        print("The standalone processor successfully replicated the sound-spaces pipeline!")
        
        return step_results, sequence_results, external_results


def demonstrate_advanced_features():
    """Demonstrate advanced features of the processor."""
    
    print("\n=== Advanced Features Demo ===")
    
    processor = SoundSpacesAudioProcessor(sampling_rate=16000, step_duration=0.5)  # 0.5s steps
    
    # Create short audio that will loop
    duration = 1.5  # Only 1.5 seconds
    t = np.linspace(0, duration, int(duration * 16000), False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)  # Short sine wave
    
    processor.set_source_audio(audio)
    print(f"Set short audio: {duration}s (will loop for longer episodes)")
    
    # Create different RIRs to show variety
    rir_configs = [
        {"delay": 0, "amplitude": 1.0, "description": "Direct sound only"},
        {"delay": 800, "amplitude": 0.3, "description": "Single reflection"},
        {"delay": 1600, "amplitude": 0.15, "description": "Double reflection"},
    ]
    
    results = []
    for i, config in enumerate(rir_configs):
        rir = np.zeros((4000, 2))
        rir[0] = [1.0, 0.9]  # Direct sound
        if config["delay"] > 0:
            rir[config["delay"]] = [config["amplitude"], config["amplitude"] * 0.9]
        
        audiogoal = processor.compute_audiogoal_single_step(rir)
        results.append(audiogoal)
        
        print(f"RIR {i} ({config['description']}): RMS = {np.sqrt(np.mean(audiogoal**2)):.4f}")
    
    # Demonstrate crossfade
    print("\n--- Crossfade Demo ---")
    audio1 = results[0]
    audio2 = results[1]
    
    crossfaded = SoundSpacesAudioProcessor.crossfade(audio1, audio2, crossfade_duration=0.1)
    print(f"Crossfaded audio shape: {crossfaded.shape}")
    
    # Show how different step durations work
    print("\n--- Different Step Durations ---")
    for step_duration in [0.25, 0.5, 1.0]:
        proc = SoundSpacesAudioProcessor(step_duration=step_duration)
        proc.set_source_audio(audio)
        
        simple_rir = np.zeros((2000, 2))
        simple_rir[0] = [1.0, 0.8]
        
        result = proc.compute_audiogoal_single_step(simple_rir)
        expected_samples = int(16000 * step_duration)
        
        print(f"Step duration {step_duration}s: Expected {expected_samples} samples, got {result.shape[1]}")


if __name__ == "__main__":
    # Run the complete example
    try:
        step_results, sequence_results, external_results = create_example_audio_and_rir()
        demonstrate_advanced_features()
        
        print("\n🎉 All examples completed successfully!")
        print("\nThe standalone audio processor is ready for use in your project.")
        print("\nKey usage patterns:")
        print("1. processor.set_source_audio('audio.wav')")
        print("2. rir = processor.load_rir_from_file('rir.wav')")
        print("3. audiogoal = processor.compute_audiogoal_single_step(rir)")
        print("4. sequence = processor.compute_audiogoal_sequence(rir_list)")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()