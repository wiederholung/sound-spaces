#!/usr/bin/env python3
"""
Example Usage of StandaloneAudioProcessor
==========================================

This script demonstrates how to use the StandaloneAudioProcessor with 
actual impulse responses and source audio, following the exact workflow
from the sound-spaces repository.

Example scenarios covered:
1. Processing pre-computed impulse responses from a navigation episode
2. Generating step-by-step spatialized audio observations  
3. Creating complete concatenated audio for the entire episode
4. Comparing with and without crossfading
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from standalone_audio_processor import StandaloneAudioProcessor
import librosa
import soundfile as sf

def create_realistic_impulse_responses(num_steps=10, rir_length=2048, sampling_rate=16000):
    """
    Create realistic-looking impulse responses that simulate movement through space.
    
    In a real scenario, these would come from your acoustic simulation or 
    pre-recorded room impulse responses.
    """
    impulse_responses = []
    
    for step in range(num_steps):
        # Simulate movement by varying the direct path and reverberation
        
        # Direct path (gets weaker and delayed as we move further from source)
        distance_factor = 1.0 + step * 0.2  # Increasing distance
        direct_delay = int(step * 3)  # Increasing delay (samples)
        direct_amplitude = 1.0 / distance_factor  # Inverse square law approximation
        
        # Create IR with direct path and reverb
        rir = np.zeros((rir_length, 2))
        
        # Direct path (stronger in left ear initially, shifts to right)
        left_direct = direct_amplitude * (1.0 - step / num_steps)
        right_direct = direct_amplitude * (step / num_steps)
        
        if direct_delay < rir_length:
            rir[direct_delay, 0] = left_direct
            rir[direct_delay, 1] = right_direct
        
        # Add realistic reverb tail
        reverb_start = direct_delay + 10
        if reverb_start < rir_length:
            reverb_length = rir_length - reverb_start
            # Exponential decay
            decay = np.exp(-np.arange(reverb_length) / (sampling_rate * 0.2))  # 200ms RT60
            
            # Random reverb with exponential decay
            reverb_left = decay * np.random.randn(reverb_length) * 0.1 * direct_amplitude
            reverb_right = decay * np.random.randn(reverb_length) * 0.1 * direct_amplitude
            
            rir[reverb_start:, 0] = reverb_left
            rir[reverb_start:, 1] = reverb_right
        
        impulse_responses.append(rir)
    
    return impulse_responses


def load_or_create_source_audio(audio_path=None, duration=5.0, sampling_rate=16000):
    """
    Load source audio file or create a synthetic one for demonstration.
    """
    if audio_path and os.path.exists(audio_path):
        print(f"Loading source audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=sampling_rate)
        return audio
    else:
        print("Creating synthetic source audio for demonstration...")
        # Create a more complex synthetic audio signal
        t = np.linspace(0, duration, int(sampling_rate * duration))
        
        # Musical chord (C major)
        c4 = 261.63  # C4
        e4 = 329.63  # E4  
        g4 = 392.00  # G4
        
        audio = (0.3 * np.sin(2 * np.pi * c4 * t) +
                0.2 * np.sin(2 * np.pi * e4 * t) +
                0.2 * np.sin(2 * np.pi * g4 * t))
        
        # Add some envelope to make it more interesting
        envelope = np.exp(-t / 2.0)  # Exponential decay
        audio = audio * envelope
        
        return audio


def demonstrate_basic_usage():
    """
    Demonstrate basic usage of the StandaloneAudioProcessor.
    """
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the processor with realistic parameters
    processor = StandaloneAudioProcessor(
        sampling_rate=16000,
        step_time=0.1,           # 100ms per step
        crossfade_enabled=True,
        crossfade_duration=0.02  # 20ms crossfade
    )
    
    # Create source audio
    source_audio = load_or_create_source_audio()
    print(f"Source audio: {len(source_audio)} samples ({len(source_audio)/16000:.2f}s)")
    
    # Create realistic impulse responses (simulating navigation episode)
    num_steps = 15
    impulse_responses = create_realistic_impulse_responses(
        num_steps=num_steps, 
        rir_length=2048,
        sampling_rate=16000
    )
    print(f"Created {len(impulse_responses)} impulse responses")
    
    # Process the audio
    print("Processing audio with impulse responses...")
    step_observations, complete_audio = processor.process_impulse_responses(
        source_audio, impulse_responses
    )
    
    # Display results
    print(f"\nResults:")
    print(f"- Generated {len(step_observations)} step observations")
    print(f"- Each step: {step_observations[0].shape} ({step_observations[0].shape[1]/16000:.3f}s)")
    print(f"- Complete audio: {complete_audio.shape} ({complete_audio.shape[1]/16000:.2f}s)")
    
    # Save outputs
    os.makedirs("/tmp/audio_outputs", exist_ok=True)
    
    # Save source audio for comparison
    sf.write("/tmp/audio_outputs/source_audio.wav", source_audio, 16000)
    
    # Save complete spatialized audio
    processor.save_audio(complete_audio, "/tmp/audio_outputs/complete_spatialized.wav")
    
    # Save individual step observations
    for i, step_audio in enumerate(step_observations):
        processor.save_audio(step_audio, f"/tmp/audio_outputs/step_{i:02d}.wav")
    
    print(f"\nSaved outputs to /tmp/audio_outputs/")
    
    return processor, source_audio, impulse_responses, step_observations, complete_audio


def compare_crossfade_effects(processor, source_audio, impulse_responses):
    """
    Compare audio processing with and without crossfading.
    """
    print("\n" + "=" * 60)
    print("CROSSFADE COMPARISON")
    print("=" * 60)
    
    # Process without crossfading
    processor_no_crossfade = StandaloneAudioProcessor(
        sampling_rate=processor.sampling_rate,
        step_time=processor.step_time,
        crossfade_enabled=False
    )
    
    print("Processing without crossfading...")
    _, complete_no_crossfade = processor_no_crossfade.process_impulse_responses(
        source_audio, impulse_responses
    )
    
    # Process with crossfading  
    processor_with_crossfade = StandaloneAudioProcessor(
        sampling_rate=processor.sampling_rate,
        step_time=processor.step_time,
        crossfade_enabled=True,
        crossfade_duration=0.05  # 50ms crossfade
    )
    
    print("Processing with crossfading...")
    _, complete_with_crossfade = processor_with_crossfade.process_impulse_responses(
        source_audio, impulse_responses
    )
    
    # Save both versions for comparison
    processor.save_audio(complete_no_crossfade, "/tmp/audio_outputs/complete_no_crossfade.wav")
    processor.save_audio(complete_with_crossfade, "/tmp/audio_outputs/complete_with_crossfade.wav")
    
    print("Saved crossfade comparison files:")
    print("- complete_no_crossfade.wav")
    print("- complete_with_crossfade.wav")
    
    # Analyze the difference
    if complete_no_crossfade.shape == complete_with_crossfade.shape:
        difference = np.abs(complete_with_crossfade - complete_no_crossfade)
        max_diff = np.max(difference)
        avg_diff = np.mean(difference)
        
        print(f"\nCrossfade impact analysis:")
        print(f"- Maximum difference: {max_diff:.6f}")
        print(f"- Average difference: {avg_diff:.6f}")
        print(f"- Relative difference: {max_diff/np.max(np.abs(complete_no_crossfade)):.2%}")


def analyze_step_by_step_audio(step_observations):
    """
    Analyze the step-by-step audio observations.
    """
    print("\n" + "=" * 60)
    print("STEP-BY-STEP ANALYSIS")
    print("=" * 60)
    
    print("Audio level analysis for each step:")
    print("Step | Left RMS  | Right RMS | Stereo Width")
    print("-" * 45)
    
    for i, step_audio in enumerate(step_observations):
        left_rms = np.sqrt(np.mean(step_audio[0] ** 2))
        right_rms = np.sqrt(np.mean(step_audio[1] ** 2))
        
        # Stereo width: correlation between channels (lower = wider)
        correlation = np.corrcoef(step_audio[0], step_audio[1])[0, 1]
        stereo_width = 1.0 - abs(correlation)  # Simple width metric
        
        print(f"{i:4d} | {left_rms:.6f} | {right_rms:.6f} | {stereo_width:.3f}")
    
    # Create a simple visualization if matplotlib is available
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot RMS levels over time
        plt.subplot(2, 1, 1)
        steps = range(len(step_observations))
        left_rms_values = [np.sqrt(np.mean(step[0] ** 2)) for step in step_observations]
        right_rms_values = [np.sqrt(np.mean(step[1] ** 2)) for step in step_observations]
        
        plt.plot(steps, left_rms_values, 'b-o', label='Left Channel RMS')
        plt.plot(steps, right_rms_values, 'r-o', label='Right Channel RMS')
        plt.xlabel('Step Number')
        plt.ylabel('RMS Level')
        plt.title('Audio Level Evolution Across Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot waveform of first few steps
        plt.subplot(2, 1, 2)
        for i in range(min(3, len(step_observations))):
            offset = i * 0.5
            time_axis = np.arange(len(step_observations[i][0])) / 16000
            plt.plot(time_axis + offset, step_observations[i][0] + i * 0.2, 
                    label=f'Step {i} (Left)', alpha=0.7)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('Waveforms of First 3 Steps (Left Channel)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/audio_outputs/analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved analysis plot to /tmp/audio_outputs/analysis.png")
        
    except ImportError:
        print("matplotlib not available for visualization")


def simulate_real_world_usage():
    """
    Simulate how this would be used in a real-world scenario with 
    pre-computed impulse responses from a navigation episode.
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD USAGE SIMULATION")
    print("=" * 60)
    
    print("Simulating scenario where you have:")
    print("1. Pre-computed impulse responses from navigation episode")
    print("2. Source audio that was playing during the episode")
    print("3. Need to generate what the agent would have heard")
    
    # Simulate loading IRs from a real episode (e.g., from your repository)
    print("\n1. Loading impulse responses...")
    print("   (In real usage: loaded from your acoustic simulation)")
    
    # This would be replaced with actual loading, e.g.:
    # impulse_responses = [np.load(f"irs/step_{i}.npy") for i in range(num_steps)]
    impulse_responses = create_realistic_impulse_responses(
        num_steps=20,
        rir_length=4096,  # Higher quality IRs
        sampling_rate=16000
    )
    
    print(f"   Loaded {len(impulse_responses)} IRs")
    
    # Simulate loading source audio
    print("\n2. Loading source audio...")
    print("   (In real usage: the audio file that was playing)")
    
    # This would be replaced with actual loading, e.g.:
    # source_audio = processor.load_source_audio("path/to/source.wav")
    source_audio = load_or_create_source_audio(duration=8.0)  # Longer audio
    
    print(f"   Loaded {len(source_audio)/16000:.1f}s of source audio")
    
    # Process with optimal settings for quality
    print("\n3. Processing with high-quality settings...")
    
    processor = StandaloneAudioProcessor(
        sampling_rate=16000,
        step_time=0.1,
        crossfade_enabled=True,
        crossfade_duration=0.03  # 30ms crossfade
    )
    
    step_observations, complete_audio = processor.process_impulse_responses(
        source_audio, impulse_responses
    )
    
    print(f"   Generated {complete_audio.shape[1]/16000:.2f}s of spatialized audio")
    
    # Save high-quality output
    processor.save_audio(complete_audio, "/tmp/audio_outputs/final_episode_audio.wav")
    
    print("\n4. Results saved:")
    print("   - final_episode_audio.wav: Complete spatialized audio for the episode")
    print("   - This is what the agent would have heard during navigation")
    
    return step_observations, complete_audio


def main():
    """
    Main demonstration function.
    """
    print("StandaloneAudioProcessor - Complete Example")
    print("=" * 60)
    print("This example demonstrates the complete workflow for processing")
    print("impulse responses with source audio to generate spatialized")
    print("audio observations, following the sound-spaces pipeline.")
    print("")
    
    try:
        # Basic usage demonstration
        processor, source_audio, impulse_responses, step_observations, complete_audio = demonstrate_basic_usage()
        
        # Compare crossfade effects
        compare_crossfade_effects(processor, source_audio, impulse_responses)
        
        # Analyze step-by-step results
        analyze_step_by_step_audio(step_observations)
        
        # Simulate real-world usage
        simulate_real_world_usage()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All example files have been saved to /tmp/audio_outputs/")
        print("")
        print("Key files:")
        print("- source_audio.wav: Original source audio")
        print("- complete_spatialized.wav: Final spatialized audio")
        print("- complete_no_crossfade.wav: Without crossfading")
        print("- complete_with_crossfade.wav: With crossfading")
        print("- step_XX.wav: Individual step observations")
        print("- final_episode_audio.wav: High-quality episode audio")
        print("")
        print("You can now listen to these files to hear the spatial audio effects!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()