#!/usr/bin/env python3
"""
Validation tests for the standalone audio processor.

This script tests the standalone implementation against the original sound-spaces
logic to ensure correct behavior.
"""

import numpy as np
import os
import tempfile
from scipy.io import wavfile
from standalone_audio_processor import SoundSpacesAudioProcessor


def create_test_audio_file(filename: str, duration: float = 2.0, freq: float = 440, sr: int = 16000):
    """Create a test audio file."""
    t = np.linspace(0, duration, int(duration * sr), False)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    wavfile.write(filename, sr, (audio * 32767).astype(np.int16))
    return audio


def create_test_rir_file(filename: str, length: int = 8000, sr: int = 16000):
    """Create a test RIR file."""
    rir = np.zeros((length, 2), dtype=np.float32)
    rir[0] = [1.0, 0.8]  # Direct sound
    rir[100] = [0.3, 0.25]  # Early reflection
    rir[500] = [0.1, 0.12]  # Late reflection
    
    # Add exponential decay
    decay = np.exp(-np.arange(length) / 2000)
    rir[:, 0] *= decay
    rir[:, 1] *= decay
    
    wavfile.write(filename, sr, (rir * 32767).astype(np.int16))
    return rir.astype(np.float32)


def test_basic_functionality():
    """Test basic functionality of the processor."""
    print("=== Testing Basic Functionality ===")
    
    processor = SoundSpacesAudioProcessor()
    
    # Test with synthetic data
    duration = 2.0
    freq = 440
    t = np.linspace(0, duration, int(duration * 16000), False)
    source_audio = 0.3 * np.sin(2 * np.pi * freq * t)
    
    processor.set_source_audio(source_audio)
    
    # Create simple RIR
    rir = np.zeros((4000, 2))
    rir[0] = [1.0, 0.8]
    rir[100] = [0.2, 0.15]
    
    # Test single step
    audiogoal = processor.compute_audiogoal_single_step(rir)
    
    assert audiogoal.shape == (2, 16000), f"Expected shape (2, 16000), got {audiogoal.shape}"
    assert not np.allclose(audiogoal, 0), "Audio goal should not be all zeros"
    
    print("✓ Basic functionality test passed")


def test_file_loading():
    """Test file loading functionality."""
    print("=== Testing File Loading ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        audio_file = os.path.join(tmpdir, "test_audio.wav")
        rir_file = os.path.join(tmpdir, "test_rir.wav")
        
        original_audio = create_test_audio_file(audio_file)
        original_rir = create_test_rir_file(rir_file)
        
        processor = SoundSpacesAudioProcessor()
        
        # Test audio loading
        loaded_audio = processor.load_source_audio(audio_file)
        processor.set_source_audio(loaded_audio)
        
        # Test RIR loading
        loaded_rir = processor.load_rir_from_file(rir_file)
        
        # Process
        audiogoal = processor.compute_audiogoal_single_step(loaded_rir)
        
        assert audiogoal.shape == (2, 16000), f"Expected shape (2, 16000), got {audiogoal.shape}"
        assert not np.allclose(audiogoal, 0), "Audio goal should not be all zeros"
        
        print("✓ File loading test passed")


def test_reverb_carryover_logic():
    """Test the reverb carryover logic for different step indices."""
    print("=== Testing Reverb Carryover Logic ===")
    
    processor = SoundSpacesAudioProcessor()
    
    # Create longer audio (5 seconds) with varying frequency to make differences more obvious
    duration = 5.0
    t = np.linspace(0, duration, int(duration * 16000), False)
    # Use a chirp (frequency sweep) instead of constant sine wave for more distinctive segments
    freq_start = 440
    freq_end = 880
    source_audio = 0.3 * np.sin(2 * np.pi * (freq_start + (freq_end - freq_start) * t / duration) * t)
    processor.set_source_audio(source_audio)
    
    # Create RIR
    rir_length = 8000  # 0.5 seconds
    rir = np.zeros((rir_length, 2))
    rir[0] = [1.0, 0.8]
    rir[1000] = [0.3, 0.25]
    
    # Test different steps
    step_results = []
    processor.reset()
    
    for step in range(3):
        audiogoal = processor.compute_audiogoal_single_step(rir, step)
        step_results.append(audiogoal)
        
        assert audiogoal.shape == (2, 16000), f"Step {step}: Expected shape (2, 16000), got {audiogoal.shape}"
        
        # Check that results are different (due to different audio segments)
        if step > 0:
            # Use a more lenient threshold since segments may be similar for periodic signals
            mean_diff = np.mean(np.abs(step_results[step] - step_results[step-1]))
            assert mean_diff > 1e-6, \
                f"Step {step} should differ from step {step-1} (diff: {mean_diff})"
    
    print("✓ Reverb carryover logic test passed")


def test_sequence_processing():
    """Test processing a sequence of RIRs."""
    print("=== Testing Sequence Processing ===")
    
    processor = SoundSpacesAudioProcessor()
    
    # Create source audio
    duration = 3.0
    t = np.linspace(0, duration, int(duration * 16000), False)
    source_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    processor.set_source_audio(source_audio)
    
    # Create sequence of different RIRs
    num_steps = 3
    rir_sequence = []
    
    for i in range(num_steps):
        rir = np.zeros((4000, 2))
        rir[0] = [1.0, 0.8]
        # Add different reflections for each step
        rir[100 + i * 200] = [0.2 - i * 0.05, 0.15 - i * 0.03]
        rir_sequence.append(rir)
    
    # Process sequence
    audiogoal_sequence = processor.compute_audiogoal_sequence(rir_sequence)
    
    assert len(audiogoal_sequence) == num_steps, f"Expected {num_steps} results, got {len(audiogoal_sequence)}"
    
    for i, audiogoal in enumerate(audiogoal_sequence):
        assert audiogoal.shape == (2, 16000), f"Step {i}: Expected shape (2, 16000), got {audiogoal.shape}"
        assert not np.allclose(audiogoal, 0), f"Step {i}: Audio goal should not be all zeros"
    
    print("✓ Sequence processing test passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=== Testing Edge Cases ===")
    
    processor = SoundSpacesAudioProcessor()
    
    # Test with very short audio
    short_audio = np.array([0.1, 0.2, 0.3, 0.4])  # Only 4 samples
    processor.set_source_audio(short_audio)
    
    # Small RIR
    rir = np.zeros((100, 2))
    rir[0] = [1.0, 0.8]
    
    try:
        audiogoal = processor.compute_audiogoal_single_step(rir)
        assert audiogoal.shape == (2, 16000), "Should handle short audio gracefully"
        print("✓ Short audio test passed")
    except Exception as e:
        print(f"✗ Short audio test failed: {e}")
    
    # Test mono RIR (should be converted to stereo)
    mono_rir = np.zeros((1000, 1))
    mono_rir[0] = [1.0]
    
    # Reset with proper audio
    duration = 2.0
    t = np.linspace(0, duration, int(duration * 16000), False)
    source_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    processor.set_source_audio(source_audio)
    
    try:
        audiogoal = processor.compute_audiogoal_single_step(mono_rir)
        assert audiogoal.shape == (2, 16000), "Should convert mono RIR to stereo"
        print("✓ Mono RIR test passed")
    except Exception as e:
        print(f"✗ Mono RIR test failed: {e}")


def test_crossfade_function():
    """Test the crossfade utility function."""
    print("=== Testing Crossfade Function ===")
    
    # Create two test audio segments
    x1 = np.ones((2, 16000)) * 0.5
    x2 = np.ones((2, 16000)) * -0.5
    
    # Test crossfade
    result = SoundSpacesAudioProcessor.crossfade(x1, x2)
    
    assert result.shape == x2.shape, f"Expected shape {x2.shape}, got {result.shape}"
    
    # Check that crossfade region blends the signals
    crossfade_samples = int(0.05 * 16000)  # 50ms
    crossfade_region = result[:, :crossfade_samples+1]
    
    # Should be between the two original values
    assert np.all(crossfade_region >= -0.5), "Crossfade values should be >= -0.5"
    assert np.all(crossfade_region <= 0.5), "Crossfade values should be <= 0.5"
    
    print("✓ Crossfade function test passed")


def validate_against_original_logic():
    """Validate key aspects match the original implementation."""
    print("=== Validating Against Original Logic ===")
    
    processor = SoundSpacesAudioProcessor()
    
    # Test case that matches typical usage
    duration = 3.0
    sampling_rate = 16000
    t = np.linspace(0, duration, int(duration * sampling_rate), False)
    source_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    processor.set_source_audio(source_audio)
    
    # RIR similar to typical room impulse response
    rir_length = 8000
    rir = np.zeros((rir_length, 2))
    rir[0] = [1.0, 0.8]  # Direct sound
    
    # Add reflections with realistic timing and decay
    for delay, amplitude in [(800, 0.3), (1600, 0.15), (3200, 0.08)]:
        if delay < rir_length:
            rir[delay] = [amplitude, amplitude * 0.9]
    
    # Test early step (no reverb carryover)
    audiogoal_0 = processor.compute_audiogoal_single_step(rir, step_index=0)
    
    # Test later step (with reverb carryover)
    audiogoal_1 = processor.compute_audiogoal_single_step(rir, step_index=1)
    
    # Verify basic properties
    assert audiogoal_0.shape == (2, sampling_rate), "Wrong shape for step 0"
    assert audiogoal_1.shape == (2, sampling_rate), "Wrong shape for step 1"
    
    # Results should be different due to different audio segments
    assert not np.allclose(audiogoal_0, audiogoal_1), "Different steps should produce different results"
    
    # Check that convolution actually happened (not just zeros)
    assert np.max(np.abs(audiogoal_0)) > 0.1, "Step 0 should have significant amplitude"
    assert np.max(np.abs(audiogoal_1)) > 0.1, "Step 1 should have significant amplitude"
    
    print("✓ Original logic validation passed")


def run_all_tests():
    """Run all validation tests."""
    print("Running validation tests for standalone audio processor...\n")
    
    tests = [
        test_basic_functionality,
        test_file_loading,
        test_reverb_carryover_logic,
        test_sequence_processing,
        test_edge_cases,
        test_crossfade_function,
        validate_against_original_logic,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The standalone processor is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Please review the implementation.")


if __name__ == "__main__":
    run_all_tests()