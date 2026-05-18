import unittest

# Adjust path to import from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.signal_engine import generate_pipeline_data
from models.signal_models import SineModel

class MockController:
    """A simple mock controller to pass into the pipeline engine without UI dependencies."""
    def __init__(self, model):
        self.model = model

class TestSignalEngine(unittest.TestCase):
    """Unit tests for the signal engine pipeline."""

    def test_generate_pipeline_data_padding_logic(self):
        """Test that the padding logic and time array cropping works perfectly."""
        # Parameters
        duration = 2.0
        max_freq = 10.0
        stft_window_size = 256
        stft_overlap = 128
        stft_window_type = 'hann'
        
        model = SineModel(frequency=max_freq)
        controllers = [MockController(model)]
        
        # Generate the pipeline data
        result = generate_pipeline_data(
            controllers, 
            duration, 
            max_freq, 
            stft_window_size, 
            stft_overlap, 
            stft_window_type
        )
        
        fs = result.fs
        expected_padding = (stft_window_size / 2) / fs
        
        # 1. t_ext and y_ext should be longer than t and y due to the padding
        self.assertGreater(len(result.t_ext), len(result.t))
        self.assertGreater(len(result.y_ext), len(result.y))

        # Verify the extended time starts at exactly -padding_time
        self.assertAlmostEqual(result.t_ext[0], -expected_padding)

        # 2. The core time vector (t) should strictly start at 0 and end before duration
        self.assertAlmostEqual(result.t[0], 0.0)
        self.assertLess(result.t[-1], duration)
        
        # Ensure that extracting the core didn't lose sample points (e.g. t is exactly [0, duration))
        expected_core_length = int(fs * duration)
        self.assertEqual(len(result.t), expected_core_length)
        
        # 3. Ensure the dimensions of the returned STFT matrices align
        # Zxx_ext shape should be (frequencies, time_segments)
        self.assertEqual(result.Zxx_ext.shape[0], len(result.f))
        self.assertEqual(result.Zxx_ext.shape[1], len(result.t_stft_ext))

        # Ensure that flux and t_stft_core lengths align
        self.assertEqual(len(result.flux), len(result.t_stft_core))

if __name__ == '__main__':
    unittest.main()
