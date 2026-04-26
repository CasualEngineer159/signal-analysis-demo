import unittest
import numpy as np

# Adjust path to import from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis import perform_fft, perform_stft

class TestAnalysisFunctions(unittest.TestCase):
    """Unit tests for the signal analysis functions."""

    def setUp(self):
        """Set up common parameters for the tests."""
        self.fs = 1000  # Sampling frequency of 1 kHz
        self.duration = 2
        self.t = np.linspace(0, self.duration, self.duration * self.fs, endpoint=False)
        self.y_empty = np.array([])

    def test_fft_nominal(self):
        """Test FFT with a simple sine wave."""
        test_freq = 50  # 50 Hz
        y_signal = np.sin(2 * np.pi * test_freq * self.t)
        
        xf, yf = perform_fft(y_signal, self.fs)
        
        # Assert that the output arrays have the correct shape
        self.assertEqual(xf.ndim, 1)
        self.assertEqual(yf.ndim, 1)
        self.assertEqual(xf.shape, yf.shape)
        
        # Assert that the peak frequency is at the correct location
        peak_freq_index = np.argmax(yf)
        self.assertAlmostEqual(xf[peak_freq_index], test_freq, delta=1.0)

    def test_fft_edge_case_empty(self):
        """Test FFT with an empty input array."""
        xf, yf = perform_fft(self.y_empty, self.fs)
        self.assertEqual(len(xf), 0)
        self.assertEqual(len(yf), 0)

    def test_stft_nominal(self):
        """Test STFT with a simple sine wave."""
        y_signal = np.sin(2 * np.pi * 50 * self.t)
        nperseg = 256
        noverlap = 128
        
        f, t, Zxx = perform_stft(y_signal, self.fs, 'hann', nperseg, noverlap)
        
        # Assert that the output arrays have the correct dimensions
        self.assertEqual(f.ndim, 1)
        self.assertEqual(t.ndim, 1)
        self.assertEqual(Zxx.ndim, 2)
        
        # Assert that the dimensions match up
        self.assertEqual(Zxx.shape[0], f.shape[0])
        self.assertEqual(Zxx.shape[1], t.shape[0])

    def test_stft_edge_case_empty(self):
        """Test STFT with an empty input array."""
        f, t, Zxx = perform_stft(self.y_empty, self.fs, 'hann', 256, 128)
        self.assertEqual(len(f), 0)
        self.assertEqual(len(t), 0)
        self.assertEqual(Zxx.size, 0)

    def test_stft_edge_case_window_too_large(self):
        """Test STFT when nperseg is larger than the signal length."""
        y_short = self.t[:100] # Signal of length 100
        nperseg = 256 # Window larger than signal
        
        # The function should handle this gracefully without error
        try:
            f, t, Zxx = perform_stft(y_short, self.fs, 'hann', nperseg, nperseg // 2)
            # Assert that the STFT was computed with a reduced window size
            self.assertEqual(Zxx.shape[0], len(y_short) // 2 + 1)
        except Exception as e:
            self.fail(f"STFT raised an exception with a large window: {e}")

if __name__ == '__main__':
    unittest.main()
