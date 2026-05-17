import unittest
import numpy as np

# Adjust path to import from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis import perform_fft, perform_stft, calculate_spectral_flux, evaluate_detection, get_matched_pairs

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

    def test_spectral_flux_nominal(self):
        """Test calculate_spectral_flux with a distinct anomaly (peak)."""
        # Create a mock STFT magnitude matrix (frequencies x time)
        # Background noise level of 1.0
        Zxx = np.ones((10, 20))
        t_stft = np.linspace(0, 1.0, 20)
        
        # Inject an anomaly at time index 10 (across all frequencies)
        Zxx[:, 10] = 100.0
        
        flux, peak_times = calculate_spectral_flux(Zxx, t_stft)
        
        # Flux array length should match time array length
        self.assertEqual(len(flux), len(t_stft))
        
        # Should detect exactly one peak
        self.assertEqual(len(peak_times), 1)
        
        # The peak time should match the injected anomaly time index (10)
        self.assertEqual(peak_times[0], t_stft[10])

    def test_spectral_flux_edge_case_empty(self):
        """Test calculate_spectral_flux with empty arrays."""
        Zxx_empty = np.array([[]])
        t_empty = np.array([])
        
        flux, peak_times = calculate_spectral_flux(Zxx_empty, t_empty)
        
        self.assertEqual(len(flux), 0)
        self.assertEqual(len(peak_times), 0)

    def test_spectral_flux_edge_case_uniform(self):
        """Test calculate_spectral_flux with a perfectly uniform matrix to test the MAD fallback logic."""
        # A perfectly clean/flat signal yields a uniform matrix
        Zxx_uniform = np.full((10, 20), 5.0)
        t_stft = np.linspace(0, 1.0, 20)
        
        # Since it's uniform, the difference between columns is zero
        # MAD will be 0.0, which triggers the fallback threshold logic to prevent false positives.
        flux, peak_times = calculate_spectral_flux(Zxx_uniform, t_stft)
        
        # Flux should be all zeros
        self.assertTrue(np.all(flux == 0.0))
        
        # No peaks should be detected because the threshold fallback requires exceeding max(flux)*0.1
        self.assertEqual(len(peak_times), 0)

    def test_evaluate_detection_perfect_match(self):
        gt = [1.0, 2.0, 3.0]
        pred = [1.0, 2.05, 2.95]
        res = evaluate_detection(gt, pred, tolerance=0.1)
        self.assertEqual(res['TP'], 3)
        self.assertEqual(res['FP'], 0)
        self.assertEqual(res['FN'], 0)
        self.assertEqual(res['Precision'], 1.0)
        self.assertEqual(res['Recall'], 1.0)
        self.assertEqual(res['F1-Score'], 1.0)

    def test_evaluate_detection_false_positives(self):
        gt = [1.0]
        pred = [1.0, 2.0, 3.0]
        res = evaluate_detection(gt, pred, tolerance=0.1)
        self.assertEqual(res['TP'], 1)
        self.assertEqual(res['FP'], 2)
        self.assertEqual(res['FN'], 0)
        self.assertAlmostEqual(res['Precision'], 1/3)
        self.assertEqual(res['Recall'], 1.0)

    def test_evaluate_detection_false_negatives(self):
        gt = [1.0, 2.0, 3.0]
        pred = [2.0]
        res = evaluate_detection(gt, pred, tolerance=0.1)
        self.assertEqual(res['TP'], 1)
        self.assertEqual(res['FP'], 0)
        self.assertEqual(res['FN'], 2)
        self.assertEqual(res['Precision'], 1.0)
        self.assertAlmostEqual(res['Recall'], 1/3)

    def test_evaluate_detection_tolerance(self):
        gt = [1.0]
        # Prediction is outside the 0.1 tolerance
        pred = [1.2]
        res = evaluate_detection(gt, pred, tolerance=0.1)
        self.assertEqual(res['TP'], 0)
        self.assertEqual(res['FP'], 1)
        self.assertEqual(res['FN'], 1)

        # Now with a larger tolerance
        res_large_tol = evaluate_detection(gt, pred, tolerance=0.3)
        self.assertEqual(res_large_tol['TP'], 1)
        self.assertEqual(res_large_tol['FP'], 0)
        self.assertEqual(res_large_tol['FN'], 0)

    def test_evaluate_detection_empty_inputs(self):
        # Empty predictions
        res1 = evaluate_detection([1.0], [])
        self.assertEqual(res1['TP'], 0)
        self.assertEqual(res1['FN'], 1)
        self.assertEqual(res1['FP'], 0)

        # Empty ground truth
        res2 = evaluate_detection([], [1.0])
        self.assertEqual(res2['TP'], 0)
        self.assertEqual(res2['FN'], 0)
        self.assertEqual(res2['FP'], 1)

        # Both empty
        res3 = evaluate_detection([], [])
        self.assertEqual(res3['TP'], 0)
        self.assertEqual(res3['FN'], 0)
        self.assertEqual(res3['FP'], 0)
        self.assertEqual(res3['F1-Score'], 0.0)

    def test_get_matched_pairs(self):
        """Test the get_matched_pairs function with various scenarios."""
        
        # Scenario 1: Perfect one-to-one match
        gt1 = [1.0, 2.0]
        pred1 = [1.05, 1.95]
        pairs1 = get_matched_pairs(gt1, pred1, tolerance=0.1)
        self.assertCountEqual(pairs1, [(1.0, 1.05), (2.0, 1.95)])

        # Scenario 2: False Positive (extra prediction)
        gt2 = [1.0]
        pred2 = [1.05, 3.0]
        pairs2 = get_matched_pairs(gt2, pred2, tolerance=0.1)
        self.assertCountEqual(pairs2, [(1.0, 1.05), (None, 3.0)])

        # Scenario 3: False Negative (missed ground truth)
        gt3 = [1.0, 4.0]
        pred3 = [1.05]
        pairs3 = get_matched_pairs(gt3, pred3, tolerance=0.1)
        self.assertCountEqual(pairs3, [(1.0, 1.05), (4.0, None)])

        # Scenario 4: Tolerance test (one match, one miss)
        gt4 = [1.0, 2.0]
        pred4 = [1.05, 2.2] # 2.2 is outside tolerance
        pairs4 = get_matched_pairs(gt4, pred4, tolerance=0.1)
        self.assertCountEqual(pairs4, [(1.0, 1.05), (2.0, None), (None, 2.2)])

        # Scenario 5: Complex case with a shared potential match
        # The algorithm should correctly assign the closest pairs
        gt5 = [1.0, 1.1]
        pred5 = [1.06]
        # 1.06 is closer to 1.1 than to 1.0.
        pairs5 = get_matched_pairs(gt5, pred5, tolerance=0.1)
        self.assertCountEqual(pairs5, [(1.1, 1.06), (1.0, None)])

        # Scenario 6: The user's failing case
        gt6 = [0.5, 1.0, 1.28, 1.5]
        pred6 = [0.5, 1.062, 1.5, 1.625]
        pairs6 = get_matched_pairs(gt6, pred6, tolerance=0.1)
        # Expected: 1.28 is a miss (FN), 1.625 is a false alarm (FP)
        self.assertCountEqual(pairs6, [
            (0.5, 0.5),
            (1.0, 1.062),
            (1.5, 1.5),
            (1.28, None),
            (None, 1.625)
        ])

        # Scenario 7: Empty inputs
        self.assertEqual(get_matched_pairs([], []), [])
        self.assertEqual(get_matched_pairs([1.0], []), [(1.0, None)])
        self.assertEqual(get_matched_pairs([], [1.0]), [(None, 1.0)])

if __name__ == '__main__':
    unittest.main()
