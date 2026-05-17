import unittest
import numpy as np
from dataclasses import fields

# Adjust path to import from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.signal_models import *
from models.anomaly_models import *

class TestSignalModel(unittest.TestCase):
    """Tests for the pure data models of signal components."""

    def setUp(self):
        self.t_nominal = np.linspace(0, 2, 2000)
        self.t_empty = np.array([])
        self.t_single = np.array([0.0])
        self.y_in = np.zeros_like(self.t_nominal)

    def _test_serialization_roundtrip(self, model_instance):
        """Helper to test the to_dict/from_dict round trip for any model."""
        model_class = model_instance.__class__
        config = model_instance.to_dict()
        
        for field in fields(model_instance):
            if field.init:
                self.assertIn(field.name, config['params'])

        recreated_model = model_class.from_dict(config)
        self.assertEqual(model_instance, recreated_model)

    def test_sine_model(self):
        model = SineModel(frequency=10)
        y_out = model.generate(self.t_nominal, self.y_in)
        self.assertEqual(y_out.shape, self.t_nominal.shape)
        self.assertTrue(np.any(y_out != 0))
        
        model_zero_freq = SineModel(frequency=0)
        y_out_zero = model_zero_freq.generate(self.t_nominal, self.y_in)
        np.testing.assert_array_almost_equal(y_out_zero, np.zeros_like(self.t_nominal))
        
        y_out_empty = model.generate(self.t_empty, np.array([]))
        self.assertEqual(y_out_empty.shape, (0,))

        self._test_serialization_roundtrip(model)

    def test_cosine_model(self):
        # A cosine wave starting at phase=0 and t=0 should have amplitude 1.0 (max)
        model = CosineModel(frequency=10, amplitude=5.0, phase=0.0)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        # Verify initial value is exactly the positive amplitude
        self.assertAlmostEqual(y_out[0], 5.0)
        self.assertEqual(y_out.shape, self.t_nominal.shape)
        
        # Verify shift by testing phase
        model_shifted = CosineModel(frequency=10, amplitude=5.0, phase=90.0)
        y_out_shifted = model_shifted.generate(self.t_nominal, self.y_in)
        self.assertAlmostEqual(y_out_shifted[0], 0.0, places=5)
        
        self._test_serialization_roundtrip(model)

    def test_square_model(self):
        amplitude = 3.0
        model = SquareModel(frequency=5, amplitude=amplitude, duty_cycle=0.5)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        # A square wave should only consist of values very close to +amplitude and -amplitude
        # (allowing for float imprecision)
        unique_values = np.unique(np.round(y_out, decimals=5))
        self.assertEqual(len(unique_values), 2)
        self.assertTrue(amplitude in unique_values)
        self.assertTrue(-amplitude in unique_values)
        
        # Test extreme duty cycle (e.g. 1.0 means it stays at +amplitude)
        model_duty_one = SquareModel(frequency=5, amplitude=amplitude, duty_cycle=1.0)
        y_out_duty_one = model_duty_one.generate(self.t_nominal, self.y_in)
        self.assertTrue(np.all(np.round(y_out_duty_one, decimals=5) == amplitude))
        
        self._test_serialization_roundtrip(model)

    def test_chirp_model(self):
        model = ChirpModel(start_freq=10, end_freq=100)
        y_out = model.generate(self.t_nominal, self.y_in)
        self.assertEqual(y_out.shape, self.t_nominal.shape)
        self.assertTrue(np.any(y_out != 0))

        y_out_empty = model.generate(self.t_empty, np.array([]))
        self.assertEqual(y_out_empty.shape, (0,))
        
        y_out_single = model.generate(self.t_single, np.zeros_like(self.t_single))
        self.assertEqual(y_out_single.shape, (1,))

        self._test_serialization_roundtrip(model)
        
    def test_sine_varying_freq_model(self):
        change_time = 1.0
        model = SineVaryingFreqModel(start_freq=10, end_freq=50, change_time=change_time)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        # Find index corresponding to change_time
        change_idx = np.searchsorted(self.t_nominal, change_time, side='right')
        
        # The first part should be a pure 10Hz sine wave
        t_first_part = self.t_nominal[:change_idx]
        expected_first_part = np.sin(2 * np.pi * 10 * t_first_part)
        np.testing.assert_array_almost_equal(y_out[:change_idx], expected_first_part)
        
        # The frequencies should differ visibly in the second part
        self.assertFalse(np.allclose(y_out[change_idx:], np.sin(2 * np.pi * 10 * self.t_nominal[change_idx:])))

        self._test_serialization_roundtrip(model)

    def test_gaussian_noise_model(self):
        std_dev = 0.5
        model = GaussianNoiseModel(std_dev=std_dev, seed=42)
        y_out1 = model.generate(self.t_nominal, self.y_in)
        
        # Verify noise is added
        self.assertFalse(np.allclose(self.y_in, y_out1))
        self.assertEqual(y_out1.shape, self.t_nominal.shape)
        
        # Verify deterministic behavior with same seed
        model_same_seed = GaussianNoiseModel(std_dev=std_dev, seed=42)
        y_out2 = model_same_seed.generate(self.t_nominal, self.y_in)
        np.testing.assert_array_equal(y_out1, y_out2)
        
        # Verify different behavior with different seed
        model_diff_seed = GaussianNoiseModel(std_dev=std_dev, seed=100)
        y_out3 = model_diff_seed.generate(self.t_nominal, self.y_in)
        self.assertFalse(np.allclose(y_out1, y_out3))
        
        self.assertEqual(model.get_anomaly_times(), [])
        
        self._test_serialization_roundtrip(model)

    def test_impulse_noise_model(self):
        amplitude = 10.0
        impulse_time = 1.0
        model = ImpulseNoiseModel(amplitude=amplitude, impulse_time=impulse_time)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        # Find index corresponding to impulse_time
        impulse_idx = np.searchsorted(self.t_nominal, impulse_time, side='right')
        
        # Verify only one point is modified
        diff = y_out - self.y_in
        non_zero_indices = np.nonzero(diff)[0]
        self.assertEqual(len(non_zero_indices), 1)
        self.assertEqual(non_zero_indices[0], impulse_idx)
        self.assertAlmostEqual(diff[impulse_idx], amplitude)
        
        self.assertEqual(model.get_anomaly_times(), [impulse_time])
        
        self._test_serialization_roundtrip(model)

    def test_outlier_model(self):
        value = 50.0
        outlier_time = 1.0
        model = OutlierModel(value=value, outlier_time=outlier_time)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        # Find index corresponding to outlier_time
        outlier_idx = np.searchsorted(self.t_nominal, outlier_time, side='right')
        
        # Verify only one point is modified
        diff = y_out - self.y_in
        non_zero_indices = np.nonzero(diff)[0]
        self.assertEqual(len(non_zero_indices), 1)
        self.assertEqual(non_zero_indices[0], outlier_idx)
        self.assertAlmostEqual(y_out[outlier_idx], value)
        
        self.assertEqual(model.get_anomaly_times(), [outlier_time])
        
        self._test_serialization_roundtrip(model)

    def test_amplitude_jump_model(self):
        jump_size = 5.0
        jump_time = 1.0
        model = AmplitudeJumpModel(jump_size=jump_size, jump_time=jump_time)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        # Find index corresponding to jump_time
        jump_idx = np.searchsorted(self.t_nominal, jump_time, side='right')
        
        # Verify values before jump_time are unchanged
        np.testing.assert_array_equal(y_out[:jump_idx], self.y_in[:jump_idx])
        
        # Verify values after jump_time are shifted by jump_size
        expected_after_jump = self.y_in[jump_idx:] + jump_size
        np.testing.assert_array_almost_equal(y_out[jump_idx:], expected_after_jump)
        
        self.assertEqual(model.get_anomaly_times(), [jump_time])
        
        self._test_serialization_roundtrip(model)

    def test_bias_model(self):
        offset = 2.5
        model = BiasModel(offset=offset)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        expected_out = self.y_in + offset
        np.testing.assert_array_almost_equal(y_out, expected_out)
        
        self.assertEqual(model.get_anomaly_times(), [])
        
        self._test_serialization_roundtrip(model)

    def test_drift_model(self):
        slope = 1.5
        model = DriftModel(slope=slope)
        y_out = model.generate(self.t_nominal, self.y_in)
        
        expected_out = self.y_in + slope * self.t_nominal
        np.testing.assert_array_almost_equal(y_out, expected_out)
        
        # Explicitly verify a specific point
        t_index = np.searchsorted(self.t_nominal, 1.0, side='right')
        expected_value_at_t1 = self.y_in[t_index] + slope * self.t_nominal[t_index]
        self.assertAlmostEqual(y_out[t_index], expected_value_at_t1)
        
        self.assertEqual(model.get_anomaly_times(), [])

        self._test_serialization_roundtrip(model)

    def test_saturation_model(self):
        model = SaturationModel(threshold=0.5)
        y_in_test = np.array([-1.0, 0.2, 0.8, 1.5])
        y_out = model.generate(self.t_nominal, y_in_test)
        np.testing.assert_array_equal(y_out, np.array([-0.5, 0.2, 0.5, 0.5]))

        model_zero = SaturationModel(threshold=0)
        y_out_zero = model_zero.generate(self.t_nominal, y_in_test)
        np.testing.assert_array_equal(y_out_zero, np.zeros_like(y_in_test))
        
        self.assertEqual(model.get_anomaly_times(), [])

        self._test_serialization_roundtrip(model)
        
    def test_dropout_model(self):
        # Test the generation logic
        model = DropoutModel(start_time=0.5, duration=0.5)
        y_in_test = np.ones(100)
        t_test = np.linspace(0, 1, 100)
        y_out = model.generate(t_test, y_in_test)
        self.assertTrue(np.all(y_out[50:99] == 0))
        self.assertEqual(y_out[49], 1)

        model_zero_dur = DropoutModel(duration=0)
        y_out_zero = model_zero_dur.generate(t_test, y_in_test)
        np.testing.assert_array_equal(y_out_zero, y_in_test)
        
        # Test the intelligent ground truth reporting
        # Case 1: Duration is longer than tolerance, expect two times
        long_dropout = DropoutModel(start_time=0.5, duration=0.5)
        self.assertEqual(long_dropout.get_anomaly_times(tolerance=0.1), [0.5, 1.0])

        # Case 2: Duration is shorter than tolerance, expect one time
        short_dropout = DropoutModel(start_time=0.5, duration=0.05)
        self.assertEqual(short_dropout.get_anomaly_times(tolerance=0.1), [0.5])
        
        # Case 3: Default behavior without tolerance (should return both)
        self.assertEqual(long_dropout.get_anomaly_times(), [0.5, 1.0])

        self._test_serialization_roundtrip(model)

    def test_time_delay_model_edge_cases(self):
        """Test TimeDelayModel with zero-duration or empty time arrays."""
        model = TimeDelayModel(delay=0.1)
        y_in = np.ones(10)
        
        # Test with empty time array
        t_empty = np.array([])
        y_out_empty = model.generate(t_empty, np.array([]))
        self.assertEqual(y_out_empty.shape, (0,))

        # Test with single-point time array (zero duration)
        t_single = np.array([1.0])
        y_out_single = model.generate(t_single, np.ones(1))
        np.testing.assert_array_equal(y_out_single, np.ones(1)) # Should return input unchanged

        # Test with zero delay
        model_zero_delay = TimeDelayModel(delay=0)
        y_out_zero_delay = model_zero_delay.generate(self.t_nominal, self.y_in)
        np.testing.assert_array_equal(y_out_zero_delay, self.y_in)

        # Test with delay longer than signal
        model_long_delay = TimeDelayModel(delay=3.0)
        y_out_long = model_long_delay.generate(self.t_nominal, self.y_in)
        np.testing.assert_array_equal(y_out_long, np.zeros_like(self.y_in))
        
        self.assertEqual(model.get_anomaly_times(), [0.1])

class TestIntegration(unittest.TestCase):
    """
    Integration test to verify the chained application of multiple components.
    """
    def test_full_pipeline(self):
        t = np.linspace(0, 2, 2000)
        y = np.zeros_like(t)

        sine_model = SineModel(frequency=10, amplitude=5)
        noise_model = GaussianNoiseModel(std_dev=0.1, seed=42)
        dropout_model = DropoutModel(start_time=1.0, duration=0.2)
        delay_model = TimeDelayModel(delay=0.1)

        y_signal = sine_model.generate(t, y)
        y_with_noise = noise_model.generate(t, y_signal)
        y_with_dropout = dropout_model.generate(t, y_with_noise)
        y_final = delay_model.generate(t, y_with_dropout)

        self.assertEqual(y_final.shape, t.shape)
        self.assertFalse(np.array_equal(y_signal, y_with_noise))

        dropout_start_idx = 1000
        dropout_end_idx = 1200
        self.assertTrue(np.all(y_with_dropout[dropout_start_idx:dropout_end_idx] == 0))
        self.assertFalse(np.all(y_with_dropout[dropout_start_idx-10:dropout_start_idx] == 0))

        delay_samples = int(0.1 * (len(t) / 2.0))
        self.assertTrue(np.all(y_final[:delay_samples] == 0))

        shifted_dropout_start = dropout_start_idx + delay_samples
        shifted_dropout_end = dropout_end_idx + delay_samples
        self.assertTrue(np.all(y_final[shifted_dropout_start:shifted_dropout_end] == 0))

if __name__ == '__main__':
    unittest.main()
