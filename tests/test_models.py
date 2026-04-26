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

    def test_saturation_model(self):
        model = SaturationModel(threshold=0.5)
        y_in_test = np.array([-1.0, 0.2, 0.8, 1.5])
        y_out = model.generate(self.t_nominal, y_in_test)
        np.testing.assert_array_equal(y_out, np.array([-0.5, 0.2, 0.5, 0.5]))

        model_zero = SaturationModel(threshold=0)
        y_out_zero = model_zero.generate(self.t_nominal, y_in_test)
        np.testing.assert_array_equal(y_out_zero, np.zeros_like(y_in_test))

        self._test_serialization_roundtrip(model)
        
    def test_dropout_model(self):
        model = DropoutModel(start_time=0.5, duration=0.5)
        y_in_test = np.ones(100)
        t_test = np.linspace(0, 1, 100)
        y_out = model.generate(t_test, y_in_test)
        self.assertTrue(np.all(y_out[50:99] == 0))
        self.assertEqual(y_out[49], 1)

        model_zero_dur = DropoutModel(duration=0)
        y_out_zero = model_zero_dur.generate(t_test, y_in_test)
        np.testing.assert_array_equal(y_out_zero, y_in_test)

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
