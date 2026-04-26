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
        
        # Check that all parameters are present
        for field in fields(model_instance):
            if field.init: # Only check fields that are part of the constructor
                self.assertIn(field.name, config['params'])

        recreated_model = model_class.from_dict(config)
        self.assertEqual(model_instance, recreated_model)

    def test_sine_model(self):
        # Nominal
        model = SineModel(frequency=10)
        y_out = model.generate(self.t_nominal, self.y_in)
        self.assertEqual(y_out.shape, self.t_nominal.shape)
        self.assertTrue(np.any(y_out != 0))
        
        # Edge Cases
        model_zero_freq = SineModel(frequency=0)
        y_out_zero = model_zero_freq.generate(self.t_nominal, self.y_in)
        np.testing.assert_array_almost_equal(y_out_zero, np.zeros_like(self.t_nominal))
        
        y_out_empty = model.generate(self.t_empty, np.array([]))
        self.assertEqual(y_out_empty.shape, (0,))

        # Serialization
        self._test_serialization_roundtrip(model)

    def test_chirp_model(self):
        # Nominal
        model = ChirpModel(start_freq=10, end_freq=100)
        y_out = model.generate(self.t_nominal, self.y_in)
        self.assertEqual(y_out.shape, self.t_nominal.shape)
        self.assertTrue(np.any(y_out != 0))

        # Edge Cases
        y_out_empty = model.generate(self.t_empty, np.array([]))
        self.assertEqual(y_out_empty.shape, (0,))
        
        y_out_single = model.generate(self.t_single, np.zeros_like(self.t_single))
        self.assertEqual(y_out_single.shape, (1,))

        # Serialization
        self._test_serialization_roundtrip(model)

    def test_saturation_model(self):
        # Nominal
        model = SaturationModel(threshold=0.5)
        y_in_test = np.array([-1.0, 0.2, 0.8, 1.5])
        y_out = model.generate(self.t_nominal, y_in_test)
        np.testing.assert_array_equal(y_out, np.array([-0.5, 0.2, 0.5, 0.5]))

        # Edge Case: Zero threshold
        model_zero = SaturationModel(threshold=0)
        y_out_zero = model_zero.generate(self.t_nominal, y_in_test)
        np.testing.assert_array_equal(y_out_zero, np.zeros_like(y_in_test))

        # Serialization
        self._test_serialization_roundtrip(model)
        
    def test_dropout_model(self):
        # Nominal
        model = DropoutModel(start_time=0.5, duration=0.5)
        y_in_test = np.ones(100)
        t_test = np.linspace(0, 1, 100)
        y_out = model.generate(t_test, y_in_test)
        self.assertTrue(np.all(y_out[50:99] == 0))
        self.assertEqual(y_out[49], 1)

        # Edge Case: Zero duration
        model_zero_dur = DropoutModel(duration=0)
        y_out_zero = model_zero_dur.generate(t_test, y_in_test)
        np.testing.assert_array_equal(y_out_zero, y_in_test)

        # Serialization
        self._test_serialization_roundtrip(model)

class TestIntegration(unittest.TestCase):
    """
    Integration test to verify the chained application of multiple components.
    """
    def test_full_pipeline(self):
        t = np.linspace(0, 2, 2000)
        y = np.zeros_like(t)

        # 1. Define models
        sine_model = SineModel(frequency=10, amplitude=5)
        noise_model = GaussianNoiseModel(std_dev=0.1, seed=42)
        dropout_model = DropoutModel(start_time=1.0, duration=0.2)
        delay_model = TimeDelayModel(delay=0.1)

        # 2. Generate signals step-by-step
        y_signal = sine_model.generate(t, y)
        y_with_noise = noise_model.generate(t, y_signal)
        y_with_dropout = dropout_model.generate(t, y_with_noise)
        y_final = delay_model.generate(t, y_with_dropout)

        # 3. Assertions
        self.assertEqual(y_final.shape, t.shape)

        # Assert that noise was added successfully
        self.assertFalse(np.array_equal(y_signal, y_with_noise))

        # Assert that the dropout section IS zero in the pre-delayed signal
        dropout_start_idx = 1000
        dropout_end_idx = 1200
        self.assertTrue(np.all(y_with_dropout[dropout_start_idx:dropout_end_idx] == 0))
        # And that the section *before* it is NOT zero (it should have noisy signal)
        self.assertFalse(np.all(y_with_dropout[dropout_start_idx-10:dropout_start_idx] == 0))

        # Assert that the delay created zero-padding at the start
        delay_samples = int(0.1 * (len(t) / 2.0))
        self.assertTrue(np.all(y_final[:delay_samples] == 0))

        # Assert that the zeroed-out dropout section was shifted by the delay
        shifted_dropout_start = dropout_start_idx + delay_samples
        shifted_dropout_end = dropout_end_idx + delay_samples
        self.assertTrue(np.all(y_final[shifted_dropout_start:shifted_dropout_end] == 0))

if __name__ == '__main__':
    unittest.main()
