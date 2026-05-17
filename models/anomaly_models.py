import numpy as np
from dataclasses import dataclass, field
from .base_model import AnomalyComponentModel

@dataclass
class GaussianNoiseModel(AnomalyComponentModel):
    """Model for Gaussian noise."""
    name: str = field(default="Gaussian Noise", init=False)
    std_dev: float = 0.1
    seed: int = 12345

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        if self.std_dev <= 0: return y_in
        rng = np.random.RandomState(self.seed)
        return y_in + rng.normal(0, self.std_dev, len(t))

@dataclass
class ImpulseNoiseModel(AnomalyComponentModel):
    """Model for impulse noise."""
    name: str = field(default="Impulse Noise", init=False)
    amplitude: float = 5.0
    impulse_time: float = 1.0

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        if len(t) == 0: return y_in
        y_out = y_in.copy()
        duration = t[-1]
        actual_time = np.clip(self.impulse_time, 0, duration)
        impulse_idx = np.searchsorted(t, actual_time, side='right')
        if 0 <= impulse_idx < len(y_out):
            y_out[impulse_idx] += self.amplitude
        return y_out

    def get_anomaly_times(self, **kwargs) -> list[float]:
        return [self.impulse_time]

@dataclass
class AmplitudeJumpModel(AnomalyComponentModel):
    """Model for an amplitude jump."""
    name: str = field(default="Amplitude Jump", init=False)
    jump_size: float = 2.0
    jump_time: float = 1.0

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        if len(t) == 0: return y_in
        y_out = y_in.copy()
        duration = t[-1]
        actual_time = np.clip(self.jump_time, 0, duration)
        jump_idx = np.searchsorted(t, actual_time, side='right')
        y_out[jump_idx:] += self.jump_size
        return y_out

    def get_anomaly_times(self, **kwargs) -> list[float]:
        return [self.jump_time]

@dataclass
class BiasModel(AnomalyComponentModel):
    """Model for a DC bias."""
    name: str = field(default="Bias", init=False)
    offset: float = 0.5

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        return y_in + self.offset

@dataclass
class DriftModel(AnomalyComponentModel):
    """Model for a linear drift."""
    name: str = field(default="Drift", init=False)
    slope: float = 0.5

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        return y_in + self.slope * t

@dataclass
class DropoutModel(AnomalyComponentModel):
    """Model for a signal dropout."""
    name: str = field(default="Signal Dropout", init=False)
    start_time: float = 0.75
    duration: float = 0.5

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        if len(t) == 0 or self.duration <= 0: return y_in
        y_out = y_in.copy()
        total_duration = t[-1]
        start = np.clip(self.start_time, 0, total_duration)
        end = np.clip(start + self.duration, 0, total_duration)
        start_idx = np.searchsorted(t, start, side='right')
        end_idx = np.searchsorted(t, end, side='right')
        y_out[start_idx:end_idx] = 0
        return y_out

    def get_anomaly_times(self, tolerance: float = 0.0, **kwargs) -> list[float]:
        """
        Returns the ground truth times for the dropout.
        If the dropout duration is less than the tolerance, only the start time is returned.
        """
        if self.duration > tolerance:
            return [self.start_time, self.start_time + self.duration]
        else:
            return [self.start_time]

@dataclass
class SaturationModel(AnomalyComponentModel):
    """Model for signal saturation."""
    name: str = field(default="Saturation", init=False)
    threshold: float = 1.0

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        # Ensure threshold is non-negative
        actual_threshold = max(0, self.threshold)
        return np.clip(y_in, -actual_threshold, actual_threshold)

@dataclass
class OutlierModel(AnomalyComponentModel):
    """Model for a single outlier."""
    name: str = field(default="Outlier", init=False)
    value: float = 10.0
    outlier_time: float = 1.0

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        if len(t) == 0: return y_in
        y_out = y_in.copy()
        duration = t[-1]
        actual_time = np.clip(self.outlier_time, 0, duration)
        outlier_idx = np.searchsorted(t, actual_time, side='right')
        if 0 <= outlier_idx < len(y_out):
            y_out[outlier_idx] = self.value
        return y_out

    def get_anomaly_times(self, **kwargs) -> list[float]:
        return [self.outlier_time]

@dataclass
class TimeDelayModel(AnomalyComponentModel):
    """Model for a time delay."""
    name: str = field(default="Time Delay", init=False)
    delay: float = 0.2

    def _apply_anomaly(self, t: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        # Edge case: not enough points to calculate duration or apply delay
        if len(t) < 2 or self.delay <= 0:
            return y_in

        # Robustly calculate sampling frequency
        duration = t[-1] - t[0]
        if duration <= 0:
            return y_in
        fs = len(t) / duration

        delay_samples = int(self.delay * fs)
        if delay_samples <= 0:
            return y_in
        
        # If delay is as long or longer than the signal, output is all zeros
        if delay_samples >= len(t):
            return np.zeros_like(y_in)

        y_out = np.roll(y_in, delay_samples)
        y_out[:delay_samples] = 0
        return y_out

    def get_anomaly_times(self, **kwargs) -> list[float]:
        return [self.delay]
