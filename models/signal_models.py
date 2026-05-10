import numpy as np
from dataclasses import dataclass, field
from scipy.signal import square, chirp
from .base_model import SignalComponentModel

@dataclass
class SineModel(SignalComponentModel):
    """Model for a standard sine wave."""
    name: str = field(default="Sine", init=False)
    amplitude: float = 1.0
    frequency: float = 5.0
    phase: float = 0.0  # Phase in degrees

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if self.frequency < 0: return np.zeros_like(t)
        phase_rad = np.deg2rad(self.phase)
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + phase_rad)

    def get_max_freq(self) -> float:
        return self.frequency

@dataclass
class CosineModel(SineModel):
    """Model for a standard cosine wave."""
    name: str = field(default="Cosine", init=False)

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if self.frequency < 0: return np.zeros_like(t)
        phase_rad = np.deg2rad(self.phase)
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + phase_rad)

@dataclass
class SquareModel(SignalComponentModel):
    """Model for a square wave."""
    name: str = field(default="Square", init=False)
    amplitude: float = 1.0
    frequency: float = 5.0
    phase: float = 0.0  # Phase in degrees
    duty_cycle: float = 0.5

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if self.frequency < 0: return np.zeros_like(t)
        phase_rad = np.deg2rad(self.phase)
        duty = np.clip(self.duty_cycle, 0.0, 1.0)
        return self.amplitude * square(2 * np.pi * self.frequency * t + phase_rad, duty=duty)

    def get_max_freq(self) -> float:
        return self.frequency

@dataclass
class ChirpModel(SignalComponentModel):
    """Model for a chirp signal."""
    name: str = field(default="Chirp", init=False)
    amplitude: float = 1.0
    start_freq: float = 1.0
    end_freq: float = 50.0

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0: return np.array([])
        duration = t[-1]
        if duration <= 0: return np.zeros_like(t)
        return self.amplitude * chirp(t, f0=self.start_freq, f1=self.end_freq, t1=duration, method='linear')

    def get_max_freq(self) -> float:
        return max(self.start_freq, self.end_freq)

@dataclass
class SineVaryingFreqModel(SignalComponentModel):
    """Model for a sine wave with a frequency jump."""
    name: str = field(default="Sine (Varying Freq)", init=False)
    amplitude: float = 1.0
    start_freq: float = 20.0
    end_freq: float = 60.0
    change_time: float = 1.0

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0: return np.array([])
        y = np.zeros_like(t)
        
        # Ensure change_time is within the signal's duration
        duration = t[-1]
        actual_change_time = np.clip(self.change_time, 0, duration)
        change_idx = np.searchsorted(t, actual_change_time, side='right')
        
        t1 = t[:change_idx]
        if len(t1) > 0:
            y[:change_idx] = self.amplitude * np.sin(2 * np.pi * self.start_freq * t1)
        
        t2 = t[change_idx:]
        if len(t2) > 0:
            # Calculate phase at the end of the first segment to ensure continuity
            last_phase = 2 * np.pi * self.start_freq * t[change_idx-1] if change_idx > 0 else 0
            # Calculate phase correction for the new frequency
            phase_correction = 2 * np.pi * self.end_freq * t[change_idx]
            y[change_idx:] = self.amplitude * np.sin(2 * np.pi * self.end_freq * t2 - phase_correction + last_phase)

        return y

    def get_max_freq(self) -> float:
        return max(self.start_freq, self.end_freq)

    def get_anomaly_times(self) -> list[float]:
        return [self.change_time]
