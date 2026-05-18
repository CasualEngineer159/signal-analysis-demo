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
        # Simple periodic signals evaluate against absolute global time
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

        # If end_time is specified, it's a windowed component.
        if self.end_time >= self.start_time:
            duration = self.end_time - self.start_time
            t_relative = t - self.start_time
        # Otherwise, it's a full-length component. Derive duration from the time slice.
        else:
            duration = t[-1] - t[0] if len(t) > 1 else 0
            t_relative = t - t[0]

        if duration <= 0: return np.zeros_like(t)

        return self.amplitude * chirp(t_relative, f0=self.start_freq, f1=self.end_freq, t1=duration, method='linear')

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
        
        # If windowed, time is relative to the component's start_time.
        # Otherwise, it's relative to the start of the provided time slice.
        t0 = self.start_time if self.end_time >= self.start_time else t[0]
        t_relative = t - t0
        actual_change_time = self.change_time

        # Phase of the first signal at the exact moment of change
        phase_at_change = 2 * np.pi * self.start_freq * actual_change_time

        # Required phase offset for the second signal to ensure continuity
        phase_offset = phase_at_change - (2 * np.pi * self.end_freq * actual_change_time)

        # Generate both signals across the entire time slice
        y1 = self.amplitude * np.sin(2 * np.pi * self.start_freq * t_relative)
        y2 = self.amplitude * np.sin(2 * np.pi * self.end_freq * t_relative + phase_offset)

        # Combine the signals based on the change time
        return np.where(t_relative < actual_change_time, y1, y2)

    def get_max_freq(self) -> float:
        return max(self.start_freq, self.end_freq)

    def get_anomaly_times(self) -> list[float]:
        # The anomaly time is the component's start time plus the relative change time.
        return [self.start_time + self.change_time]
