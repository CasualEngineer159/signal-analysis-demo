import numpy as np
from dataclasses import dataclass, field
from scipy.signal import square, chirp
from .base_model import SignalComponentModel

@dataclass
class SineModel(SignalComponentModel):
    """
    Model for a standard sine wave.

    Args:
        amplitude (float): The amplitude of the sine wave.
        frequency (float): The frequency of the sine wave in Hz.
        phase (float): The phase of the sine wave in degrees.
    """
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
    """
    Model for a standard cosine wave.
    """
    name: str = field(default="Cosine", init=False)

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if self.frequency < 0: return np.zeros_like(t)
        phase_rad = np.deg2rad(self.phase)
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + phase_rad)

@dataclass
class SquareModel(SignalComponentModel):
    """
    Model for a square wave.

    Args:
        amplitude (float): The amplitude of the square wave.
        frequency (float): The frequency of the square wave in Hz.
        phase (float): The phase of the square wave in degrees.
        duty_cycle (float): The duty cycle of the square wave (0.0 to 1.0).
    """
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
    """
    Model for a chirp signal.

    Args:
        amplitude (float): The amplitude of the chirp signal.
        start_freq (float): The starting frequency in Hz.
        end_freq (float): The ending frequency in Hz.
        duration (float): The duration of the chirp.
    """
    name: str = field(default="Chirp", init=False)
    amplitude: float = 1.0
    start_freq: float = 1.0
    end_freq: float = 50.0
    duration: float = 2.0

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0: return np.array([])
        
        # Make time relative to the component's start time for the chirp calculation.
        # This ensures the chirp starts sweeping at self.start_time in the absolute timeline.
        t_relative = t - self.start_time
        
        # Calculate the proper duration of the sweep (time to reach end_freq)
        if self.end_time >= self.start_time:
            sweep_duration = self.end_time - self.start_time
        else:
            # If end_time is -1, it runs until the end of the global observation duration
            sweep_duration = self.duration - self.start_time
            
        if sweep_duration <= 0: return np.zeros_like(t)

        return self.amplitude * chirp(t_relative, f0=self.start_freq, f1=self.end_freq, t1=sweep_duration, method='linear')

    def get_max_freq(self) -> float:
        return max(self.start_freq, self.end_freq)

@dataclass
class SineVaryingFreqModel(SignalComponentModel):
    """
    Model for a sine wave with a frequency jump.

    Args:
        amplitude (float): The amplitude of the sine wave.
        start_freq (float): The starting frequency in Hz.
        end_freq (float): The ending frequency in Hz.
        change_time (float): The time at which the frequency changes.
    """
    name: str = field(default="Sine (Varying Freq)", init=False)
    amplitude: float = 1.0
    start_freq: float = 20.0
    end_freq: float = 60.0
    change_time: float = 1.0

    def _generate_signal(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0: return np.array([])
        
        # Always make time relative to the component's start time.
        t_relative = t - self.start_time
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
