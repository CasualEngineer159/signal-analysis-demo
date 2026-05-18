from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields
import numpy as np

@dataclass
class BaseComponentModel(ABC):
    """
    Abstract base class for a pure, UI-agnostic data model of a signal component.
    It holds parameters as primitive types and contains the core mathematical logic.
    """
    name: str = "Base Component"

    @abstractmethod
    def generate(self, t: 'np.ndarray', y_in: 'np.ndarray') -> 'np.ndarray':
        """
        Generates the output of this component.
        
        Args:
            t (np.ndarray): The time vector.
            y_in (np.ndarray): The input signal from the previous stage.
        
        Returns:
            np.ndarray: The output signal.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """
        Serializes the model's parameters to a dictionary.
        The `name` field is used to identify the component type.
        """
        params = asdict(self)
        if 'name' in params:
            del params['name']
        return {'type': self.name, 'params': params}

    @classmethod
    def from_dict(cls, config: dict) -> 'BaseComponentModel':
        """
        Creates a model instance from a dictionary configuration.
        """
        param_config = config.get('params', {})
        known_fields = {f.name for f in fields(cls)}
        filtered_params = {k: v for k, v in param_config.items() if k in known_fields}
        return cls(**filtered_params)

    def get_anomaly_times(self) -> list[float]:
        """Returns a list of specific times where this component introduces an anomaly/change."""
        return []

@dataclass
class SignalComponentModel(BaseComponentModel):
    """Base for models that generate a primary signal."""
    start_time: float = 0.0
    end_time: float = -1.0  # -1 indicates to the end of the signal

    def generate(self, t: 'np.ndarray', y_in: 'np.ndarray') -> 'np.ndarray':
        """Generates the component's signal and adds it to the input signal within the specified time window."""
        if len(t) == 0:
            return y_in

        component_signal = self._generate_signal(t)
        effective_end_time = self.end_time if self.end_time >= 0 else t[-1]

        start_idx = np.searchsorted(t, self.start_time, side='left')
        # Use 'left' to make the interval exclusive of the end point, preventing overlap
        end_idx = np.searchsorted(t, effective_end_time, side='left')

        if start_idx < end_idx:
            y_out = y_in.copy()
            # Add the component signal to the input
            y_out[start_idx:end_idx] += component_signal[start_idx:end_idx]
            return y_out
        else:
            return y_in

    @abstractmethod
    def _generate_signal(self, t: 'np.ndarray') -> 'np.ndarray':
        raise NotImplementedError

    def get_max_freq(self) -> float:
        """Returns the model's maximum frequency content."""
        return 0

@dataclass
class AnomalyComponentModel(BaseComponentModel):
    """Base for models that modify an incoming signal."""
    def generate(self, t: 'np.ndarray', y_in: 'np.ndarray') -> 'np.ndarray':
        return self._apply_anomaly(t, y_in)

    @abstractmethod
    def _apply_anomaly(self, t: 'np.ndarray', y_in: 'np.ndarray') -> 'np.ndarray':
        raise NotImplementedError
