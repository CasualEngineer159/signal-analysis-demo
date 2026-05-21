from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields
import numpy as np

@dataclass
class BaseComponentModel(ABC):
    """
    Abstract base class for a pure, UI-agnostic data model of a signal component.
    It holds parameters as primitive types and contains the core mathematical logic.

    Args:
        name (str): The name of the component.
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

        Returns:
            dict: The serialized model parameters.
        """
        params = asdict(self)
        if 'name' in params:
            del params['name']
        return {'type': self.name, 'params': params}

    @classmethod
    def from_dict(cls, config: dict) -> 'BaseComponentModel':
        """
        Creates a model instance from a dictionary configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            BaseComponentModel: The created model instance.
        """
        param_config = config.get('params', {})
        known_fields = {f.name for f in fields(cls)}
        filtered_params = {k: v for k, v in param_config.items() if k in known_fields}
        return cls(**filtered_params)

    def get_anomaly_times(self) -> list[float]:
        """
        Returns a list of specific times where this component introduces an anomaly/change.

        Returns:
            list[float]: The list of anomaly times.
        """
        return []

@dataclass
class SignalComponentModel(BaseComponentModel):
    """
    Base for models that generate a primary signal.

    Args:
        start_time (float): The start time of the signal component.
        end_time (float): The end time of the signal component.
    """
    start_time: float = 0.0
    end_time: float = -1.0  # -1 indicates to the end of the signal

    def generate(self, t: 'np.ndarray', y_in: 'np.ndarray') -> 'np.ndarray':
        """
        Generates the component's signal and adds it to the input signal within the specified time window.

        Args:
            t (np.ndarray): The time vector.
            y_in (np.ndarray): The input signal.

        Returns:
            np.ndarray: The resulting signal.
        """
        if len(t) == 0:
            return y_in

        component_signal = self._generate_signal(t)

        # Fix for left padding: if start_time is 0, extend it to include the negative padding time.
        # This prevents edge artifacts when STFT is calculated with zero-padding on the left.
        effective_start_time = t[0] if (self.start_time <= 0.0 and t[0] < 0.0) else self.start_time
        
        # Similar fix for right padding: if end_time is -1.0, extend it to the very end of t_ext.
        # Actually, end_time = -1.0 logic already covered this by going to t[-1], but let's be safe:
        effective_end_time = t[-1] + 1.0 if self.end_time < 0 else self.end_time

        start_idx = np.searchsorted(t, effective_start_time, side='left')
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
        """
        Internal method to generate the signal.

        Args:
            t (np.ndarray): The time vector.

        Returns:
            np.ndarray: The generated signal.
        """
        raise NotImplementedError

    def get_max_freq(self) -> float:
        """
        Returns the model's maximum frequency content.

        Returns:
            float: The maximum frequency.
        """
        return 0

@dataclass
class AnomalyComponentModel(BaseComponentModel):
    """
    Base for models that modify an incoming signal.
    """
    def generate(self, t: 'np.ndarray', y_in: 'np.ndarray') -> 'np.ndarray':
        """
        Applies the anomaly to the input signal.

        Args:
            t (np.ndarray): The time vector.
            y_in (np.ndarray): The input signal.

        Returns:
            np.ndarray: The output signal.
        """
        return self._apply_anomaly(t, y_in)

    @abstractmethod
    def _apply_anomaly(self, t: 'np.ndarray', y_in: 'np.ndarray') -> 'np.ndarray':
        """
        Internal method to apply the anomaly.

        Args:
            t (np.ndarray): The time vector.
            y_in (np.ndarray): The input signal.

        Returns:
            np.ndarray: The output signal.
        """
        raise NotImplementedError
