import tkinter as tk
from .base import ComponentController
from core.utils import create_slider_entry
from models.signal_models import SineModel, SquareModel, ChirpModel, SineVaryingFreqModel

class SineController(ComponentController):
    """
    Controller for the Sine component.

    Args:
        model (SineModel): The model for the sine wave.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: SineModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Sine component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['frequency'] = tk.DoubleVar(value=self.model.frequency)
        self.vars['phase'] = tk.DoubleVar(value=self.model.phase)
        self.vars['start_time'] = tk.DoubleVar(value=self.model.start_time)
        self.vars['end_time'] = tk.DoubleVar(value=self.model.end_time)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Frequency (Hz):", self.vars['frequency'], 0, 100, self.update_model_from_vars)
        create_slider_entry(parent, "Phase (deg):", self.vars['phase'], 0, 360, self.update_model_from_vars)
        
        start_time_frame = create_slider_entry(parent, "Start Time (s):", self.vars['start_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(start_time_frame)
        
        end_time_frame = create_slider_entry(parent, "End Time (s):", self.vars['end_time'], -1, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(end_time_frame)

class CosineController(SineController):
    """
    Controller for the Cosine component.
    """
    pass # UI is identical to Sine

class SquareController(ComponentController):
    """
    Controller for the Square component.

    Args:
        model (SquareModel): The model for the square wave.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: SquareModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Square component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['frequency'] = tk.DoubleVar(value=self.model.frequency)
        self.vars['phase'] = tk.DoubleVar(value=self.model.phase)
        self.vars['duty_cycle'] = tk.DoubleVar(value=self.model.duty_cycle)
        self.vars['start_time'] = tk.DoubleVar(value=self.model.start_time)
        self.vars['end_time'] = tk.DoubleVar(value=self.model.end_time)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Frequency (Hz):", self.vars['frequency'], 0, 100, self.update_model_from_vars)
        create_slider_entry(parent, "Phase (deg):", self.vars['phase'], 0, 360, self.update_model_from_vars)
        create_slider_entry(parent, "Duty Cycle:", self.vars['duty_cycle'], 0, 1, self.update_model_from_vars)
        
        start_time_frame = create_slider_entry(parent, "Start Time (s):", self.vars['start_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(start_time_frame)
        
        end_time_frame = create_slider_entry(parent, "End Time (s):", self.vars['end_time'], -1, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(end_time_frame)

class ChirpController(ComponentController):
    """
    Controller for the Chirp component.

    Args:
        model (ChirpModel): The model for the chirp signal.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: ChirpModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Chirp component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['start_freq'] = tk.DoubleVar(value=self.model.start_freq)
        self.vars['end_freq'] = tk.DoubleVar(value=self.model.end_freq)
        self.vars['start_time'] = tk.DoubleVar(value=self.model.start_time)
        self.vars['end_time'] = tk.DoubleVar(value=self.model.end_time)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Start Frequency (Hz):", self.vars['start_freq'], 1, 100, self.update_model_from_vars)
        create_slider_entry(parent, "End Frequency (Hz):", self.vars['end_freq'], 1, 100, self.update_model_from_vars)
        
        start_time_frame = create_slider_entry(parent, "Start Time (s):", self.vars['start_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(start_time_frame)
        
        end_time_frame = create_slider_entry(parent, "End Time (s):", self.vars['end_time'], -1, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(end_time_frame)

    def update_model_from_vars(self, *args):
        """
        Updates the model from the variables.
        """
        super().update_model_from_vars(*args)
        # Update duration on the model from the global UI state
        if hasattr(self.model, 'duration'):
            self.model.duration = self.get_duration()

    def update_slider_ranges(self):
        """
        Updates the slider ranges.
        """
        super().update_slider_ranges()
        # Also update the duration on the model when global duration changes
        if hasattr(self.model, 'duration'):
            self.model.duration = self.get_duration()

class SineVaryingFreqController(ComponentController):
    """
    Controller for the SineVaryingFreq component.

    Args:
        model (SineVaryingFreqModel): The model for the sine wave with varying frequency.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: SineVaryingFreqModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the SineVaryingFreq component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['start_freq'] = tk.DoubleVar(value=self.model.start_freq)
        self.vars['end_freq'] = tk.DoubleVar(value=self.model.end_freq)
        self.vars['change_time'] = tk.DoubleVar(value=self.model.change_time)
        self.vars['start_time'] = tk.DoubleVar(value=self.model.start_time)
        self.vars['end_time'] = tk.DoubleVar(value=self.model.end_time)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Start Frequency (Hz):", self.vars['start_freq'], 1, 100, self.update_model_from_vars)
        create_slider_entry(parent, "End Frequency (Hz):", self.vars['end_freq'], 1, 100, self.update_model_from_vars)
        
        change_time_frame = create_slider_entry(parent, "Change Time (s):", self.vars['change_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(change_time_frame)
        
        start_time_frame = create_slider_entry(parent, "Start Time (s):", self.vars['start_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(start_time_frame)
        
        end_time_frame = create_slider_entry(parent, "End Time (s):", self.vars['end_time'], -1, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(end_time_frame)
