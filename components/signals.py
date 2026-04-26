import tkinter as tk
from .base import ComponentController
from utils import create_slider_entry
from models.signal_models import SineModel, CosineModel, SquareModel, ChirpModel, SineVaryingFreqModel

class SineController(ComponentController):
    def __init__(self, model: SineModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['frequency'] = tk.DoubleVar(value=self.model.frequency)
        self.vars['phase'] = tk.DoubleVar(value=self.model.phase)
        
        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Frequency (Hz):", self.vars['frequency'], 0, 100, self.update_model_from_vars)
        create_slider_entry(parent, "Phase (deg):", self.vars['phase'], 0, 360, self.update_model_from_vars)

class CosineController(SineController):
    pass # UI is identical to Sine

class SquareController(ComponentController):
    def __init__(self, model: SquareModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['frequency'] = tk.DoubleVar(value=self.model.frequency)
        self.vars['phase'] = tk.DoubleVar(value=self.model.phase)
        self.vars['duty_cycle'] = tk.DoubleVar(value=self.model.duty_cycle)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Frequency (Hz):", self.vars['frequency'], 0, 100, self.update_model_from_vars)
        create_slider_entry(parent, "Phase (deg):", self.vars['phase'], 0, 360, self.update_model_from_vars)
        create_slider_entry(parent, "Duty Cycle:", self.vars['duty_cycle'], 0, 1, self.update_model_from_vars)

class ChirpController(ComponentController):
    def __init__(self, model: ChirpModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['start_freq'] = tk.DoubleVar(value=self.model.start_freq)
        self.vars['end_freq'] = tk.DoubleVar(value=self.model.end_freq)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Start Frequency (Hz):", self.vars['start_freq'], 1, 100, self.update_model_from_vars)
        create_slider_entry(parent, "End Frequency (Hz):", self.vars['end_freq'], 1, 100, self.update_model_from_vars)

class SineVaryingFreqController(ComponentController):
    def __init__(self, model: SineVaryingFreqModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['start_freq'] = tk.DoubleVar(value=self.model.start_freq)
        self.vars['end_freq'] = tk.DoubleVar(value=self.model.end_freq)
        self.vars['change_time'] = tk.DoubleVar(value=self.model.change_time)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 10, self.update_model_from_vars)
        create_slider_entry(parent, "Start Frequency (Hz):", self.vars['start_freq'], 1, 100, self.update_model_from_vars)
        create_slider_entry(parent, "End Frequency (Hz):", self.vars['end_freq'], 1, 100, self.update_model_from_vars)
        frame = create_slider_entry(parent, "Change Time (s):", self.vars['change_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)
