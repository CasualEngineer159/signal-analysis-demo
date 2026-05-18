import tkinter as tk
from tkinter import ttk
import numpy as np
from .base import ComponentController
from core.utils import create_slider_entry
from models.anomaly_models import (GaussianNoiseModel, WhiteNoiseModel, ImpulseNoiseModel, AmplitudeJumpModel, 
                                 BiasModel, DriftModel, DropoutModel, SaturationModel, 
                                 OutlierModel, TimeDelayModel)

class GaussianNoiseController(ComponentController):
    def __init__(self, model: GaussianNoiseModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['std_dev'] = tk.DoubleVar(value=self.model.std_dev)
        self.vars['seed'] = tk.IntVar(value=self.model.seed)

        create_slider_entry(parent, "Std. Deviation:", self.vars['std_dev'], 0, 2, self.update_model_from_vars)
        
        seed_frame = ttk.Frame(parent)
        seed_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT, anchor=tk.W)
        
        seed_entry = ttk.Entry(seed_frame, textvariable=self.vars['seed'], width=10)
        seed_entry.pack(side=tk.LEFT, padx=(5,0), expand=True, fill=tk.X)
        seed_entry.bind("<Return>", lambda e: self.update_model_from_vars())

        regenerate_btn = ttk.Button(seed_frame, text="Regenerate", command=self.regenerate_seed)
        regenerate_btn.pack(side=tk.RIGHT, padx=(5,0))

    def regenerate_seed(self):
        self.vars['seed'].set(np.random.randint(0, 10000))
        self.update_model_from_vars()

class WhiteNoiseController(ComponentController):
    def __init__(self, model: WhiteNoiseModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['seed'] = tk.IntVar(value=self.model.seed)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 2, self.update_model_from_vars)
        
        seed_frame = ttk.Frame(parent)
        seed_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT, anchor=tk.W)
        
        seed_entry = ttk.Entry(seed_frame, textvariable=self.vars['seed'], width=10)
        seed_entry.pack(side=tk.LEFT, padx=(5,0), expand=True, fill=tk.X)
        seed_entry.bind("<Return>", lambda e: self.update_model_from_vars())

        regenerate_btn = ttk.Button(seed_frame, text="Regenerate", command=self.regenerate_seed)
        regenerate_btn.pack(side=tk.RIGHT, padx=(5,0))

    def regenerate_seed(self):
        self.vars['seed'].set(np.random.randint(0, 10000))
        self.update_model_from_vars()


class ImpulseNoiseController(ComponentController):
    def __init__(self, model: ImpulseNoiseModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['impulse_time'] = tk.DoubleVar(value=self.model.impulse_time)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 20, self.update_model_from_vars)
        frame = create_slider_entry(parent, "Time (s):", self.vars['impulse_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)

class AmplitudeJumpController(ComponentController):
    def __init__(self, model: AmplitudeJumpModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['jump_size'] = tk.DoubleVar(value=self.model.jump_size)
        self.vars['jump_time'] = tk.DoubleVar(value=self.model.jump_time)

        create_slider_entry(parent, "Jump Size:", self.vars['jump_size'], -10, 10, self.update_model_from_vars)
        frame = create_slider_entry(parent, "Time (s):", self.vars['jump_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)

class BiasController(ComponentController):
    def __init__(self, model: BiasModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['offset'] = tk.DoubleVar(value=self.model.offset)
        create_slider_entry(parent, "Offset:", self.vars['offset'], -5, 5, self.update_model_from_vars)

class DriftController(ComponentController):
    def __init__(self, model: DriftModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['slope'] = tk.DoubleVar(value=self.model.slope)
        create_slider_entry(parent, "Slope (units/sec):", self.vars['slope'], -5, 5, self.update_model_from_vars)

class DropoutController(ComponentController):
    def __init__(self, model: DropoutModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['start_time'] = tk.DoubleVar(value=self.model.start_time)
        self.vars['duration'] = tk.DoubleVar(value=self.model.duration)

        frame1 = create_slider_entry(parent, "Start Time (s):", self.vars['start_time'], 0, self.get_duration(), self.update_model_from_vars)
        frame2 = create_slider_entry(parent, "Duration (s):", self.vars['duration'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.extend([frame1, frame2])

class SaturationController(ComponentController):
    def __init__(self, model: SaturationModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['threshold'] = tk.DoubleVar(value=self.model.threshold)
        create_slider_entry(parent, "Threshold:", self.vars['threshold'], 0, 10, self.update_model_from_vars)

class OutlierController(ComponentController):
    def __init__(self, model: OutlierModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['value'] = tk.DoubleVar(value=self.model.value)
        self.vars['outlier_time'] = tk.DoubleVar(value=self.model.outlier_time)

        create_slider_entry(parent, "Value:", self.vars['value'], -20, 20, self.update_model_from_vars)
        frame = create_slider_entry(parent, "Time (s):", self.vars['outlier_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)

class TimeDelayController(ComponentController):
    def __init__(self, model: TimeDelayModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        super().get_config_frame(parent)
        self.vars['delay'] = tk.DoubleVar(value=self.model.delay)
        frame = create_slider_entry(parent, "Delay (s):", self.vars['delay'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)
