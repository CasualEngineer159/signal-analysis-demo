import tkinter as tk
from tkinter import ttk
import numpy as np
from .base import ComponentController
from core.utils import create_slider_entry
from models.anomaly_models import (GaussianNoiseModel, WhiteNoiseModel, ImpulseNoiseModel, AmplitudeJumpModel, 
                                 BiasModel, DriftModel, DropoutModel, SaturationModel, 
                                 OutlierModel, TimeDelayModel)

class GaussianNoiseController(ComponentController):
    """
    Controller for the Gaussian Noise component.

    Args:
        model (GaussianNoiseModel): The model for the Gaussian noise.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: GaussianNoiseModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Gaussian Noise component.

        Args:
            parent: The parent widget.
        """
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
        """
        Regenerates the seed for the random number generator.
        """
        self.vars['seed'].set(np.random.randint(0, 10000))
        self.update_model_from_vars()

class WhiteNoiseController(ComponentController):
    """
    Controller for the White Noise component.

    Args:
        model (WhiteNoiseModel): The model for the white noise.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: WhiteNoiseModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the White Noise component.

        Args:
            parent: The parent widget.
        """
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
        """
        Regenerates the seed for the random number generator.
        """
        self.vars['seed'].set(np.random.randint(0, 10000))
        self.update_model_from_vars()


class ImpulseNoiseController(ComponentController):
    """
    Controller for the Impulse Noise component.

    Args:
        model (ImpulseNoiseModel): The model for the impulse noise.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: ImpulseNoiseModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Impulse Noise component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['amplitude'] = tk.DoubleVar(value=self.model.amplitude)
        self.vars['impulse_time'] = tk.DoubleVar(value=self.model.impulse_time)

        create_slider_entry(parent, "Amplitude:", self.vars['amplitude'], 0, 20, self.update_model_from_vars)
        frame = create_slider_entry(parent, "Time (s):", self.vars['impulse_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)

class AmplitudeJumpController(ComponentController):
    """
    Controller for the Amplitude Jump component.

    Args:
        model (AmplitudeJumpModel): The model for the amplitude jump.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: AmplitudeJumpModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Amplitude Jump component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['jump_size'] = tk.DoubleVar(value=self.model.jump_size)
        self.vars['jump_time'] = tk.DoubleVar(value=self.model.jump_time)

        create_slider_entry(parent, "Jump Size:", self.vars['jump_size'], -10, 10, self.update_model_from_vars)
        frame = create_slider_entry(parent, "Time (s):", self.vars['jump_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)

class BiasController(ComponentController):
    """
    Controller for the Bias component.

    Args:
        model (BiasModel): The model for the bias.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: BiasModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Bias component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['offset'] = tk.DoubleVar(value=self.model.offset)
        create_slider_entry(parent, "Offset:", self.vars['offset'], -5, 5, self.update_model_from_vars)

class DriftController(ComponentController):
    """
    Controller for the Drift component.

    Args:
        model (DriftModel): The model for the drift.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: DriftModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Drift component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['slope'] = tk.DoubleVar(value=self.model.slope)
        create_slider_entry(parent, "Slope (units/sec):", self.vars['slope'], -5, 5, self.update_model_from_vars)

class DropoutController(ComponentController):
    """
    Controller for the Dropout component.

    Args:
        model (DropoutModel): The model for the dropout.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: DropoutModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Dropout component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['start_time'] = tk.DoubleVar(value=self.model.start_time)
        self.vars['duration'] = tk.DoubleVar(value=self.model.duration)

        frame1 = create_slider_entry(parent, "Start Time (s):", self.vars['start_time'], 0, self.get_duration(), self.update_model_from_vars)
        frame2 = create_slider_entry(parent, "Duration (s):", self.vars['duration'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.extend([frame1, frame2])

class SaturationController(ComponentController):
    """
    Controller for the Saturation component.

    Args:
        model (SaturationModel): The model for the saturation.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: SaturationModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Saturation component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['lower_threshold'] = tk.DoubleVar(value=self.model.lower_threshold)
        self.vars['upper_threshold'] = tk.DoubleVar(value=self.model.upper_threshold)

        create_slider_entry(parent, "Lower Threshold:", self.vars['lower_threshold'], -10, 0, self.update_model_from_vars)
        create_slider_entry(parent, "Upper Threshold:", self.vars['upper_threshold'], 0, 10, self.update_model_from_vars)

class OutlierController(ComponentController):
    """
    Controller for the Outlier component.

    Args:
        model (OutlierModel): The model for the outlier.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: OutlierModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Outlier component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['value'] = tk.DoubleVar(value=self.model.value)
        self.vars['outlier_time'] = tk.DoubleVar(value=self.model.outlier_time)

        create_slider_entry(parent, "Value:", self.vars['value'], -20, 20, self.update_model_from_vars)
        frame = create_slider_entry(parent, "Time (s):", self.vars['outlier_time'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)

class TimeDelayController(ComponentController):
    """
    Controller for the Time Delay component.

    Args:
        model (TimeDelayModel): The model for the time delay.
        update_callback (callable): The callback function to update the UI.
        get_duration (callable): A function to get the duration of the signal.
    """
    def __init__(self, model: TimeDelayModel, update_callback, get_duration):
        super().__init__(model, update_callback, get_duration)

    def get_config_frame(self, parent):
        """
        Creates the configuration frame for the Time Delay component.

        Args:
            parent: The parent widget.
        """
        super().get_config_frame(parent)
        self.vars['delay'] = tk.DoubleVar(value=self.model.delay)
        frame = create_slider_entry(parent, "Delay (s):", self.vars['delay'], 0, self.get_duration(), self.update_model_from_vars)
        self.config_widgets.append(frame)
