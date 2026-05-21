import tkinter as tk
from abc import ABC, abstractmethod

class ComponentController(ABC):
    """
    Abstract base class for a UI Controller.
    It owns a data model and manages the UI for editing that model's parameters.

    Args:
        model: The data model instance.
        update_callback (callable): A function to call when parameters change.
        get_duration (callable): A function that returns the global signal duration.
    """
    def __init__(self, model, update_callback, get_duration):
        self.model = model
        self.update_callback = update_callback
        self.get_duration = get_duration
        self.config_widgets = []
        self.vars = {} # To hold tk.DoubleVar, tk.IntVar, etc.

    @abstractmethod
    def get_config_frame(self, parent):
        """
        Creates the UI frame for configuring the model's parameters.

        Args:
            parent: The parent widget.

        Returns:
            None
        """
        self.config_widgets = []
        self.vars = {}

    def update_model_from_vars(self, *args):
        """
        Updates the model's primitive types from the UI's tk.Vars.

        Args:
            *args: Variable length argument list.

        Returns:
            None
        """
        for name, var in self.vars.items():
            if hasattr(self.model, name):
                setattr(self.model, name, var.get())
        self.update_callback()

    def update_vars_from_model(self, *args):
        """
        Updates the UI's tk.Vars from the model's primitive types.

        Args:
            *args: Variable length argument list.

        Returns:
            None
        """
        for name, var in self.vars.items():
            if hasattr(self.model, name):
                var.set(getattr(self.model, name))

    def update_slider_ranges(self):
        """
        Updates the 'to' value of any time-based sliders.

        Returns:
            None
        """
        duration = self.get_duration()
        for widget in self.config_widgets:
            if widget.winfo_exists():
                widget.slider.config(to=duration)
    
    @property
    def id(self):
        """
        Returns a unique identifier for the component's model instance
        based on its memory address.

        Returns:
            int: The unique identifier.
        """
        return id(self.model)

    def __str__(self):
        """
        String representation for the UI, using the last 4 digits of the
        model's ID for a short but unique display.

        Returns:
            str: The string representation.
        """
        return f"{self.model.name} ({self.id % 10000:04d})"
