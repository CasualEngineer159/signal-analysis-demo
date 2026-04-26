import tkinter as tk
from tkinter import ttk

class RoundedStringVar(tk.StringVar):
    """
    A custom tkinter StringVar that automatically rounds the displayed value
    to two decimal places when it's not being actively edited by the user.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_editing = False

    def set(self, value):
        try:
            if not self.is_editing:
                value = f"{float(value):.2f}"
        except (ValueError, TypeError):
            pass
        super().set(value)

def create_slider_entry(parent, label, var, from_, to, command):
    """
    Creates a composite widget with a label, a slider, and a text entry.
    The plot updates dynamically as the slider is moved, but allows for
    stable manual text entry.

    Args:
        parent (tk.Widget): The parent widget.
        label (str): The text label for the widget.
        var (tk.DoubleVar): The tkinter variable to link to.
        from_ (float): The minimum value of the slider.
        to (float): The maximum value of the slider.
        command (callable): Function to call to update the plot.
    """
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=5, padx=5)
    ttk.Label(frame, text=label).pack(side=tk.TOP, anchor=tk.W)

    string_var = RoundedStringVar(value=f"{var.get():.2f}")

    # --- Trace functions to ONLY sync the variables, not update the plot ---
    def update_var_from_string(*args):
        """When the entry text changes, update the DoubleVar."""
        try:
            var.set(float(string_var.get()))
        except (ValueError, TypeError):
            pass

    def update_string_from_var(*args):
        """When the DoubleVar changes (from slider), update the entry text."""
        if not string_var.is_editing:
            string_var.set(var.get())

    var.trace_add("write", update_string_from_var)
    string_var.trace_add("write", update_var_from_string)

    # --- Entry Widget ---
    entry = ttk.Entry(frame, textvariable=string_var, width=10)
    entry.pack(side=tk.RIGHT, padx=(5, 0))

    def on_entry_focus_in(event):
        string_var.is_editing = True

    def on_entry_focus_out(event):
        string_var.is_editing = False
        string_var.set(var.get()) # Re-apply rounding
        command() # Update plot when focus is lost

    entry.bind("<FocusIn>", on_entry_focus_in)
    entry.bind("<FocusOut>", on_entry_focus_out)
    entry.bind("<Return>", lambda e: command()) # Update plot on Enter

    # --- Slider Widget ---
    # The slider's command provides the live update while dragging.
    slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var, command=lambda e: command())
    slider.pack(side=tk.RIGHT, expand=True, fill=tk.X)
    
    # Store slider on the frame so its range can be updated externally
    frame.slider = slider
    return frame
