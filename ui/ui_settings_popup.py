import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SettingsPopup:
    def __init__(self, parent, controller, get_duration_func, on_close_callback):
        self.parent = parent
        self.controller = controller
        self.get_duration = get_duration_func
        self.on_close_callback = on_close_callback

        self.popup = tk.Toplevel(parent)
        self.popup.title(f"Settings: {self.controller.model.name}")
        self.popup.transient(parent)
        self.popup.grab_set()
        self.popup.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- UI Creation ---
        # Parameters Frame
        config_panel = ttk.LabelFrame(self.popup, text="Parameters")
        config_panel.pack(fill=tk.X, expand=True, padx=10, pady=10)
        config_frame = ttk.Frame(config_panel)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        self.controller.get_config_frame(config_frame)

        # Preview Frame
        preview_frame = ttk.LabelFrame(self.popup, text="Component Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.preview_fig, self.preview_ax = plt.subplots(figsize=(4, 2.5))
        self.preview_fig.tight_layout()
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Positioning and Sizing ---
        self.popup.update_idletasks() # Update widgets to get correct size
        
        # Set fixed width, dynamic height
        width = 400
        height = self.popup.winfo_reqheight()
        self.popup.resizable(False, False)

        # Center the window
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        x = parent_x + (parent_width // 2) - (width // 2)
        y = parent_y + (parent_height // 2) - (height // 2)
        
        self.popup.geometry(f'{width}x{height}+{x}+{y}')

        # --- Callback Handling ---
        self.original_callback = self.controller.update_callback
        self.controller.update_callback = self.update_preview
        
        self.update_preview() # Initial draw

    def update_preview(self):
        # Trigger the main plot update first
        self.original_callback()

        # Now, redraw the preview plot inside the popup
        self.preview_ax.clear()
        duration = self.get_duration()
        # Use a reasonable number of points for the preview
        t = np.linspace(0, duration, int(duration * 500) if duration > 0 else 1)
        
        # Generate the signal for the preview using the public generate method,
        # which correctly handles start/end time windowing.
        y_preview = self.controller.model.generate(t, np.zeros_like(t))

        self.preview_ax.plot(t, y_preview, color='darkorange')
        self.preview_ax.set_title(f"Preview: {self.controller.model.name}", fontsize=9)
        self.preview_ax.set_xlim(0, duration if duration > 0 else 1)
        
        if np.any(y_preview):
            min_val, max_val = np.min(y_preview), np.max(y_preview)
            padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
            self.preview_ax.set_ylim((min_val - padding), (max_val + padding))

        self.preview_ax.tick_params(axis='x', labelsize=8)
        self.preview_ax.tick_params(axis='y', labelsize=8)
        self.preview_canvas.draw()

    def on_close(self):
        """Restore original callback and destroy the popup."""
        self.controller.update_callback = self.original_callback
        self.popup.destroy()
        # Notify the main app that the popup is closed
        if self.on_close_callback:
            self.on_close_callback()
