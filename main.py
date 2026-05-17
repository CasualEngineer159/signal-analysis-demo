import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import json
import csv
from tkinter import filedialog, messagebox

from utils import create_slider_entry, ScrollableFrame
from analysis import perform_fft, perform_stft, calculate_spectral_flux
from components.signals import (SineController, CosineController, SquareController, 
                              ChirpController, SineVaryingFreqController)
from components.anomalies import (GaussianNoiseController, ImpulseNoiseController, 
                                AmplitudeJumpController, BiasController, DriftController,
                                DropoutController, SaturationController, OutlierController,
                                TimeDelayController)
from models.signal_models import *
from models.anomaly_models import *
from plot_manager import PlotManager
from config_manager import ConfigManager
from signal_engine import generate_pipeline_data
from ui_pipeline_list import PipelineListPanel
from ui_settings_panel import SettingsPanel

# --- Component Mappings ---
COMPONENT_MAP = {
    "Sine": (SineModel, SineController),
    "Sine (Varying Freq)": (SineVaryingFreqModel, SineVaryingFreqController),
    "Cosine": (CosineModel, CosineController),
    "Square": (SquareModel, SquareController),
    "Chirp": (ChirpModel, ChirpController),
    "Gaussian Noise": (GaussianNoiseModel, GaussianNoiseController),
    "Amplitude Jump": (AmplitudeJumpModel, AmplitudeJumpController),
    "Impulse Noise": (ImpulseNoiseModel, ImpulseNoiseController),
    "Outlier": (OutlierModel, OutlierController),
    "Bias": (BiasModel, BiasController),
    "Drift": (DriftModel, DriftController),
    "Signal Dropout": (DropoutModel, DropoutController),
    "Saturation": (SaturationModel, SaturationController),
    "Time Delay": (TimeDelayModel, TimeDelayController),
}
SIGNAL_TYPES = {k for k, (m, c) in COMPONENT_MAP.items() if issubclass(m, SignalComponentModel)}

# --- Main Application Class ---
class SignalGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Analysis Tool")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.controllers = []
        
        self._last_flux_data = None

        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_container = ttk.Frame(main_frame, width=350)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_container.pack_propagate(False)

        left_scroll = ScrollableFrame(left_container, h_scroll=False)
        left_scroll.pack(fill=tk.BOTH, expand=True)
        left_panel = left_scroll.scrollable_frame

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.setup_file_io_panel(left_panel)
        
        # Setup config panel early so pipeline list can use it
        self.config_panel = ttk.LabelFrame(left_panel, text="Parameters")
        
        self.pipeline_panel = PipelineListPanel(
            parent=left_panel,
            controllers=self.controllers,
            on_pipeline_changed=self.update_plot,
            config_panel=self.config_panel,
            component_map=COMPONENT_MAP,
            signal_types=SIGNAL_TYPES,
            on_add_component=self.add_component
        )
        
        self.setup_preview_panel(left_panel)
        
        self.config_panel.pack(fill=tk.X, pady=(10, 5))
        
        self.settings_panel = SettingsPanel(
            parent=left_panel,
            on_params_changed=self.handle_settings_change,
            get_last_flux_data=lambda: self._last_flux_data
        )

        self.plot_manager = PlotManager(right_panel)
        self.update_plot()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def setup_file_io_panel(self, parent):
        io_frame = ttk.LabelFrame(parent, text="Configuration")
        io_frame.pack(fill=tk.X, pady=(10, 5))
        btn_frame = ttk.Frame(io_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        clear_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_all)
        clear_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        save_btn = ttk.Button(btn_frame, text="Save Config", command=self.save_configuration)
        save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        load_btn = ttk.Button(btn_frame, text="Load Config", command=self.load_configuration)
        load_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 0))

    def setup_preview_panel(self, parent):
        preview_frame = ttk.LabelFrame(parent, text="Component Preview")
        preview_frame.pack(fill=tk.X, pady=(10, 5))
        self.preview_fig, self.preview_ax = plt.subplots(figsize=(4, 1.5))
        self.preview_fig.tight_layout()
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def handle_settings_change(self, source=None):
        """Synchronize time settings and trigger plot update."""
        if source == "periods":
            max_freq = self._get_max_freq()
            new_duration = self.settings_panel.periods.get() / max_freq if max_freq > 0 else 2
            self.settings_panel.duration_seconds.set(new_duration)
        elif source == "duration":
            max_freq = self._get_max_freq()
            new_periods = self.settings_panel.duration_seconds.get() * max_freq
            self.settings_panel.periods.set(new_periods)
            
        self.update_plot()

    def add_component(self, comp_name, model_config=None):
        model_class, controller_class = COMPONENT_MAP[comp_name]
        model = model_class.from_dict(model_config) if model_config else model_class()
        controller = controller_class(model, self.update_plot, lambda: self.settings_panel.duration_seconds.get())
        self.controllers.append(controller)
        self.pipeline_panel.add_to_listbox(controller)
        self.update_plot()
        return controller

    def clear_all(self):
        self.controllers.clear()
        self.pipeline_panel.clear_listbox()

        # Reset settings to default using the apply_settings method for consistency
        self.settings_panel.apply_settings({}, {})
        self.update_plot()

    def save_configuration(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not filepath: return
        try:
            ui_settings, spectral_flux_settings = self.settings_panel.get_settings()
            components = [c.model.to_dict() for c in self.controllers]
            ConfigManager.save_to_file(filepath, ui_settings, components, spectral_flux_settings)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration file:\n{e}")

    def load_configuration(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not filepath: return
        try:
            ui_settings, components, spectral_flux_settings = ConfigManager.load_from_file(filepath)
            
            self.controllers.clear()
            self.pipeline_panel.clear_listbox()

            self.settings_panel.apply_settings(ui_settings, spectral_flux_settings)

            for comp_conf in components:
                comp_type = comp_conf.get('type')
                if comp_type in COMPONENT_MAP:
                    self.add_component(comp_type, comp_conf)
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load or parse configuration file:\n{e}")

    def _get_max_freq(self):
        signal_models = [c.model for c in self.controllers if isinstance(c.model, SignalComponentModel)]
        signal_freqs = [m.get_max_freq() for m in signal_models if m.get_max_freq() > 0]
        return max(signal_freqs) if signal_freqs else 1.0

    def update_plot(self, event=None):
        if not hasattr(self, 'settings_panel'):
            return

        duration = self.settings_panel.duration_seconds.get()
        max_freq = self._get_max_freq()
        
        selected_controller = self.pipeline_panel.selected_controller
        if selected_controller: 
            selected_controller.update_slider_ranges()
        
        result = generate_pipeline_data(
            self.controllers,
            duration,
            max_freq,
            self.settings_panel.stft_window_size.get(),
            self.settings_panel.stft_overlap.get(),
            self.settings_panel.stft_window_type.get(),
            rectify=self.settings_panel.spectral_flux_rectify.get()
        )

        self._last_flux_data = (result.t_stft_core, result.flux) if len(result.flux) > 0 else None

        if hasattr(self, 'plot_manager'):
            self.plot_manager.draw_plots(
                duration, max_freq, result.fs, result.t, result.y, result.t_ext, result.y_ext,
                result.t_stft_ext, result.f, result.Zxx_ext, result.t_stft_core, result.flux,
                result.peak_times, result.xf, result.yf, result.ground_truth_times, result.evaluation_metrics,
                result.fft_peak_freqs, result.fft_peak_amps, result.matched_pairs
            )

            self.preview_ax.clear()
            if selected_controller:
                y_preview = selected_controller.model.generate(result.t, np.zeros_like(result.t))
                self.preview_ax.plot(result.t, y_preview, color='darkorange')
                self.preview_ax.set_title(f"Preview: {selected_controller.model.name}", fontsize=9)
                self.preview_ax.set_xlim(self.plot_manager.ax.get_xlim())
                self.preview_ax.set_ylim(self.plot_manager.ax.get_ylim())
            else:
                self.preview_ax.set_title("No Component Selected", fontsize=9)
            self.preview_ax.tick_params(axis='x', labelsize=8)
            self.preview_ax.tick_params(axis='y', labelsize=8)
            self.preview_canvas.draw()

if __name__ == '__main__':
    root = tk.Tk()
    app = SignalGeneratorApp(root)
    root.mainloop()