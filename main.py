import tkinter as tk
import os
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt

from core.utils import ScrollableFrame
from components.signals import (SineController, CosineController, SquareController, 
                              ChirpController, SineVaryingFreqController)
from components.anomalies import (GaussianNoiseController, ImpulseNoiseController, 
                                AmplitudeJumpController, BiasController, DriftController,
                                DropoutController, SaturationController, OutlierController,
                                TimeDelayController)
from models.signal_models import *
from models.anomaly_models import *
from core.plot_manager import PlotManager
from core.config_manager import ConfigManager
from core.signal_engine import generate_pipeline_data
from ui.ui_pipeline_list import PipelineListPanel
from ui.ui_settings_panel import SettingsPanel
from ui.ui_settings_popup import SettingsPopup

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
        self._last_pipeline_result = None
        self.settings_popup_instance = None
        self.configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        os.makedirs(self.configs_dir, exist_ok=True)

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
        
        self.pipeline_panel = PipelineListPanel(
            parent=left_panel,
            controllers=self.controllers,
            on_pipeline_changed=self.update_plot,
            on_open_settings=self.open_settings_popup,
            component_map=COMPONENT_MAP,
            signal_types=SIGNAL_TYPES,
            on_add_component=self.add_component
        )
        
        self.settings_panel = SettingsPanel(
            parent=left_panel,
            on_params_changed=self.handle_settings_change,
            get_last_flux_data=lambda: self._last_flux_data,
            on_debug_signal=self.show_debug_signal
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
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(btn_frame, text="Save Config", command=self.save_configuration).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(btn_frame, text="Load Config", command=self.load_configuration).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 0))

    def open_settings_popup(self):
        selected_controller = self.pipeline_panel.selected_controller
        if not selected_controller:
            return

        if self.settings_popup_instance:
            self.settings_popup_instance.on_close()

        self.settings_popup_instance = SettingsPopup(
            parent=self.root,
            controller=selected_controller,
            get_duration_func=self.settings_panel.duration_seconds.get,
            on_close_callback=self.on_popup_closed
        )

    def on_popup_closed(self):
        self.settings_popup_instance = None
        self.update_plot()

    def handle_settings_change(self, source=None):
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
        self.settings_panel.apply_settings({}, {})
        self.update_plot()

    def save_configuration(self):
        filepath = filedialog.asksaveasfilename(
            initialdir=self.configs_dir,
            defaultextension=".json", 
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath: return
        try:
            ui_settings, spectral_flux_settings = self.settings_panel.get_settings()
            components = [c.model.to_dict() for c in self.controllers]
            ConfigManager.save_to_file(filepath, ui_settings, components, spectral_flux_settings)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration file:\n{e}")

    def load_configuration(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.configs_dir,
            filetypes=[("JSON files", "*.json")]
        )
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

    def show_debug_signal(self):
        if self._last_pipeline_result is not None:
            result = self._last_pipeline_result
            duration = self.settings_panel.duration_seconds.get()
            
            fig = plt.figure(figsize=(10, 5))
            plt.plot(result.t_ext, result.y_ext, label="Extended Signal (with padding)", color='blue')
            plt.axvline(x=0, color='r', linestyle='--', label="Start of core signal")
            plt.axvline(x=duration, color='r', linestyle='--', label="End of core signal")
            
            # Highlight the core area
            plt.axvspan(0, duration, color='gray', alpha=0.2, label="Core signal area")
            
            plt.title("Debug Signal: Extended Time Range")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def update_plot(self, event=None):
        if not hasattr(self, 'settings_panel'):
            return

        duration = self.settings_panel.duration_seconds.get()
        max_freq = self._get_max_freq()
        
        # Update slider ranges for all controllers, not just the selected one
        for controller in self.controllers:
            controller.update_slider_ranges()
        
        result = generate_pipeline_data(
            self.controllers,
            duration,
            max_freq,
            self.settings_panel.stft_window_size.get(),
            self.settings_panel.stft_overlap.get(),
            self.settings_panel.stft_window_type.get(),
            rectify=self.settings_panel.spectral_flux_rectify.get()
        )
        
        self._last_pipeline_result = result
        self._last_flux_data = (result.t_stft_core, result.flux) if len(result.flux) > 0 else None

        if hasattr(self, 'plot_manager'):
            self.plot_manager.draw_plots(
                duration, max_freq, result.fs, result.t, result.y, result.t_ext, result.y_ext,
                result.t_stft_ext, result.f, result.Zxx_ext, result.t_stft_core, result.flux,
                result.peak_times, result.xf, result.yf, result.ground_truth_times, result.evaluation_metrics,
                result.fft_peak_freqs, result.fft_peak_amps, result.matched_pairs
            )

if __name__ == '__main__':
    root = tk.Tk()
    app = SignalGeneratorApp(root)
    root.mainloop()