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
        
        # --- Left Container: Configuration, Pipeline, Settings ---
        left_container = ttk.Frame(main_frame, width=350)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_container.pack_propagate(False)

        left_scroll = ScrollableFrame(left_container, h_scroll=False)
        left_scroll.pack(fill=tk.BOTH, expand=True)
        left_panel = left_scroll.scrollable_frame
        
        # --- Thick Vertical Separator ---
        separator = tk.Frame(main_frame, width=3, bg='gray')
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # --- Right Container: Plots and Tables ---
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
        # Regenerate data every time the button is pressed
        duration = self.settings_panel.duration_seconds.get()
        max_freq = self._get_max_freq()
        
        result = generate_pipeline_data(
            self.controllers,
            duration,
            max_freq,
            self.settings_panel.stft_window_size.get(),
            self.settings_panel.stft_overlap.get(),
            self.settings_panel.stft_window_type.get(),
            rectify=self.settings_panel.spectral_flux_rectify.get()
        )

        # --- Window 1: Extended Time-Domain Signal ---
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(result.t_ext, result.y_ext, label="Extended Signal (with padding)", color='blue')
        ax1.axvline(x=0, color='r', linestyle='--', label="Start of core signal")
        ax1.axvline(x=duration, color='r', linestyle='--', label="End of core signal")
        ax1.axvspan(0, duration, color='gray', alpha=0.2, label="Core signal area")
        ax1.set_title("Extended Signal with Padding")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True)
        fig1.tight_layout()

        # --- Window 2: Main Analysis Plots (Time-Domain, STFT, Flux) ---
        fig2, (ax2_td, ax2_stft, ax2_flux) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))

        # Time-Domain Plot
        ax2_td.plot(result.t, result.y)
        ax2_td.set_title("Time-Domain Signal (Core)")
        ax2_td.set_ylabel("Amplitude")
        ax2_td.grid(True)
        ax2_td.set_xlim(0, duration)

        # STFT Plot
        if result.Zxx_ext.size > 0:
            ax2_stft.pcolormesh(result.t_stft_ext, result.f, result.Zxx_ext, shading='gouraud')
        ax2_stft.set_title("Short-Time Fourier Transform (STFT)")
        ax2_stft.set_ylabel("Frequency [Hz]")
        ax2_stft.set_ylim(0, max_freq * 2 if max_freq > 0 else 100)
        ax2_stft.set_xlim(0, duration)

        # Spectral Flux Plot
        if len(result.flux) > 0:
            ax2_flux.plot(result.t_stft_core, result.flux, color='purple')
            if result.peak_times is not None and len(result.peak_times) > 0:
                for i, peak_time in enumerate(result.peak_times):
                    ax2_flux.axvline(x=peak_time, color='red', linestyle='--', alpha=0.7, label='Detected Anomaly' if i == 0 else "")
                ax2_flux.legend(loc='upper right', fontsize='small')
        ax2_flux.set_title("Spectral Flux")
        ax2_flux.set_xlabel("Time [s]")
        ax2_flux.set_ylabel("Spectral Flux")
        ax2_flux.set_xlim(0, duration)
        ax2_flux.grid(True)
        
        fig2.tight_layout()

        # --- Window 3: FFT Spectrum ---
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(result.xf, result.yf)
        if result.fft_peak_freqs is not None and result.fft_peak_amps is not None:
            for i, (freq, amp) in enumerate(zip(result.fft_peak_freqs, result.fft_peak_amps)):
                ax3.axvline(x=freq, color='red', linestyle='--', alpha=0.7, label='Detected Peak' if i == 0 else "")
                ax3.plot(freq, amp, "rx")
            if len(result.fft_peak_freqs) > 0:
                ax3.legend(loc='upper right', fontsize='small')
        ax3.set_title("FFT Spectrum (Core Signal)")
        ax3.set_xlabel("Frequency [Hz]")
        ax3.set_ylabel("Amplitude")
        ax3.grid(True)
        ax3.set_xlim(0, max_freq * 2 if max_freq > 0 else 100)
        fig3.tight_layout()

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