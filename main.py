import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import json
from tkinter import filedialog, messagebox

from utils import create_slider_entry, ScrollableFrame
from analysis import perform_fft, perform_stft
from components.signals import (SineController, CosineController, SquareController, 
                              ChirpController, SineVaryingFreqController)
from components.anomalies import (GaussianNoiseController, ImpulseNoiseController, 
                                AmplitudeJumpController, BiasController, DriftController,
                                DropoutController, SaturationController, OutlierController,
                                TimeDelayController)
from models.signal_models import *
from models.anomaly_models import *

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
        self.selected_controller = None
        self.config_frame = None
        
        self.periods = tk.DoubleVar(value=5.0)
        self.duration_seconds = tk.DoubleVar(value=2.0)
        self._is_updating_sliders = False
        
        self.stft_window_size = tk.IntVar(value=256)
        self.stft_overlap = tk.IntVar(value=128)
        self.stft_window_type = tk.StringVar(value='hann')

        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.setup_file_io_panel(left_panel)
        self.signal_listbox, self.anomaly_listbox = self.setup_component_panels(left_panel)
        self.setup_preview_panel(left_panel)
        self.config_panel = ttk.LabelFrame(left_panel, text="Parameters")
        self.config_panel.pack(fill=tk.X, pady=5)
        
        self.setup_plot_panels(right_panel)
        self.update_plot()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def get_duration(self):
        return self.duration_seconds.get()

    def setup_file_io_panel(self, parent):
        io_frame = ttk.LabelFrame(parent, text="Configuration")
        io_frame.pack(fill=tk.X, pady=5)
        btn_frame = ttk.Frame(io_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        save_btn = ttk.Button(btn_frame, text="Save Config", command=self.save_configuration)
        save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        load_btn = ttk.Button(btn_frame, text="Load Config", command=self.load_configuration)
        load_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

    def setup_component_panels(self, parent):
        sig_listbox = self._create_component_panel(parent, "Signal Sources", {k:v for k,v in COMPONENT_MAP.items() if k in SIGNAL_TYPES})
        anom_listbox = self._create_component_panel(parent, "Anomalies & Noise", {k:v for k,v in COMPONENT_MAP.items() if k not in SIGNAL_TYPES})
        sig_listbox.bind('<<ListboxSelect>>', lambda e: self.on_component_select(e, anom_listbox, sig_listbox))
        anom_listbox.bind('<<ListboxSelect>>', lambda e: self.on_component_select(e, sig_listbox, anom_listbox))
        return sig_listbox, anom_listbox

    def _create_component_panel(self, parent, title, classes):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.X, pady=5)
        listbox = tk.Listbox(frame, height=5, exportselection=False)
        listbox.pack(fill=tk.X, padx=5, pady=5)
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        comp_type_var = tk.StringVar(value=list(classes.keys())[0])
        add_combo = ttk.Combobox(btn_frame, textvariable=comp_type_var, values=list(classes.keys()), state='readonly')
        add_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        add_cmd = lambda: self.add_component(comp_type_var.get(), listbox)
        rem_cmd = lambda: self.remove_component(listbox)
        ttk.Button(btn_frame, text="Add", command=add_cmd).pack(side=tk.LEFT, padx=(5,0))
        ttk.Button(btn_frame, text="Remove", command=rem_cmd).pack(side=tk.LEFT, padx=(5,0))
        return listbox

    def setup_preview_panel(self, parent):
        preview_frame = ttk.LabelFrame(parent, text="Component Preview")
        preview_frame.pack(fill=tk.X, pady=5)
        self.preview_fig, self.preview_ax = plt.subplots(figsize=(4, 1.5))
        self.preview_fig.tight_layout()
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_plot_panels(self, parent):
        scrollable_area = ScrollableFrame(parent)
        scrollable_area.pack(fill="both", expand=True)
        container = scrollable_area.scrollable_frame

        signal_frame = ttk.LabelFrame(container, text="Time-Domain Signal")
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)
        self.fig, self.ax = plt.subplots(figsize=(8, 2.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=signal_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        view_frame = ttk.Frame(signal_frame)
        view_frame.pack(fill=tk.X, pady=(5,0))
        create_slider_entry(view_frame, "Visible Periods:", self.periods, 1, 50, self.on_periods_changed)
        create_slider_entry(view_frame, "Duration (s):", self.duration_seconds, 0.1, 10, self.on_duration_changed)

        fft_frame = ttk.LabelFrame(container, text="FFT Spectrum")
        fft_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        self.fft_fig, self.fft_ax = plt.subplots(figsize=(8, 2.5))
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=fft_frame)
        self.fft_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        stft_outer_frame = ttk.LabelFrame(container, text="STFT Analysis")
        stft_outer_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0), padx=5)
        stft_controls_frame = ttk.Frame(stft_outer_frame, width=200)
        stft_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 10), pady=5)
        stft_plot_frame = ttk.Frame(stft_outer_frame)
        stft_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.stft_fig, self.stft_ax = plt.subplots(figsize=(8, 4))
        self.stft_canvas = FigureCanvasTkAgg(self.stft_fig, master=stft_plot_frame)
        self.stft_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        create_slider_entry(stft_controls_frame, "Window Size:", self.stft_window_size, 32, 512, self.update_plot)
        create_slider_entry(stft_controls_frame, "Overlap:", self.stft_overlap, 0, 511, self.update_plot)
        ttk.Label(stft_controls_frame, text="Window Type:").pack(pady=(10,0))
        window_combo = ttk.Combobox(stft_controls_frame, textvariable=self.stft_window_type, 
                                    values=['hann', 'hamming', 'blackman', 'bartlett', 'boxcar'], state='readonly')
        window_combo.pack(fill=tk.X)
        window_combo.bind('<<ComboboxSelected>>', self.update_plot)

    def on_periods_changed(self, event=None):
        if self._is_updating_sliders: return
        self._is_updating_sliders = True
        max_freq = self._get_max_freq()
        new_duration = self.periods.get() / max_freq if max_freq > 0 else 2
        self.duration_seconds.set(new_duration)
        self.update_plot()
        self._is_updating_sliders = False

    def on_duration_changed(self, event=None):
        if self._is_updating_sliders: return
        self._is_updating_sliders = True
        max_freq = self._get_max_freq()
        new_periods = self.duration_seconds.get() * max_freq
        self.periods.set(new_periods)
        self.update_plot()
        self._is_updating_sliders = False

    def add_component(self, comp_name, listbox, model_config=None):
        model_class, controller_class = COMPONENT_MAP[comp_name]
        model = model_class.from_dict(model_config) if model_config else model_class()
        controller = controller_class(model, self.update_plot, self.get_duration)
        self.controllers.append(controller)
        listbox.insert(tk.END, str(controller))
        return controller

    def remove_component(self, listbox):
        idxs = listbox.curselection()
        if not idxs: return
        selected_str = listbox.get(idxs[0])
        listbox.delete(idxs[0])
        controller_to_remove = next((c for c in self.controllers if str(c) == selected_str), None)
        if controller_to_remove:
            self.controllers.remove(controller_to_remove)
            if self.selected_controller and self.selected_controller.id == controller_to_remove.id:
                if self.config_frame: self.config_frame.destroy()
                self.selected_controller = None
        self.update_plot()

    def on_component_select(self, event, other_listbox, active_listbox):
        other_listbox.selection_clear(0, tk.END)
        idxs = active_listbox.curselection()
        if not idxs:
            if self.config_frame: self.config_frame.destroy()
            self.selected_controller = None
            self.update_plot()
            return
        selected_str = active_listbox.get(idxs[0])
        self.selected_controller = next((c for c in self.controllers if str(c) == selected_str), None)
        if self.config_frame: self.config_frame.destroy()
        self.config_frame = ttk.Frame(self.config_panel)
        self.config_frame.pack(fill=tk.X, padx=5, pady=5)
        if self.selected_controller:
            self.selected_controller.get_config_frame(self.config_frame)
        self.update_plot()

    def save_configuration(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not filepath: return
        try:
            ui_settings = {
                "periods": self.periods.get(),
                "duration_seconds": self.duration_seconds.get(),
                "stft_window_size": self.stft_window_size.get(),
                "stft_overlap": self.stft_overlap.get(),
                "stft_window_type": self.stft_window_type.get()
            }
            config = {
                'ui_settings': ui_settings,
                'components': [c.model.to_dict() for c in self.controllers]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except (IOError, TypeError) as e:
            messagebox.showerror("Save Error", f"Failed to save configuration file:\n{e}")

    def load_configuration(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.controllers.clear()
            self.signal_listbox.delete(0, tk.END)
            self.anomaly_listbox.delete(0, tk.END)
            if self.config_frame: self.config_frame.destroy()
            self.selected_controller = None

            if 'ui_settings' in config:
                settings = config['ui_settings']
                self.periods.set(settings.get('periods', 5.0))
                self.duration_seconds.set(settings.get('duration_seconds', 2.0))
                self.stft_window_size.set(settings.get('stft_window_size', 256))
                self.stft_overlap.set(settings.get('stft_overlap', 128))
                self.stft_window_type.set(settings.get('stft_window_type', 'hann'))

            for comp_conf in config.get('components', []):
                comp_type = comp_conf.get('type')
                if comp_type in COMPONENT_MAP:
                    listbox = self.signal_listbox if comp_type in SIGNAL_TYPES else self.anomaly_listbox
                    self.add_component(comp_type, listbox, comp_conf)
            self.update_plot()
        except (IOError, json.JSONDecodeError, KeyError) as e:
            messagebox.showerror("Load Error", f"Failed to load or parse configuration file:\n{e}")

    def _get_max_freq(self):
        signal_models = [c.model for c in self.controllers if isinstance(c.model, SignalComponentModel)]
        signal_freqs = [m.get_max_freq() for m in signal_models if m.get_max_freq() > 0]
        return max(signal_freqs) if signal_freqs else 1.0

    def update_plot(self, event=None):
        duration = self.get_duration()
        max_freq = self._get_max_freq()
        fs = max(1000, max_freq * 40)
        if self.selected_controller: self.selected_controller.update_slider_ranges()
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        if len(t) == 0: return
        
        y = np.zeros_like(t)
        for controller in self.controllers:
            y = controller.model.generate(t, y)
            
        self.ax.clear(); self.fft_ax.clear(); self.stft_ax.clear()
        
        self.ax.plot(t, y)
        self.ax.set_title("Time-Domain Signal"); self.ax.set_xlabel("Time [s]"); self.ax.set_ylabel("Amplitude")
        self.ax.grid(True); self.ax.set_xlim(0, duration)
        self.canvas.draw()
        
        xf, yf = perform_fft(y, fs)
        self.fft_ax.plot(xf, yf)
        self.fft_ax.set_title("FFT Spectrum"); self.fft_ax.set_xlabel("Frequency [Hz]"); self.fft_ax.set_ylabel("Amplitude")
        self.fft_ax.grid(True); self.fft_ax.set_xlim(0, max_freq * 2 if max_freq > 0 else 100)
        self.fft_canvas.draw()

        f, t_stft, Zxx = perform_stft(y, fs, self.stft_window_type.get(), self.stft_window_size.get(), self.stft_overlap.get())
        if Zxx.size > 0:
            self.stft_ax.pcolormesh(t_stft, f, Zxx, shading='gouraud')
        self.stft_ax.set_title(""); self.stft_ax.set_xlabel("Time [s]"); self.stft_ax.set_ylabel("Frequency [Hz]")
        self.stft_ax.set_ylim(0, max_freq * 2 if max_freq > 0 else 100)
        self.stft_canvas.draw()

        self.preview_ax.clear()
        if self.selected_controller:
            y_preview = self.selected_controller.model.generate(t, np.zeros_like(t))
            self.preview_ax.plot(t, y_preview, color='darkorange')
            self.preview_ax.set_title(f"Preview: {self.selected_controller.model.name}", fontsize=9)
            self.preview_ax.set_xlim(self.ax.get_xlim()); self.preview_ax.set_ylim(self.ax.get_ylim())
        else:
            self.preview_ax.set_title("No Component Selected", fontsize=9)
        self.preview_ax.tick_params(axis='x', labelsize=8); self.preview_ax.tick_params(axis='y', labelsize=8)
        self.preview_canvas.draw()

if __name__ == '__main__':
    root = tk.Tk()
    app = SignalGeneratorApp(root)
    root.mainloop()
