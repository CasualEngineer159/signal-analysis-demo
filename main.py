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
        self.stft_auto_overlap = tk.BooleanVar(value=True)
        self.stft_overlap_widget = None

        self._last_flux_data = None
        self._drag_data = {"item_index": None}

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
        self.setup_component_panels(left_panel)
        self.setup_preview_panel(left_panel)
        
        self.config_panel = ttk.LabelFrame(left_panel, text="Parameters")
        self.config_panel.pack(fill=tk.X, pady=(10, 5))
        
        self.setup_time_settings_panel(left_panel)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)
        
        self.setup_stft_settings_panel(left_panel)
        
        self.setup_plot_panels(right_panel)
        self.update_plot()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def get_duration(self):
        return self.duration_seconds.get()

    def setup_file_io_panel(self, parent):
        io_frame = ttk.LabelFrame(parent, text="Configuration")
        io_frame.pack(fill=tk.X, pady=(10, 5))
        btn_frame = ttk.Frame(io_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        save_btn = ttk.Button(btn_frame, text="Save Config", command=self.save_configuration)
        save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        load_btn = ttk.Button(btn_frame, text="Load Config", command=self.load_configuration)
        load_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

    def setup_component_panels(self, parent):
        # Execution Pipeline
        pipeline_frame = ttk.LabelFrame(parent, text="Execution Pipeline")
        pipeline_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.pipeline_listbox = tk.Listbox(pipeline_frame, height=6, exportselection=False)
        self.pipeline_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.pipeline_listbox.bind('<<ListboxSelect>>', self.on_component_select)
        
        # Drag and Drop bindings
        self.pipeline_listbox.bind('<Button-1>', self.on_drag_start)
        self.pipeline_listbox.bind('<B1-Motion>', self.on_drag_motion)
        self.pipeline_listbox.bind('<ButtonRelease-1>', self.on_drag_release)
        
        btn_frame = ttk.Frame(pipeline_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Button(btn_frame, text="Move Up", command=self.move_component_up).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(btn_frame, text="Move Down", command=self.move_component_down).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 2))
        ttk.Button(btn_frame, text="Remove", command=self.remove_component).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # Add Signal
        sig_frame = ttk.LabelFrame(parent, text="Add Signal")
        sig_frame.pack(fill=tk.X, pady=(5, 5))
        sig_btn_frame = ttk.Frame(sig_frame)
        sig_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        sig_classes = {k:v for k,v in COMPONENT_MAP.items() if k in SIGNAL_TYPES}
        sig_var = tk.StringVar(value=list(sig_classes.keys())[0])
        ttk.Combobox(sig_btn_frame, textvariable=sig_var, values=list(sig_classes.keys()), state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sig_btn_frame, text="Add", command=lambda: self.add_component(sig_var.get())).pack(side=tk.LEFT, padx=(5,0))

        # Add Anomaly
        anom_frame = ttk.LabelFrame(parent, text="Add Anomaly & Noise")
        anom_frame.pack(fill=tk.X, pady=(5, 5))
        anom_btn_frame = ttk.Frame(anom_frame)
        anom_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        anom_classes = {k:v for k,v in COMPONENT_MAP.items() if k not in SIGNAL_TYPES}
        anom_var = tk.StringVar(value=list(anom_classes.keys())[0])
        ttk.Combobox(anom_btn_frame, textvariable=anom_var, values=list(anom_classes.keys()), state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(anom_btn_frame, text="Add", command=lambda: self.add_component(anom_var.get())).pack(side=tk.LEFT, padx=(5,0))

    def on_drag_start(self, event):
        """Record the item's starting index."""
        index = self.pipeline_listbox.nearest(event.y)
        if index >= 0:
            self._drag_data["item_index"] = index

    def on_drag_motion(self, event):
        """Move the item visually in the listbox as it is dragged."""
        current_index = self._drag_data["item_index"]
        if current_index is None:
            return
            
        new_index = self.pipeline_listbox.nearest(event.y)
        
        if new_index != current_index and new_index >= 0 and new_index < self.pipeline_listbox.size():
            # Update Listbox visually
            text = self.pipeline_listbox.get(current_index)
            self.pipeline_listbox.delete(current_index)
            self.pipeline_listbox.insert(new_index, text)
            
            # Update controllers array to match visual order
            controller = self.controllers.pop(current_index)
            self.controllers.insert(new_index, controller)
            
            # Maintain selection if it was the selected item
            if self.selected_controller and self.selected_controller.id == controller.id:
                self.pipeline_listbox.selection_clear(0, tk.END)
                self.pipeline_listbox.selection_set(new_index)
            else:
                # If we moved an item past the currently selected one, we need to fix the visual selection
                if self.selected_controller:
                    try:
                        sel_idx = self.controllers.index(self.selected_controller)
                        self.pipeline_listbox.selection_clear(0, tk.END)
                        self.pipeline_listbox.selection_set(sel_idx)
                    except ValueError:
                        pass
                        
            self._drag_data["item_index"] = new_index

    def on_drag_release(self, event):
        """Finalize drag and trigger an update."""
        if self._drag_data["item_index"] is not None:
            self._drag_data["item_index"] = None
            self.update_plot()

    def setup_preview_panel(self, parent):
        preview_frame = ttk.LabelFrame(parent, text="Component Preview")
        preview_frame.pack(fill=tk.X, pady=(10, 5))
        self.preview_fig, self.preview_ax = plt.subplots(figsize=(4, 1.5))
        self.preview_fig.tight_layout()
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_time_settings_panel(self, parent):
        time_frame = ttk.LabelFrame(parent, text="Global Time Settings")
        time_frame.pack(fill=tk.X, pady=(10, 5))
        create_slider_entry(time_frame, "Visible Periods:", self.periods, 1, 50, self.on_periods_changed)
        create_slider_entry(time_frame, "Duration (s):", self.duration_seconds, 0.1, 10, self.on_duration_changed)

    def setup_stft_settings_panel(self, parent):
        stft_controls_frame = ttk.LabelFrame(parent, text="STFT Settings")
        stft_controls_frame.pack(fill=tk.X, pady=(10, 5))
        
        create_slider_entry(stft_controls_frame, "Window Size:", self.stft_window_size, 32, 512, self.on_stft_params_changed)
        self.stft_overlap_widget = create_slider_entry(stft_controls_frame, "Overlap:", self.stft_overlap, 0, 511, self.on_stft_params_changed)

        auto_overlap_check = ttk.Checkbutton(
            stft_controls_frame,
            text="Auto Overlap (50%)",
            variable=self.stft_auto_overlap,
            command=self.on_stft_params_changed
        )
        auto_overlap_check.pack(pady=(5, 0), anchor='w', padx=5)

        ttk.Label(stft_controls_frame, text="Window Type:").pack(pady=(5,0), padx=5, anchor='w')
        window_combo = ttk.Combobox(stft_controls_frame, textvariable=self.stft_window_type, 
                                    values=['hann', 'hamming', 'blackman', 'bartlett', 'boxcar'], state='readonly')
        window_combo.pack(fill=tk.X, padx=5, pady=(0, 5))
        window_combo.bind('<<ComboboxSelected>>', self.on_stft_params_changed)
        
        ttk.Button(stft_controls_frame, text="Export Flux Data to CSV", command=self.export_flux_to_csv).pack(fill=tk.X, padx=5, pady=(10, 5))
        
        self.on_stft_params_changed()

    def export_flux_to_csv(self):
        if self._last_flux_data is None:
            messagebox.showwarning("No Data", "No spectral flux data is available to export.")
            return
            
        t_stft_core, flux = self._last_flux_data
        
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
            
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Time [s]', 'Spectral Flux'])
                for t_val, f_val in zip(t_stft_core, flux):
                    writer.writerow([t_val, f_val])
            messagebox.showinfo("Export Successful", f"Successfully exported spectral flux data to {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{e}")

    def setup_plot_panels(self, parent):
        scrollable_area = ScrollableFrame(parent, h_scroll=True)
        scrollable_area.pack(fill="both", expand=True)
        container = scrollable_area.scrollable_frame

        left_graphs = ttk.Frame(container)
        left_graphs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_graphs = ttk.Frame(container)
        right_graphs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Combined Time-Domain, STFT, and Spectral Flux plot
        signal_frame = ttk.LabelFrame(left_graphs, text="Time-Domain & STFT Analysis")
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)
        self.fig, (self.ax, self.stft_ax, self.flux_ax) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 6.5))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=signal_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # FFT Spectrum
        fft_frame = ttk.LabelFrame(right_graphs, text="FFT Spectrum")
        fft_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5, anchor='n')
        self.fft_fig, self.fft_ax = plt.subplots(figsize=(6, 2.5))
        self.fft_fig.tight_layout()
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=fft_frame)
        self.fft_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Detection Results Table
        results_frame = ttk.LabelFrame(right_graphs, text="Detection Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 5), padx=5)
        
        # Setup Treeview inside the frame
        columns = ("Algorithm", "Detected", "Time [s]")
        self.results_table = ttk.Treeview(results_frame, columns=columns, show="headings", height=5)
        
        for col in columns:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=120, anchor=tk.CENTER)
            
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)


    def on_stft_params_changed(self, event=None):
        """Handles all changes to STFT parameters and updates UI state."""
        if not hasattr(self, 'stft_overlap_widget') or self.stft_overlap_widget is None:
            if hasattr(self, 'ax'):
                self.update_plot()
            return

        window_size = self.stft_window_size.get()
        self.stft_overlap_widget.slider.config(to=window_size - 1)

        if self.stft_auto_overlap.get():
            self.stft_overlap.set(window_size // 2)
            for widget in self.stft_overlap_widget.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Scale)):
                    widget.config(state='disabled')
        else:
            for widget in self.stft_overlap_widget.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Scale)):
                    widget.config(state='normal')

        if self.stft_overlap.get() >= window_size:
            self.stft_overlap.set(window_size - 1)

        if hasattr(self, 'ax'):
            self.update_plot()

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

    def add_component(self, comp_name, model_config=None):
        model_class, controller_class = COMPONENT_MAP[comp_name]
        model = model_class.from_dict(model_config) if model_config else model_class()
        controller = controller_class(model, self.update_plot, self.get_duration)
        self.controllers.append(controller)
        self.pipeline_listbox.insert(tk.END, str(controller))
        self.update_plot()
        return controller

    def remove_component(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        self.pipeline_listbox.delete(idx)
        controller_to_remove = self.controllers.pop(idx)
        if self.selected_controller and self.selected_controller.id == controller_to_remove.id:
            if self.config_frame: self.config_frame.destroy()
            self.selected_controller = None
        self.update_plot()

    def move_component_up(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        if idx == 0: return # Already at top
        
        # Swap in controllers
        self.controllers[idx - 1], self.controllers[idx] = self.controllers[idx], self.controllers[idx - 1]
        
        # Swap in listbox
        text = self.pipeline_listbox.get(idx)
        self.pipeline_listbox.delete(idx)
        self.pipeline_listbox.insert(idx - 1, text)
        self.pipeline_listbox.selection_set(idx - 1)
        
        self.update_plot()

    def move_component_down(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        if idx == len(self.controllers) - 1: return # Already at bottom
        
        # Swap in controllers
        self.controllers[idx + 1], self.controllers[idx] = self.controllers[idx], self.controllers[idx + 1]
        
        # Swap in listbox
        text = self.pipeline_listbox.get(idx)
        self.pipeline_listbox.delete(idx)
        self.pipeline_listbox.insert(idx + 1, text)
        self.pipeline_listbox.selection_set(idx + 1)
        
        self.update_plot()

    def on_component_select(self, event=None):
        idxs = self.pipeline_listbox.curselection()
        if not idxs:
            if self.config_frame: self.config_frame.destroy()
            self.selected_controller = None
            self.update_plot()
            return
        idx = idxs[0]
        self.selected_controller = self.controllers[idx]
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
                "stft_window_type": self.stft_window_type.get(),
                "stft_auto_overlap": self.stft_auto_overlap.get()
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
            self.pipeline_listbox.delete(0, tk.END)
            if self.config_frame: self.config_frame.destroy()
            self.selected_controller = None

            if 'ui_settings' in config:
                settings = config['ui_settings']
                self.periods.set(settings.get('periods', 5.0))
                self.duration_seconds.set(settings.get('duration_seconds', 2.0))
                self.stft_window_size.set(settings.get('stft_window_size', 256))
                self.stft_overlap.set(settings.get('stft_overlap', 128))
                self.stft_window_type.set(settings.get('stft_window_type', 'hann'))
                self.stft_auto_overlap.set(settings.get('stft_auto_overlap', True))

            for comp_conf in config.get('components', []):
                comp_type = comp_conf.get('type')
                if comp_type in COMPONENT_MAP:
                    self.add_component(comp_type, comp_conf)
            self.on_stft_params_changed()
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
        
        # Calculate padding
        padding_time = (self.stft_window_size.get() / 2) / fs
        
        # Extended time vector for generating the padded signal
        t_ext = np.linspace(-padding_time, duration + padding_time, int(fs * (duration + 2 * padding_time)), endpoint=False)
        if len(t_ext) == 0: return
        
        y_ext = np.zeros_like(t_ext)
        for controller in self.controllers:
            y_ext = controller.model.generate(t_ext, y_ext)

        # Core time vector for plotting standard Time-Domain and FFT
        t_core_idx = (t_ext >= 0) & (t_ext < duration)
        t = t_ext[t_core_idx]
        y = y_ext[t_core_idx]

        self.ax.clear(); self.fft_ax.clear(); self.stft_ax.clear(); self.flux_ax.clear()

        # Time-Domain Plot
        self.ax.plot(t, y)
        self.ax.set_title("Time-Domain Signal")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(0, duration) # Force strict limits
        self.ax.grid(True)

        # STFT Plot on PADDED signal
        f, t_stft_ext, Zxx_ext = perform_stft(y_ext, fs, self.stft_window_type.get(), self.stft_window_size.get(), self.stft_overlap.get())
        
        # Shift the STFT time axis back to start at -padding_time
        t_stft_ext = t_stft_ext - padding_time
        
        # 1. Plot the EXTENDED STFT. This prevents any white gaps on the edges because 
        # Matplotlib's xlim will visually crop it perfectly to the axis boundaries.
        if Zxx_ext.size > 0:
            self.stft_ax.pcolormesh(t_stft_ext, f, Zxx_ext, shading='gouraud')
        self.stft_ax.set_ylabel("Frequency [Hz]")
        self.stft_ax.set_ylim(0, max_freq * 2 if max_freq > 0 else 100)
        self.stft_ax.set_xlim(0, duration) # Force strict visual limits

        # 2. Extract the absolutely clean CORE area to prevent edge artifacts in Spectral Flux
        stft_core_idx = (t_stft_ext >= 0) & (t_stft_ext <= duration)
        t_stft_core = t_stft_ext[stft_core_idx]
        Zxx_core = Zxx_ext[:, stft_core_idx]

        # Spectral Flux Plot calculated ONLY on clean core data
        flux, peak_times = calculate_spectral_flux(Zxx_core, t_stft_core)

        # Store for CSV export
        self._last_flux_data = (t_stft_core, flux) if len(flux) > 0 else None

        if len(flux) > 0:
            self.flux_ax.plot(t_stft_core, flux, color='purple')
            for i, peak_time in enumerate(peak_times):
                self.flux_ax.axvline(x=peak_time, color='red', linestyle='--', alpha=0.7, label='Detected Anomaly' if i == 0 else "")
            
            if len(peak_times) > 0:
                self.flux_ax.legend(loc='upper right', fontsize='small')
                
        self.flux_ax.set_xlabel("Time [s]")
        self.flux_ax.set_ylabel("Spectral Flux")
        self.flux_ax.set_xlim(0, duration) # Force strict visual limits
        self.flux_ax.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # FFT Plot
        xf, yf = perform_fft(y, fs)
        self.fft_ax.plot(xf, yf)
        self.fft_ax.set_title("FFT Spectrum")
        self.fft_ax.set_xlabel("Frequency [Hz]")
        self.fft_ax.set_ylabel("Amplitude")
        self.fft_ax.grid(True)
        self.fft_ax.set_xlim(0, max_freq * 2 if max_freq > 0 else 100)
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()

        # Update Results Table
        self.results_table.delete(*self.results_table.get_children())
        self.results_table.insert("", tk.END, values=("FFT Global", "Yes" if len(yf) > 0 and np.max(yf) > 0 else "No", "N/A"))
        
        if len(peak_times) > 0:
            peak_str = ", ".join([f"{pt:.3f}" for pt in peak_times])
            detected_str = "Yes"
        else:
            peak_str = "N/A"
            detected_str = "No"
            
        self.results_table.insert("", tk.END, values=("STFT Spectral Flux", detected_str, peak_str))

        # Preview Plot
        self.preview_ax.clear()
        if self.selected_controller:
            y_preview = self.selected_controller.model.generate(t, np.zeros_like(t))
            self.preview_ax.plot(t, y_preview, color='darkorange')
            self.preview_ax.set_title(f"Preview: {self.selected_controller.model.name}", fontsize=9)
            self.preview_ax.set_xlim(self.ax.get_xlim())
            self.preview_ax.set_ylim(self.ax.get_ylim())
        else:
            self.preview_ax.set_title("No Component Selected", fontsize=9)
        self.preview_ax.tick_params(axis='x', labelsize=8)
        self.preview_ax.tick_params(axis='y', labelsize=8)
        self.preview_canvas.draw()

if __name__ == '__main__':
    root = tk.Tk()
    app = SignalGeneratorApp(root)
    root.mainloop()
