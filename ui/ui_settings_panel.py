import tkinter as tk
from tkinter import ttk
import csv
import os
from tkinter import filedialog, messagebox
from core.utils import create_slider_entry

class SettingsPanel:
    """
    Manages the UI for the global settings.

    Args:
        parent: The parent widget.
        on_params_changed (callable): The callback function for when parameters change.
        get_last_flux_data (callable): The callback function to get the last flux data.
        on_debug_signal (callable, optional): The callback function to debug a signal. Defaults to None.
    """
    def __init__(self, parent, on_params_changed, get_last_flux_data, on_debug_signal=None):
        self.on_params_changed = on_params_changed
        self.get_last_flux_data = get_last_flux_data
        self.on_debug_signal = on_debug_signal

        self.periods = tk.DoubleVar(value=5.0)
        self.duration_seconds = tk.DoubleVar(value=2.0)
        self._is_updating_sliders = False
        
        self.stft_window_size = tk.IntVar(value=256)
        self.stft_overlap_percent = tk.DoubleVar(value=50.0)
        self.stft_window_type = tk.StringVar(value='hann')
        self.stft_overlap_widget = None

        self.spectral_flux_rectify = tk.BooleanVar(value=False)
        self.show_peaks = tk.BooleanVar(value=True)
        
        # Determine the data directory for exporting CSVs
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.setup_time_settings_panel(parent)
        
        # Thicker Horizontal Separator
        separator = tk.Frame(parent, height=5, bg='black', relief=tk.SUNKEN, bd=2)
        separator.pack(fill=tk.X, pady=10)
        
        self.setup_plot_settings_panel(parent)
        
        self.setup_stft_settings_panel(parent)
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)

        self.setup_spectral_flux_settings_panel(parent)

    def setup_time_settings_panel(self, parent):
        """
        Sets up the time settings panel.

        Args:
            parent: The parent widget.
        """
        time_frame = ttk.LabelFrame(parent, text="Global Time Settings")
        time_frame.pack(fill=tk.X, pady=(10, 5))
        create_slider_entry(time_frame, "Visible Periods:", self.periods, 1, 50, self.on_periods_changed)
        create_slider_entry(time_frame, "Duration (s):", self.duration_seconds, 0.1, 10, self.on_duration_changed)
        
        if self.on_debug_signal:
            ttk.Button(time_frame, text="Debug Signal", command=self.on_debug_signal).pack(fill=tk.X, padx=5, pady=(5, 5))

    def setup_plot_settings_panel(self, parent):
        """
        Sets up the plot settings panel.

        Args:
            parent: The parent widget.
        """
        plot_settings_frame = ttk.LabelFrame(parent, text="Plot Settings")
        plot_settings_frame.pack(fill=tk.X, pady=(0, 5))
        
        show_peaks_check = ttk.Checkbutton(
            plot_settings_frame,
            text="Show Peak Detection",
            variable=self.show_peaks,
            command=self.on_params_changed
        )
        show_peaks_check.pack(fill=tk.X, padx=5, pady=5)

    def setup_stft_settings_panel(self, parent):
        """
        Sets up the STFT settings panel.

        Args:
            parent: The parent widget.
        """
        stft_controls_frame = ttk.LabelFrame(parent, text="STFT Settings")
        stft_controls_frame.pack(fill=tk.X, pady=(10, 5))
        
        create_slider_entry(stft_controls_frame, "Window Size:", self.stft_window_size, 32, 2048, self.on_stft_params_changed)
        self.stft_overlap_widget = create_slider_entry(stft_controls_frame, "Overlap (%):", self.stft_overlap_percent, 0, 99, self.on_stft_params_changed)

        ttk.Label(stft_controls_frame, text="Window Type:").pack(pady=(5,0), padx=5, anchor='w')
        window_combo = ttk.Combobox(stft_controls_frame, textvariable=self.stft_window_type, 
                                    values=['hann', 'hamming', 'blackman', 'bartlett', 'boxcar'], state='readonly')
        window_combo.pack(fill=tk.X, padx=5, pady=(0, 5))
        window_combo.bind('<<ComboboxSelected>>', self.on_stft_params_changed)
        
        self.on_stft_params_changed()

    def setup_spectral_flux_settings_panel(self, parent):
        """
        Sets up the spectral flux settings panel.

        Args:
            parent: The parent widget.
        """
        flux_controls_frame = ttk.LabelFrame(parent, text="Spectral Flux Settings")
        flux_controls_frame.pack(fill=tk.X, pady=(10, 5))

        rectify_check = ttk.Checkbutton(
            flux_controls_frame,
            text="Half-wave Rectify",
            variable=self.spectral_flux_rectify,
            command=self.on_params_changed
        )
        rectify_check.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(flux_controls_frame, text="Export Flux Data to CSV", command=self.export_flux_to_csv).pack(fill=tk.X, padx=5, pady=(10, 5))

    def get_settings(self) -> tuple[dict, dict]:
        """
        Returns the current UI settings as dictionaries.

        Returns:
            A tuple containing:
            - ui_settings (dict): The UI settings.
            - spectral_flux_settings (dict): The spectral flux settings.
        """
        ui_settings = {
            'periods': self.periods.get(),
            'duration_seconds': self.duration_seconds.get(),
            'stft_window_size': self.stft_window_size.get(),
            'stft_overlap_percent': self.stft_overlap_percent.get(),
            'stft_window_type': self.stft_window_type.get(),
            'show_peaks': self.show_peaks.get(),
        }
        spectral_flux_settings = {
            'rectify': self.spectral_flux_rectify.get()
        }
        return ui_settings, spectral_flux_settings

    def apply_settings(self, ui_settings: dict, spectral_flux_settings: dict):
        """
        Applies loaded settings to the UI, with defaults for missing values.

        Args:
            ui_settings (dict): The UI settings.
            spectral_flux_settings (dict): The spectral flux settings.
        """
        # Time settings
        self.periods.set(ui_settings.get('periods', 5.0))
        self.duration_seconds.set(ui_settings.get('duration_seconds', 2.0))

        # STFT settings
        self.stft_window_size.set(ui_settings.get('stft_window_size', 256))
        self.stft_overlap_percent.set(ui_settings.get('stft_overlap_percent', 50.0))
        self.stft_window_type.set(ui_settings.get('stft_window_type', 'hann'))

        # Plot settings
        self.show_peaks.set(ui_settings.get('show_peaks', True))

        # Spectral Flux settings
        self.spectral_flux_rectify.set(spectral_flux_settings.get('rectify', False))

        # Notify that parameters have changed to trigger a plot update
        self.on_params_changed()

    @property
    def stft_overlap(self):
        """
        Calculates the absolute overlap value in samples based on the window size and percentage.

        Returns:
            tk.IntVar: The overlap value in samples.
        """
        window_size = self.stft_window_size.get()
        percent = self.stft_overlap_percent.get()
        return tk.IntVar(value=int(window_size * (percent / 100.0)))

    def export_flux_to_csv(self):
        """
        Exports the spectral flux data to a CSV file.
        """
        last_flux_data = self.get_last_flux_data()
        if last_flux_data is None:
            messagebox.showwarning("No Data", "No spectral flux data is available to export.")
            return
            
        t_stft_core, flux = last_flux_data
        
        filepath = filedialog.asksaveasfilename(
            initialdir=self.data_dir,
            defaultextension=".csv", 
            filetypes=[("CSV files", "*.csv")]
        )
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

    def on_stft_params_changed(self, event=None):
        """
        Handles all changes to STFT parameters and updates UI state.

        Args:
            event: The event that triggered the call.
        """
        self.on_params_changed()

    def on_periods_changed(self, event=None):
        """
        Handles changes to the periods parameter.

        Args:
            event: The event that triggered the call.
        """
        if self._is_updating_sliders: return
        self._is_updating_sliders = True
        self.on_params_changed(source="periods")
        self._is_updating_sliders = False

    def on_duration_changed(self, event=None):
        """
        Handles changes to the duration parameter.

        Args:
            event: The event that triggered the call.
        """
        if self._is_updating_sliders: return
        self._is_updating_sliders = True
        self.on_params_changed(source="duration")
        self._is_updating_sliders = False