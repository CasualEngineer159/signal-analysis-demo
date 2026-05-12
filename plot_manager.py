import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from utils import ScrollableFrame

class PlotManager:
    def __init__(self, parent_frame):
        scrollable_area = ScrollableFrame(parent_frame, h_scroll=True)
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

        # FFT Peaks Results Table
        self.fft_results_frame = ttk.LabelFrame(right_graphs, text="FFT Peaks")
        self.fft_results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 5), padx=5)
        
        fft_columns = ("Frequency [Hz]", "Amplitude")
        self.fft_results_table = ttk.Treeview(self.fft_results_frame, columns=fft_columns, show="headings", height=3)
        for col in fft_columns:
            self.fft_results_table.heading(col, text=col)
            self.fft_results_table.column(col, width=120, anchor=tk.CENTER)
            
        fft_scrollbar = ttk.Scrollbar(self.fft_results_frame, orient=tk.VERTICAL, command=self.fft_results_table.yview)
        self.fft_results_table.configure(yscrollcommand=fft_scrollbar.set)
        fft_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.fft_results_table.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Detection Results Table
        self.results_frame = ttk.LabelFrame(right_graphs, text="Detection Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 5), padx=5)
        
        # Setup Treeview inside the frame
        columns = ("Algorithm", "Detected", "Time [s]")
        self.results_table = ttk.Treeview(self.results_frame, columns=columns, show="headings", height=5)
        
        for col in columns:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=120, anchor=tk.CENTER)
            
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_table.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Setup Evaluation Metrics Panel
        self.eval_frame = ttk.Frame(self.results_frame)
        self.eval_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.eval_labels = {}
        metrics = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(self.eval_frame, text=f"{metric}:").grid(row=row, column=col, padx=(10 if col > 0 else 0, 5), pady=2, sticky=tk.W)
            value_label = ttk.Label(self.eval_frame, text="-")
            value_label.grid(row=row, column=col+1, padx=5, pady=2, sticky=tk.W)
            self.eval_labels[metric] = value_label

    def draw_plots(self, duration, max_freq, fs, t, y, t_ext, y_ext, t_stft_ext, f, Zxx_ext, t_stft_core, flux, peak_times, xf, yf, ground_truth_times=None, evaluation_metrics=None, fft_peak_freqs=None, fft_peak_amps=None):
        self.ax.clear(); self.fft_ax.clear(); self.stft_ax.clear(); self.flux_ax.clear()

        # Time-Domain Plot
        self.ax.plot(t, y)
        self.ax.set_title("Time-Domain Signal")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(0, duration) # Force strict limits
        self.ax.grid(True)

        # 1. Plot the EXTENDED STFT. This prevents any white gaps on the edges because 
        # Matplotlib's xlim will visually crop it perfectly to the axis boundaries.
        if Zxx_ext.size > 0:
            self.stft_ax.pcolormesh(t_stft_ext, f, Zxx_ext, shading='gouraud')
        self.stft_ax.set_ylabel("Frequency [Hz]")
        self.stft_ax.set_ylim(0, max_freq * 2 if max_freq > 0 else 100)
        self.stft_ax.set_xlim(0, duration) # Force strict visual limits

        # Spectral Flux Plot calculated ONLY on clean core data
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
        self.fft_ax.plot(xf, yf)
        if fft_peak_freqs is not None and fft_peak_amps is not None:
            for i, (freq, amp) in enumerate(zip(fft_peak_freqs, fft_peak_amps)):
                self.fft_ax.axvline(x=freq, color='red', linestyle='--', alpha=0.7, label='Detected Peak' if i == 0 else "")
                self.fft_ax.plot(freq, amp, "rx")
            
            if len(fft_peak_freqs) > 0:
                self.fft_ax.legend(loc='upper right', fontsize='small')

        self.fft_ax.set_title("FFT Spectrum")
        self.fft_ax.set_xlabel("Frequency [Hz]")
        self.fft_ax.set_ylabel("Amplitude")
        self.fft_ax.grid(True)
        self.fft_ax.set_xlim(0, max_freq * 2 if max_freq > 0 else 100)
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()

        # Update FFT Results Table
        self.fft_results_table.delete(*self.fft_results_table.get_children())
        if fft_peak_freqs is not None and fft_peak_amps is not None:
            for freq, amp in zip(fft_peak_freqs, fft_peak_amps):
                self.fft_results_table.insert("", tk.END, values=(f"{freq:.3f}", f"{amp:.3f}"))

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
        
        # Add Ground Truth Times to the Results Table
        if ground_truth_times is not None:
            if len(ground_truth_times) > 0:
                gt_str = ", ".join([f"{pt:.3f}" for pt in ground_truth_times])
                gt_detected = "Yes"
            else:
                gt_str = "N/A"
                gt_detected = "No"
                
            self.results_table.insert("", tk.END, values=("Ground Truth (Actual)", gt_detected, gt_str))

        
        # Update Evaluation Metrics
        if evaluation_metrics:
            for metric, label in self.eval_labels.items():
                if metric in evaluation_metrics:
                    val = evaluation_metrics[metric]
                    if isinstance(val, float):
                        label.config(text=f"{val:.3f}")
                    else:
                        label.config(text=str(val))
                else:
                    label.config(text="-")