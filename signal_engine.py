import numpy as np
from dataclasses import dataclass, field
from analysis import perform_stft, calculate_spectral_flux, perform_fft, evaluate_detection, find_fft_peaks

@dataclass
class PipelineResult:
    fs: float
    t: np.ndarray
    y: np.ndarray
    t_ext: np.ndarray
    y_ext: np.ndarray
    t_stft_ext: np.ndarray
    f: np.ndarray
    Zxx_ext: np.ndarray
    t_stft_core: np.ndarray
    flux: np.ndarray
    peak_times: np.ndarray
    xf: np.ndarray
    yf: np.ndarray
    fft_peak_freqs: np.ndarray
    fft_peak_amps: np.ndarray
    ground_truth_times: list[float] = field(default_factory=list)
    evaluation_metrics: dict = field(default_factory=dict)

def generate_pipeline_data(controllers, duration, max_freq, stft_window_size, stft_overlap, stft_window_type, rectify=False) -> PipelineResult:
    fs = max(1000, max_freq * 40)
    
    # Calculate padding
    padding_time = (stft_window_size / 2) / fs
    
    # Extended time vector for generating the padded signal
    t_ext = np.linspace(-padding_time, duration + padding_time, int(fs * (duration + 2 * padding_time)), endpoint=False)
    
    y_ext = np.zeros_like(t_ext)
    
    ground_truth_times = []
    if len(t_ext) > 0:
        for controller in controllers:
            y_ext = controller.model.generate(t_ext, y_ext)
            if hasattr(controller.model, 'get_anomaly_times'):
                ground_truth_times.extend(controller.model.get_anomaly_times())

    # Core time vector for plotting standard Time-Domain and FFT
    t_core_idx = (t_ext >= 0) & (t_ext < duration)
    t = t_ext[t_core_idx]
    y = y_ext[t_core_idx]

    # STFT Plot on PADDED signal
    f, t_stft_ext, Zxx_ext = perform_stft(y_ext, fs, stft_window_type, stft_window_size, stft_overlap)
    
    # Shift the STFT time axis back to start at -padding_time
    t_stft_ext = t_stft_ext - padding_time
    
    # 2. Extract the absolutely clean CORE area to prevent edge artifacts in Spectral Flux
    stft_core_idx = (t_stft_ext >= 0) & (t_stft_ext <= duration)
    t_stft_core = t_stft_ext[stft_core_idx]
    Zxx_core = Zxx_ext[:, stft_core_idx]

    # Spectral Flux Plot calculated ONLY on clean core data
    flux, peak_times = calculate_spectral_flux(Zxx_core, t_stft_core, rectify=rectify)
    
    # Evaluate anomaly detection
    # Default tolerance could be roughly 0.1 seconds, or dependent on window size
    tolerance = (stft_window_size / fs) * 2  # Example adaptive tolerance
    evaluation_metrics = evaluate_detection(ground_truth_times, peak_times.tolist(), tolerance=max(0.1, tolerance))

    # FFT Plot
    xf, yf = perform_fft(y, fs)
    fft_peak_freqs, fft_peak_amps = find_fft_peaks(xf, yf)

    return PipelineResult(
        fs=fs,
        t=t,
        y=y,
        t_ext=t_ext,
        y_ext=y_ext,
        t_stft_ext=t_stft_ext,
        f=f,
        Zxx_ext=Zxx_ext,
        t_stft_core=t_stft_core,
        flux=flux,
        peak_times=peak_times,
        xf=xf,
        yf=yf,
        fft_peak_freqs=fft_peak_freqs,
        fft_peak_amps=fft_peak_amps,
        ground_truth_times=ground_truth_times,
        evaluation_metrics=evaluation_metrics
    )