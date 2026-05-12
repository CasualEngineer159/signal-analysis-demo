import numpy as np
from scipy.signal import stft, get_window, find_peaks

def perform_fft(y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a Fast Fourier Transform on a given signal.

    Args:
        y (np.ndarray): The input signal array.
        fs (float): The sampling frequency of the signal.

    Returns:
        A tuple containing:
        - The frequency bins (x-axis).
        - The corresponding amplitudes (y-axis).
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(n, 1 / fs)[:n//2]
    
    # Normalize the amplitude
    amplitude = 2.0/n * np.abs(yf[0:n//2])
    
    return xf, amplitude


def find_fft_peaks(xf: np.ndarray, amplitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if amplitude.size == 0 or len(xf) == 0:
        return np.array([]), np.array([])

    # 1. Dynamic Threshold: Set height relative to the max amplitude.
    # This ensures that peaks are found regardless of the overall signal strength.
    max_amp = np.max(amplitude) if amplitude.size > 0 else 0
    threshold = max_amp * 0.05

    # 2. Dynamic Prominence: Also set prominence relative to the max amplitude.
    # This helps to avoid detecting noisy peaks in a strong signal.
    prominence = max_amp * 0.05

    # 3. Distance: Minimum distance between peaks in samples (bins).
    # This prevents detecting multiple peaks on the same spectral feature.
    min_distance_bins = 5

    # Find peaks with the dynamic parameters
    peaks, _ = find_peaks(
        amplitude,
        height=threshold,
        prominence=prominence,
        distance=min_distance_bins
    )

    peak_freqs = xf[peaks]
    peak_amps = amplitude[peaks]

    return peak_freqs, peak_amps

def perform_stft(y: np.ndarray, fs: float, window_type: str, nperseg: int, noverlap: int) -> tuple:
    """
    Performs a Short-Time Fourier Transform on a given signal.

    Args:
        y (np.ndarray): The input signal array.
        fs (float): The sampling frequency.
        window_type (str): The type of window to use (e.g., 'hann').
        nperseg (int): The number of samples per segment.
        noverlap (int): The number of samples to overlap between segments.

    Returns:
        A tuple containing:
        - f (np.ndarray): Array of sample frequencies.
        - t (np.ndarray): Array of segment times.
        - Zxx (np.ndarray): The complex STFT result.
    """
    n = len(y)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # Ensure window size is not greater than the signal length
    nperseg = min(nperseg, n)
    
    # Ensure overlap is less than the window size
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    try:
        window = get_window(window_type, nperseg)
    except ValueError:
        # Fallback to a default window if the type is invalid
        window = get_window('hann', nperseg)

    f, t, Zxx = stft(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)

def calculate_spectral_flux(Zxx: np.ndarray, t_stft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the spectral flux of an STFT result and finds anomalies.

    Args:
        Zxx (np.ndarray): The STFT magnitude matrix.
        t_stft (np.ndarray): The array of segment times.

    Returns:
        A tuple containing:
        - The 1D flux array.
        - An array of times in seconds where anomalies (peaks) were detected.
    """
    if Zxx.size == 0 or len(t_stft) == 0:
        return np.array([]), np.array([])

    # Calculate absolute difference between adjacent columns
    diff = np.abs(np.diff(Zxx, axis=1))
    
    # Sum differences along the frequency axis
    flux = np.sum(diff, axis=0)
    
    # np.diff reduces the array length by 1. To match t_stft length, 
    # prepend the FIRST value of flux to itself instead of 0.0, 
    # so the graph doesn't artificially start at zero:
    if len(flux) > 0:
        flux = np.insert(flux, 0, flux[0])

    # Robustní výpočet prahu pomocí Mediánu a MAD
    median_flux = np.median(flux)
    mad_flux = np.median(np.abs(flux - median_flux))

    # Pokud je MAD nula (což se může stát u absolutně čistého signálu bez šumu), přidáme pojistku
    if mad_flux < 1e-6:
        mad_flux = np.max(flux) * 0.1

    # Práh nastavíme na medián + N-násobek MAD (např. 5x až 10x)
    threshold = median_flux + 3 * mad_flux

    # (Original threshold calculation is not used directly, standardizing on find_peaks argument if needed later)
    peaks, _ = find_peaks(flux, height=threshold, prominence=0.3)
    peak_times = t_stft[peaks]
    
    return flux, peak_times

def evaluate_detection(ground_truth_times: list[float], predicted_times: list[float], tolerance: float = 0.1) -> dict:
    """
    Evaluates anomaly detection performance.

    Args:
        ground_truth_times (list[float]): The actual times anomalies occurred.
        predicted_times (list[float]): The times anomalies were predicted.
        tolerance (float): The maximum time difference to consider a match valid.

    Returns:
        dict: A dictionary containing TP, FP, FN, Precision, Recall, and F1-Score.
    """
    tp = 0
    fp = 0
    fn = 0
    
    matched_predictions = set()
    
    for gt in ground_truth_times:
        # Find all predictions within tolerance
        valid_preds = [i for i, p in enumerate(predicted_times) if abs(p - gt) <= tolerance and i not in matched_predictions]
        
        if valid_preds:
            # Pick the closest one
            closest_pred_idx = min(valid_preds, key=lambda i: abs(predicted_times[i] - gt))
            matched_predictions.add(closest_pred_idx)
            tp += 1
        else:
            fn += 1
            
    fp = len(predicted_times) - len(matched_predictions)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    }