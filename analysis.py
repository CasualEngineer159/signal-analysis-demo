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
    threshold = median_flux + 10 * mad_flux

    peaks, _ = find_peaks(flux, height=threshold)
    peak_times = t_stft[peaks]
    
    return flux, peak_times
