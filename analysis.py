import numpy as np
from scipy.signal import stft, get_window

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
