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

def calculate_spectral_flux(Zxx: np.ndarray, t_stft: np.ndarray, rectify: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the spectral flux of an STFT result and finds anomalies.

    Args:
        Zxx (np.ndarray): The STFT magnitude matrix.
        t_stft (np.ndarray): The array of segment times.
        rectify (bool): If True, applies half-wave rectification.

    Returns:
        A tuple containing:
        - The 1D flux array.
        - An array of times in seconds where anomalies (peaks) were detected.
    """
    if Zxx.size == 0 or len(t_stft) == 0:
        return np.array([]), np.array([])

    # Calculate difference between adjacent columns
    diff = np.diff(Zxx, axis=1)
    
    # Apply half-wave rectification if specified
    if rectify:
        diff = np.maximum(0, diff)
    else:
        diff = np.abs(diff)

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
        mad_flux = np.max(flux) * 0.1 if np.max(flux) > 0 else 0.1

    # Práh nastavíme na medián + N-násobek MAD (např. 5x až 10x)
    threshold = median_flux + 3 * mad_flux

    # (Original threshold calculation is not used directly, standardizing on find_peaks argument if needed later)
    peaks, _ = find_peaks(flux, height=threshold, prominence=0.3)
    peak_times = t_stft[peaks]
    
    return flux, peak_times

def get_matched_pairs(ground_truth_times: list[float], predicted_times: list[float], tolerance: float = 0.1) -> list[tuple[float | None, float | None]]:
    """
    Matches ground truth times to predicted times within a tolerance, ensuring a one-to-one mapping.

    Args:
        ground_truth_times (list[float]): The actual times anomalies occurred.
        predicted_times (list[float]): The times anomalies were predicted.
        tolerance (float): The maximum time difference to consider a match valid.

    Returns:
        A list of tuples, where each tuple is a pair of (ground_truth, predicted) times.
        If a time is unmatched, its corresponding pair will be None.
    """
    # Handle edge cases where one or both lists are empty
    if not ground_truth_times and not predicted_times:
        return []
    if not ground_truth_times:
        return sorted([(None, p) for p in predicted_times], key=lambda x: x[1])
    if not predicted_times:
        return sorted([(gt, None) for gt in ground_truth_times], key=lambda x: x[0])

    gt_list = sorted(ground_truth_times)
    pred_list = sorted(predicted_times)
    
    potential_matches = {} # key: gt_idx, value: list of (pred_idx, diff)
    
    # Step 1: Find all potential matches for each ground truth within tolerance
    for i, gt in enumerate(gt_list):
        for j, p in enumerate(pred_list):
            diff = abs(gt - p)
            if diff <= tolerance:
                if i not in potential_matches:
                    potential_matches[i] = []
                potential_matches[i].append((j, diff))

    # Step 2: Create a list of confirmed one-to-one matches
    matches = [] # list of (gt_idx, pred_idx)
    used_preds = set()

    # Sort by ground truth index to process in order
    for gt_idx in sorted(potential_matches.keys()):
        # Sort this GT's potential matches by difference, smallest first
        sorted_preds = sorted(potential_matches[gt_idx], key=lambda x: x[1])
        
        # Find the best, unused prediction for this ground truth
        for pred_idx, diff in sorted_preds:
            if pred_idx not in used_preds:
                # This is a valid match. However, we must check if another GT
                # could also claim this prediction with an even better score.
                is_best_claim = True
                for other_gt_idx in potential_matches:
                    if other_gt_idx != gt_idx:
                        for other_pred_idx, other_diff in potential_matches[other_gt_idx]:
                            if other_pred_idx == pred_idx and other_diff < diff:
                                is_best_claim = False
                                break
                    if not is_best_claim:
                        break
                
                if is_best_claim:
                    matches.append((gt_idx, pred_idx))
                    used_preds.add(pred_idx)
                    break # Move to the next ground truth

    # Step 3: Build the final list of pairs
    final_pairs = []
    matched_gt_indices = {m[0] for m in matches}
    matched_pred_indices = {m[1] for m in matches}

    # Add the confirmed matches
    for gt_idx, pred_idx in matches:
        final_pairs.append((gt_list[gt_idx], pred_list[pred_idx]))

    # Add unmatched ground truths (False Negatives)
    for i in range(len(gt_list)):
        if i not in matched_gt_indices:
            final_pairs.append((gt_list[i], None))

    # Add unmatched predictions (False Positives)
    for i in range(len(pred_list)):
        if i not in matched_pred_indices:
            final_pairs.append((None, pred_list[i]))

    # Sort the final list for a clean, chronological display
    final_pairs.sort(key=lambda x: x[0] if x[0] is not None else x[1])
    
    return final_pairs

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
    
    # Use the new matching logic to get TP, FP, FN
    matched_pairs = get_matched_pairs(ground_truth_times, predicted_times, tolerance)
    
    tp = sum(1 for gt, pred in matched_pairs if gt is not None and pred is not None)
    fn = sum(1 for gt, pred in matched_pairs if gt is not None and pred is None)
    fp = sum(1 for gt, pred in matched_pairs if gt is None and pred is not None)
    
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