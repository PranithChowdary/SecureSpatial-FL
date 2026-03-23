import numpy as np
from scipy import signal

def low_pass_filter(csi_data, cutoff=30, fs=1000, order=5):
    """
    Apply a low-pass Butterworth filter to reduce noise in CSI signals.
    :param csi_data: np.array, raw CSI amplitudes
    :param cutoff: float, cutoff frequency in Hz
    :param fs: float, sampling rate in Hz
    :param order: int, filter order
    :return: np.array, filtered CSI data
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, csi_data, axis=0)
    return filtered_data

def cfr_modeling(csi_data):
    """
    Channel Frequency Response (CFR) Modeling.
    Isolate human gait signatures by removing static components.
    :param csi_data: np.array, complex CSI data
    :return: np.array, dynamic CFR components
    """
    # Remove static components (DC component)
    dynamic_csi = csi_data - np.mean(csi_data, axis=0)
    return dynamic_csi

def extract_periodogram(csi_data, fs=1000):
    """
    Convert processed signals into 2D periodograms for spatial-temporal analysis.
    :param csi_data: np.array, filtered CSI data (time x subcarriers)
    :param fs: float, sampling rate
    :return: np.array, periodogram (frequency x time)
    """
    # Use Welch's method or spectrogram for each subcarrier
    # For simplicity, we can use a basic FFT-based periodogram or spectrogram
    # Here we simulate the 2D spatial-temporal representation
    n_subcarriers = csi_data.shape[1]
    all_specs = []
    
    for i in range(n_subcarriers):
        f, t, Sxx = signal.spectrogram(csi_data[:, i], fs=fs, nperseg=64, noverlap=32)
        all_specs.append(Sxx)
        
    # Average across subcarriers or stack them
    # For human movement, we often look at the average or dominant components
    combined_spec = np.mean(all_specs, axis=0)
    return f, t, combined_spec

if __name__ == "__main__":
    # Test with dummy data
    dummy_csi = np.random.randn(1000, 30) + 1j * np.random.randn(1000, 30)
    filtered = low_pass_filter(np.abs(dummy_csi))
    dynamic = cfr_modeling(dummy_csi)
    f, t, spec = extract_periodogram(np.abs(dynamic))
    print(f"Spec shape: {spec.shape}")
