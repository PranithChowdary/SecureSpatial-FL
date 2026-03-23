import matplotlib.pyplot as plt
import numpy as np
import torch
from preprocessing import low_pass_filter, cfr_modeling, extract_periodogram

def plot_csi_signals(raw_csi, filtered_csi, title="CSI Amplitude"):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(raw_csi)[:, 0], label='Raw')
    plt.title(f"{title} (Subcarrier 0)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(filtered_csi[:, 0], label='Filtered', color='orange')
    plt.title("Filtered CSI")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("./logs/csi_filtering.png")
    plt.show()

def plot_periodogram(f, t, spec, title="2D Periodogram"):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(t, f, 10 * np.log10(spec), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')
    plt.savefig("./logs/periodogram.png")
    plt.show()

def main():
    # Simulate signal visualization
    t = np.linspace(0, 1, 1000)
    # Simulate a human movement signal at ~2Hz
    signal_main = np.sin(2 * np.pi * 2 * t) + 0.5 * np.random.randn(1000)
    dummy_csi = np.tile(signal_main, (30, 1)).T
    
    filtered = low_pass_filter(dummy_csi)
    plot_csi_signals(dummy_csi, filtered)
    
    f, t_spec, spec = extract_periodogram(filtered)
    plot_periodogram(f, t_spec, spec)
    print("Visualizations saved to ./logs/")

if __name__ == "__main__":
    import os
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    main()
