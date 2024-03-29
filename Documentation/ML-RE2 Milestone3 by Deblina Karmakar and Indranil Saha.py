import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Load the dataset
df = pd.read_csv('C:\\Users\\Theertha\\PycharmProjects\\AIS_ML\\Dataset\\adc_1m_hard_surface.csv')
df = df.iloc[:, 17:]  # Adjust this as necessary for your dataset

# Initialize array to store calculated distances to the highest peak
distances_highest_peak = np.zeros((len(df.index),), dtype=float)

# Define constants
velocity_of_sound = 343  # Speed of sound in air in m/s
dt = 0.05  # Sampling rate in microseconds (adjust according to your dataset)

for i in range(len(df.index)):
    f = df.iloc[i, :]

    # Perform FFT and calculate Power Spectral Density (PSD)
    fhat = np.fft.fft(f)
    PSD = fhat * np.conj(fhat) / len(f)
    indices = PSD > 1.5  # Threshold value, adjust based on your data
    fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)

    # Use the Hilbert transform to find the envelope of the filtered signal
    analytical_signal = hilbert(ffilt.real)
    env = np.abs(analytical_signal)

    # Find the index of the highest peak in the envelope
    highest_peak_index = np.argmax(env)

    # Calculate the distance to the highest peak
    pos_highest_peak = highest_peak_index * dt
    distance = 0.5 * pos_highest_peak * velocity_of_sound / 1e6  # Convert microseconds to seconds
    distances_highest_peak[i] = distance

# Select the first row for plotting
f = df.iloc[0, :]

# Repeat the processing for the selected row
fhat = np.fft.fft(f)
PSD = fhat * np.conj(fhat) / len(f)
indices = PSD > 1.5
fhat = indices * fhat
ffilt = np.fft.ifft(fhat)
analytical_signal = hilbert(ffilt.real)
env = np.abs(analytical_signal)
highest_peak_index = np.argmax(env)

# Time vector for plotting
time = np.arange(len(f)) * dt

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Original signal
axs[0].plot(time, f, label='Original Signal')
axs[0].set_title('Original Signal')
axs[0].set_xlabel('Time (microseconds)')
axs[0].set_ylabel('Amplitude')
axs[0].legend()

# FFT - Power Spectral Density
freq = (1 / (dt * len(f))) * np.arange(len(f))
axs[1].plot(freq[:len(f) // 2], PSD[:len(f) // 2], label='PSD of Original Signal')
axs[1].set_title('FFT - Power Spectral Density')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Power')
axs[1].legend()

# Filtered Signal
axs[2].plot(time, ffilt.real, label='Filtered Signal')
axs[2].set_title('Filtered Signal')
axs[2].set_xlabel('Time (microseconds)')
axs[2].set_ylabel('Amplitude')
axs[2].legend()

# Envelope with Highest Peak
axs[3].plot(time, env, label='Envelope')
axs[3].plot(time[highest_peak_index], env[highest_peak_index], "x", label='Highest Peak')
axs[3].set_title('Envelope with Highest Peak')
axs[3].set_xlabel('Time (microseconds)')
axs[3].set_ylabel('Amplitude')
axs[3].legend()

plt.tight_layout()
plt.show()

# Print the calculated distance for the first row
print("Distance to the highest peak for the first row:", distances_highest_peak[0], "meters")
