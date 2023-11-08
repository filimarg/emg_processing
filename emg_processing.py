import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from datetime import datetime

# load the raw EMG data
def load_data(file_name):
    time = []
    emg = []
    with open(file_name, 'r') as f:
        for line in f:
            t, e = line.strip().split(']')
            t = t[1:]
            # convert the time string to a datetime object
            t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
            e = e.strip()
            time.append(t)
            emg.append(float(e))
    return np.array(time), np.array(emg)

# extract the Maximum Voluntary Contraction (MVC)
# window_size is the number of consecutive data points used to calculate the rolling average
def get_mvc(file_name, window_size=10):
    _, emg = load_data(file_name)
    emg = emg + 300  # TO BE REMOVED if real mvc data
    emg_series = pd.Series(emg)
    rolling_avg = emg_series.rolling(window=window_size).mean()
    return rolling_avg.max()

# normalize the signal with respect to MVC and rectify
def normalize_rectify(emg, mvc):
    normalized_emg = emg / mvc
    normalized_emg[normalized_emg > 1] = 1  # cut emg values bigger than mvc
    rectified_emg = np.abs(normalized_emg)
    return rectified_emg


# plot the processed EMG data in the time domain
def plot_time_domain(time, emg, title):
    plt.figure(figsize=(10, 6))
    plt.plot(time, emg)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# analyze the processed EMG data in the frequency domain
def plot_frequency_domain(time, emg):
    n = len(time)
    # convert the time difference to seconds
    T = (time[1] - time[0]).total_seconds()
    yf = fft(emg)
    xf = fftfreq(n, T)[:n//2]
    plt.figure(figsize=(10, 6))
    plt.plot(xf[1:], 2.0/n * np.abs(yf[1:n//2]))
    plt.title('Processed EMG Signal in Frequency Domain - Without DC')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()

# load logged emg data
time, raw_emg = load_data('emg_data.log')
# plot raw emg data
plot_time_domain(time, raw_emg, 'Raw EMG Signal in Time Domain')
# load logged emg data for mvc extraction
MVC = get_mvc('mvc_data.log')
# normalize and rectify data
rectified_emg = normalize_rectify(raw_emg, MVC)
# plot processed data in time and frequency domain
plot_time_domain(time, rectified_emg, 'Processed EMG Signal in Time Domain')
plot_frequency_domain(time, rectified_emg)