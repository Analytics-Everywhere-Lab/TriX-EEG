import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.signal import butter, lfilter, iirnotch, welch, resample_poly
import warnings
warnings.filterwarnings('ignore')
from pyedflib import highlevel, FILETYPE_EDFPLUS

SAMPLING_RATE = 2048
TARGET_SAMPLING_RATE = 256


def resample_signal(data, original_fs=SAMPLING_RATE, target_fs=TARGET_SAMPLING_RATE):
    """Resample a signal while preserving its duration."""
    divisor = np.gcd(original_fs, target_fs)
    return resample_poly(data, target_fs // divisor, original_fs // divisor)


def get_physical_range(signal):
    """Return an EDF-compatible physical range, including for constant signals."""
    physical_min = float(np.min(signal))
    physical_max = float(np.max(signal))
    if physical_min == physical_max:
        padding = max(abs(physical_min) * 0.01, 1.0)
        physical_min -= padding
        physical_max += padding
    return physical_min, physical_max

def dc_blocking_filter(data, alpha=0.99):
    y = np.zeros_like(data, dtype=np.float32)
    y[0] = data[0]
    for i in range(1, len(data)):
        y[i] = data[i] - data[i-1] + alpha * y[i-1]
    return y

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)


def notch_filter(data, notch_freq, fs, Q=30.0):
    w0 = notch_freq / (0.5 * fs)
    b, a = iirnotch(w0, Q)
    return lfilter(b, a, data)

# EEG filters
def preprocess_eeg(data, fs=SAMPLING_RATE):

    # Step 1: DC Blocking
    data = dc_blocking_filter(data)

    # Step 2: Notch Filtering
    notch_60 = notch_filter(data, 60, fs, Q=12)
    notch_50 = notch_filter(notch_60, 50, fs, Q=5)

    # Step 3: Bandpass Filtering (1–35 Hz)
    filtered = bandpass_filter(notch_50, 1, 100, fs)
    

    return filtered

motion_data_dir = "data_motion_artifact/EEG-csv"
motion_data_edf_dir = "data_motion_artifact/EEG-edf"
os.makedirs(motion_data_edf_dir, exist_ok=True)

for file in os.listdir(motion_data_dir)[:5]:
    if file.endswith(".csv"):
        file_path = os.path.join(motion_data_dir, file)
        edf_path = os.path.join(motion_data_edf_dir, file.replace('.csv', '.edf'))
        data = pd.read_csv(file_path)
        print(f"Processing file: {file}")
        
        # Assuming the EEG data is in the first column
        eeg_data_channel_1 = data.iloc[:, 1].values # Clean EEG data
        eeg_data_channel_2 = data.iloc[:, 2].values # Motion artifact EEG data
        # Plotting the EEG data
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(eeg_data_channel_1, color='blue')
        plt.title(f'EEG Data Channel 1 - {file}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(eeg_data_channel_2, color='orange')
        plt.title(f'EEG Data Channel 2 - {file}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()


        # Plot filtered EEG data
        filtered_eeg_channel_1 = preprocess_eeg(eeg_data_channel_1)
        filtered_eeg_channel_2 = preprocess_eeg(eeg_data_channel_2)
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(filtered_eeg_channel_1, color='blue')
        plt.title(f'Filtered EEG Data Channel 1 - {file}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude') 
        plt.subplot(2, 1, 2)
        plt.plot(filtered_eeg_channel_2, color='orange')
        plt.title(f'Filtered EEG Data Channel 2 - {file}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.tight_layout()  
        plt.show()    

        signals = np.vstack([
            resample_signal(eeg_data_channel_1),
            resample_signal(eeg_data_channel_2),
            resample_signal(filtered_eeg_channel_1),
            resample_signal(filtered_eeg_channel_2),
        ])

        # Create channel metadata
        signal_headers = []
        for label, signal in zip(['clean_raw', 'motion_artifact_raw', 'clean_filtered', 'motion_artifact_filtered'], signals):
            physical_min, physical_max = get_physical_range(signal)
            signal_headers.append({
                'label': label,
                'dimension': 'uV',
                'sample_frequency': TARGET_SAMPLING_RATE,
                'physical_min': physical_min,
                'physical_max': physical_max,
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': '',
                'prefilter': ''
            })

        # Write EDF file
        highlevel.write_edf(
            edf_path,
            signals,
            signal_headers,
            file_type=FILETYPE_EDFPLUS
        )
