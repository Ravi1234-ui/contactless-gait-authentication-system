# preprocessing/windowing.py

import numpy as np


class SensorWindowing:
    """
    Creates sliding windows for accelerometer + gyroscope data.
    Expected columns:
    acc_x, acc_y, acc_z,
    gyro_x, gyro_y, gyro_z
    """

    def __init__(self, window_size=128, overlap=0.5):
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))

    def create_windows(self, df):

        required_cols = [
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        data = df[required_cols].values

        windows = []

        for start in range(0, len(data) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window = data[start:end]
            windows.append(window)

        return np.array(windows)

    def normalize_windows(self, windows):

        mean = np.mean(windows, axis=(1, 2), keepdims=True)
        std = np.std(windows, axis=(1, 2), keepdims=True) + 1e-8

        return (windows - mean) / std