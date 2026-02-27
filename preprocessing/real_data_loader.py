# preprocessing/real_data_loader.py

import pandas as pd
import numpy as np
import os
from scipy.signal import resample
from preprocessing.windowing import SensorWindowing


class RealWorldDataLoader:

    def __init__(self,
                 base_path="data/raw/real_world",
                 target_sampling_rate=50):

        self.base_path = base_path
        self.target_sampling_rate = target_sampling_rate
        self.windowing = SensorWindowing(window_size=128, overlap=0.5)

    # ----------------------------
    # CLEAN CSV FILE
    # ----------------------------
    def clean_file(self, file_path, is_gyro=False):

        # Try auto delimiter detection
        df = pd.read_csv(
            file_path,
            comment="#",
            sep=None,
            engine="python"
        )

        df = df.dropna()

        # Print detected columns for debugging
        print("Detected columns:", df.columns)

        # Keep only first 4 numeric columns after time
        if is_gyro:
            df = df.iloc[:, 0:4]
            df.columns = ["time", "gyro_x", "gyro_y", "gyro_z"]
        else:
            df = df.iloc[:, 0:4]
            df.columns = ["time", "acc_x", "acc_y", "acc_z"]

        return df

    # ----------------------------
    # RESAMPLE TO 50 Hz
    # ----------------------------
    def resample_signal(self, df):

        duration = df["time"].iloc[-1] - df["time"].iloc[0]

        new_length = int(duration * self.target_sampling_rate)

        resampled_values = resample(
            df.drop(columns=["time"]).values,
            new_length
        )

        resampled_time = np.linspace(
            df["time"].iloc[0],
            df["time"].iloc[-1],
            new_length
        )

        new_df = pd.DataFrame(
            resampled_values,
            columns=df.columns[1:]
        )

        new_df.insert(0, "time", resampled_time)

        return new_df

    # ----------------------------
    # LOAD ONE PERSON
    # ----------------------------
    def load_person(self, person_folder):

        person_path = os.path.join(self.base_path, person_folder)

        gyro_path = os.path.join(person_path, "gyroscope.csv")
        acc_path = os.path.join(person_path, "accelerometer.csv")

        if not os.path.exists(gyro_path) or not os.path.exists(acc_path):
            print(f"Skipping {person_folder} (missing files)")
            return None

        print(f"Processing {person_folder}...")

        gyro_df = self.clean_file(gyro_path, is_gyro=True)
        acc_df = self.clean_file(acc_path, is_gyro=False)

        # Resample both to 50 Hz
        gyro_df = self.resample_signal(gyro_df)
        acc_df = self.resample_signal(acc_df)

        # Merge based on nearest time
        merged = pd.merge_asof(
            acc_df.sort_values("time"),
            gyro_df.sort_values("time"),
            on="time"
        )

        merged = merged.dropna()

        # Windowing
        windows = self.windowing.create_windows(merged)
        windows = self.windowing.normalize_windows(windows)

        print(f"{person_folder} → {windows.shape[0]} windows created")

        return windows

    # ----------------------------
    # LOAD ALL PERSONS
    # ----------------------------
    def load_all(self):

        all_windows = []
        all_labels = []

        persons = os.listdir(self.base_path)

        for person in persons:
            person_path = os.path.join(self.base_path, person)

            if os.path.isdir(person_path):

                windows = self.load_person(person)

                if windows is not None:
                    for window in windows:
                        all_windows.append(window)
                        all_labels.append(person)

        return np.array(all_windows), np.array(all_labels)


# ----------------------------
# TEST MODULE
# ----------------------------
if __name__ == "__main__":

    loader = RealWorldDataLoader()
    windows, labels = loader.load_all()

    print("\nFinal Output:")
    print("Windows shape:", windows.shape)
    print("Unique persons:", set(labels))