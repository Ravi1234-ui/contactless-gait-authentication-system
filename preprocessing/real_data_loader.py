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

        df = pd.read_csv(
            file_path,
            comment="#",
            sep=None,
            engine="python"
        )

        df = df.dropna()

        print("Detected columns:", df.columns)

        # Keep first 4 columns (time + x,y,z)
        df = df.iloc[:, 0:4]

        if is_gyro:
            df.columns = ["time", "gyro_x", "gyro_y", "gyro_z"]
        else:
            df.columns = ["time", "acc_x", "acc_y", "acc_z"]

        # Ensure numeric type
        df = df.astype(np.float32)

        return df

    # ----------------------------
    # REMOVE GRAVITY COMPONENT
    # ----------------------------
    def remove_gravity(self, acc_df):
        """
        Physics Toolbox accelerometer includes gravity.
        Remove DC component (mean subtraction).
        """

        acc_df["acc_x"] -= acc_df["acc_x"].mean()
        acc_df["acc_y"] -= acc_df["acc_y"].mean()
        acc_df["acc_z"] -= acc_df["acc_z"].mean()

        return acc_df

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

        return new_df.astype(np.float32)

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

        # Remove gravity from accelerometer
        acc_df = self.remove_gravity(acc_df)

        # Resample to 50Hz
        gyro_df = self.resample_signal(gyro_df)
        acc_df = self.resample_signal(acc_df)

        # Merge based on nearest time
        merged = pd.merge_asof(
            acc_df.sort_values("time"),
            gyro_df.sort_values("time"),
            on="time"
        )

        merged = merged.dropna()

        # Create windows (NO normalization here)
        windows = self.windowing.create_windows(merged)

        print(f"{person_folder} → {windows.shape[0]} windows created")

        return windows.astype(np.float32)

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

        return np.array(all_windows, dtype=np.float32), np.array(all_labels)


# ----------------------------
# TEST MODULE
# ----------------------------
if __name__ == "__main__":

    loader = RealWorldDataLoader()
    windows, labels = loader.load_all()

    print("\nFinal Output:")
    print("Windows shape:", windows.shape)
    print("Unique persons:", set(labels))