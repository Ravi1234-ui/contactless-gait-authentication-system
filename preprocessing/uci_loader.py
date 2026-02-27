# preprocessing/uci_loader.py

import os
import numpy as np


class UCILoader:
    """
    Loads raw accelerometer + gyroscope inertial signals
    from UCI HAR Dataset.

    Uses:
        body_acc_x/y/z
        body_gyro_x/y/z

    Filters only WALKING activity (label = 1).
    """

    def __init__(self, base_path):
        """
        base_path example:
        data/raw/uci/UCI HAR Dataset
        """
        self.base_path = base_path

    def load_split(self, split="train"):
        """
        Loads either train or test split.
        Returns:
            windows: shape (N, 128, 6)
            subject_ids: shape (N,)
        """

        inertial_path = os.path.join(
            self.base_path, split, "Inertial Signals"
        )

        # Accelerometer
        acc_x = np.loadtxt(os.path.join(inertial_path, f"body_acc_x_{split}.txt"))
        acc_y = np.loadtxt(os.path.join(inertial_path, f"body_acc_y_{split}.txt"))
        acc_z = np.loadtxt(os.path.join(inertial_path, f"body_acc_z_{split}.txt"))

        # Gyroscope
        gyro_x = np.loadtxt(os.path.join(inertial_path, f"body_gyro_x_{split}.txt"))
        gyro_y = np.loadtxt(os.path.join(inertial_path, f"body_gyro_y_{split}.txt"))
        gyro_z = np.loadtxt(os.path.join(inertial_path, f"body_gyro_z_{split}.txt"))

        # Stack into (N, 128, 6)
        windows = np.stack(
            [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
            axis=2
        )

        # Load activity labels
        y = np.loadtxt(
            os.path.join(self.base_path, split, f"y_{split}.txt")
        )

        # Load subject IDs
        subjects = np.loadtxt(
            os.path.join(self.base_path, split, f"subject_{split}.txt")
        )

        # Keep only WALKING (label == 1)
        walking_mask = (y == 1)

        windows = windows[walking_mask]
        subjects = subjects[walking_mask]

        return windows, subjects

    def load_all(self):
        """
        Loads train + test together.
        """

        train_windows, train_subjects = self.load_split("train")
        test_windows, test_subjects = self.load_split("test")

        windows = np.concatenate([train_windows, test_windows], axis=0)
        subjects = np.concatenate([train_subjects, test_subjects], axis=0)

        return windows, subjects


if __name__ == "__main__":
    # Example test

    base_path = "data/raw/uci/UCI HAR Dataset"

    loader = UCILoader(base_path)

    windows, subjects = loader.load_all()

    print("Windows shape:", windows.shape)
    print("Subjects shape:", subjects.shape)
    print("Unique subjects:", np.unique(subjects))