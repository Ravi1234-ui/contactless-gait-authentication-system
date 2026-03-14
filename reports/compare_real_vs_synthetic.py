# reports/compare_real_vs_synthetic.py

import numpy as np
import matplotlib.pyplot as plt
import os

from preprocessing.uci_loader import UCILoader

UCI_PATH = "data/raw/uci/UCI HAR Dataset"
SYN_PATH = "data/raw/synthetic/generated_windows"


def load_real_sample():

    loader = UCILoader(UCI_PATH)
    windows, labels = loader.load_all()

    # pick first sample
    return windows[0]


def load_synthetic_sample():

    files = os.listdir(SYN_PATH)

    for f in files:
        if f.endswith(".npy"):
            data = np.load(os.path.join(SYN_PATH, f))
            return data[0]


def compare_signals(real, synthetic):

    time = np.arange(128)

    fig, axs = plt.subplots(3, 2, figsize=(12,8))

    titles = [
        "Acc X", "Acc Y", "Acc Z",
        "Gyro X", "Gyro Y", "Gyro Z"
    ]

    for i in range(6):

        r = real[:, i]
        s = synthetic[:, i]

        row = i // 2
        col = i % 2

        axs[row, col].plot(time, r, label="Real", linewidth=2)
        axs[row, col].plot(time, s, label="Synthetic", linestyle="--")

        axs[row, col].set_title(titles[i])
        axs[row, col].legend()

    plt.suptitle("Real vs Synthetic Gait Signals")
    plt.tight_layout()
    plt.show()


def print_statistics(real, synthetic):

    print("\nSignal Statistics Comparison\n")

    channels = [
        "Acc X", "Acc Y", "Acc Z",
        "Gyro X", "Gyro Y", "Gyro Z"
    ]

    for i in range(6):

        r = real[:, i]
        s = synthetic[:, i]

        print(channels[i])

        print("Real Mean:", round(np.mean(r),4),
              "Std:", round(np.std(r),4))

        print("Synthetic Mean:", round(np.mean(s),4),
              "Std:", round(np.std(s),4))

        print("-"*40)


if __name__ == "__main__":

    real = load_real_sample()
    synthetic = load_synthetic_sample()

    compare_signals(real, synthetic)

    print_statistics(real, synthetic)