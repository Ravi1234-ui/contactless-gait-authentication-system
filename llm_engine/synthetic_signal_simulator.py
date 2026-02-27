# llm_engine/synthetic_signal_simulator.py

import numpy as np
import json
import os


class SyntheticSignalSimulator:
    """
    Converts biomechanical profiles into synthetic
    accelerometer + gyroscope time-series signals.
    """

    def __init__(self,
                 profile_path="data/raw/synthetic/synthetic_profiles.json",
                 output_path="data/raw/synthetic/generated_windows"):

        self.profile_path = profile_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def load_profiles(self):
        with open(self.profile_path, "r") as f:
            profiles = json.load(f)
        return profiles

    def simulate_window(self, profile, timesteps=128, sampling_rate=50):
        """
        Generate one synthetic window (128, 6)
        """

        t = np.linspace(0, timesteps / sampling_rate, timesteps)

        cadence = profile["cadence_hz"]
        acc_amp = profile["acc_amplitude_g"]
        gyro_peak = profile["gyro_peak_rad_s"]
        vertical_var = profile["vertical_variation"]
        arm_swing = profile["arm_swing_factor"]
        hip_rot = profile["hip_rotation_factor"]

        # Accelerometer signals
        acc_x = acc_amp * np.sin(2 * np.pi * cadence * t)
        acc_y = vertical_var * np.cos(2 * np.pi * cadence * t)
        acc_z = 0.5 * acc_amp * np.sin(4 * np.pi * cadence * t)

        # Add noise
        acc_x += np.random.normal(0, 0.05, timesteps)
        acc_y += np.random.normal(0, 0.05, timesteps)
        acc_z += np.random.normal(0, 0.05, timesteps)

        # Gyroscope signals
        gyro_x = gyro_peak * np.sin(2 * np.pi * cadence * t) * arm_swing
        gyro_y = gyro_peak * np.cos(2 * np.pi * cadence * t) * hip_rot
        gyro_z = 0.5 * gyro_peak * np.sin(4 * np.pi * cadence * t)

        gyro_x += np.random.normal(0, 0.02, timesteps)
        gyro_y += np.random.normal(0, 0.02, timesteps)
        gyro_z += np.random.normal(0, 0.02, timesteps)

        window = np.stack(
            [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
            axis=1
        )

        return window

    def generate_identity_windows(self, profile, windows_per_identity=20):
        """
        Generate multiple windows for one synthetic identity
        """
        identity_id = profile["identity_id"]

        identity_windows = []

        for _ in range(windows_per_identity):
            window = self.simulate_window(profile)
            identity_windows.append(window)

        identity_windows = np.array(identity_windows)

        output_file = os.path.join(self.output_path, f"{identity_id}.npy")
        np.save(output_file, identity_windows)

        return identity_windows

    def generate_all(self, windows_per_identity=20):
        profiles = self.load_profiles()

        total_windows = 0

        for profile in profiles:
            windows = self.generate_identity_windows(
                profile,
                windows_per_identity=windows_per_identity
            )
            total_windows += len(windows)

        print(f"Generated {total_windows} synthetic windows.")


if __name__ == "__main__":
    simulator = SyntheticSignalSimulator()
    simulator.generate_all(windows_per_identity=30)