# llm_engine/synthetic_signal_simulator.py

import numpy as np
import json
import os


class SyntheticSignalSimulator:
    """
    Physics-informed synthetic gait signal generator.

    Converts biomechanical profiles into realistic
    6-axis accelerometer + gyroscope signals.
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

        t = np.linspace(0, timesteps / sampling_rate, timesteps)
        cadence = profile["cadence_hz"]

        acc_v = profile["acc_vertical_g"]
        acc_h = profile["acc_horizontal_g"]
        gyro_s = profile["gyro_sagittal_rad_s"]
        gyro_f = profile["gyro_frontal_rad_s"]
        asymmetry = profile["step_asymmetry"]
        heel_sharp = profile["heel_strike_sharpness"]

        # --- Random phase for intra-class diversity ---
        phase_shift = np.random.uniform(0, 2*np.pi)

        # --- Fundamental + Harmonics ---
        acc_z = (
            acc_v * np.sin(2*np.pi*cadence*t + phase_shift) +
            0.3 * acc_v * np.sin(4*np.pi*cadence*t + phase_shift) +
            0.1 * acc_v * np.sin(6*np.pi*cadence*t + phase_shift)
        )

        acc_x = acc_h * np.sin(2*np.pi*cadence*t + phase_shift/2)
        acc_y = 0.6 * acc_h * np.cos(2*np.pi*cadence*t + phase_shift)

        # --- Heel Strike Transient ---
        step_times = np.arange(0, timesteps/sampling_rate, 1/cadence)

        for st in step_times:
            idx = int(st * sampling_rate)
            if idx < timesteps:
                duration = int(0.1 * sampling_rate)
                for d in range(min(duration, timesteps-idx)):
                    acc_z[idx+d] += heel_sharp * 0.5 * np.exp(-d * 0.35)

        # --- Step Asymmetry ---
        for i, st in enumerate(step_times):
            if i % 2 == 1:
                idx = int(st * sampling_rate)
                if idx < timesteps:
                    acc_z[idx:idx+10] *= (1 + asymmetry * 0.25)

        # --- Controlled noise ---
        acc_x += np.random.normal(0, 0.03, timesteps)
        acc_y += np.random.normal(0, 0.03, timesteps)
        acc_z += np.random.normal(0, 0.04, timesteps)

        # --- Gyroscope Signals ---
        gyro_x = gyro_s * np.sin(2*np.pi*cadence*t + phase_shift)
        gyro_y = gyro_f * np.cos(2*np.pi*cadence*t + phase_shift)
        gyro_z = 0.5 * gyro_s * np.sin(4*np.pi*cadence*t)

        gyro_x += np.random.normal(0, 0.02, timesteps)
        gyro_y += np.random.normal(0, 0.02, timesteps)
        gyro_z += np.random.normal(0, 0.02, timesteps)

        window = np.stack(
            [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
            axis=1
        )

        return window

    def generate_identity_windows(self, profile, windows_per_identity=60):

        identity_id = profile["identity_id"]
        identity_windows = []

        for _ in range(windows_per_identity):
            window = self.simulate_window(profile)
            identity_windows.append(window)

        identity_windows = np.array(identity_windows)

        output_file = os.path.join(self.output_path, f"{identity_id}.npy")
        np.save(output_file, identity_windows)

        return identity_windows

    def generate_all(self, windows_per_identity=60):

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
    simulator.generate_all(windows_per_identity=60)