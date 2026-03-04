# llm_engine/biomechanical_profile_generator.py

import json
import os
import random
import numpy as np


class BiomechanicalProfileGenerator:
    """
    Physics-informed biomechanical walking profile generator.

    This module generates correlated human walking profiles grounded
    in UCI HAR population statistics (50Hz waist-mounted smartphone).

    These profiles are later converted into accelerometer + gyroscope
    signals by the synthetic signal simulator.
    """

    def __init__(self, output_path="data/raw/synthetic"):

        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Population statistics derived from UCI HAR dataset
        self.population_stats = {
            "cadence_mean": 1.87,
            "cadence_std": 0.15,
            "cadence_min": 1.4,
            "cadence_max": 2.2,
            "acc_mean_g": 0.98,
            "acc_std_g": 0.18,
            "gyro_mean": 0.42,
            "gyro_std": 0.11
        }

    # -------------------------------------------------------
    # Utility function
    # -------------------------------------------------------

    def clamp(self, value, min_val, max_val):
        return max(min(value, max_val), min_val)

    # -------------------------------------------------------
    # Generate correlated biomechanical profile
    # -------------------------------------------------------

    def generate_profile(self, identity_id):

        # ----------------------------
        # Demographic parameters
        # ----------------------------

        age = random.randint(19, 48)
        height_m = round(random.uniform(1.55, 1.90), 2)
        weight_kg = round(random.uniform(50, 95), 1)

        # ----------------------------
        # Cadence generation
        # ----------------------------

        cadence = np.random.normal(
            self.population_stats["cadence_mean"],
            self.population_stats["cadence_std"]
        )

        # Age correlation (older → slower cadence)
        cadence -= (age - 25) * 0.005

        # Height correlation (taller → longer stride → lower cadence)
        cadence -= (height_m - 1.70) * 0.25

        cadence = self.clamp(
            cadence,
            self.population_stats["cadence_min"],
            self.population_stats["cadence_max"]
        )

        # ----------------------------
        # Stride length (height based)
        # ----------------------------

        stride_length = 0.43 * height_m + np.random.normal(0, 0.03)
        stride_length = self.clamp(stride_length, 0.5, 0.9)

        # ----------------------------
        # Vertical acceleration
        # ----------------------------

        acc_vertical = np.random.normal(
            self.population_stats["acc_mean_g"],
            self.population_stats["acc_std_g"]
        )

        # heavier people produce stronger impacts
        acc_vertical += (weight_kg - 70) * 0.003

        acc_vertical = self.clamp(acc_vertical, 0.6, 1.5)

        # Horizontal acceleration (forward motion)
        acc_horizontal = acc_vertical * random.uniform(0.6, 0.85)

        # ----------------------------
        # Gyroscope signals
        # ----------------------------

        gyro_sagittal = np.random.normal(
            self.population_stats["gyro_mean"],
            self.population_stats["gyro_std"]
        )

        gyro_sagittal += (stride_length - 0.7) * 0.4
        gyro_sagittal = self.clamp(gyro_sagittal, 0.2, 1.0)

        gyro_frontal = gyro_sagittal * random.uniform(0.6, 1.2)

        # ----------------------------
        # Step asymmetry
        # ----------------------------

        step_asymmetry = random.uniform(0.05, 0.25)

        # older people slightly more asymmetric
        step_asymmetry += (age - 30) * 0.002

        step_asymmetry = self.clamp(step_asymmetry, 0.0, 0.4)

        # ----------------------------
        # Heel strike sharpness
        # ----------------------------

        heel_strike_sharpness = random.uniform(0.3, 0.9)

        # heavier individuals produce stronger impacts
        heel_strike_sharpness += (weight_kg - 70) * 0.004

        heel_strike_sharpness = self.clamp(heel_strike_sharpness, 0.2, 1.0)

        # ----------------------------
        # Final profile
        # ----------------------------

        profile = {
            "identity_id": identity_id,
            "age": age,
            "height_m": height_m,
            "weight_kg": weight_kg,
            "cadence_hz": round(cadence, 3),
            "stride_length_m": round(stride_length, 3),
            "acc_vertical_g": round(acc_vertical, 3),
            "acc_horizontal_g": round(acc_horizontal, 3),
            "gyro_sagittal_rad_s": round(gyro_sagittal, 3),
            "gyro_frontal_rad_s": round(gyro_frontal, 3),
            "step_asymmetry": round(step_asymmetry, 3),
            "heel_strike_sharpness": round(heel_strike_sharpness, 3)
        }

        return profile

    # -------------------------------------------------------
    # Generate multiple identities
    # -------------------------------------------------------

    def generate_multiple_profiles(self, num_profiles=1000):

        profiles = []

        for i in range(num_profiles):
            identity_id = f"SYNTH_{i+1:04d}"
            profile = self.generate_profile(identity_id)
            profiles.append(profile)

        output_file = os.path.join(
            self.output_path,
            "synthetic_profiles.json"
        )

        with open(output_file, "w") as f:
            json.dump(profiles, f, indent=4)

        print(f"{num_profiles} synthetic profiles saved to {output_file}")

        return profiles


# -------------------------------------------------------
# Script execution
# -------------------------------------------------------

if __name__ == "__main__":

    generator = BiomechanicalProfileGenerator()
    generator.generate_multiple_profiles(num_profiles=1000)