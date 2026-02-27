# llm_engine/biomechanical_profile_generator.py

import json
import os
import random


class BiomechanicalProfileGenerator:
    """
    Generates biomechanical walking profiles.
    This simulates LLM-driven generation of realistic
    gait parameter sets.
    """

    def __init__(self, output_path="data/raw/synthetic"):
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def generate_profile(self, identity_id):
        """
        Generate one synthetic biomechanical profile.
        """

        profile = {
            "identity_id": identity_id,
            "cadence_hz": round(random.uniform(1.4, 2.2), 2),
            "stride_length_m": round(random.uniform(0.5, 0.9), 2),
            "acc_amplitude_g": round(random.uniform(0.6, 1.4), 2),
            "gyro_peak_rad_s": round(random.uniform(0.2, 1.0), 2),
            "vertical_variation": round(random.uniform(0.05, 0.25), 2),
            "arm_swing_factor": round(random.uniform(0.3, 1.2), 2),
            "hip_rotation_factor": round(random.uniform(0.3, 1.5), 2),
        }

        return profile

    def generate_multiple_profiles(self, num_profiles=100):
        """
        Generate multiple synthetic identities.
        """

        profiles = []

        for i in range(num_profiles):
            identity_id = f"SYNTH_{i+1:04d}"
            profile = self.generate_profile(identity_id)
            profiles.append(profile)

        # Save to JSON
        output_file = os.path.join(self.output_path, "synthetic_profiles.json")

        with open(output_file, "w") as f:
            json.dump(profiles, f, indent=4)

        print(f"{num_profiles} synthetic profiles saved to {output_file}")

        return profiles


if __name__ == "__main__":
    generator = BiomechanicalProfileGenerator()
    generator.generate_multiple_profiles(num_profiles=1000)