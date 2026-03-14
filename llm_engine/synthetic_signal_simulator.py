# llm_engine/synthetic_signal_simulator.py

import numpy as np
import json
import os


class SyntheticSignalSimulator:
    """
    Physics-informed synthetic gait signal generator.

    Converts biomechanical walking profiles (from api_profile_generator)
    into realistic accelerometer + gyroscope time-series signals.

    Reads the fixed 12-field profile schema:
        identity_id, age, height_m, weight_kg,
        cadence_hz, stride_length_m,
        acc_vertical_g, acc_horizontal_g,
        gyro_sagittal_rad_s, gyro_frontal_rad_s,
        step_asymmetry, heel_strike_sharpness

    Signal model (no raw sinusoids):
        - Heel-strike loading transient  → Ricker (Mexican-hat) wavelet
        - Mid-stance body-weight acceptance → Gaussian bell
        - Toe-off push-off impulse        → Gaussian bell
        - Mediolateral sway               → alternating Gaussian per step
        - Intra-window cadence variability → Ornstein–Uhlenbeck random walk
        - Gravity DC offset               → trunk lean derived from stride length
        - Postural sway                   → low-frequency (0.1–0.5 Hz) sinusoid
        - Sensor noise                    → pink (1/f) + white Gaussian

    Output per identity: numpy array of shape (W, 128, 6)
        columns → [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        frame   → waist-mounted smartphone, UCI HAR convention
                  acc_x = anteroposterior (forward)
                  acc_y = mediolateral
                  acc_z = vertical (includes ~1 g gravity DC)
    """

    SAMPLING_RATE       = 50    # Hz  — UCI HAR convention
    TIMESTEPS           = 128   # samples per window
    WINDOWS_PER_IDENTITY = 60   # default windows generated per person

    def __init__(
        self,
        profile_path: str = "data/raw/synthetic/api_synthetic_profiles.json",
        output_path:  str = "data/raw/synthetic/generated_windows",
    ):
        self.profile_path = profile_path
        self.output_path  = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────

    def load_profiles(self):
        with open(self.profile_path, "r") as f:
            return json.load(f)

    # ─────────────────────────────────────────────────────────────────────────
    # Waveform primitives
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ricker(n: int, center: float, width: float, amplitude: float) -> np.ndarray:
        """
        Ricker (Mexican-hat) wavelet.
        Models the double-peak heel-strike loading transient seen in
        vertical GRF and waist-mounted accelerometry.
        """
        t     = np.arange(n, dtype=float) - center
        sigma = max(width / 2.5, 0.5)
        u     = (t / sigma) ** 2
        return amplitude * (1.0 - u) * np.exp(-0.5 * u)

    @staticmethod
    def _gauss(n: int, center: float, width: float, amplitude: float) -> np.ndarray:
        """
        Gaussian bell.
        Models smooth muscle-force events: mid-stance load, toe-off push, ML sway.
        """
        t     = np.arange(n, dtype=float) - center
        sigma = max(width / 2.0, 0.5)
        return amplitude * np.exp(-0.5 * (t / sigma) ** 2)

    @staticmethod
    def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Pink (1/f) noise via spectral shaping.
        Better matches real MEMS IMU noise floor than white Gaussian noise.
        """
        n = int(n)
        f        = np.fft.rfftfreq(n)
        f[0]     = 1e-10
        power    = 1.0 / np.sqrt(f)
        power[0] = 0.0
        phases   = rng.uniform(0, 2 * np.pi, len(power))
        sig      = np.fft.irfft(power * np.exp(1j * phases), n=n)
        return sig / (sig.std() + 1e-12)

    # ─────────────────────────────────────────────────────────────────────────
    # Intra-window cadence trajectory
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cadence_trajectory(
        cadence_mean: float, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Ornstein–Uhlenbeck random walk on cadence across steps.
        CV ≈ 2–4 % from healthy adults (Menz et al. 2003).
        """
        cv      = rng.uniform(0.015, 0.040)
        sigma   = cadence_mean * cv
        theta   = 0.35
        traj    = np.empty(n_steps)
        traj[0] = cadence_mean + rng.normal(0, sigma)
        for i in range(1, n_steps):
            traj[i] = (traj[i - 1]
                       + theta * (cadence_mean - traj[i - 1])
                       + rng.normal(0, sigma))
        return np.clip(traj, cadence_mean * 0.80, cadence_mean * 1.20)

    # ─────────────────────────────────────────────────────────────────────────
    # Core simulation — one window
    # ─────────────────────────────────────────────────────────────────────────

    def simulate_window(
        self, profile: dict, rng: np.random.Generator = None
    ) -> np.ndarray:
        """
        Generate one (128, 6) gait window from a profile dict.

        Returns
        -------
        np.ndarray, shape (128, 6), dtype float32
        """
        if rng is None:
            rng = np.random.default_rng()

        N  = self.TIMESTEPS
        fs = self.SAMPLING_RATE
        dt = 1.0 / fs

        # ── Unpack profile (fixed schema only) ────────────────────────────
        cadence    = float(profile["cadence_hz"])
        stride_len = float(profile["stride_length_m"])
        acc_v      = float(profile["acc_vertical_g"])
        acc_h      = float(profile["acc_horizontal_g"])
        gyro_s     = float(profile["gyro_sagittal_rad_s"])
        gyro_f     = float(profile["gyro_frontal_rad_s"])
        asymmetry  = float(profile["step_asymmetry"])
        heel_sharp = float(profile["heel_strike_sharpness"])

        # ── Secondary timing (derived, no extra profile fields) ───────────
        duration    = N * dt                                   # 2.56 s
        stance_frac = float(np.clip(
            0.60 - (cadence - 1.87) * 0.04, 0.52, 0.68
        ))

        # Trunk lean estimated from stride length
        trunk_lean_rad = np.deg2rad(
            float(np.clip(3.0 + (stride_len - 0.70) * 8.0, 1.0, 12.0))
        )

        # Noise floor typical of waist-mounted MEMS IMU
        noise_acc  = 0.030
        noise_gyro = 0.018

        # ── Step event times ──────────────────────────────────────────────
        n_steps_est = int(np.ceil(cadence * duration)) + 3
        cad_traj    = self._cadence_trajectory(cadence, n_steps_est, rng)

        step_times = []
        t_cur      = rng.uniform(0.0, 0.5 / cadence)   # random phase
        for i in range(n_steps_est):
            step_times.append(t_cur)
            period = 1.0 / cad_traj[i]
            # Asymmetric step durations (left vs right)
            period *= (1.0 - asymmetry * 0.5) if i % 2 == 1 else (1.0 + asymmetry * 0.5)
            t_cur += period

        step_times = [st for st in step_times if st < duration + 0.25]

        # ── Signal arrays ─────────────────────────────────────────────────
        acc_x  = np.zeros(N)
        acc_y  = np.zeros(N)
        acc_z  = np.zeros(N)
        gyro_x = np.zeros(N)
        gyro_y = np.zeros(N)
        gyro_z = np.zeros(N)

        stance_samples = max(4, int(stance_frac / cadence * fs))

        # ── Per-step event injection ──────────────────────────────────────
        for step_i, st in enumerate(step_times):

            idx     = int(st * fs)
            ml_sign = 1.0 if step_i % 2 == 0 else -1.0

            # ── Heel-strike loading transient ─────────────────────────────
            # Width narrows as heel_strike_sharpness increases
            hs_width = max(2, int(fs * 0.10 * (1.0 - heel_sharp * 0.45)))
            hs_amp_z = acc_v * heel_sharp * rng.uniform(0.90, 1.10)
            hs_amp_x = acc_h * 0.40       * rng.uniform(0.85, 1.15)   # braking

            acc_z += self._ricker(N, idx, hs_width,  hs_amp_z)
            acc_x += self._gauss( N, idx, hs_width, -hs_amp_x)

            # ── Mid-stance body-weight acceptance ─────────────────────────
            ms_idx   = idx + int(0.30 * stance_samples)
            ms_width = max(4, int(0.28 * stance_samples))
            ms_amp_z = acc_v * (1.0 - heel_sharp * 0.35) * rng.uniform(0.90, 1.10)

            acc_z += self._gauss(N, ms_idx, ms_width, ms_amp_z)

            # ── Toe-off push-off impulse ──────────────────────────────────
            to_idx   = idx + int(0.60 * stance_samples)
            to_width = max(3, int(fs * 0.07))
            to_amp_x = acc_h * 0.65 * rng.uniform(0.90, 1.10)
            to_amp_z = acc_v * 0.22 * rng.uniform(0.80, 1.20)

            acc_x += self._gauss(N, to_idx, to_width,  to_amp_x)
            acc_z += self._gauss(N, to_idx, to_width,  to_amp_z)

            # ── Mediolateral sway (alternates L/R each step) ──────────────
            ml_amp   = acc_h * 0.35 * rng.uniform(0.85, 1.15)
            ml_idx   = idx + int(0.40 * stance_samples)
            ml_width = max(4, int(0.32 * stance_samples))

            acc_y += self._gauss(N, ml_idx, ml_width, ml_sign * ml_amp)

            # ── Gyro — sagittal (hip/knee flex-extend) ────────────────────
            gx_width = max(5, int(0.45 * stance_samples))
            gyro_x  += self._gauss(N, idx + int(0.20 * gx_width),
                                   gx_width,
                                   gyro_s * rng.uniform(0.88, 1.12))
            gyro_x  += self._gauss(N, to_idx, to_width,
                                   -gyro_s * 0.68 * rng.uniform(0.88, 1.12))

            # ── Gyro — frontal (pelvic obliquity) ─────────────────────────
            gyro_y += self._gauss(N, ms_idx, ms_width,
                                  ml_sign * gyro_f * rng.uniform(0.88, 1.12))

            # ── Gyro — transverse (pelvic rotation) ───────────────────────
            gyro_z += self._gauss(N,
                                  idx + int(0.50 * stance_samples),
                                  max(4, int(0.60 * stance_samples)),
                                  ml_sign * gyro_f * 0.45 * rng.uniform(0.80, 1.20))

        # ── Gravity DC offset ─────────────────────────────────────────────
        acc_z += np.cos(trunk_lean_rad)   # ≈ 1 g
        acc_x += np.sin(trunk_lean_rad)   # small forward tilt

        # ── Slow postural sway (0.1–0.5 Hz) ──────────────────────────────
        sway_f   = rng.uniform(0.10, 0.45)
        sway_amp = rng.uniform(0.005, 0.022)
        t_vec    = np.linspace(0, duration, N)
        acc_x   += sway_amp        * np.sin(2 * np.pi * sway_f * t_vec)
        acc_y   += sway_amp * 0.55 * np.cos(2 * np.pi * sway_f * t_vec)

        # ── Pink + white sensor noise ─────────────────────────────────────
        for sig, w_sigma, p_sigma in [
            (acc_x,  noise_acc  * 0.65, noise_acc  * 0.35),
            (acc_y,  noise_acc  * 0.65, noise_acc  * 0.35),
            (acc_z,  noise_acc  * 0.65, noise_acc  * 0.35),
            (gyro_x, noise_gyro * 0.65, noise_gyro * 0.35),
            (gyro_y, noise_gyro * 0.65, noise_gyro * 0.35),
            (gyro_z, noise_gyro * 0.65, noise_gyro * 0.35),
        ]:
            sig += rng.normal(0, w_sigma, N)
            sig += self._pink_noise(N, rng) * p_sigma

        return np.stack(
            [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1
        ).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Identity-level generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_identity_windows(
        self,
        profile: dict,
        windows_per_identity: int = WINDOWS_PER_IDENTITY,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Generate `windows_per_identity` windows for one identity.

        Returns
        -------
        np.ndarray, shape (windows_per_identity, 128, 6)
        Saved to:  <output_path>/<identity_id>.npy
        """
        if rng is None:
            rng = np.random.default_rng()

        windows = np.stack(
            [self.simulate_window(profile, rng=rng)
             for _ in range(windows_per_identity)],
            axis=0,
        )

        out_file = os.path.join(self.output_path, f"{profile['identity_id']}.npy")
        np.save(out_file, windows)
        return windows

    # ─────────────────────────────────────────────────────────────────────────
    # Full dataset generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        windows_per_identity: int = WINDOWS_PER_IDENTITY,
        seed: int = None,
    ) -> None:
        """
        Simulate signals for every profile in self.profile_path.

        Each identity's windows are saved as a separate .npy file:
            <output_path>/SYNTH_XXXX.npy   shape (W, 128, 6)

        Parameters
        ----------
        windows_per_identity : windows generated per profile
        seed                 : global RNG seed (reproducibility)
        """
        profiles = self.load_profiles()
        rng      = np.random.default_rng(seed)
        total    = 0

        print(f"Simulating signals for {len(profiles)} identities "
              f"× {windows_per_identity} windows each …\n")

        for i, profile in enumerate(profiles, 1):
            windows = self.generate_identity_windows(
                profile,
                windows_per_identity=windows_per_identity,
                rng=rng,
            )
            total += len(windows)

            if i % 100 == 0 or i == len(profiles):
                print(f"  [{i:>4} / {len(profiles)}]  "
                      f"total windows so far: {total:,}")

        print(f"\n✓ Done. {total:,} windows across {len(profiles)} identities "
              f"saved to {self.output_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# Script execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    simulator = SyntheticSignalSimulator(
        profile_path="data/raw/synthetic/api_synthetic_profiles.json",
        output_path ="data/raw/synthetic/generated_windows",
    )
    simulator.generate_all(windows_per_identity=60, seed=42)