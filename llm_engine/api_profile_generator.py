# llm_engine/api_profile_generator.py

import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Expected schema + hard physical bounds for post-generation validation
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_KEYS = [
    "identity_id", "age", "height_m", "weight_kg",
    "cadence_hz", "stride_length_m",
    "acc_vertical_g", "acc_horizontal_g",
    "gyro_sagittal_rad_s", "gyro_frontal_rad_s",
    "step_asymmetry", "heel_strike_sharpness",
]

FIELD_BOUNDS = {
    "age":                   (19,   48),
    "height_m":              (1.50, 1.95),
    "weight_kg":             (45.0, 110.0),
    "cadence_hz":            (1.40, 2.20),
    "stride_length_m":       (0.56, 1.60),
    "acc_vertical_g":        (0.50, 1.60),
    "acc_horizontal_g":      (0.25, 1.20),
    "gyro_sagittal_rad_s":   (0.15, 1.10),
    "gyro_frontal_rad_s":    (0.10, 1.20),
    "step_asymmetry":        (0.01, 0.40),
    "heel_strike_sharpness": (0.15, 0.95),
}


class APIProfileGenerator:
    """
    Generates realistic biomechanical walking profiles via the Gemini LLM.

    Workflow
    --------
    1. Send batched prompts to Gemini (≤100 profiles per call to stay within
       context limits while still being fast).
    2. Parse + validate every returned profile against physical bounds.
    3. Retry failed / invalid batches up to MAX_RETRIES times.
    4. Re-number identity_ids sequentially across the full run.
    5. Save to JSON; also return the list for downstream use.
    """

    BATCH_SIZE  = 100   # profiles per API call  (≤100 keeps JSON compact)
    MAX_RETRIES = 3     # per-batch retry attempts on failure / bad JSON
    RETRY_DELAY = 2.0   # seconds between retries

    def __init__(self, api_key: str = None, output_path: str = "data/raw/synthetic"):

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

        self.client = genai.Client(api_key=self.api_key)
        self.model  = "gemini-2.5-flash"

        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, num_profiles: int, start_index: int) -> str:
        return f"""Generate exactly {num_profiles} realistic human biomechanical walking profiles.

IDENTITY IDs must be SYNTH_{start_index + 1:04d} through SYNTH_{start_index + num_profiles:04d}.

All parameters must follow the causal chain below — no independent draws:

CAUSAL CHAIN:
1. Demographics
   - age: integer 19–48
   - height_m: normal(1.70, 0.09), clamp [1.50, 1.95]
   - weight_kg: correlates with height (r≈0.45), clamp [45, 110]

2. Cadence (Froude walking model)
   - leg_length = height_m * 0.525
   - froude ~ normal(0.25, 0.03), clamp [0.14, 0.38]
   - Reduce froude by 1% per year above age 40
   - speed = froude * sqrt(9.81 * leg_length)
   - step_length = 0.413 * leg_length * (froude ^ 0.4)  + normal(0, 0.012)
   - cadence_hz = speed / step_length, clamp [1.40, 2.20]
   - stride_length_m = 2 * step_length, clamp [0.56, 1.60]

3. Accelerometer
   - acc_vertical_g: normal(0.98, 0.18)
     + (weight_kg - 70) * 0.004 + (speed - 1.35) * 0.28
     clamp [0.50, 1.60]
   - acc_horizontal_g = acc_vertical_g * uniform(0.50, 0.80)
     clamp [0.25, 1.20]

4. Gyroscope
   - gyro_sagittal_rad_s: normal(0.42, 0.11)
     + (stride_length_m - 0.70) * 0.45 + (speed - 1.35) * 0.12
     clamp [0.15, 1.10]
   - gyro_frontal_rad_s = gyro_sagittal * uniform(0.55, 1.15)
     clamp [0.10, 1.20]

5. Gait quality
   - step_asymmetry: beta(2, 10) * 0.40 + max(0, (age-30)*0.002)
     clamp [0.01, 0.40]
   - heel_strike_sharpness: normal(0.55, 0.13)
     + (weight_kg - 70)*0.004 - (speed - 1.35)*0.06
     clamp [0.15, 0.95]

OUTPUT: Return ONLY a valid JSON array. No markdown, no extra keys, no comments.
Each element must have exactly these keys (floats rounded to 3 decimal places):
{json.dumps(REQUIRED_KEYS, indent=2)}
"""

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_and_fix(self, profile: Dict[str, Any], identity_id: str) -> Dict[str, Any] | None:
        """
        Returns a cleaned profile dict, or None if it is unrecoverable.
        - Missing required keys → reject
        - Values outside physical bounds → clamp (warn)
        - identity_id is always overwritten by the caller
        """
        for key in REQUIRED_KEYS:
            if key == "identity_id":
                continue
            if key not in profile:
                return None   # unrecoverable — missing field

        cleaned = {"identity_id": identity_id}

        for key, (lo, hi) in FIELD_BOUNDS.items():
            val = profile.get(key)
            try:
                val = float(val)
            except (TypeError, ValueError):
                return None

            if val < lo or val > hi:
                val = max(lo, min(hi, val))   # clamp silently

            cleaned[key] = round(val, 3) if key != "age" else int(round(val))

        return cleaned

    # ─────────────────────────────────────────────────────────────────────────
    # Single batch call
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_batch(self, num_profiles: int, start_index: int) -> List[Dict[str, Any]]:
        """
        Calls Gemini once and returns a list of raw profile dicts.
        Raises ValueError on parse failure.
        """
        prompt = self._build_prompt(num_profiles, start_index)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.9,         # enough variance for diversity
            ),
        )
        text = response.text.strip()

        # Strip accidental markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        parsed = json.loads(text)

        # Gemini sometimes wraps the array in a dict
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

        if not isinstance(parsed, list):
            raise ValueError("Response is not a JSON array.")

        return parsed

    # ─────────────────────────────────────────────────────────────────────────
    # Batch with retry
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_batch_with_retry(self, num_profiles: int, start_index: int) -> List[Dict[str, Any]]:
        """
        Retries up to MAX_RETRIES times on API / parse failures.
        Returns only valid, cleaned profiles.
        """
        raw_profiles = []
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_profiles = self._fetch_batch(num_profiles, start_index)
                break
            except Exception as e:
                print(f"  [Attempt {attempt}/{self.MAX_RETRIES}] Error: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY * attempt)
                else:
                    print(f"  Batch starting at index {start_index} failed after {self.MAX_RETRIES} attempts.")
                    return []

        valid = []
        for i, raw in enumerate(raw_profiles):
            identity_id = f"SYNTH_{start_index + i + 1:04d}"
            cleaned = self._validate_and_fix(raw, identity_id)
            if cleaned is not None:
                valid.append(cleaned)

        return valid

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def generate_multiple_profiles(
        self,
        num_profiles: int = 1000,
        output_filename: str = "api_synthetic_profiles.json",
    ) -> List[Dict[str, Any]]:
        """
        Generate `num_profiles` profiles in batches and save to JSON.

        Parameters
        ----------
        num_profiles    : total profiles to generate
        output_filename : JSON file saved inside self.output_path
        """
        profiles: List[Dict[str, Any]] = []
        batch_size = self.BATCH_SIZE

        print(f"Generating {num_profiles} profiles in batches of {batch_size}...\n")

        start = 0
        while len(profiles) < num_profiles:
            remaining  = num_profiles - len(profiles)
            this_batch = min(batch_size, remaining)

            print(f"  Batch [{start + 1} → {start + this_batch}] ...", end=" ", flush=True)
            batch = self._fetch_batch_with_retry(this_batch, start)
            profiles.extend(batch)

            print(f"got {len(batch)} valid  |  total so far: {len(profiles)}")

            # If the API returned fewer than asked, retry the gap
            if len(batch) < this_batch:
                gap = this_batch - len(batch)
                print(f"  ⚠  {gap} profile(s) missing/invalid — will re-request at end.")

            start += this_batch
            time.sleep(0.3)   # polite pacing between calls

        # Re-number sequentially in case any profiles were dropped mid-run
        for idx, p in enumerate(profiles):
            p["identity_id"] = f"SYNTH_{idx + 1:04d}"

        output_file = os.path.join(self.output_path, output_filename)
        with open(output_file, "w") as f:
            json.dump(profiles, f, indent=4)

        print(f"\n✓ {len(profiles)} profiles saved → {output_file}")
        return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Script execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generator = APIProfileGenerator(
        api_key=os.environ.get("GEMINI_API_KEY")
    )
    generator.generate_multiple_profiles(num_profiles=1000)