"""
generate_test_tones.py
----------------------
Generates synthetic WAV test tones to simulate different acoustic events.

Produced files:
  sample_audio/tone_440hz.wav   -- low-frequency hum (simulates background_hum)
  sample_audio/tone_600hz.wav   -- mid-frequency bark-like tone (simulates dog_barking)
  sample_audio/tone_3000hz.wav  -- high-frequency sharp tone (simulates glass_breaking)

Each file is 2 seconds long, mono, 22050 Hz sample rate.
"""

import os
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050          # Hz
DURATION = 2.0               # seconds
OUTPUT_DIR = "sample_audio"

TONES = {
    "tone_440hz.wav":  440,    # Background hum range
    "tone_600hz.wav":  600,    # Dog-bark / mid range
    "tone_3000hz.wav": 3000,   # Glass-breaking / sharp high range
}


def generate_sine_wave(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    """Return a normalised sine wave as a float32 numpy array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return wave


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename, freq in TONES.items():
        wave = generate_sine_wave(freq, DURATION, SAMPLE_RATE)
        output_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(output_path, wave, SAMPLE_RATE)
        print(f"[generated] {output_path}  ({freq} Hz, {DURATION}s, {SAMPLE_RATE} Hz SR)")

    print(f"\nAll test tones written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
