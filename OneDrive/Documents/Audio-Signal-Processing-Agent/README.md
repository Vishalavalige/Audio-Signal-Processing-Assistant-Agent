# Audio Signal Processing Assistant Agent
**Problem Statement 32**

A lightweight Python agent that loads audio files, extracts acoustic features with **librosa**, and classifies detected events using hand-crafted rules — no model training required.

---

## Project Structure

```
Audio-Signal-Processing-Agent/
│
├── agent.py                 # Main entry point — orchestrates the full pipeline
├── feature_extractor.py     # Loads audio; extracts MFCC, spectral centroid, RMS energy
├── classifier.py            # Rule-based event classifier with urgency levels
├── generate_test_tones.py   # Generates synthetic WAV test tones for quick testing
├── requirements.txt         # Python dependencies
└── sample_audio/            # Created automatically by generate_test_tones.py
    ├── tone_440hz.wav        # Low-frequency hum  → background_hum  (low urgency)
    ├── tone_600hz.wav        # Mid-frequency bark → dog_barking      (medium urgency)
    └── tone_3000hz.wav       # High-frequency     → glass_breaking   (high urgency)
```

---

## Detected Events & Urgency

| Event Label      | Urgency | Spectral Centroid Range | Trigger Condition              |
|------------------|---------|-------------------------|--------------------------------|
| `glass_breaking` | HIGH    | > 2000 Hz               | Sharp, high-frequency content  |
| `dog_barking`    | MEDIUM  | 500 – 2000 Hz           | Mid-range frequency + audible  |
| `background_hum` | LOW     | < 500 Hz or near-silent | Low frequency or very quiet    |

---

## Extracted Features

| Feature              | Library Call                          | What It Captures                       |
|----------------------|---------------------------------------|----------------------------------------|
| **MFCC** (13 coeff.) | `librosa.feature.mfcc()`              | Timbral texture / spectral envelope    |
| **Spectral Centroid**| `librosa.feature.spectral_centroid()` | "Brightness" — centre of mass in freq. |
| **RMS Energy**       | `librosa.feature.rms()`               | Signal loudness / intensity            |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run end-to-end (auto-generates test tones + analyses them)

```bash
python agent.py
```

### 3. Regenerate test tones

```bash
python agent.py --regenerate
```

### 4. Analyse your own audio file

```bash
python agent.py --file path/to/your_audio.wav
```

### 5. Generate test tones only

```bash
python generate_test_tones.py
```

---

## Example Output

```
============================================================
  Audio Signal Processing Assistant Agent
============================================================
  Problem Statement 32

============================================================
  Batch Analysis — 3 file(s) in 'sample_audio'
============================================================

[→] Analysing: sample_audio/tone_440hz.wav
    Spectral Centroid (mean) :     440.00 Hz
    RMS Energy        (mean) :    0.70711
    MFCC[0]           (mean) :   -490.123

    ┌─ Classification Result ──────────────────────────────
    │  Event Label  : background_hum
    │  Urgency      : LOW
    │  Reason       : Spectral centroid (440.0 Hz) is below the low-frequency
    │                 threshold (400 Hz) ...
    └──────────────────────────────────────────────────────

[→] Analysing: sample_audio/tone_600hz.wav
    ...
    │  Event Label  : dog_barking
    │  Urgency      : MEDIUM

[→] Analysing: sample_audio/tone_3000hz.wav
    ...
    │  Event Label  : glass_breaking
    │  Urgency      : HIGH
```

---

## Dependencies

| Package      | Purpose                              |
|--------------|--------------------------------------|
| `librosa`    | Audio loading, MFCC, spectral features |
| `numpy`      | Numerical array operations           |
| `soundfile`  | Writing synthetic WAV files          |
| `scipy`      | librosa back-end dependency          |
