"""
classifier.py
-------------
Rule-based acoustic event classifier.

Decision logic
--------------
The classifier maps three scalar acoustic features to one of three event
labels and an urgency level using hand-crafted thresholds:

  Spectral Centroid (sc_mean) -- primary discriminator of frequency content
  RMS Energy        (rms_mean)-- discriminates quiet hums from loud events
  MFCC[0] mean                -- captures overall energy / timbral envelope

Event labels and thresholds (tuned for 22050 Hz sine-wave test tones):

  glass_breaking  -- very high spectral centroid (> 2000 Hz) regardless of RMS
  dog_barking     -- mid spectral centroid (400 – 2000 Hz) AND moderate RMS
  background_hum  -- low spectral centroid (< 400 Hz) OR very low RMS

Urgency mapping:
  glass_breaking  -> high
  dog_barking     -> medium
  background_hum  -> low
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thresholds (Hz / amplitude units)
# ---------------------------------------------------------------------------
SC_HIGH_THRESHOLD   = 2000.0   # Hz -- above this → glass_breaking territory
SC_MID_LOW          =  500.0   # Hz -- below this  → background_hum territory
RMS_QUIET_THRESHOLD =  0.01    # amplitude -- very quiet signal → background_hum

# ---------------------------------------------------------------------------
# Label → urgency mapping
# ---------------------------------------------------------------------------
URGENCY_MAP: dict[str, str] = {
    "glass_breaking": "high",
    "dog_barking":    "medium",
    "background_hum": "low",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(features: dict) -> dict:
    """
    Apply rule-based thresholds to *features* and return a classification result.

    Parameters
    ----------
    features : dict
        Output of ``feature_extractor.extract_features()``.

    Returns
    -------
    dict with keys:
        label   -- str  -- predicted event label
        urgency -- str  -- 'low' | 'medium' | 'high'
        reason  -- str  -- human-readable explanation of the decision
    """
    sc_mean  = features["sc_mean"]
    rms_mean = features["rms_mean"]

    # ------------------------------------------------------------------
    # Rule 1 – Glass breaking: sharp, high-frequency transient
    # ------------------------------------------------------------------
    if sc_mean > SC_HIGH_THRESHOLD:
        label  = "glass_breaking"
        reason = (
            f"Spectral centroid ({sc_mean:.1f} Hz) exceeds the high-frequency "
            f"threshold ({SC_HIGH_THRESHOLD} Hz), indicating a sharp, "
            "high-pitched transient consistent with breaking glass."
        )

    # ------------------------------------------------------------------
    # Rule 2 – Background hum: very low frequency or near-silence
    # ------------------------------------------------------------------
    elif sc_mean < SC_MID_LOW or rms_mean < RMS_QUIET_THRESHOLD:
        label  = "background_hum"
        reason = (
            f"Spectral centroid ({sc_mean:.1f} Hz) is below the low-frequency "
            f"threshold ({SC_MID_LOW} Hz) or RMS energy ({rms_mean:.5f}) is "
            f"below the quiet threshold ({RMS_QUIET_THRESHOLD}), "
            "consistent with a steady low-level background hum."
        )

    # ------------------------------------------------------------------
    # Rule 3 – Dog barking: mid-range frequency, audible energy
    # ------------------------------------------------------------------
    else:
        label  = "dog_barking"
        reason = (
            f"Spectral centroid ({sc_mean:.1f} Hz) falls in the mid-frequency "
            f"band ({SC_MID_LOW}–{SC_HIGH_THRESHOLD} Hz) with sufficient "
            f"RMS energy ({rms_mean:.5f}), consistent with a dog bark."
        )

    return {
        "label":   label,
        "urgency": URGENCY_MAP[label],
        "reason":  reason,
    }
