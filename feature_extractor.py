"""
feature_extractor.py
--------------------
Loads an audio file and extracts three acoustic feature sets used by the
rule-based classifier:

  * MFCC (Mel-Frequency Cepstral Coefficients) -- captures timbral texture
  * Spectral Centroid                           -- describes the "brightness"
  * RMS Energy                                  -- measures loudness / intensity

All features are returned as scalar summary statistics (mean / std) so the
classifier can apply simple threshold rules without dealing with time-series.
"""

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(audio_path: str, sr: int = 22050) -> dict:
    """
    Load *audio_path* and return a flat dictionary of summary features.

    Parameters
    ----------
    audio_path : str
        Path to any audio file supported by librosa (wav, mp3, flac, …).
    sr : int
        Target sample rate.  Audio is resampled if necessary.

    Returns
    -------
    dict with keys:
        mfcc_mean    -- np.ndarray, shape (n_mfcc,)  -- per-coefficient mean
        mfcc_std     -- np.ndarray, shape (n_mfcc,)  -- per-coefficient std
        sc_mean      -- float  -- mean spectral centroid (Hz)
        sc_std       -- float  -- std  spectral centroid (Hz)
        rms_mean     -- float  -- mean RMS energy
        rms_std      -- float  -- std  RMS energy
    """
    # --- Load audio -------------------------------------------------------
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # --- MFCC (13 coefficients) -------------------------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)          # (13, T)
    mfcc_mean = np.mean(mfcc, axis=1)                            # (13,)
    mfcc_std  = np.std(mfcc,  axis=1)                            # (13,)

    # --- Spectral Centroid ------------------------------------------------
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)           # (1, T)
    sc_mean = float(np.mean(sc))
    sc_std  = float(np.std(sc))

    # --- RMS Energy -------------------------------------------------------
    rms = librosa.feature.rms(y=y)                               # (1, T)
    rms_mean = float(np.mean(rms))
    rms_std  = float(np.std(rms))

    return {
        "mfcc_mean": mfcc_mean,
        "mfcc_std":  mfcc_std,
        "sc_mean":   sc_mean,
        "sc_std":    sc_std,
        "rms_mean":  rms_mean,
        "rms_std":   rms_std,
    }
