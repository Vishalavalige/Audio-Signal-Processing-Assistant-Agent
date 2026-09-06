"""
agent.py
--------
Audio Signal Processing Assistant Agent
Problem Statement 32

Entry point that ties together:
  1. Synthetic test-tone generation  (generate_test_tones.py)
  2. Acoustic feature extraction     (feature_extractor.py)
  3. Rule-based event classification (classifier.py)

Usage
-----
  # Run end-to-end with auto-generated test tones:
  python agent.py

  # Analyse a specific audio file:
  python agent.py --file path/to/audio.wav

  # Regenerate test tones before running:
  python agent.py --regenerate
"""

from __future__ import annotations

import argparse
import os
import sys

import generate_test_tones
import feature_extractor
import classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 60


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def analyse_file(audio_path: str) -> None:
    """Extract features from *audio_path*, classify, and print a report."""
    if not os.path.isfile(audio_path):
        print(f"[error] File not found: {audio_path}", file=sys.stderr)
        return

    print(f"\n[>>] Analysing: {audio_path}")

    # -- Feature extraction ------------------------------------------------
    features = feature_extractor.extract_features(audio_path)

    print(f"    Spectral Centroid (mean) : {features['sc_mean']:>10.2f} Hz")
    print(f"    RMS Energy        (mean) : {features['rms_mean']:>10.5f}")
    print(f"    MFCC[0]           (mean) : {features['mfcc_mean'][0]:>10.3f}")

    # -- Classification ----------------------------------------------------
    result = classifier.classify(features)

    print(f"\n    +-- Classification Result ---------------------------------")
    print(f"    |  Event Label  : {result['label']}")
    print(f"    |  Urgency      : {result['urgency'].upper()}")
    print(f"    |  Reason       : {result['reason']}")
    print(f"    +----------------------------------------------------------")


def run_batch(audio_dir: str) -> None:
    """Analyse every WAV file found inside *audio_dir*."""
    wav_files = sorted(
        f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")
    )

    if not wav_files:
        print(f"[warn] No WAV files found in '{audio_dir}'.")
        return

    print_section(f"Batch Analysis — {len(wav_files)} file(s) in '{audio_dir}'")
    for fname in wav_files:
        analyse_file(os.path.join(audio_dir, fname))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audio Signal Processing Assistant Agent (Problem Statement 32)"
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Path to a single audio file to analyse.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="(Re)generate synthetic test tones before analysing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print_section("Audio Signal Processing Assistant Agent")
    print("  Problem Statement 32")

    # ------------------------------------------------------------------
    # Step 1 – Ensure test tones exist (or regenerate on request)
    # ------------------------------------------------------------------
    sample_dir = generate_test_tones.OUTPUT_DIR
    has_tones  = (
        os.path.isdir(sample_dir)
        and any(f.endswith(".wav") for f in os.listdir(sample_dir))
    )

    if args.regenerate or not has_tones:
        print_section("Generating Synthetic Test Tones")
        generate_test_tones.main()

    # ------------------------------------------------------------------
    # Step 2 – Analyse
    # ------------------------------------------------------------------
    if args.file:
        # Single-file mode
        print_section("Single-File Analysis")
        analyse_file(args.file)
    else:
        # Batch mode: analyse all test tones
        run_batch(sample_dir)

    print(f"\n{SEPARATOR}")
    print("  Analysis complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
