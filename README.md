# Audio Signal Processing Assistant Agent

**AICTE Internship 2026-27 — Problem Statement No. 32**
Electronics and Telecommunications Engineering Track — Agentic AI Project

Prepared by **Vishala Valige**, as part of the IBM SkillsBuild Internship (via Edunet Foundation).

## Overview

This project builds an agentic AI system that listens to an audio stream, classifies the type of sound event it hears, and generates an appropriate, context-aware response for the user — functioning as an assistant that interprets real-world audio signals and communicates their meaning in natural language.

## Architecture

| Layer | Tool/Tech | Purpose |
|---|---|---|
| Hardware / Capture | ESP32 + I2S microphone (Wokwi simulation) | Audio capture design |
| Signal Processing | Python, librosa | Feature extraction (MFCC, spectral centroid, RMS energy) |
| Classification | Rule-based classifier | Maps features → event label + urgency |
| Agent | IBM Granite-4.0-H-Small via IBM watsonx.ai | Generates natural-language notifications |

## Files

- `Audio_Agent_Project_Report.docx` — full project write-up: problem statement, architecture, implementation details, results, and challenges
- `audio-processing-notebook.ipynb` — Jupyter notebook with the working pipeline: feature extraction, rule-based classification, and the watsonx.ai agent call

## Results

The pipeline was validated on three representative audio events, each producing a distinct, appropriately-toned response from the agent:

| Event | Spectral Centroid | Urgency | Agent Response Summary |
|---|---|---|---|
| Background hum | 124.7 Hz | Low | Calm status update, no action needed |
| Dog barking | 607.9 Hz | Medium | Friendly notification, suggests checking on the pet |
| Glass breaking | 3007.4 Hz | High | Urgent alert, recommends checking surroundings |

See the full report for details, example prompts, and the engineering challenges encountered (notably, spectral-centroid sensitivity to background noise in the rule-based classifier).

## Tools & Platform

- Wokwi — ESP32 + I2S microphone hardware/firmware simulation
- Python, librosa, NumPy, soundfile — audio loading and feature extraction
- IBM watsonx.ai — Prompt Lab and Notebook environment
- IBM Granite-4.0-H-Small — foundation model for agent response generation

## Future Work

- Replace synthetic test tones with real recorded `.wav` samples from the ESP32/I2S hardware
- Replace the rule-based classifier with a trained ML model (e.g., a CNN on mel-spectrograms) for robustness to real-world noise
- Register formal tool definitions so the agent can directly trigger actions (e.g., `send_alert`)
- Deploy the pipeline as a watsonx.ai deployment endpoint for real-time inference
