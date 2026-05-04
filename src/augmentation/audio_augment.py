#!/usr/bin/env python3
"""
Audio augmentation script.
Implements various augmentation techniques to expand limited training data.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
SAMPLE_RATE = 16000
ORIGINAL_DIRS = {
    "target": PROJECT_ROOT / "dataset" / "target",
    "non_target": PROJECT_ROOT / "dataset" / "non-target",
}
OUTPUT_DIRS = {
    "target": PROJECT_ROOT / "dataset" / "augmented" / "target_train_aug",
    "non_target": PROJECT_ROOT / "dataset" / "augmented" / "non_target_train_aug",
}

# Augmentation parameters
SPEED_FACTORS = [0.7, 0.9, 1.1, 1.3]
VOLUME_FACTORS = [0.5, 1.5, 2.0]
NOISE_SNRS = [10, 20, 30]  # dB
TIME_SHIFTS = [-0.15, -0.1, 0.1, 0.15]  # seconds


def speed_perturbation(input_file, output_file, speed_factor):
    """Change speed of audio using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-filter:a",
        f"atempo={speed_factor}",
        "-ar",
        str(SAMPLE_RATE),
        output_file,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def volume_perturbation(input_file, output_file, volume_factor):
    """Change volume of audio using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-filter:a",
        f"volume={volume_factor}",
        output_file,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def add_noise(input_file, output_file, snr_db):
    """Add Gaussian noise at specified SNR."""
    data, sr = sf.read(input_file)
    if len(data.shape) > 1:
        data = data[:, 0]

    signal_power = np.mean(data**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(data))

    augmented = data + noise
    sf.write(output_file, augmented, sr)


def time_shift(input_file, output_file, shift_seconds):
    """Shift audio in time."""
    data, sr = sf.read(input_file)
    if len(data.shape) > 1:
        data = data[:, 0]

    shift_samples = int(shift_seconds * sr)
    if shift_samples > 0:
        augmented = np.concatenate([data[shift_samples:], np.zeros(shift_samples)])
    else:
        augmented = np.concatenate([np.zeros(-shift_samples), data[:shift_samples]])

    sf.write(output_file, augmented, sr)


def augment_file(wav_file, output_dir, basename):
    """Apply all augmentations to a single audio file."""
    count = 0
    os.makedirs(output_dir, exist_ok=True)

    # Speed perturbation
    for speed in SPEED_FACTORS:
        name = f"speed{int(speed * 10)}"
        try:
            speed_perturbation(
                str(wav_file), str(output_dir / f"{basename}_{name}.wav"), speed
            )
            count += 1
        except Exception:
            pass

    # Volume perturbation
    for vol in VOLUME_FACTORS:
        name = f"vol{int(vol * 10)}"
        try:
            volume_perturbation(
                str(wav_file), str(output_dir / f"{basename}_{name}.wav"), vol
            )
            count += 1
        except Exception:
            pass

    # Noise injection
    for snr in NOISE_SNRS:
        name = f"noise{snr}"
        try:
            add_noise(str(wav_file), str(output_dir / f"{basename}_{name}.wav"), snr)
            count += 1
        except Exception:
            pass

    # Time shift
    for shift in TIME_SHIFTS:
        name = f"shift{int(shift * 100)}"
        try:
            time_shift(str(wav_file), str(output_dir / f"{basename}_{name}.wav"), shift)
            count += 1
        except Exception:
            pass

    return count


def main():
    total_original = 0
    total_augmented = 0

    for category in ["target", "non_target"]:
        input_dir = ORIGINAL_DIRS[category]
        output_dir = OUTPUT_DIRS[category]

        os.makedirs(output_dir, exist_ok=True)

        wav_files = sorted(input_dir.glob("*.wav"))
        print(f"\nProcessing {category} audio: {len(wav_files)}")

        for wav_file in wav_files:
            basename = wav_file.stem
            try:
                total_original += 1
                count = augment_file(wav_file, output_dir, basename)
                total_augmented += count
                print(f"  {wav_file.name} -> {count} augmented")
            except Exception as e:
                print(f"  Error processing {wav_file.name}: {e}")

    print("\n=== Summary ===")
    print(f"Original files: {total_original}")
    print(f"Augmented files: {total_augmented}")
    print(f"Total files: {total_original + total_augmented}")


if __name__ == "__main__":
    main()
