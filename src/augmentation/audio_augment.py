#!/usr/bin/env python3
"""
Audio augmentation script for SUR project 2025/2026.
Implements various augmentation techniques to expand limited training data.
"""

import os
from pathlib import Path

import numpy as np
import soundfile as sf

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
SAMPLE_RATE = 16000
ORIGINAL_DIRS = {
    "target": PROJECT_ROOT / "dataset" / "target_train",
    "non_target": PROJECT_ROOT / "dataset" / "non_target_train",
}
OUTPUT_DIRS = {
    "target": PROJECT_ROOT / "dataset" / "augmented" / "target_train_aug",
    "non_target": PROJECT_ROOT / "dataset" / "augmented" / "non_target_train_aug",
}

# Augmentation parameters
SPEED_FACTORS = [0.9, 1.1]
VOLUME_FACTORS = [0.5, 2.0]
NOISE_SNRS = [10, 20, 30]  # dB
TIME_SHIFTS = [-0.1, 0.1]  # seconds
COMBINED = [("speed09", "noise20"), ("vol05", "noise20")]


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


def apply_augmentation(input_file, output_dir, augment_name, augment_fn, *args):
    """Apply augmentation and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(input_file).stem
    output_file = output_dir / f"{basename}_{augment_name}.wav"

    try:
        augment_fn(input_file, output_file, *args)
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def augment_class(class_name, original_dir, output_dir):
    """Augment all files in a class directory."""
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)

    wav_files = list(original_dir.glob("*.wav"))
    print(f"\nAugmenting {class_name} ({len(wav_files)} files)...")

    count = 0
    for wav_file in wav_files:
        # Speed perturbation
        for speed in SPEED_FACTORS:
            name = f"speed{int(speed * 10)}"
            if apply_augmentation(
                wav_file, output_dir, name, speed_perturbation, speed
            ):
                count += 1

        # Volume perturbation
        for vol in VOLUME_FACTORS:
            name = f"vol{int(vol * 10)}"
            if apply_augmentation(wav_file, output_dir, name, volume_perturbation, vol):
                count += 1

        # Noise injection
        for snr in NOISE_SNRS:
            name = f"noise{snr}"
            if apply_augmentation(wav_file, output_dir, name, add_noise, snr):
                count += 1

        # Time shift
        for shift in TIME_SHIFTS:
            name = f"shift{int(shift * 100)}"
            if apply_augmentation(wav_file, output_dir, name, time_shift, shift):
                count += 1

    print(f"  Created {count} augmented files")


def main():
    """Run audio augmentation for all classes."""
    print("Starting audio augmentation...")
    print(f"Project root: {PROJECT_ROOT}")

    for class_name, original_dir in ORIGINAL_DIRS.items():
        output_dir = OUTPUT_DIRS[class_name]
        augment_class(class_name, original_dir, output_dir)

    print("\nAugmentation complete!")


if __name__ == "__main__":
    import subprocess

    main()
