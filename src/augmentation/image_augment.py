#!/usr/bin/env python3
"""
Image augmentation script for SUR project 2025/2026.
Implements various augmentation techniques to expand limited training data.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
ORIGINAL_DIRS = {
    "target": PROJECT_ROOT / "dataset" / "target_train",
    "non_target": PROJECT_ROOT / "dataset" / "non_target_train",
}
OUTPUT_DIRS = {
    "target": PROJECT_ROOT / "dataset" / "augmented" / "target_image_aug",
    "non_target": PROJECT_ROOT / "dataset" / "augmented" / "non_target_image_aug",
}

# Augmentation parameters
ROTATION_ANGLES = [-10, -5, 5, 10]
SHIFT_PIXELS = [-8, -5, 5, 8]
SCALE_FACTORS = [0.9, 1.1]
NOISE_SNRS = [10, 20, 30]
FLIP_HORIZONTAL = True
BRIGHTNESS_FACTORS = [0.7, 1.3]
CONTRAST_FACTORS = [0.7, 1.3]


def add_gaussian_noise(image, snr_db):
    """Add gaussian noise with specified SNR."""
    img_array = np.array(image, dtype=np.float64)
    signal_power = np.mean(img_array**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), img_array.shape)
    noisy = img_array + noise
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def rotate_image(image, angle):
    """Rotate image by angle (degrees)."""
    return image.rotate(angle, fillcolor=(128, 128, 128))


def shift_image(image, shift_x, shift_y):
    """Shift image by pixels."""
    if shift_x == 0 and shift_y == 0:
        return image

    shifted = Image.new("RGB", image.size, (128, 128, 128))
    shifted.paste(image, (shift_x, shift_y))
    return shifted


def scale_image(image, factor):
    """Scale image and resize back to original."""
    w, h = image.size
    new_w, new_h = int(w * factor), int(h * factor)
    scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Center crop/pad to original size
    result = Image.new("RGB", (w, h), (128, 128, 128))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    result.paste(scaled, (paste_x, paste_y))
    return result


def adjust_brightness(image, factor):
    """Adjust image brightness."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    """Adjust image contrast."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def augment_image(image, output_dir, basename):
    """Apply all augmentations to a single image."""
    count = 0
    os.makedirs(output_dir, exist_ok=True)

    # Rotation
    for angle in ROTATION_ANGLES:
        output_name = f"{basename}_rot{angle}.png"
        output_path = os.path.join(output_dir, output_name)
        aug_img = rotate_image(image, angle)
        aug_img.save(output_path)
        count += 1

    # Shift (translation)
    for shift_x in SHIFT_PIXELS:
        for shift_y in SHIFT_PIXELS:
            if shift_x == 0 and shift_y == 0:
                continue
            output_name = f"{basename}_shift{shift_x}_{shift_y}.png"
            output_path = os.path.join(output_dir, output_name)
            aug_img = shift_image(image, shift_x, shift_y)
            aug_img.save(output_path)
            count += 1

    # Scale
    for factor in SCALE_FACTORS:
        output_name = f"{basename}_scale{int(factor * 10)}.png"
        output_path = os.path.join(output_dir, output_name)
        aug_img = scale_image(image, factor)
        aug_img.save(output_path)
        count += 1

    # Gaussian noise
    for snr in NOISE_SNRS:
        output_name = f"{basename}_noise{snr}.png"
        output_path = os.path.join(output_dir, output_name)
        aug_img = add_gaussian_noise(image, snr)
        aug_img.save(output_path)
        count += 1

    # Horizontal flip
    if FLIP_HORIZONTAL:
        output_name = f"{basename}_flip.png"
        output_path = os.path.join(output_dir, output_name)
        aug_img = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        aug_img.save(output_path)
        count += 1

    # Brightness
    for factor in BRIGHTNESS_FACTORS:
        output_name = f"{basename}_bright{int(factor * 10)}.png"
        output_path = os.path.join(output_dir, output_name)
        aug_img = adjust_brightness(image, factor)
        aug_img.save(output_path)
        count += 1

    # Contrast
    for factor in CONTRAST_FACTORS:
        output_name = f"{basename}_contrast{int(factor * 10)}.png"
        output_path = os.path.join(output_dir, output_name)
        aug_img = adjust_contrast(image, factor)
        aug_img.save(output_path)
        count += 1

    return count


def main():
    total_original = 0
    total_augmented = 0

    for category in ["target", "non_target"]:
        input_dir = str(ORIGINAL_DIRS[category])
        output_dir = str(OUTPUT_DIRS[category])

        os.makedirs(output_dir, exist_ok=True)

        # Get all PNG files
        png_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
        print(f"\nProcessing {category} images: {len(png_files)}")

        for png_file in sorted(png_files):
            input_path = os.path.join(input_dir, png_file)
            basename = png_file.replace(".png", "")

            try:
                image = Image.open(input_path).convert("RGB")
                total_original += 1
                count = augment_image(image, output_dir, basename)
                total_augmented += count
                print(f"  {png_file} -> {count} augmented")
            except Exception as e:
                print(f"  Error processing {png_file}: {e}")

    print("\n=== Summary ===")
    print(f"Original images: {total_original}")
    print(f"Augmented images: {total_augmented}")
    print(f"Total images: {total_original + total_augmented}")


if __name__ == "__main__":
    main()
