"""

Utility functions for SUR project 2025/2026.
Data loading, session parsing, and prediction utilities.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve


def parse_filename(filename):
    """
    Parse filename to extract components.
    Format: {personID}_{session}_{camera}_{i0}_{0}.png
    Returns: dict with person_id, session, camera, base_name, extension
    """
    name = Path(filename).stem
    parts = name.split("_")
    return {
        "person_id": parts[0],
        "session": parts[1],
        "camera": parts[2] if len(parts) > 2 else None,
        "base_name": name,
        "extension": Path(filename).suffix,
    }


def parse_session(filename):
    """Extract session ID from filename."""
    return parse_filename(filename)["session"]


def load_images(data_dir, extensions=None, limit=None):
    """
    Load images from directory.

    Args:
        data_dir: Path to directory containing images
        extensions: List of extensions to filter (default: ['.png'])
        limit: Maximum number of images to load (None for all)

    Returns:
        tuple: (images, filenames) where
            images: numpy array of shape (n, height, width, channels)
            filenames: list of filenames without extension
    """
    if extensions is None:
        extensions = [".png"]

    data_dir = Path(data_dir)
    files = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in extensions])

    if limit:
        files = files[:limit]

    images = []
    filenames = []

    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            img_array = np.array(img)
            images.append(img_array)
            filenames.append(f.stem)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return np.array(images), filenames


def load_dataset(base_dir, use_augmented=True):
    """
    Load complete dataset (target and non-target).

    Args:
        base_dir: Base dataset directory
        use_augmented: Whether to include augmented data

    Returns:
        tuple: (images, labels, filenames) where
            images: numpy array
            labels: 1 for target, 0 for non-target
            filenames: list of filenames
    """
    base_dir = Path(base_dir)
    images = []
    labels = []
    filenames = []

    # Target training data
    target_dir = base_dir / "target_train"
    if target_dir.exists():
        imgs, fnames = load_images(target_dir)
        images.append(imgs)
        labels.extend([1] * len(imgs))
        filenames.extend(fnames)
        print(f"Loaded {len(imgs)} target training images")

    # Non-target training data
    non_target_dir = base_dir / "non_target_train"
    if non_target_dir.exists():
        imgs, fnames = load_images(non_target_dir)
        images.append(imgs)
        labels.extend([0] * len(imgs))
        filenames.extend(fnames)
        print(f"Loaded {len(imgs)} non-target training images")

    # Augmented data
    if use_augmented:
        aug_dir = base_dir / "augmented"
        if aug_dir.exists():
            # Target augmented
            target_aug = aug_dir / "target_image_aug"
            if target_aug.exists():
                imgs, fnames = load_images(target_aug)
                images.append(imgs)
                labels.extend([1] * len(imgs))
                print(f"Loaded {len(imgs)} target augmented images")
                filenames.extend(fnames)

            # Non-target augmented
            non_target_aug = aug_dir / "non_target_image_aug"
            if non_target_aug.exists():
                imgs, fnames = load_images(non_target_aug)
                images.append(imgs)
                labels.extend([0] * len(imgs))
                print(f"Loaded {len(imgs)} non-target augmented images")
                filenames.extend(fnames)

    if images:
        images = np.vstack(
            [img.reshape(1, *img.shape) for img in np.concatenate(images)]
        )

    return np.array(images), np.array(labels), filenames


def preprocess_image(img, target_size=(80, 80)):
    """
    Preprocess image: resize, normalize.

    Args:
        img: numpy array (H, W, C) or (H, W)
        target_size: Target size (height, width)

    Returns:
        Preprocessed image as numpy array
    """
    if len(img.shape) == 3:
        # RGB to grayscale
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # Normalize to [0, 1]
    if img.max() > 1:
        img = img / 255.0

    return img.astype(np.float32)


def compute_eer_threshold(labels, scores):
    """
    Compute optimal EER threshold where FPR ≈ FNR.

    Args:
        labels: Ground truth labels (0 or 1)
        scores: Prediction scores (higher = more confident target)

    Returns:
        tuple: (optimal_threshold, eer_value)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Find threshold where FPR ≈ FNR
    optimal_idx = np.argmin(np.abs(fpr - fnr))
    optimal_threshold = thresholds[optimal_idx]
    eer = (fpr[optimal_idx] + fnr[optimal_idx]) / 2

    return optimal_threshold, eer


def save_predictions(filename, filenames, scores, threshold=0.5):
    """
    Save predictions in required format.

    Args:
        filename: Output file path
        filenames: List of filenames (without extension)
        scores: Prediction scores (higher = more confident target)
        threshold: Threshold for hard decision (P(target)=0.5)
    """
    hard_decisions = (scores > threshold).astype(int)

    with open(filename, "w") as f:
        for fname, score, decision in zip(filenames, scores, hard_decisions):
            f.write(f"{fname} {score:.6f} {decision}\n")

    print(f"Saved predictions to {filename}")
