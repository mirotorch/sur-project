"""

Utility functions for SUR project 2025/2026.
Data loading, session parsing, and prediction utilities.
"""

import os
import joblib
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_threshold_cache_path(method, cv_strategy, n_splits, n_components=None):
    """Generate cache file path for CV mean threshold."""
    n_comp_str = f"pca{n_components}" if n_components is not None else "nopca"
    filename = f"threshold_{method}_{cv_strategy}_{n_splits}_{n_comp_str}.npy"
    return os.path.join(CACHE_DIR, filename)


def save_cv_threshold(threshold, method, cv_strategy, n_splits, n_components=None):
    """Save mean CV threshold to cache."""
    ensure_cache_dir()
    cache_path = get_threshold_cache_path(method, cv_strategy, n_splits, n_components)
    np.save(cache_path, np.array([threshold]))
    print(f"  Saved CV mean threshold to {cache_path}")


def load_cv_threshold(method, cv_strategy, n_splits, n_components=None):
    """
    Load mean CV threshold from cache.

    Returns:
        Threshold value or None if not found
    """
    cache_path = get_threshold_cache_path(method, cv_strategy, n_splits, n_components)
    if os.path.exists(cache_path):
        threshold = np.load(cache_path)[0]
        print(f"  Loaded CV mean threshold from {cache_path}: {threshold:.4f}")
        return threshold
    return None


def get_model_cache_path(method, model_type, cv_strategy=None, n_splits=None, n_components=None, fold=None):
    """Generate cache file path for models."""
    n_comp_str = f"pca{n_components}" if n_components is not None else "nopca"
    if cv_strategy and n_splits:
        if fold is not None:
            filename = f"model_{method}_{model_type}_fold{fold}_{cv_strategy}_{n_splits}_{n_comp_str}.joblib"
        else:
            filename = f"model_{method}_{model_type}_{cv_strategy}_{n_splits}_{n_comp_str}.joblib"
    else:
        filename = f"model_{method}_{model_type}_dev_{n_comp_str}.joblib"
    return os.path.join(CACHE_DIR, filename)


def save_model_cache(model, method, model_type, cv_strategy=None, n_splits=None, n_components=None, fold=None):
    """Save model to cache."""
    ensure_cache_dir()
    cache_path = get_model_cache_path(method, model_type, cv_strategy, n_splits, n_components, fold)
    joblib.dump(model, cache_path)
    print(f"  Saved {model_type} to {cache_path}")


def load_model_cache(method, model_type, cv_strategy=None, n_splits=None, n_components=None, fold=None):
    """Load model from cache."""
    cache_path = get_model_cache_path(method, model_type, cv_strategy, n_splits, n_components, fold)
    if os.path.exists(cache_path):
        model = joblib.load(cache_path)
        print(f"  Loaded {model_type} from {cache_path}")
        return model
    return None


def compute_dataset_hash(images):
    """Compute simple hash of dataset for cache invalidation."""
    return f"{len(images)}_{images.shape[1]}_{images.shape[2]}"


def get_feature_cache_path(method, params_str, dataset_hash):
    """Generate cache file path for features."""
    filename = f"{method}_{params_str}_{dataset_hash}.npy"
    return os.path.join(CACHE_DIR, filename)


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

    # Target data (from target/ and non-target/ directories)
    target_dir = base_dir / "target"
    if target_dir.exists():
        imgs, fnames = load_images(target_dir)
        images.append(imgs)
        labels.extend([1] * len(imgs))
        filenames.extend(fnames)
        print(f"Loaded {len(imgs)} target images")

    # Non-target data
    non_target_dir = base_dir / "non-target"
    if non_target_dir.exists():
        imgs, fnames = load_images(non_target_dir)
        images.append(imgs)
        labels.extend([0] * len(imgs))
        filenames.extend(fnames)
        print(f"Loaded {len(imgs)} non-target images")

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
