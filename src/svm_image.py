"""
Image-based person detector using HOG and LBP features with SVM.
Implements session-based cross-validation and generates prediction files.
"""

import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cross_validation.session_cv import session_based_cv
from extractors.hog_extractor import extract_hog_batch
from extractors.lbp_extractor import extract_lbp_batch
from utils import load_dataset, load_images, save_predictions

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(method, params_str, dataset_hash):
    """Generate cache file path for features."""
    filename = f"{method}_{params_str}_{dataset_hash}.npy"
    return os.path.join(CACHE_DIR, filename)


def compute_dataset_hash(images):
    """Compute simple hash of dataset for cache invalidation."""
    return f"{len(images)}_{images.shape[1]}_{images.shape[2]}"


def extract_features_cached(images, method="hog", use_cache=True, **kwargs):
    """
    Extract features with caching support.

    Args:
        images: Batch of images
        method: 'hog' or 'lbp'
        use_cache: Whether to use cached features
        **kwargs: Feature extraction parameters

    Returns:
        Feature matrix
    """
    ensure_cache_dir()

    # Create parameter string for cache key
    params_str = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
    dataset_hash = compute_dataset_hash(images)
    cache_path = get_cache_path(method, params_str, dataset_hash)

    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached {method} features from {cache_path}")
        return np.load(cache_path)

    # Extract features
    print(f"Extracting {method} features...")
    start = time.time()

    if method == "hog":
        features = extract_hog_batch(images, **kwargs)
    elif method == "lbp":
        features = extract_lbp_batch(images, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start
    print(f"  Extracted {features.shape} in {elapsed:.2f}s")

    # Save to cache
    if use_cache:
        np.save(cache_path, features)
        print(f"  Cached to {cache_path}")

    return features


def train_svm(X_train, y_train, C=1.0):
    """
    Train SVM classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization parameter

    Returns:
        Tuple of (trained SVM, fitted scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM with balanced class weights
    svm = LinearSVC(C=C, class_weight="balanced", max_iter=10000)
    svm.fit(X_train_scaled, y_train)

    return svm, scaler


def evaluate_svm(svm, scaler, X_test, y_test):
    """
    Evaluate SVM classifier.

    Args:
        svm: Trained SVM
        scaler: Feature scaler
        X_test: Test features
        y_test: Test labels

    Returns:
        dict with accuracy, auc, and scores
    """
    X_test_scaled = scaler.transform(X_test)

    # Get decision scores
    scores = svm.decision_function(X_test_scaled)

    # Convert to probabilities (simple min-max normalization)
    score_min, score_max = scores.min(), scores.max()
    if score_max - score_min > 1e-10:
        calibrated_scores = (scores - score_min) / (score_max - score_min)
    else:
        calibrated_scores = np.ones_like(scores) * 0.5

    # Predictions
    y_pred = svm.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, calibrated_scores)
    else:
        auc = 0.5

    return {
        "accuracy": acc,
        "auc": auc,
        "scores": scores,
        "calibrated_scores": calibrated_scores,
        "predictions": y_pred,
    }


def cross_validate(
    images,
    labels,
    filenames,
    method="hog",
    n_splits=2,
    C=1.0,
    use_cache=True,
    **feature_kwargs,
):
    """
    Perform session-based cross-validation.

    Args:
        images, labels, filenames: Dataset
        method: Feature extraction method
        n_splits: Number of CV folds
        C: SVM regularization
        use_cache: Whether to use feature caching
        **feature_kwargs: Arguments for feature extraction

    Returns:
        List of CV results
    """
    # Extract features once (cached)
    features = extract_features_cached(images, method, use_cache, **feature_kwargs)

    cv_splits = session_based_cv(filenames, labels, n_splits)
    results = []

    print(f"\n=== {method.upper()} Cross-Validation ({len(cv_splits)} folds) ===")

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold + 1}:")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        print(f"  Feature shape: {X_train.shape}")

        # Train and evaluate
        svm, scaler = train_svm(X_train, y_train, C)
        result = evaluate_svm(svm, scaler, X_val, y_val)

        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC: {result['auc']:.4f}")

        results.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "accuracy": result["accuracy"],
                "auc": result["auc"],
                "svm": svm,
                "scaler": scaler,
                "features": features,
            }
        )

    # Summary
    accuracies = [r["accuracy"] for r in results]
    aucs = [r["auc"] for r in results]
    print("\n=== Summary ===")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    return results


def train_final_model(
    images, labels, method="hog", C=1.0, use_cache=True, **feature_kwargs
):
    """
    Train final model on all training data.

    Args:
        images, labels: Full training dataset
        method: Feature extraction method
        C: SVM regularization
        use_cache: Whether to use feature caching
        **feature_kwargs: Arguments for feature extraction

    Returns:
        tuple: (svm, scaler, feature_size)
    """
    print(f"\n=== Training final {method.upper()} model on all data ===")

    # Extract features
    X = extract_features_cached(images, method, use_cache, **feature_kwargs)
    print(f"Feature shape: {X.shape}")

    # Train
    svm, scaler = train_svm(X, labels, C)

    return svm, scaler, X.shape[1]


def predict_on_dev(svm, scaler, method, use_cache=True, **feature_kwargs):
    """
    Generate predictions on dev set and save results.

    Args:
        svm, scaler: Trained model and scaler
        method: Feature extraction method
        use_cache: Whether to use feature caching
        **feature_kwargs: Arguments for feature extraction

    Returns:
        dict with predictions and scores
    """
    base_dir = PROJECT_ROOT / "dataset"

    print(f"\nEvaluating on dev set with {method.upper()}...")

    # Load dev data
    dev_images, dev_filenames = load_images(os.path.join(base_dir, "target_dev"))
    non_target_dev, non_target_fnames = load_images(
        os.path.join(base_dir, "non_target_dev")
    )

    dev_all = np.concatenate([dev_images, non_target_dev])
    dev_fnames_all = dev_filenames + non_target_fnames
    dev_labels = np.array([1] * len(dev_images) + [0] * len(non_target_dev))

    # Extract features
    dev_features = extract_features_cached(dev_all, method, use_cache, **feature_kwargs)

    # Evaluate
    result = evaluate_svm(svm, scaler, dev_features, dev_labels)

    print(f"  Dev Accuracy: {result['accuracy']:.4f}")
    print(f"  Dev AUC: {result['auc']:.4f}")

    # Save predictions
    output_dir = PROJECT_ROOT / "results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"image_{method}.txt")
    save_predictions(output_file, dev_fnames_all, result["calibrated_scores"])

    return {
        "filenames": dev_fnames_all,
        "labels": dev_labels,
        "scores": result["calibrated_scores"],
        "accuracy": result["accuracy"],
        "auc": result["auc"],
    }


def main():
    base_dir = str(PROJECT_ROOT / "dataset")

    # Load dataset
    print("Loading dataset...")
    images, labels, filenames = load_dataset(base_dir, use_augmented=True)
    print(
        f"Total: {len(images)} images, {sum(labels)} target, {len(labels) - sum(labels)} non-target"
    )

    # HOG parameters
    hog_params = {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "normalize": True,
    }

    # LBP parameters
    lbp_params = {
        "radius": 1,
        "neighbors": 8,
        "use_uniform": True,
        "grid_size": (4, 4),  # Use spatial grid for better accuracy
        "normalize": True,
    }

    # HOG Cross-Validation
    print("\n" + "=" * 50)
    hog_results = cross_validate(
        images, labels, filenames, method="hog", n_splits=2, C=1.0, **hog_params
    )

    # LBP Cross-Validation
    print("\n" + "=" * 50)
    lbp_results = cross_validate(
        images, labels, filenames, method="lbp", n_splits=2, C=1.0, **lbp_params
    )

    # Train final models
    print("\n" + "=" * 50)
    hog_svm, hog_scaler, hog_dim = train_final_model(
        images, labels, method="hog", C=1.0, **hog_params
    )
    print(f"HOG feature dimension: {hog_dim}")

    print("\n" + "=" * 50)
    lbp_svm, lbp_scaler, lbp_dim = train_final_model(
        images, labels, method="lbp", C=1.0, **lbp_params
    )
    print(f"LBP feature dimension: {lbp_dim}")

    # Evaluate on dev set
    hog_dev = predict_on_dev(hog_svm, hog_scaler, "hog", **hog_params)
    lbp_dev = predict_on_dev(lbp_svm, lbp_scaler, "lbp", **lbp_params)

    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"  HOG - Dev Accuracy: {hog_dev['accuracy']:.4f}, AUC: {hog_dev['auc']:.4f}")
    print(f"  LBP - Dev Accuracy: {lbp_dev['accuracy']:.4f}, AUC: {lbp_dev['auc']:.4f}")
    print(f"\nResults saved to {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    main()
