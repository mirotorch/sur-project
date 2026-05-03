"""
Image-based person detector using HOG and LBP features with SVM.
Implements session-based cross-validation and generates prediction files.
"""

import os
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from extractors.hog_extractor import extract_hog_batch
from extractors.lbp_extractor import extract_lbp_batch
from session_cv import k_fold, loso
from utils import (compute_eer_threshold, load_dataset, load_images,
                   save_predictions)

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
        X_train: Training features (already scaled/transformed)
        y_train: Training labels
        C: Regularization parameter

    Returns:
        Trained SVM
    """
    svm = LinearSVC(C=C, class_weight="balanced", max_iter=10000)
    svm.fit(X_train, y_train)
    return svm


def evaluate_svm(svm, X_test, y_test):
    """
    Evaluate SVM classifier.

    Args:
        svm: Trained SVM
        X_test: Test features (already scaled/transformed)
        y_test: Test labels

    Returns:
        dict with accuracy, auc, and scores
    """
    scores = svm.decision_function(X_test)

    # Convert to probabilities (simple min-max normalization)
    score_min, score_max = scores.min(), scores.max()
    if score_max - score_min > 1e-10:
        calibrated_scores = (scores - score_min) / (score_max - score_min)
    else:
        calibrated_scores = np.ones_like(scores) * 0.5

    y_pred = svm.predict(X_test)

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
    cv_strategy="kfold",
    n_splits=2,
    C=1.0,
    n_components=None,
    use_cache=True,
    **feature_kwargs,
):
    """
    Perform session-based cross-validation with SVM classifiers.

    Supports multiple cross-validation strategies to prevent data leakage
    from samples in the same recording session.

    Args:
        images: Image array
        labels: Labels array
        filenames: List of filenames
        method: Feature extraction method ('hog' or 'lbp')
        cv_strategy: Cross-validation strategy
            - "kfold": Session-Aware K-Fold (default, faster)
            - "loso": Leave-One-Session-Out (more comprehensive)
        n_splits: Number of folds (only used for kfold strategy)
        C: SVM regularization parameter
        n_components: PCA components (int for number, float for variance ratio, None for no PCA)
        use_cache: Whether to use feature caching
        **feature_kwargs: Arguments for feature extraction

    Returns:
        List of CV results (dict with fold info, metrics, and models)
    """
    features = extract_features_cached(images, method, use_cache, **feature_kwargs)

    if cv_strategy.lower() == "kfold":
        cv_splits = k_fold(filenames, labels, n_splits)
    elif cv_strategy.lower() == "loso":
        cv_splits = loso(filenames, labels)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}. Use 'kfold' or 'loso'")

    results = []

    print(
        f"\n=== {method.upper()} Cross-Validation ({cv_strategy.upper()}, "
        f"{len(cv_splits)} folds) ==="
    )
    if n_components is not None:
        print(f"  Using PCA with n_components={n_components}")

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold + 1}/{len(cv_splits)}:")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print(f"  Skipping fold - only one class present")
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Apply PCA if requested
        pca = None
        if n_components is not None:
            pca = PCA(n_components=n_components)
            X_train_transformed = pca.fit_transform(X_train_scaled)
            X_val_transformed = pca.transform(X_val_scaled)
            print(f"  Feature shape after PCA: {X_train_transformed.shape}")
        else:
            X_train_transformed = X_train_scaled
            X_val_transformed = X_val_scaled

        print(f"  Feature shape: {X_train_transformed.shape}")

        svm = train_svm(X_train_transformed, y_train, C)
        result = evaluate_svm(svm, X_val_transformed, y_val)

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
                "pca": pca,
                "features": features,
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
        )

    accuracies = [r["accuracy"] for r in results]
    aucs = [r["auc"] for r in results]
    print("\n=== Summary ===")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    return results


def train_final_model(
    images, labels, method="hog", C=1.0, n_components=None, use_cache=True, **feature_kwargs
):
    """
    Train final model on all training data.

    Args:
        images, labels: Full training dataset
        method: Feature extraction method
        C: SVM regularization
        n_components: PCA components (int for number, float for variance ratio, None for no PCA)
        use_cache: Whether to use feature caching
        **feature_kwargs: Arguments for feature extraction

    Returns:
        tuple: (svm, scaler, pca, feature_size)
    """
    print(f"\n=== Training final {method.upper()} model on all data ===")
    if n_components is not None:
        print(f"  Using PCA with n_components={n_components}")

    X = extract_features_cached(images, method, use_cache, **feature_kwargs)
    print(f"Feature shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = None
    if n_components is not None:
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X_scaled)
        print(f"Feature shape after PCA: {X_transformed.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_transformed = X_scaled

    svm = train_svm(X_transformed, labels, C)

    return svm, scaler, pca, X_transformed.shape[1]


def predict_on_dev(svm, scaler, method, pca=None, use_cache=True, **feature_kwargs):
    """
    Generate predictions on dev set, compute EER threshold, and save results.

    Args:
        svm, scaler: Trained model and scaler
        method: Feature extraction method
        pca: Fitted PCA object (or None if not using PCA)
        use_cache: Whether to use feature caching
        **feature_kwargs: Arguments for feature extraction

    Returns:
        dict with predictions, scores, and optimal threshold
    """
    base_dir = PROJECT_ROOT / "dataset"

    print(f"\nEvaluating on dev set with {method.upper()}...")

    dev_images, dev_filenames = load_images(os.path.join(base_dir, "target"))
    non_target_dev, non_target_fnames = load_images(
        os.path.join(base_dir, "non-target")
    )

    dev_all = np.concatenate([dev_images, non_target_dev])
    dev_fnames_all = dev_filenames + non_target_fnames
    dev_labels = np.array([1] * len(dev_images) + [0] * len(non_target_dev))

    dev_features = extract_features_cached(dev_all, method, use_cache, **feature_kwargs)
    dev_features_scaled = scaler.transform(dev_features)

    if pca is not None:
        dev_features_transformed = pca.transform(dev_features_scaled)
    else:
        dev_features_transformed = dev_features_scaled

    result = evaluate_svm(svm, dev_features_transformed, dev_labels)

    optimal_threshold, eer = compute_eer_threshold(
        dev_labels, result["calibrated_scores"]
    )
    print(f"  Dev EER: {eer:.4f}, Optimal Threshold: {optimal_threshold:.4f}")

    print(f"  Dev Accuracy: {result['accuracy']:.4f}")
    print(f"  Dev AUC: {result['auc']:.4f}")

    output_dir = PROJECT_ROOT / "results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"image_{method}.txt")
    save_predictions(
        output_file,
        dev_fnames_all,
        result["calibrated_scores"],
        threshold=optimal_threshold,
    )

    return {
        "filenames": dev_fnames_all,
        "labels": dev_labels,
        "scores": result["calibrated_scores"],
        "accuracy": result["accuracy"],
        "auc": result["auc"],
        "optimal_threshold": optimal_threshold,
        "eer": eer,
    }


def predict_on_eval(svm, scaler, optimal_threshold, method, pca=None, use_cache=True, **feature_kwargs):
    """
    Evaluate on eval set using EER threshold from dev.

    Args:
        svm, scaler: Trained model and scaler
        optimal_threshold: Threshold computed from dev set
        method: Feature extraction method
        pca: Fitted PCA object (or None if not using PCA)
        use_cache: Whether to use feature caching
        **feature_kwargs: Arguments for feature extraction

    Returns:
        dict with predictions and scores
    """
    base_dir = PROJECT_ROOT / "dataset"

    print(f"\nEvaluating on eval set with {method.upper()}...")

    print(f"  Using threshold from dev set: {optimal_threshold:.4f}")

    eval_dir = os.path.join(base_dir, "eval")
    if not os.path.exists(eval_dir):
        print("Eval directory not found. Skipping eval.")
        return None

    eval_images, eval_filenames = load_images(eval_dir)

    eval_features = extract_features_cached(eval_images, method, use_cache, **feature_kwargs)
    eval_features_scaled = scaler.transform(eval_features)

    if pca is not None:
        eval_features_transformed = pca.transform(eval_features_scaled)
    else:
        eval_features_transformed = eval_features_scaled

    eval_scores_raw = svm.decision_function(eval_features_transformed)

    score_min, score_max = eval_scores_raw.min(), eval_scores_raw.max()
    if score_max - score_min > 1e-10:
        eval_calibrated = (eval_scores_raw - score_min) / (score_max - score_min)
    else:
        eval_calibrated = np.ones_like(eval_scores_raw) * 0.5

    predictions = (eval_calibrated > optimal_threshold).astype(int)

    output_dir = PROJECT_ROOT / "results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"image_{method}_eval.txt")
    save_predictions(output_file, eval_filenames, eval_calibrated, threshold=optimal_threshold)

    print(f"  Eval predictions saved to {output_file}")

    return {
        "filenames": eval_filenames,
        "scores": eval_calibrated,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold,
    }


def main(mode="train", cv_strategy="kfold", n_splits=2, n_components=None):
    """
    Main training pipeline.

    Args:
        mode: One of "train" (cross-validation), "dev" (train on all dev data),
             or "eval" (train on dev and predict on eval)
        cv_strategy: Cross-validation strategy ("kfold" or "loso")
        n_splits: Number of folds for kfold strategy
        n_components: PCA components (int for number, float for variance ratio, None for no PCA)
    """
    base_dir = str(PROJECT_ROOT / "dataset")

    print("Loading dataset...")
    images, labels, filenames = load_dataset(base_dir, use_augmented=True)
    print(
        f"Total: {len(images)} images, {sum(labels)} target, {len(labels) - sum(labels)} non-target"
    )

    hog_params = {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "normalize": True,
    }

    lbp_params = {
        "radius": 1,
        "neighbors": 8,
        "use_uniform": True,
        "grid_size": (4, 4),
        "normalize": True,
    }

    if mode == "train":
        print("\n" + "=" * 50)
        hog_results = cross_validate(
            images,
            labels,
            filenames,
            method="hog",
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            C=1.0,
            n_components=n_components,
            **hog_params,
        )

        print("\n" + "=" * 50)
        lbp_results = cross_validate(
            images,
            labels,
            filenames,
            method="lbp",
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            C=1.0,
            n_components=n_components,
            **lbp_params,
        )
    elif mode == "dev":
        print("\n" + "=" * 50)
        hog_svm, hog_scaler, hog_pca, hog_dim = train_final_model(
            images, labels, method="hog", C=1.0, n_components=n_components, **hog_params
        )
        print(f"HOG feature dimension: {hog_dim}")

        print("\n" + "=" * 50)
        lbp_svm, lbp_scaler, lbp_pca, lbp_dim = train_final_model(
            images, labels, method="lbp", C=1.0, n_components=n_components, **lbp_params
        )
        print(f"LBP feature dimension: {lbp_dim}")

        hog_dev = predict_on_dev(hog_svm, hog_scaler, "hog", pca=hog_pca, **hog_params)
        lbp_dev = predict_on_dev(lbp_svm, lbp_scaler, "lbp", pca=lbp_pca, **lbp_params)

        print("\n" + "=" * 50)
        print("Final Results:")
        print(
            f"  HOG - Dev Accuracy: {hog_dev['accuracy']:.4f}, AUC: {hog_dev['auc']:.4f}"
        )
        print(
            f"  LBP - Dev Accuracy: {lbp_dev['accuracy']:.4f}, AUC: {lbp_dev['auc']:.4f}"
        )
        print(f"\nResults saved to {PROJECT_ROOT / 'results'}")
    elif mode == "eval":
        print("\n" + "=" * 50)
        hog_svm, hog_scaler, hog_pca, hog_dim = train_final_model(
            images, labels, method="hog", C=1.0, n_components=n_components, **hog_params
        )
        print(f"HOG feature dimension: {hog_dim}")

        print("\n" + "=" * 50)
        lbp_svm, lbp_scaler, lbp_pca, lbp_dim = train_final_model(
            images, labels, method="lbp", C=1.0, n_components=n_components, **lbp_params
        )
        print(f"LBP feature dimension: {lbp_dim}")

        hog_dev = predict_on_dev(hog_svm, hog_scaler, 'hog', pca=hog_pca, **hog_params)
        hog_threshold = hog_dev['optimal_threshold']

        lbp_dev = predict_on_dev(lbp_svm, lbp_scaler, 'lbp', pca=lbp_pca, **lbp_params)
        lbp_threshold = lbp_dev['optimal_threshold']

        hog_eval = predict_on_eval(hog_svm, hog_scaler, hog_threshold, "hog", pca=hog_pca, **hog_params)
        lbp_eval = predict_on_eval(lbp_svm, lbp_scaler, lbp_threshold, "lbp", pca=lbp_pca, **lbp_params)

        print("\n" + "=" * 50)
        print("Final Results (Eval):")
        print(
            f"  HOG predictions saved to {PROJECT_ROOT / 'results' / 'image_hog_eval.txt'}"
        )
        print(
            f"  LBP predictions saved to {PROJECT_ROOT / 'results' / 'image_lbp_eval.txt'}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SVM-based image person detector with session-aware cross-validation"
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        choices=["kfold", "loso"],
        default="kfold",
        help="Cross-validation strategy to use (default: kfold)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=2,
        help="Number of folds for kfold strategy (default: 2)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "dev", "eval"],
        default="train",
        help="train: cross-validation, dev: train on all dev data and evaluate on dev, eval: train on dev and predict on eval",
    )
    parser.add_argument(
        "--pca-components",
        type=str,
        default=None,
        help="PCA components: int (e.g., 150) or float 0-1 for variance (e.g., 0.95). None for no PCA.",
    )

    args = parser.parse_args()

    n_components = None
    if args.pca_components is not None:
        try:
            n_components = int(args.pca_components)
        except ValueError:
            n_components = float(args.pca_components)

    main(mode=args.mode, cv_strategy=args.cv_strategy, n_splits=args.n_splits, n_components=n_components)
