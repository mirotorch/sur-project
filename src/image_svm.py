"""
Image-based person detector using HOG and LBP features with SVM.
Supports combined HOG+LBP features with PCA dimensionality reduction.
"""

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
from utils import (compute_dataset_hash, compute_eer_threshold,
                   load_cv_threshold, load_dataset, load_images,
                   load_model_cache, save_cv_threshold, save_model_cache,
                   save_predictions)

PROJECT_ROOT = Path(__file__).parent.parent


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
    from utils import get_feature_cache_path

    params_str = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
    dataset_hash = compute_dataset_hash(images)
    cache_path = get_feature_cache_path(method, params_str, dataset_hash)

    if use_cache and Path(cache_path).exists():
        print(f"Loading cached {method} features from {cache_path}")
        return np.load(cache_path)

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

    if use_cache:
        np.save(cache_path, features)
        print(f"  Cached to {cache_path}")

    return features


def extract_combined_features_cached(
    images, use_cache=True, hog_kwargs=None, lbp_kwargs=None
):
    """
    Extract HOG and LBP features and concatenate them.

    Args:
        images: Batch of images
        use_cache: Whether to use cached features
        hog_kwargs: Parameters for HOG extraction
        lbp_kwargs: Parameters for LBP extraction

    Returns:
        Combined feature matrix (n_samples, n_hog + n_lbp)
    """
    if hog_kwargs is None:
        hog_kwargs = {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "normalize": True,
        }
    if lbp_kwargs is None:
        lbp_kwargs = {
            "radius": 1,
            "neighbors": 8,
            "use_uniform": True,
            "grid_size": (4, 4),
            "normalize": True,
        }

    hog_features = extract_features_cached(
        images, method="hog", use_cache=use_cache, **hog_kwargs
    )
    lbp_features = extract_features_cached(
        images, method="lbp", use_cache=use_cache, **lbp_kwargs
    )

    combined = np.concatenate([hog_features, lbp_features], axis=1)
    print(f"  Combined features shape: {combined.shape}")

    return combined


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
        dict with accuracy, auc, scores, and calibrated scores
    """
    scores = svm.decision_function(X_test)

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
    hog_kwargs=None,
    lbp_kwargs=None,
    **feature_kwargs,
):
    """
    Perform session-based cross-validation with SVM classifiers.

    Args:
        images: Image array
        labels: Labels array
        filenames: List of filenames
        method: Feature extraction method ('hog', 'lbp', or 'combined')
        cv_strategy: Cross-validation strategy ('kfold' or 'loso')
        n_splits: Number of folds (only used for kfold strategy)
        C: SVM regularization parameter
        n_components: PCA components (int for number, float for variance ratio, None for no PCA)
        use_cache: Whether to use feature caching
        hog_kwargs: Parameters for HOG extraction (used when method='combined')
        lbp_kwargs: Parameters for LBP extraction (used when method='combined')
        **feature_kwargs: Arguments for feature extraction (for single method)

    Returns:
        List of CV results with mean threshold saved to cache
    """
    if method == "combined":
        features = extract_combined_features_cached(
            images, use_cache=use_cache, hog_kwargs=hog_kwargs, lbp_kwargs=lbp_kwargs
        )
    else:
        features = extract_features_cached(images, method, use_cache, **feature_kwargs)

    if cv_strategy.lower() == "kfold":
        cv_splits = k_fold(filenames, labels, n_splits)
    elif cv_strategy.lower() == "loso":
        cv_splits = loso(filenames, labels)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}. Use 'kfold' or 'loso'")

    results = []
    fold_thresholds = []

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
            print("  Skipping fold - only one class present")
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

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

        # Compute EER threshold for this fold
        fold_threshold, fold_eer = compute_eer_threshold(
            y_val, result["calibrated_scores"]
        )
        fold_thresholds.append(fold_threshold)

        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  Fold EER: {fold_eer:.4f}, Threshold: {fold_threshold:.4f}")

        # Save fold models to cache
        # save_model_cache(svm, method, "svm", cv_strategy, n_splits, n_components, fold)
        # save_model_cache(
        #    scaler, method, "scaler", cv_strategy, n_splits, n_components, fold
        # )
        # if pca is not None:
        #   save_model_cache(
        #      pca, method, "pca", cv_strategy, n_splits, n_components, fold
        # )

        results.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "accuracy": result["accuracy"],
                "auc": result["auc"],
                "threshold": fold_threshold,
                "eer": fold_eer,
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

    if fold_thresholds:
        mean_threshold = np.mean(fold_thresholds)
        print(f"\nMean CV Threshold: {mean_threshold:.4f}")
        save_cv_threshold(mean_threshold, method, cv_strategy, n_splits, n_components)

    return results


def train_final_model(
    images,
    labels,
    method="hog",
    C=1.0,
    n_components=None,
    use_cache=True,
    hog_kwargs=None,
    lbp_kwargs=None,
    **feature_kwargs,
):
    """
    Train final model on all training data.

    Args:
        images, labels: Full training dataset
        method: Feature extraction method ('hog', 'lbp', or 'combined')
        C: SVM regularization
        n_components: PCA components (int for number, float for variance ratio, None for no PCA)
        use_cache: Whether to use feature caching
        hog_kwargs: Parameters for HOG extraction (used when method='combined')
        lbp_kwargs: Parameters for LBP extraction (used when method='combined')
        **feature_kwargs: Arguments for feature extraction (for single method)

    Returns:
        tuple: (svm, scaler, pca, feature_size)
    """
    print(f"\n=== Training final {method} model on all data ===")
    if n_components is not None:
        print(f"  Using PCA with n_components={n_components}")

    if method == "combined":
        X = extract_combined_features_cached(
            images, use_cache=use_cache, hog_kwargs=hog_kwargs, lbp_kwargs=lbp_kwargs
        )
    else:
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

    # Save models to cache
    save_model_cache(svm, method, "svm")
    save_model_cache(scaler, method, "scaler")
    if pca is not None:
        save_model_cache(pca, method, "pca")

    return svm, scaler, pca, X_transformed.shape[1]


def predict_on_dev(
    svm,
    scaler,
    method,
    pca=None,
    use_cache=True,
    hog_kwargs=None,
    lbp_kwargs=None,
    cv_strategy="kfold",
    n_splits=2,
    n_components=None,
    **feature_kwargs,
):
    """
    Generate predictions on dev set using mean CV threshold and save results.

    Args:
        svm, scaler: Trained model and scaler
        method: Feature extraction method ('hog', 'lbp', or 'combined')
        pca: Fitted PCA object (or None if not using PCA)
        use_cache: Whether to use feature caching
        hog_kwargs: Parameters for HOG extraction (used when method='combined')
        lbp_kwargs: Parameters for LBP extraction (used when method='combined')
        cv_strategy: CV strategy used (for loading threshold)
        n_splits: Number of splits used (for loading threshold)
        n_components: PCA components used (for loading threshold)
        **feature_kwargs: Arguments for feature extraction (for single method)

    Returns:
        dict with predictions, scores, and threshold used
    """
    base_dir = PROJECT_ROOT / "dataset"

    print(f"\nEvaluating on dev set with {method.upper()}...")

    dev_images, dev_filenames = load_images(str(base_dir / "target"))
    non_target_dev, non_target_fnames = load_images(str(base_dir / "non-target"))

    dev_all = np.concatenate([dev_images, non_target_dev])
    dev_fnames_all = dev_filenames + non_target_fnames
    dev_labels = np.array([1] * len(dev_images) + [0] * len(non_target_dev))

    if method == "combined":
        dev_features = extract_combined_features_cached(
            dev_all, use_cache=use_cache, hog_kwargs=hog_kwargs, lbp_kwargs=lbp_kwargs
        )
    else:
        dev_features = extract_features_cached(
            dev_all, method, use_cache, **feature_kwargs
        )

    dev_features_scaled = scaler.transform(dev_features)

    if pca is not None:
        dev_features_transformed = pca.transform(dev_features_scaled)
    else:
        dev_features_transformed = dev_features_scaled

    result = evaluate_svm(svm, dev_features_transformed, dev_labels)

    # Use mean CV threshold if available, otherwise default to 0.5
    optimal_threshold = load_cv_threshold(method, cv_strategy, n_splits, n_components)
    threshold_source = "CV cache"

    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5
        threshold_source = "default (0.5)"
    else:
        print(f"  Using threshold from {threshold_source}: {optimal_threshold:.4f}")

    dev_eer, _ = compute_eer_threshold(dev_labels, result["calibrated_scores"])

    print(
        f"  Dev EER: {dev_eer:.4f}, Using Threshold: {optimal_threshold:.4f} (from {threshold_source})"
    )
    print(f"  Dev Accuracy: {result['accuracy']:.4f}")
    print(f"  Dev AUC: {result['auc']:.4f}")

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"image_{method}.txt"
    # save_predictions(
    #     str(output_file),
    #     dev_fnames_all,
    #     result["calibrated_scores"],
    #     threshold=optimal_threshold,
    # )

    return {
        "filenames": dev_fnames_all,
        "labels": dev_labels,
        "scores": result["calibrated_scores"],
        "accuracy": result["accuracy"],
        "auc": result["auc"],
        "optimal_threshold": optimal_threshold,
        "eer": dev_eer,
    }


def predict_on_eval(
    svm,
    scaler,
    method,
    pca=None,
    use_cache=True,
    hog_kwargs=None,
    lbp_kwargs=None,
    cv_strategy="kfold",
    n_splits=2,
    n_components=None,
    **feature_kwargs,
):
    """
    Evaluate on eval set using mean CV threshold.

    Args:
        svm, scaler: Trained model and scaler
        method: Feature extraction method ('hog', 'lbp', or 'combined')
        pca: Fitted PCA object (or None if not using PCA)
        use_cache: Whether to use feature caching
        hog_kwargs: Parameters for HOG extraction (used when method='combined')
        lbp_kwargs: Parameters for LBP extraction (used when method='combined')
        cv_strategy: CV strategy used (for loading threshold)
        n_splits: Number of splits used (for loading threshold)
        n_components: PCA components used (for loading threshold)
        **feature_kwargs: Arguments for feature extraction (for single method)

    Returns:
        dict with predictions, scores, and metrics (if labels available)
    """
    base_dir = PROJECT_ROOT / "dataset"

    print(f"\nEvaluating on eval set with {method.upper()}...")

    eval_dir = base_dir / "eval"
    if not eval_dir.exists():
        print("Eval directory not found. Skipping eval.")
        return None

    # Use mean CV threshold if available, otherwise default to 0.5
    optimal_threshold = load_cv_threshold(method, cv_strategy, n_splits, n_components)
    threshold_source = "CV cache"

    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5
        threshold_source = "default (0.5)"
    else:
        print(f"  Using threshold from {threshold_source}: {optimal_threshold:.4f}")

    eval_images, eval_filenames = load_images(str(eval_dir))

    if method == "combined":
        eval_features = extract_combined_features_cached(
            eval_images,
            use_cache=use_cache,
            hog_kwargs=hog_kwargs,
            lbp_kwargs=lbp_kwargs,
        )
    else:
        eval_features = extract_features_cached(
            eval_images, method, use_cache, **feature_kwargs
        )

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
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"image_{method}_eval.txt"
    save_predictions(
        str(output_file), eval_filenames, eval_calibrated, threshold=optimal_threshold
    )

    print(f"  Eval predictions saved to {output_file}")

    return {
        "filenames": eval_filenames,
        "scores": eval_calibrated,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold,
    }


def predict_on_eval_cv(
    images,
    labels,
    filenames,
    method="hog",
    cv_strategy="kfold",
    n_splits=2,
    C=1.0,
    n_components=None,
    use_cache=True,
    hog_kwargs=None,
    lbp_kwargs=None,
    **feature_kwargs,
):
    """
    Evaluate on eval set using ensemble of CV-trained models.

    Returns:
        dict with predictions, scores, and metrics
    """
    base_dir = PROJECT_ROOT / "dataset"

    print(f"\nEvaluating on eval set with {method.upper()} (CV Ensemble)...")

    # Load CV models from cache
    cv_models = []
    for fold in range(n_splits if cv_strategy == "kfold" else 10):
        svm = load_model_cache(method, "svm", cv_strategy, n_splits, n_components, fold)
        scaler = load_model_cache(
            method, "scaler", cv_strategy, n_splits, n_components, fold
        )
        pca = load_model_cache(method, "pca", cv_strategy, n_splits, n_components, fold)

        if svm is None or scaler is None:
            print(f"  Warning: Could not load models for fold {fold}, skipping...")
            continue

        cv_models.append({"svm": svm, "scaler": scaler, "pca": pca})

    if not cv_models:
        print("  No CV models found in cache. Run 'train' mode first.")
        return None

    print(f"  Loaded {len(cv_models)} CV models")

    # Use mean CV threshold
    optimal_threshold = load_cv_threshold(method, cv_strategy, n_splits, n_components)
    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5

    eval_dir = base_dir / "eval"
    if not eval_dir.exists():
        print("Eval directory not found. Skipping eval.")
        return None

    eval_images, eval_filenames = load_images(str(eval_dir))

    if method == "combined":
        eval_features = extract_combined_features_cached(
            eval_images,
            use_cache=use_cache,
            hog_kwargs=hog_kwargs,
            lbp_kwargs=lbp_kwargs,
        )
    else:
        eval_features = extract_features_cached(
            eval_images, method, use_cache, **feature_kwargs
        )

    # Ensemble: average predictions from all CV models
    all_scores = []
    for model_dict in cv_models:
        svm = model_dict["svm"]
        scaler = model_dict["scaler"]
        pca = model_dict["pca"]

        eval_scaled = scaler.transform(eval_features)
        if pca is not None:
            eval_transformed = pca.transform(eval_scaled)
        else:
            eval_transformed = eval_scaled

        scores = svm.decision_function(eval_transformed)
        all_scores.append(scores)

    # Average ensemble scores
    eval_scores_raw = np.mean(all_scores, axis=0)

    score_min, score_max = eval_scores_raw.min(), eval_scores_raw.max()
    if score_max - score_min > 1e-10:
        eval_calibrated = (eval_scores_raw - score_min) / (score_max - score_min)
    else:
        eval_calibrated = np.ones_like(eval_scores_raw) * 0.5

    predictions = (eval_calibrated > optimal_threshold).astype(int)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"image_{method}_eval_cv.txt"
    save_predictions(
        str(output_file), eval_filenames, eval_calibrated, threshold=optimal_threshold
    )

    print(f"  Eval-CV predictions saved to {output_file}")

    return {
        "filenames": eval_filenames,
        "scores": eval_calibrated,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold,
    }


def main(
    mode="train",
    cv_strategy="kfold",
    n_splits=2,
    n_components=None,
    method="hog",
    C=1.0,
):
    """
    Main training pipeline.

    Args:
        mode: One of "train" (cross-validation with caching), "dev" (train on all dev data with caching),
             "eval-dev" (use dev-trained model for eval), or "eval-cv" (use ensemble of CV-trained models)
        cv_strategy: Cross-validation strategy ("kfold" or "loso")
        n_splits: Number of folds for kfold strategy
        n_components: PCA components (int for number, float for variance ratio, None for no PCA)
        method: Feature extraction method ('hog', 'lbp', or 'combined')
        C: SVM regularization parameter
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
        if method == "combined":
            cross_validate(
                images,
                labels,
                filenames,
                method="combined",
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                C=C,
                n_components=n_components,
                use_cache=True,
                hog_kwargs=hog_params,
                lbp_kwargs=lbp_params,
            )
        elif method == "hog":
            cross_validate(
                images,
                labels,
                filenames,
                method="hog",
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                C=C,
                n_components=n_components,
                **hog_params,
            )
        elif method == "lbp":
            cross_validate(
                images,
                labels,
                filenames,
                method="lbp",
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                C=C,
                n_components=n_components,
                **lbp_params,
            )

    elif mode == "dev":
        print("\n" + "=" * 50)
        if method == "combined":
            svm, scaler, pca, feat_dim = train_final_model(
                images,
                labels,
                method="combined",
                C=C,
                n_components=n_components,
                hog_kwargs=hog_params,
                lbp_kwargs=lbp_params,
            )
        elif method == "hog":
            svm, scaler, pca, feat_dim = train_final_model(
                images,
                labels,
                method="hog",
                C=C,
                n_components=n_components,
                **hog_params,
            )
        elif method == "lbp":
            svm, scaler, pca, feat_dim = train_final_model(
                images,
                labels,
                method="lbp",
                C=C,
                n_components=n_components,
                **lbp_params,
            )

        print(f"{method.upper()} feature dimension: {feat_dim}")

        if method == "combined":
            dev_result = predict_on_dev(
                svm,
                scaler,
                method="combined",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                hog_kwargs=hog_params,
                lbp_kwargs=lbp_params,
            )
        elif method == "hog":
            dev_result = predict_on_dev(
                svm,
                scaler,
                method="hog",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                **hog_params,
            )
        elif method == "lbp":
            dev_result = predict_on_dev(
                svm,
                scaler,
                method="lbp",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                **lbp_params,
            )

        print("\n" + "=" * 50)
        print("Final Results:")
        print(
            f"  {method.upper()} - Dev Accuracy: {dev_result['accuracy']:.4f}, AUC: {dev_result['auc']:.4f}"
        )
        print(f"\nResults saved to {PROJECT_ROOT / 'results'}")

    elif mode == "eval":
        print("\n" + "=" * 50)
        # Try to load dev-trained model from cache
        svm = load_model_cache(method, "svm")
        scaler = load_model_cache(method, "scaler")
        pca = load_model_cache(method, "pca")

        if svm is None or scaler is None:
            print("Dev model not found in cache. Training new model...")
            if method == "combined":
                svm, scaler, pca, feat_dim = train_final_model(
                    images,
                    labels,
                    method="combined",
                    C=C,
                    n_components=n_components,
                    hog_kwargs=hog_params,
                    lbp_kwargs=lbp_params,
                )
            elif method == "hog":
                svm, scaler, pca, feat_dim = train_final_model(
                    images,
                    labels,
                    method="hog",
                    C=C,
                    n_components=n_components,
                    **hog_params,
                )
            elif method == "lbp":
                svm, scaler, pca, feat_dim = train_final_model(
                    images,
                    labels,
                    method="lbp",
                    C=C,
                    n_components=n_components,
                    **lbp_params,
                )
            print(f"{method.upper()} feature dimension: {feat_dim}")

        # Evaluate on dev
        if method == "combined":
            dev_result = predict_on_dev(
                svm,
                scaler,
                method="combined",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                hog_kwargs=hog_params,
                lbp_kwargs=lbp_params,
            )
        elif method == "hog":
            dev_result = predict_on_dev(
                svm,
                scaler,
                method="hog",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                **hog_params,
            )
        elif method == "lbp":
            dev_result = predict_on_dev(
                svm,
                scaler,
                method="lbp",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                **lbp_params,
            )

        # Predict on eval using dev model
        if method == "combined":
            eval_result = predict_on_eval(
                svm,
                scaler,
                method="combined",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                hog_kwargs=hog_params,
                lbp_kwargs=lbp_params,
            )
        elif method == "hog":
            eval_result = predict_on_eval(
                svm,
                scaler,
                method="hog",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                **hog_params,
            )
        elif method == "lbp":
            eval_result = predict_on_eval(
                svm,
                scaler,
                method="lbp",
                pca=pca,
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                n_components=n_components,
                **lbp_params,
            )

        print("\n" + "=" * 50)
        print("Final Results (Eval-Dev):")
        print(
            f"  {method.upper()} predictions saved to {PROJECT_ROOT / 'results' / f'image_{method}_eval.txt'}"
        )
    elif mode == "eval-cv":
        print("\n" + "=" * 50)
        # Evaluate on dev using CV ensemble
        if method == "combined":
            dev_result = predict_on_eval_cv(
                images,
                labels,
                filenames,
                method="combined",
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                C=C,
                n_components=n_components,
                hog_kwargs=hog_params,
                lbp_kwargs=lbp_params,
            )
        elif method == "hog":
            dev_result = predict_on_eval_cv(
                images,
                labels,
                filenames,
                method="hog",
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                C=C,
                n_components=n_components,
                **hog_params,
            )
        elif method == "lbp":
            dev_result = predict_on_eval_cv(
                images,
                labels,
                filenames,
                method="lbp",
                cv_strategy=cv_strategy,
                n_splits=n_splits,
                C=C,
                n_components=n_components,
                **lbp_params,
            )

        print("\n" + "=" * 50)
        print("Final Results (Eval-CV):")
        print(
            f"  {method.upper()} predictions saved to {PROJECT_ROOT / 'results' / f'image_{method}_eval_cv.txt'}"
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
        default="loso",
        help="Cross-validation strategy to use (default: loso)",
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
        help="train: cross-validation with model caching, dev: train on all dev data with model caching and evaluate on dev, eval: use dev-trained model for eval",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hog", "lbp", "combined"],
        default="combined",
        help="Feature extraction method (default: hog). 'combined' concatenates HOG+LBP with PCA",
    )
    parser.add_argument(
        "--pca-components",
        type=str,
        default=None,
        help="PCA components: int (e.g., 150) or float 0-1 for variance (e.g., 0.95). None for no PCA.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=0.01,
        help="SVM regularization parameter C (default: 0.01)",
    )

    args = parser.parse_args()

    n_components = None
    if args.pca_components is not None:
        try:
            n_components = int(args.pca_components)
        except ValueError:
            n_components = float(args.pca_components)

    main(
        mode=args.mode,
        cv_strategy=args.cv_strategy,
        n_splits=args.n_splits,
        n_components=n_components,
        method=args.method,
        C=args.C,
    )
