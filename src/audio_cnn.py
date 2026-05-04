"""
CNN-based audio person detector using MFCC features.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from extractors.mfcc_extractor import MFCCExtractor, load_audio_dataset
from session_cv import k_fold, loso
from spec_augment import SpecAugment
from utils import compute_eer_threshold, save_predictions

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_threshold_cache_path(cv_strategy, n_splits):
    """Generate cache file path for CV mean threshold."""
    filename = f"threshold_cnn_{cv_strategy}_{n_splits}.npy"
    return os.path.join(CACHE_DIR, filename)


def save_cv_threshold(threshold, cv_strategy, n_splits):
    """Save mean CV threshold to cache."""
    ensure_cache_dir()
    cache_path = get_threshold_cache_path(cv_strategy, n_splits)
    np.save(cache_path, np.array([threshold]))
    print(f"  Saved CV mean threshold to {cache_path}")


def load_cv_threshold(cv_strategy, n_splits):
    """
    Load mean CV threshold from cache.

    Returns:
        Threshold value or None if not found
    """
    cache_path = get_threshold_cache_path(cv_strategy, n_splits)
    if os.path.exists(cache_path):
        threshold = np.load(cache_path)[0]
        print(f"  Loaded CV mean threshold from {cache_path}: {threshold:.4f}")
        return threshold
    return None


def get_model_cache_path(model_type, cv_strategy=None, n_splits=None, fold=None):
    """Generate cache file path for models."""
    if cv_strategy and n_splits:
        if fold is not None:
            filename = (
                f"model_cnn_{model_type}_fold{fold}_{cv_strategy}_{n_splits}.joblib"
            )
        else:
            filename = f"model_cnn_{model_type}_{cv_strategy}_{n_splits}.joblib"
    else:
        filename = f"model_cnn_{model_type}_dev.joblib"
    return os.path.join(CACHE_DIR, filename)


def save_model_cache(model, model_type, cv_strategy=None, n_splits=None, fold=None):
    """Save model to cache."""
    ensure_cache_dir()
    cache_path = get_model_cache_path(model_type, cv_strategy, n_splits, fold)
    joblib.dump(model, cache_path)
    print(f"  Saved {model_type} to {cache_path}")


def load_model_cache(model_type, cv_strategy=None, n_splits=None, fold=None):
    """Load model from cache."""
    cache_path = get_model_cache_path(model_type, cv_strategy, n_splits, fold)
    if os.path.exists(cache_path):
        model = joblib.load(cache_path)
        print(f"  Loaded {model_type} from {cache_path}")
        return model
    return None


class AudioDataset(Dataset):
    """Dataset for audio MFCC features."""

    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx]
        if self.transform:
            feat = self.transform(feat)
        return feat, self.labels[idx]


class ShallowCNN(nn.Module):
    """Shallow 2D CNN with ReLU, BatchNorm, and Dropout."""

    def __init__(self, n_channels=3, n_mfcc=40, dropout_conv=0.3, dropout_fc=0.5):
        """
        Initialize shallow CNN.

        Args:
            n_channels: Number of input channels (MFCC + delta + delta2)
            n_mfcc: Number of MFCC coefficients (feature height)
            dropout_conv: Dropout rate after conv layers
            dropout_fc: Dropout rate after FC layers
        """
        super(ShallowCNN, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_conv)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout_conv)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout_conv)

        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.fc_dropout = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = self.pool_final(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x.squeeze(1)


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing for binary classification."""

    def __init__(self, pos_weight, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        if self.smoothing > 0:
            smoothed_targets = (
                targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
            )
            return self.criterion(inputs, smoothed_targets)
        return self.criterion(inputs, targets)


class CNNTrainer:
    """Trainer for CNN audio detector."""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def train(
        self,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=1e-2,
        patience=10,
    ):
        """
        Train the CNN model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience

        Returns:
            dict with training history
        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            val_loss, val_auc = self.evaluate(val_loader, criterion)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val AUC={val_auc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "/tmp/best_cnn_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.model.load_state_dict(torch.load("/tmp/best_cnn_model.pth"))
        return history

    def evaluate(self, data_loader, criterion=None):
        """Evaluate model on data loader."""
        self.model.eval()
        total_loss = 0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)

                if criterion:
                    loss = criterion(outputs, batch_labels)
                    total_loss += loss.item()

                scores = torch.sigmoid(outputs)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader) if criterion else 0

        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_scores)
        else:
            auc = 0.5

        return avg_loss, auc

    def predict(self, data_loader):
        """Get predictions for data loader."""
        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for batch_features, _ in data_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                scores = torch.sigmoid(outputs)
                all_scores.extend(scores.cpu().numpy())

        return np.array(all_scores)


def cross_validate(
    features,
    labels,
    filenames,
    cv_strategy="loso",
    n_splits=2,
    epochs=50,
    spec_augment=None,
):
    """
    Perform session-based cross-validation with CNN.

    Args:
        features: MFCC features array
        labels: Labels array
        filenames: List of filenames
        cv_strategy: Cross-validation strategy to use
            - "kfold": Session-Aware K-Fold (faster)
            - "loso": Leave-One-Session-Out (more comprehensive)
        n_splits: Number of folds (only used for kfold strategy)
        epochs: Training epochs per fold

    Returns:
        list of CV results (dict with fold info, metrics, and predictions)
    """
    if cv_strategy.lower() == "kfold":
        cv_splits = k_fold(filenames, labels, n_splits)
    elif cv_strategy.lower() == "loso":
        cv_splits = loso(filenames, labels)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}. Use 'kfold' or 'loso'")

    results = []
    fold_thresholds = []

    print(
        f"\n=== CNN Cross-Validation ({cv_strategy.upper()}, "
        f"{len(cv_splits)} folds) ==="
    )

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold + 1}/{len(cv_splits)}:")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_dataset = AudioDataset(X_train, y_train, transform=spec_augment)
        val_dataset = AudioDataset(X_val, y_val)

        use_cuda = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=use_cuda,
            num_workers=4 if use_cuda else 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=use_cuda,
            num_workers=4 if use_cuda else 0,
        )

        model = ShallowCNN(n_channels=features.shape[1], n_mfcc=features.shape[2])
        trainer = CNNTrainer(model)

        history = trainer.train(train_loader, val_loader, epochs=epochs)

        val_loss, val_auc = trainer.evaluate(val_loader)
        scores = trainer.predict(val_loader)

        # Compute EER threshold for this fold
        fold_threshold, fold_eer = compute_eer_threshold(y_val, scores)
        fold_thresholds.append(fold_threshold)

        fold_predictions = (scores > fold_threshold).astype(int)
        fold_acc = accuracy_score(y_val, fold_predictions)

        # Save fold model to cache
        save_model_cache(model.state_dict(), "audio", cv_strategy, n_splits, fold=fold)

        results.append(
            {
                "fold": fold,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "val_accuracy": fold_acc,
                "scores": scores,
                "model": model,
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
        )

        print(f"  Validation Accuracy: {fold_acc:.4f}")
        print(f"  Validation AUC: {val_auc:.4f}")
        print(f"  Fold EER: {fold_eer:.4f}, Threshold: {fold_threshold:.4f}")

    aucs = [r["val_auc"] for r in results]
    accs = [r["val_accuracy"] for r in results]
    print("\n=== Summary ===")
    print(f"Mean Accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    if fold_thresholds:
        mean_threshold = np.mean(fold_thresholds)
        print(f"\nMean CV Threshold: {mean_threshold:.4f}")
        save_cv_threshold(mean_threshold, cv_strategy, n_splits)

    return results


def train_final_model(features, labels, epochs=100, spec_augment=None):
    """
    Train final CNN model on all data.

    Args:
        features: All training features
        labels: All training labels
        epochs: Training epochs
        spec_augment: Optional SpecAugment transform for training

    Returns:
        Trained model
    """
    print("\n=== Training Final CNN Model ===")

    dataset = AudioDataset(features, labels, transform=spec_augment)
    use_cuda = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    model = ShallowCNN(n_channels=features.shape[1], n_mfcc=features.shape[2])
    trainer = CNNTrainer(model)

    dummy_loader = DataLoader(
        dataset, batch_size=64, pin_memory=use_cuda, num_workers=4 if use_cuda else 0
    )
    history = trainer.train(loader, dummy_loader, epochs=epochs)

    # Save model to cache
    save_model_cache(model.state_dict(), "audio")

    return model, trainer


def predict_on_dev(model, trainer, base_dir=None, cv_strategy="kfold", n_splits=2):
    """
    Generate predictions on dev set using mean CV threshold and save results.

    Args:
        model: Trained CNN model
        trainer: CNNTrainer instance
        base_dir: Dataset base directory (default: PROJECT_ROOT / "dataset")
        cv_strategy: CV strategy used (for loading threshold)
        n_splits: Number of splits used (for loading threshold)

    Returns:
        dict with predictions and optimal threshold
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "dataset"
    else:
        base_dir = Path(base_dir)

    print("\nEvaluating on dev set...")

    extractor = MFCCExtractor()

    target_dev_dir = base_dir / "target"
    if not target_dev_dir.exists():
        target_dev_dir = base_dir / "target_dev"

    non_target_dev_dir = base_dir / "non-target"
    if not non_target_dev_dir.exists():
        non_target_dev_dir = base_dir / "non_target_dev"

    target_paths, target_fnames = _get_audio_files(str(target_dev_dir))
    non_target_paths, non_target_fnames = _get_audio_files(str(non_target_dev_dir))

    all_paths = target_paths + non_target_paths
    all_fnames = target_fnames + non_target_fnames
    all_labels = [1] * len(target_paths) + [0] * len(non_target_paths)

    features = extractor.extract_batch(all_paths, use_cache=True)

    dataset = AudioDataset(features, all_labels)
    use_cuda = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    scores = trainer.predict(loader)

    # Use mean CV threshold if available, otherwise default to 0.5
    optimal_threshold = load_cv_threshold(cv_strategy, n_splits)
    threshold_source = "CV cache"

    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5
        threshold_source = "default (0.5)"
    else:
        print(f"  Using threshold from {threshold_source}: {optimal_threshold:.4f}")

    dev_eer, _ = compute_eer_threshold(all_labels, scores)

    print(
        f"  Dev EER: {dev_eer:.4f}, Using Threshold: {optimal_threshold:.4f} (from {threshold_source})"
    )

    auc = roc_auc_score(all_labels, scores)
    predictions = (scores > optimal_threshold).astype(int)
    acc = accuracy_score(all_labels, predictions)

    print(f"  Dev Accuracy: {acc:.4f}")
    print(f"  Dev AUC: {auc:.4f}")

    output_dir = PROJECT_ROOT / "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / "audio_cnn.txt"
    save_predictions(str(output_file), all_fnames, scores, threshold=optimal_threshold)

    return {
        "filenames": all_fnames,
        "labels": all_labels,
        "scores": scores,
        "accuracy": acc,
        "auc": auc,
        "optimal_threshold": optimal_threshold,
        "eer": dev_eer,
    }


def _get_audio_files(directory):
    """Get all WAV files in directory with their filenames."""
    directory = Path(directory)
    files = sorted(directory.glob("*.wav"))
    paths = [str(f) for f in files]
    fnames = [f.stem for f in files]
    return paths, fnames


def predict_on_eval(model, trainer, base_dir=None, cv_strategy="kfold", n_splits=2):
    """
    Evaluate on eval set using mean CV threshold.

    Args:
        model: Trained CNN model
        trainer: CNNTrainer instance
        base_dir: Dataset base directory (default: PROJECT_ROOT / "dataset")
        cv_strategy: CV strategy used (for loading threshold)
        n_splits: Number of splits used (for loading threshold)

    Returns:
        dict with predictions
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "dataset"
    else:
        base_dir = Path(base_dir)

    print("\nEvaluating on eval set...")

    extractor = MFCCExtractor()

    # Use mean CV threshold if available, otherwise default to 0.5
    optimal_threshold = load_cv_threshold(cv_strategy, n_splits)
    threshold_source = "CV cache"

    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5
        threshold_source = "default (0.5)"
    else:
        print(f"  Using threshold from {threshold_source}: {optimal_threshold:.4f}")

    # Predict on eval set
    eval_dir = base_dir / "eval"
    if not eval_dir.exists():
        print("Eval directory not found. Skipping eval.")
        return None

    eval_paths, eval_fnames = _get_audio_files(str(eval_dir))
    eval_features = extractor.extract_batch(eval_paths, use_cache=True)

    # Dummy labels since eval set has no ground truth
    eval_dataset = AudioDataset(eval_features, [0] * len(eval_paths))
    use_cuda = torch.cuda.is_available()
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    scores = trainer.predict(eval_loader)
    predictions = (scores > optimal_threshold).astype(int)

    output_dir = PROJECT_ROOT / "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / "audio_cnn_eval.txt"
    save_predictions(str(output_file), eval_fnames, scores, threshold=optimal_threshold)

    print(f"  Eval predictions saved to {output_file}")

    return {
        "filenames": eval_fnames,
        "scores": scores,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold,
    }


def predict_on_eval_cv(
    features,
    labels,
    filenames,
    cv_strategy="kfold",
    n_splits=2,
    epochs=50,
    spec_augment=None,
):
    """
    Evaluate on eval set using ensemble of CV-trained models.
    """
    base_dir = PROJECT_ROOT / "dataset"

    print("\nEvaluating on eval set with CNN (CV Ensemble)...")

    # Load CV models from cache
    cv_models = []
    for fold in range(n_splits if cv_strategy == "kfold" else 10):
        model_state = load_model_cache("audio", cv_strategy, n_splits, fold)
        if model_state is not None:
            model = ShallowCNN(n_channels=features.shape[1], n_mfcc=features.shape[2])
            model.load_state_dict(model_state)
            cv_models.append(model)
        else:
            print(f"  Warning: Could not load model for fold {fold}, skipping...")

    if not cv_models:
        print("  No CV models found in cache. Run 'train' mode first.")
        return None

    print(f"  Loaded {len(cv_models)} CV models")

    # Use mean CV threshold
    optimal_threshold = load_cv_threshold(cv_strategy, n_splits)
    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5

    eval_dir = base_dir / "eval"
    if not eval_dir.exists():
        print("Eval directory not found. Skipping eval.")
        return None

    extractor = MFCCExtractor()
    eval_paths, eval_fnames = _get_audio_files(str(eval_dir))
    eval_features = extractor.extract_batch(eval_paths, use_cache=True)

    # Dummy labels since eval set has no ground truth
    eval_dataset = AudioDataset(eval_features, [0] * len(eval_paths))
    use_cuda = torch.cuda.is_available()
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    # Ensemble: average predictions from all CV models
    all_scores = []
    for model in cv_models:
        trainer = CNNTrainer(model)
        scores = trainer.predict(eval_loader)
        all_scores.append(scores)

    # Average ensemble scores
    scores = np.mean(all_scores, axis=0)
    predictions = (scores > optimal_threshold).astype(int)

    output_dir = PROJECT_ROOT / "results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir / "audio_cnn_eval_cv.txt"
    save_predictions(str(output_file), eval_fnames, scores, threshold=optimal_threshold)

    print(f"  Eval-CV predictions saved to {output_file}")

    return {
        "filenames": eval_fnames,
        "scores": scores,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold,
    }


def main(mode="train", cv_strategy="kfold", n_splits=2, spec_augment=None):
    """
    Main training pipeline.

    Args:
        mode: One of "train" (cross-validation with caching), "dev" (train on all dev data with caching),
             "eval-dev" (use dev-trained model for eval), or "eval-cv" (use ensemble of CV-trained models)
        cv_strategy: Cross-validation strategy ("kfold" or "loso")
        n_splits: Number of folds for kfold strategy
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for GPU acceleration")

    base_dir = str(PROJECT_ROOT / "dataset")

    print("Loading audio dataset with MFCC features...")
    features, labels, filenames = load_audio_dataset(base_dir, use_augmented=True)
    print(
        f"Total: {len(features)} samples, "
        f"Target: {sum(labels)}, "
        f"Non-target: {len(labels) - sum(labels)}"
    )
    print(f"Feature shape: {features.shape}")

    if mode == "train":
        cnn_results = cross_validate(
            features,
            labels,
            filenames,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            epochs=10,
            spec_augment=spec_augment,
        )

    elif mode == "dev":
        print("\n" + "=" * 50)
        model, trainer = train_final_model(
            features, labels, epochs=20, spec_augment=spec_augment
        )

        dev_results = predict_on_dev(
            model, trainer, base_dir, cv_strategy=cv_strategy, n_splits=n_splits
        )

        print("\n" + "=" * 50)
        print("Final Results:")
        print(f"  Dev Accuracy: {dev_results['accuracy']:.4f}")
        print(f"  Dev AUC: {dev_results['auc']:.4f}")
        print(f"\nResults saved to {PROJECT_ROOT / 'results' / 'audio_cnn.txt'}")

    elif mode == "eval-dev":
        print("\n" + "=" * 50)
        # Try to load dev-trained model from cache
        model_state = load_model_cache("audio")
        if model_state is not None:
            model = ShallowCNN(n_channels=features.shape[1], n_mfcc=features.shape[2])
            model.load_state_dict(model_state)
            trainer = CNNTrainer(model)
        else:
            print("Dev model not found in cache. Training new model...")
            model, trainer = train_final_model(
                features, labels, epochs=20, spec_augment=spec_augment
            )

        dev_results = predict_on_dev(
            model, trainer, base_dir, cv_strategy=cv_strategy, n_splits=n_splits
        )

        eval_results = predict_on_eval(
            model, trainer, base_dir, cv_strategy=cv_strategy, n_splits=n_splits
        )

        print("\n" + "=" * 50)
        print("Final Results (Eval-Dev):")
        print(
            f"  Eval predictions saved to {PROJECT_ROOT / 'results' / 'audio_cnn_eval.txt'}"
        )

    elif mode == "eval-cv":
        print("\n" + "=" * 50)
        eval_results = predict_on_eval_cv(
            features,
            labels,
            filenames,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            epochs=20,
            spec_augment=spec_augment,
        )

        print("\n" + "=" * 50)
        print("Final Results (Eval-CV):")
        print(
            f"  Eval predictions saved to {PROJECT_ROOT / 'results' / 'audio_cnn_eval_cv.txt'}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        choices=["train", "dev", "eval-dev", "eval-cv"],
        default="train",
        help="train: cross-validation with model caching, dev: train on all dev data with model caching and evaluate on dev, eval-dev: use dev-trained model for eval, eval-cv: use ensemble of CV-trained models for eval",
    )
    parser.add_argument(
        "--spec-aug",
        action="store_true",
        help="Enable SpecAugment (frequency + time masking)",
    )
    parser.add_argument(
        "--freq-mask",
        type=int,
        default=10,
        help="Maximum frequency mask width in MFCC bins (default: 10)",
    )
    parser.add_argument(
        "--time-mask",
        type=int,
        default=40,
        help="Maximum time mask width in time steps (default: 40)",
    )
    parser.add_argument(
        "--num-freq-masks",
        type=int,
        default=2,
        help="Number of frequency masks per sample (default: 2)",
    )
    parser.add_argument(
        "--num-time-masks",
        type=int,
        default=2,
        help="Number of time masks per sample (default: 2)",
    )

    args = parser.parse_args()

    spec_aug = None
    if args.spec_aug:
        spec_aug = SpecAugment(
            freq_mask_param=args.freq_mask,
            time_mask_param=args.time_mask,
            num_freq_masks=args.num_freq_masks,
            num_time_masks=args.num_time_masks,
        )
        print(
            f"SpecAugment enabled: freq_mask={args.freq_mask}, "
            f"time_mask={args.time_mask}, "
            f"num_freq_masks={args.num_freq_masks}, "
            f"num_time_masks={args.num_time_masks}"
        )

    main(
        mode=args.mode,
        cv_strategy=args.cv_strategy,
        n_splits=args.n_splits,
        spec_augment=spec_aug,
    )
