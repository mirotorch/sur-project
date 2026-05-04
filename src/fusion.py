"""
Fusion model combining audio CNN and image features (fusion).
Audio: Reuse CNN without final fc2 layer -> 64-dim.
Image: HOG/LBP/combined -> PCA -> MLP -> 64-dim.
Fusion: 128-dim (concat) -> classifier -> output.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from audio_cnn import ShallowCNN
from extractors.hog_extractor import extract_hog_batch
from extractors.lbp_extractor import extract_lbp_batch
from extractors.mfcc_extractor import MFCCExtractor, load_audio_dataset
from session_cv import k_fold, loso
from utils import (_get_audio_files, compute_dataset_hash,
                   compute_eer_threshold, load_cv_threshold, load_images,
                   load_model_cache, save_cv_threshold, save_model_cache,
                   save_predictions)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def get_feature_cache_path(method, params_str, dataset_hash):
    """Generate cache file path for features."""
    filename = f"fusion_{method}_{params_str}_{dataset_hash}.npy"
    return str(CACHE_DIR / filename)


def extract_image_features_cached(images, method="hog", use_cache=True, **kwargs):
    """
    Extract image features with caching support.
    """
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
    elif method == "combined":
        hog_kwargs = kwargs.get("hog_kwargs", {})
        lbp_kwargs = kwargs.get("lbp_kwargs", {})
        hog_features = extract_hog_batch(images, **hog_kwargs)
        lbp_features = extract_lbp_batch(images, **lbp_kwargs)
        features = np.concatenate([hog_features, lbp_features], axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - start
    print(f"  Extracted {features.shape} in {elapsed:.2f}s")

    if use_cache:
        np.save(cache_path, features)
        print(f"  Cached to {cache_path}")

    return features


class FusionDataset(Dataset):
    """Dataset for fusion of audio and image features."""

    def __init__(self, audio_features, image_features, labels):
        self.audio_features = torch.FloatTensor(audio_features)
        self.image_features = torch.FloatTensor(image_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_features[idx], self.image_features[idx], self.labels[idx]


class FusionModel(nn.Module):
    def __init__(self, audio_cnn, image_feat_dim):
        super(FusionModel, self).__init__()
        # Reuse audio CNN as the audio stream
        self.audio_stream = audio_cnn
        # Remove the final classification layer of the audio CNN
        self.audio_stream.fc2 = nn.Identity()

        # Image stream for HOG/LBP/combined features
        self.image_stream = nn.Sequential(
            nn.Linear(image_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 32),  # 64 from audio, 64 from image
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, audio_x, image_x):
        audio_feat = self.audio_stream(audio_x)  # 64-dim
        image_feat = self.image_stream(image_x)  # 64-dim

        combined = torch.cat((audio_feat, image_feat), dim=1)
        return self.classifier(combined).squeeze(1)


class FusionTrainer:
    """Trainer for fusion model."""

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
        weight_decay=1e-4,
        patience=10,
    ):
        """Train the fusion model."""
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

            for batch_audio, batch_image, batch_labels in train_loader:
                batch_audio = batch_audio.to(self.device)
                batch_image = batch_image.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_audio, batch_image)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            val_loss, val_auc = self.evaluate(val_loader)
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
                torch.save(self.model.state_dict(), "/tmp/best_fusion_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.model.load_state_dict(torch.load("/tmp/best_fusion_model.pth"))
        return history

    def evaluate(self, data_loader, criterion=None):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch_audio, batch_image, batch_labels in data_loader:
                batch_audio = batch_audio.to(self.device)
                batch_image = batch_image.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_audio, batch_image)

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
        """Get predictions."""
        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for batch_audio, batch_image, _ in data_loader:
                batch_audio = batch_audio.to(self.device)
                batch_image = batch_image.to(self.device)
                outputs = self.model(batch_audio, batch_image)
                scores = torch.sigmoid(outputs)
                all_scores.extend(scores.cpu().numpy())

        return np.array(all_scores)


def prepare_data(base_dir, method="hog", n_components=None, use_cache=True):
    """Prepare audio and image data for fusion."""
    print("Loading dataset...")

    # Load audio data
    audio_features, labels, filenames = load_audio_dataset(base_dir, use_augmented=True)
    print(f"Audio: {len(audio_features)} samples, shape={audio_features.shape}")

    # Load image data
    from utils import load_dataset

    images, _, img_filenames = load_dataset(base_dir, use_augmented=True)
    print(f"Images: {len(images)} samples")

    # Extract image features
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

    print(f"\nExtracting {method} features...")
    if method == "combined":
        image_features = extract_image_features_cached(
            images,
            method="combined",
            use_cache=use_cache,
            hog_kwargs=hog_params,
            lbp_kwargs=lbp_params,
        )
    elif method == "hog":
        image_features = extract_image_features_cached(
            images, method="hog", use_cache=use_cache, **hog_params
        )
    elif method == "lbp":
        image_features = extract_image_features_cached(
            images, method="lbp", use_cache=use_cache, **lbp_params
        )

    print(f"Image features shape: {image_features.shape}")

    # Apply PCA and scaling to image features
    scaler = StandardScaler()
    image_features_scaled = scaler.fit_transform(image_features)

    pca = None
    if n_components is not None:
        print(f"\nApplying PCA with n_components={n_components}...")
        pca = PCA(n_components=n_components)
        image_features_transformed = pca.fit_transform(image_features_scaled)
        print(f"Image features after PCA: {image_features_transformed.shape}")
    else:
        image_features_transformed = image_features_scaled

    return {
        "audio_features": audio_features,
        "image_features": image_features_transformed,
        "labels": labels,
        "filenames": filenames,
        "scaler": scaler,
        "pca": pca,
    }


def cross_validate(
    audio_features,
    image_features,
    labels,
    filenames,
    method="hog",
    cv_strategy="kfold",
    n_splits=2,
    epochs=50,
    n_components=None,
):
    """Perform cross-validation for fusion model."""
    if cv_strategy.lower() == "kfold":
        cv_splits = k_fold(filenames, labels, n_splits)
    elif cv_strategy.lower() == "loso":
        cv_splits = loso(filenames, labels)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")

    results = []
    fold_thresholds = []

    print(
        f"\n=== Fusion Cross-Validation ({cv_strategy.upper()}, "
        f"{len(cv_splits)} folds) ==="
    )

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold + 1}/{len(cv_splits)}:")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_audio_train, X_audio_val = audio_features[train_idx], audio_features[val_idx]
        X_image_train, X_image_val = image_features[train_idx], image_features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_dataset = FusionDataset(X_audio_train, X_image_train, y_train)
        val_dataset = FusionDataset(X_audio_val, X_image_val, y_val)

        use_cuda = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=use_cuda,
            num_workers=4 if use_cuda else 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            pin_memory=use_cuda,
            num_workers=4 if use_cuda else 0,
        )

        # Create audio CNN for this fold
        audio_cnn = ShallowCNN(
            n_channels=audio_features.shape[1],
            n_mfcc=audio_features.shape[2],
        )

        model = FusionModel(
            audio_cnn=audio_cnn,
            image_feat_dim=X_image_train.shape[1],
        )
        trainer = FusionTrainer(model)

        history = trainer.train(train_loader, val_loader, epochs=epochs)

        scores = trainer.predict(val_loader)
        val_loss, val_auc = trainer.evaluate(val_loader)
        fold_predictions = (scores > 0.5).astype(int)
        fold_acc = accuracy_score(y_val, fold_predictions)

        fold_threshold, fold_eer = compute_eer_threshold(y_val, scores)
        fold_thresholds.append(fold_threshold)

        # Save fold model to cache
        save_model_cache(
            model.state_dict(),
            "fusion",
            "model",
            cv_strategy,
            n_splits,
            n_components,
            fold,
        )

        results.append(
            {
                "fold": fold,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "val_accuracy": fold_acc,
                "scores": scores,
            }
        )

        print(f"  Validation Accuracy: {fold_acc:.4f}")
        print(f"  Validation AUC: {val_auc:.4f}")
        print(f"  Fold EER: {fold_eer:.4f}, Threshold: {fold_threshold:.4f}")

    accs = [r["val_accuracy"] for r in results]
    aucs = [r["val_auc"] for r in results]
    print("\n=== Summary ===")
    print(f"Mean Accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    if fold_thresholds:
        mean_threshold = np.mean(fold_thresholds)
        print(f"\nMean CV Threshold: {mean_threshold:.4f}")
        save_cv_threshold(mean_threshold, "fusion", cv_strategy, n_splits, n_components)

    return results


def train_final_model(
    audio_features,
    image_features,
    labels,
    epochs=100,
):
    """Train final fusion model on all data."""
    print("\n=== Training Final fusion Model ===")

    dataset = FusionDataset(audio_features, image_features, labels)
    use_cuda = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    audio_cnn = ShallowCNN(
        n_channels=audio_features.shape[1],
        n_mfcc=audio_features.shape[2],
    )

    model = FusionModel(
        audio_cnn=audio_cnn,
        image_feat_dim=image_features.shape[1],
    )
    trainer = FusionTrainer(model)

    dummy_loader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )
    history = trainer.train(loader, dummy_loader, epochs=epochs)

    # Save model to cache
    save_model_cache(model.state_dict(), "fusion", "model")

    return model, trainer


def prepare_eval_data(
    base_dir, method="hog", n_components=None, scaler=None, pca=None, use_cache=True
):
    print("Loading eval dataset...")
    base_dir = Path(base_dir)
    eval_dir = base_dir / "eval"

    audio_paths, eval_filenames = _get_audio_files(eval_dir)
    extractor = MFCCExtractor()
    audio_features = extractor.extract_batch(audio_paths, use_cache=True)
    print(f"Eval Audio: {len(audio_features)} samples")

    eval_images, img_filenames = load_images(eval_dir)
    print(f"Eval Images: {len(eval_images)} samples")

    if len(eval_images) == 0:
        raise ValueError(f"Nenalezeny žádné obrázky v {eval_dir}!")

    # Extract image features
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

    print(f"\nExtracting {method} features for eval...")
    if method == "combined":
        eval_image_features = extract_image_features_cached(
            eval_images,
            method="combined",
            use_cache=use_cache,
            hog_kwargs=hog_params,
            lbp_kwargs=lbp_params,
        )
    elif method == "hog":
        eval_image_features = extract_image_features_cached(
            eval_images, method="hog", use_cache=use_cache, **hog_params
        )
    elif method == "lbp":
        eval_image_features = extract_image_features_cached(
            eval_images, method="lbp", use_cache=use_cache, **lbp_params
        )

    # Apply same scaling and PCA as training data
    if scaler is not None:
        eval_image_features_scaled = scaler.transform(eval_image_features)
    else:
        eval_image_features_scaled = eval_image_features

    if pca is not None:
        eval_image_features_transformed = pca.transform(eval_image_features_scaled)
    else:
        eval_image_features_transformed = eval_image_features_scaled

    return {
        "audio_features": audio_features,
        "image_features": eval_image_features_transformed,
        "filenames": eval_filenames,
    }


def predict_on_dev(
    model,
    trainer,
    base_dir=None,
    method="hog",
    n_components=None,
    cv_strategy="kfold",
    n_splits=2,
):
    """Generate predictions on dev set using CV threshold."""
    if base_dir is None:
        base_dir = PROJECT_ROOT / "dataset"

    print("\nEvaluating on dev set...")

    data = prepare_data(base_dir, method, n_components, use_cache=True)

    dataset = FusionDataset(
        data["audio_features"], data["image_features"], data["labels"]
    )
    use_cuda = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    scores = trainer.predict(loader)

    # Use CV threshold if available, otherwise default to 0.5
    optimal_threshold = load_cv_threshold("fusion", cv_strategy, n_splits, n_components)
    threshold_source = "CV cache"

    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5
        threshold_source = "default (0.5)"
    else:
        print(f"  Using threshold from {threshold_source}: {optimal_threshold:.4f}")

    dev_eer, _ = compute_eer_threshold(data["labels"], scores)

    print(
        f"  Dev EER: {dev_eer:.4f}, Using Threshold: {optimal_threshold:.4f} (from {threshold_source})"
    )

    auc = roc_auc_score(data["labels"], scores)
    predictions = (scores > optimal_threshold).astype(int)
    acc = accuracy_score(data["labels"], predictions)

    print(f"  Dev Accuracy: {acc:.4f}")
    print(f"  Dev AUC: {auc:.4f}")

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "fusion.txt"
    # save_predictions(
    #     str(output_file), data["filenames"], scores, threshold=optimal_threshold
    # )

    return {
        "filenames": data["filenames"],
        "labels": data["labels"],
        "scores": scores,
        "accuracy": acc,
        "auc": auc,
        "optimal_threshold": optimal_threshold,
        "eer": dev_eer,
    }


def predict_on_eval(
    model,
    trainer,
    base_dir=None,
    method="hog",
    n_components=None,
    scaler=None,
    pca=None,
    cv_strategy="kfold",
    n_splits=2,
):
    """Evaluate on eval set using dev-trained model and CV threshold."""
    if base_dir is None:
        base_dir = PROJECT_ROOT / "dataset"

    print("\nEvaluating on eval set...")

    # Use CV threshold if available, otherwise default to 0.5
    optimal_threshold = load_cv_threshold("fusion", cv_strategy, n_splits, n_components)
    threshold_source = "CV cache"

    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5
        threshold_source = "default (0.5)"
    else:
        print(f"  Using threshold from {threshold_source}: {optimal_threshold:.4f}")

    # Load eval data
    eval_data = prepare_eval_data(
        base_dir, method, n_components, scaler, pca, use_cache=True
    )

    eval_dataset = FusionDataset(
        eval_data["audio_features"],
        eval_data["image_features"],
        [0] * len(eval_data["filenames"]),
    )
    use_cuda = torch.cuda.is_available()
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    scores = trainer.predict(eval_loader)
    predictions = (scores > optimal_threshold).astype(int)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "fusion_eval.txt"
    save_predictions(
        str(output_file), eval_data["filenames"], scores, threshold=optimal_threshold
    )

    print(f"  Eval predictions saved to {output_file}")

    return {
        "filenames": eval_data["filenames"],
        "scores": scores,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold,
    }


def predict_on_eval_cv(
    audio_features,
    image_features,
    labels,
    filenames,
    method="hog",
    cv_strategy="kfold",
    n_splits=2,
    epochs=50,
    n_components=None,
    scaler=None,
    pca=None,
):
    """Evaluate on eval set using ensemble of CV-trained fusion models."""
    base_dir = PROJECT_ROOT / "dataset"

    print(f"\nEvaluating on eval set with Fusion (CV Ensemble)...")

    # Load CV models from cache
    cv_models = []
    for fold in range(n_splits if cv_strategy == "kfold" else 10):
        model_state = load_model_cache(
            "fusion", "model", cv_strategy, n_splits, n_components, fold
        )
        if model_state is not None:
            audio_cnn = ShallowCNN(
                n_channels=audio_features.shape[1],
                n_mfcc=audio_features.shape[2],
            )
            model = FusionModel(
                audio_cnn=audio_cnn,
                image_feat_dim=image_features.shape[1],
            )
            model.load_state_dict(model_state)
            cv_models.append(model)
        else:
            print(f"  Warning: Could not load model for fold {fold}, skipping...")

    if not cv_models:
        print("  No CV models found in cache. Run 'train' mode first.")
        return None

    print(f"  Loaded {len(cv_models)} CV models")

    # Use CV threshold
    optimal_threshold = load_cv_threshold("fusion", cv_strategy, n_splits, n_components)
    if optimal_threshold is None:
        print("  No CV threshold found in cache, defaulting to 0.5")
        optimal_threshold = 0.5

    # Load eval data
    eval_data = prepare_eval_data(
        base_dir, method, n_components, scaler, pca, use_cache=True
    )

    eval_dataset = FusionDataset(
        eval_data["audio_features"],
        eval_data["image_features"],
        [0] * len(eval_data["filenames"]),
    )
    use_cuda = torch.cuda.is_available()
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=4 if use_cuda else 0,
    )

    # Ensemble: average predictions from all CV models
    all_scores = []
    for model in cv_models:
        trainer = FusionTrainer(model)
        scores = trainer.predict(eval_loader)
        all_scores.append(scores)

    # Average ensemble scores
    scores = np.mean(all_scores, axis=0)
    predictions = (scores > optimal_threshold).astype(int)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "fusion_eval_cv.txt"
    save_predictions(
        str(output_file), eval_data["filenames"], scores, threshold=optimal_threshold
    )

    print(f"  Eval-CV predictions saved to {output_file}")

    return {
        "filenames": eval_data["filenames"],
        "scores": scores,
        "predictions": predictions,
        "optimal_threshold": optimal_threshold,
    }


def main(
    mode="train", cv_strategy="kfold", n_splits=2, method="hog", n_components=None
):
    """
    Main fusion pipeline.

    Args:
        mode: "train", "dev", "eval-dev", or "eval-cv"
        cv_strategy: Cross-validation strategy
        n_splits: Number of folds
        method: Image feature extraction method
        n_components: PCA components for image features
    """
    base_dir = str(PROJECT_ROOT / "dataset")

    if mode == "train":
        data = prepare_data(base_dir, method, n_components)

        cross_validate(
            data["audio_features"],
            data["image_features"],
            data["labels"],
            data["filenames"],
            method=method,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            epochs=10,
            n_components=n_components,
        )

    elif mode == "dev":
        print("\n" + "=" * 50)
        data = prepare_data(base_dir, method, n_components)

        model, trainer = train_final_model(
            data["audio_features"],
            data["image_features"],
            data["labels"],
            epochs=20,
        )

        dev_results = predict_on_dev(
            model,
            trainer,
            base_dir,
            method=method,
            n_components=n_components,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
        )

        print("\n" + "=" * 50)
        print("Final Results:")
        print(f"  Dev Accuracy: {dev_results['accuracy']:.4f}")
        print(f"  Dev AUC: {dev_results['auc']:.4f}")
        print(f"\nResults saved to {PROJECT_ROOT / 'results' / 'fusion.txt'}")

    elif mode == "eval":
        print("\n" + "=" * 50)
        # Try to load dev-trained model from cache
        model_state = load_model_cache("fusion", "model")
        if model_state is not None:
            data = prepare_data(base_dir, method, n_components)
            audio_cnn = ShallowCNN(
                n_channels=data["audio_features"].shape[1],
                n_mfcc=data["audio_features"].shape[2],
            )
            model = FusionModel(
                audio_cnn=audio_cnn,
                image_feat_dim=data["image_features"].shape[1],
            )
            model.load_state_dict(model_state)
            trainer = FusionTrainer(model)
        else:
            print("Dev model not found in cache. Training new model...")
            data = prepare_data(base_dir, method, n_components)
            model, trainer = train_final_model(
                data["audio_features"],
                data["image_features"],
                data["labels"],
                epochs=20,
            )

        dev_results = predict_on_dev(
            model,
            trainer,
            base_dir,
            method=method,
            n_components=n_components,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
        )

        eval_results = predict_on_eval(
            model,
            trainer,
            base_dir,
            method=method,
            n_components=n_components,
            scaler=data["scaler"],
            pca=data["pca"],
            cv_strategy=cv_strategy,
            n_splits=n_splits,
        )

        print("\n" + "=" * 50)
        print("Final Results (Eval-Dev):")
        print(
            f"  Eval predictions saved to {PROJECT_ROOT / 'results' / 'fusion_eval.txt'}"
        )

    elif mode == "eval-cv":
        print("\n" + "=" * 50)
        data = prepare_data(base_dir, method, n_components)

        eval_results = predict_on_eval_cv(
            data["audio_features"],
            data["image_features"],
            data["labels"],
            data["filenames"],
            method=method,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            epochs=20,
            n_components=n_components,
            scaler=data["scaler"],
            pca=data["pca"],
        )

        print("\n" + "=" * 50)
        print("Final Results (Eval-CV):")
        print(
            f"  Eval predictions saved to {PROJECT_ROOT / 'results' / 'fusion_eval_cv.txt'}"
        )


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="fusion model combining audio CNN and image features"
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        choices=["kfold", "loso"],
        default="loso",
        help="Cross-validation strategy",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=2,
        help="Number of folds for kfold strategy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "dev", "eval"],
        default="train",
        help="train: cross-validation, dev: train on entire dataset, eval: use dev-trained model on eval data",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hog", "lbp", "combined"],
        default="combined",
        help="Image feature extraction method",
    )
    parser.add_argument(
        "--pca-components",
        type=str,
        default=None,
        help="PCA components: int (e.g., 150) or float 0-1 for variance",
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
        method=args.method,
        n_components=n_components,
    )
