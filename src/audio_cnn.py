"""
CNN-based audio person detector using MFCC features.
Implements shallow 2D CNN with ReLU, BatchNorm, and Dropout.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from cross_validation.session_cv import session_based_cv
from extractors.mfcc_extractor import MFCCExtractor, load_audio_dataset
from utils import save_predictions

PROJECT_ROOT = Path(__file__).parent.parent

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("cuDNN benchmark enabled for GPU acceleration")


class AudioDataset(Dataset):
    """Dataset for audio MFCC features."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


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

        self._get_conv_output_size(n_mfcc)

        self.fc1 = nn.Linear(self.conv_output_size, 64)
        self.fc_dropout = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(64, 1)

    def _get_conv_output_size(self, n_mfcc):
        """Calculate flattened size after conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, n_mfcc, 300)
            dummy = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool2(F.relu(self.bn2(self.conv2(dummy))))
            dummy = self.pool3(F.relu(self.bn3(self.conv3(dummy))))
            self.conv_output_size = dummy.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x.squeeze(1)


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
        weight_decay=1e-4,
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


def cross_validate_cnn(features, labels, filenames, n_splits=2, epochs=50):
    """
    Perform session-based cross-validation with CNN.

    Args:
        features: MFCC features array
        labels: Labels array
        filenames: List of filenames
        n_splits: Number of CV folds
        epochs: Training epochs per fold

    Returns:
        list of CV results
    """
    cv_splits = session_based_cv(filenames, labels, n_splits)
    results = []

    print(f"\n=== CNN Cross-Validation ({len(cv_splits)} folds) ===")

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold + 1}:")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_dataset = AudioDataset(X_train, y_train)
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

        results.append(
            {
                "fold": fold,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "scores": scores,
                "model": model,
            }
        )

        print(f"  Validation AUC: {val_auc:.4f}")

    aucs = [r["val_auc"] for r in results]
    print("\n=== Summary ===")
    print(f"Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    return results


def train_final_model(features, labels, epochs=100):
    """
    Train final CNN model on all data.

    Args:
        features: All training features
        labels: All training labels
        epochs: Training epochs

    Returns:
        Trained model
    """
    print("\n=== Training Final CNN Model ===")

    dataset = AudioDataset(features, labels)
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

    return model, trainer


def predict_on_dev(model, trainer, base_dir=None):
    """
    Generate predictions on dev set.

    Args:
        model: Trained CNN model
        trainer: CNNTrainer instance
        base_dir: Dataset base directory (default: PROJECT_ROOT / "dataset")

    Returns:
        dict with predictions
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "dataset"
    else:
        base_dir = Path(base_dir)

    print("\nEvaluating on dev set...")

    extractor = MFCCExtractor()

    target_dev_dir = base_dir / "target_dev"
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

    auc = roc_auc_score(all_labels, scores)
    predictions = (scores > 0.5).astype(int)
    acc = accuracy_score(all_labels, predictions)

    print(f"  Dev Accuracy: {acc:.4f}")
    print(f"  Dev AUC: {auc:.4f}")

    output_dir = PROJECT_ROOT / "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / "audio_cnn.txt"
    save_predictions(str(output_file), all_fnames, scores)

    return {
        "filenames": all_fnames,
        "labels": all_labels,
        "scores": scores,
        "accuracy": acc,
        "auc": auc,
    }


def _get_audio_files(directory):
    """Get all WAV files in directory with their filenames."""
    directory = Path(directory)
    files = sorted(directory.glob("*.wav"))
    paths = [str(f) for f in files]
    fnames = [f.stem for f in files]
    return paths, fnames


def main():
    base_dir = str(PROJECT_ROOT / "dataset")

    print("Loading audio dataset with MFCC features...")
    features, labels, filenames = load_audio_dataset(base_dir, use_augmented=True)
    print(
        f"Total: {len(features)} samples, "
        f"Target: {sum(labels)}, "
        f"Non-target: {len(labels) - sum(labels)}"
    )
    print(f"Feature shape: {features.shape}")

    cnn_results = cross_validate_cnn(features, labels, filenames, n_splits=2, epochs=10)

    print("\n" + "=" * 50)
    model, trainer = train_final_model(features, labels, epochs=20)

    dev_results = predict_on_dev(model, trainer, base_dir)

    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"  Dev Accuracy: {dev_results['accuracy']:.4f}")
    print(f"  Dev AUC: {dev_results['auc']:.4f}")
    print(f"\nResults saved to {PROJECT_ROOT / 'results' / 'audio_cnn.txt'}")


if __name__ == "__main__":
    main()
