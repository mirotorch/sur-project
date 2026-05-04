"""
Audio preprocessing for SUR project 2025/2026.
Extracts MFCC features with deltas for CNN-based audio recognition.
"""

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

PROJECT_ROOT = Path(__file__).parent.parent.parent


class MFCCExtractor:
    """Extract MFCC features with deltas for audio files."""

    def __init__(
        self,
        sample_rate=16000,
        n_mfcc=40,
        n_fft=400,
        hop_length=160,
        n_mels=128,
        fixed_time_steps=300,
        use_gpu=True,
    ):
        """
        Initialize MFCC extractor.

        Args:
            sample_rate: Audio sample rate (default: 16000)
            n_mfcc: Number of MFCC coefficients (default: 40)
            n_fft: FFT window size (default: 400 = 25ms at 16kHz)
            hop_length: Hop length (default: 160 = 10ms at 16kHz)
            n_mels: Number of mel filters (default: 128)
            fixed_time_steps: Fixed time dimension (default: 300)
            use_gpu: Whether to use GPU for MFCC extraction (default: True)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.fixed_time_steps = fixed_time_steps
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "center": True,
            },
        ).to(self.device)

    def extract(self, audio_path):
        """
        Extract MFCC features with deltas from audio file.

        Args:
            audio_path: Path to WAV file

        Returns:
            numpy array of shape (3, n_mfcc, time_steps)
            where channels are: MFCC, delta, delta-delta
        """
        try:
            audio, sr = sf.read(audio_path, dtype="float32")

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            if sr != self.sample_rate:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            waveform = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # (1, time)

            mfcc = self.mfcc_transform(waveform)  # (1, n_mfcc, time)

            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)

            features = torch.cat([mfcc, delta, delta2], dim=0)  # (3, n_mfcc, time)

            features = self._fix_time_dimension(features)

            return features.cpu().numpy()

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def _fix_time_dimension(self, features):
        """Fix time dimension to fixed_time_steps."""
        _, _, time_steps = features.shape

        if time_steps > self.fixed_time_steps:
            start = (time_steps - self.fixed_time_steps) // 2
            features = features[:, :, start : start + self.fixed_time_steps]
        elif time_steps < self.fixed_time_steps:
            pad_amount = self.fixed_time_steps - time_steps
            left_pad = pad_amount // 2
            right_pad = pad_amount - left_pad
            features = torch.nn.functional.pad(
                features, (left_pad, right_pad), mode="constant", value=0
            )

        return features

    def to_cpu(self):
        """Move transform back to CPU if needed."""
        self.mfcc_transform.to("cpu")
        self.device = torch.device("cpu")
        self.use_gpu = False

    def extract_batch(self, audio_paths, use_cache=True, cache_dir=None):
        """
        Extract features for multiple files with caching.

        Args:
            audio_paths: List of audio file paths
            use_cache: Whether to use cached features
            cache_dir: Cache directory path

        Returns:
            numpy array of shape (n_files, 3, n_mfcc, time_steps)
        """
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / "cache"

        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(
            cache_dir, f"mfcc_n{self.n_mfcc}_t{self.fixed_time_steps}.npy"
        )
        cache_meta_file = os.path.join(
            cache_dir, f"mfcc_n{self.n_mfcc}_t{self.fixed_time_steps}_meta.npy"
        )

        if use_cache and os.path.exists(cache_file) and os.path.exists(cache_meta_file):
            cached_paths = np.load(cache_meta_file, allow_pickle=True)
            if len(cached_paths) == len(audio_paths) and all(
                str(a) == str(b) for a, b in zip(cached_paths, audio_paths)
            ):
                print(f"Loading cached MFCC features from {cache_file}")
                return np.load(cache_file)
            else:
                print("Cache mismatch detected, regenerating features...")

        print(f"Extracting MFCC features for {len(audio_paths)} files...")

        print(f"Extracting MFCC features for {len(audio_paths)} files...")
        features = []

        for i, path in enumerate(audio_paths):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(audio_paths)}")
            feat = self.extract(path)
            if feat is not None:
                features.append(feat)
            else:
                features.append(np.zeros((3, self.n_mfcc, self.fixed_time_steps)))

        features = np.array(features)

        if use_cache:
            np.save(cache_file, features)
            np.save(cache_meta_file, np.array(audio_paths, dtype=object))
            print(f"Cached features to {cache_file}")

        return features


def load_audio_dataset(base_dir=None, use_augmented=True, fixed_time_steps=300):
    """
    Load audio dataset with MFCC features.

    Args:
        base_dir: Base dataset directory (default: PROJECT_ROOT / "dataset")
        use_augmented: Whether to include augmented audio
        fixed_time_steps: Fixed time dimension for features

    Returns:
        tuple: (features, labels, filenames)
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "dataset"
    else:
        base_dir = Path(base_dir)

    all_paths = []
    all_labels = []
    all_filenames = []

    extractor = MFCCExtractor(fixed_time_steps=fixed_time_steps)

    # Target training data
    target_dir = base_dir / "target_train"
    if not target_dir.exists():
        target_dir = base_dir / "target"
    if target_dir.exists():
        paths, fnames = _get_audio_files(target_dir)
        all_paths.extend(paths)
        all_labels.extend([1] * len(paths))
        all_filenames.extend(fnames)
        print(f"Loaded {len(paths)} target training audio files")

    # Non-target training data
    non_target_dir = base_dir / "non_target_train"
    if not non_target_dir.exists():
        non_target_dir = base_dir / "non-target"
    if non_target_dir.exists():
        paths, fnames = _get_audio_files(non_target_dir)
        all_paths.extend(paths)
        all_labels.extend([0] * len(paths))
        all_filenames.extend(fnames)
        print(f"Loaded {len(paths)} non-target training audio files")

    if use_augmented:
        aug_dir = base_dir / "augmented"
        if aug_dir.exists():
            target_aug = aug_dir / "target_train_aug"
            if target_aug.exists():
                paths, fnames = _get_audio_files(target_aug)
                all_paths.extend(paths)
                all_labels.extend([1] * len(paths))
                all_filenames.extend(fnames)
                print(f"Loaded {len(paths)} target augmented audio files")

            non_target_aug = aug_dir / "non_target_train_aug"
            if non_target_aug.exists():
                paths, fnames = _get_audio_files(non_target_aug)
                all_paths.extend(paths)
                all_labels.extend([0] * len(paths))
                all_filenames.extend(fnames)
                print(f"Loaded {len(paths)} non-target augmented audio files")

    features = extractor.extract_batch(all_paths)

    return features, np.array(all_labels), all_filenames


def _get_audio_files(directory):
    """Get all WAV files in directory with their filenames."""
    directory = Path(directory)
    files = sorted(directory.glob("*.wav"))
    paths = [str(f) for f in files]
    fnames = [f.stem for f in files]
    return paths, fnames


if __name__ == "__main__":
    base_dir = PROJECT_ROOT / "dataset"
    features, labels, fnames = load_audio_dataset(base_dir)
    print(f"\nDataset shape: {features.shape}")
    print(
        f"Labels: {len(labels)} (target: {sum(labels)}, non-target: {len(labels) - sum(labels)})"
    )
