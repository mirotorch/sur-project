"""
Session-based cross-validation strategy.
Ensures images from the same session are not split across train/validation sets.
"""

from collections import defaultdict

import numpy as np

from utils import parse_session


def session_based_cv(filenames, labels, n_splits=3):
    """
    Create session-based cross-validation splits.
    Ensures images from the same session are not split across train/val.
    Also ensures both classes are present in train and val sets.

    Args:
        filenames: List of filenames
        labels: Array of labels
        n_splits: Number of CV folds (limited by available sessions)

    Returns:
        List of (train_idx, val_idx) tuples
    """
    # Group indices by session AND by class
    session_class_to_indices = defaultdict(list)
    for idx, fname in enumerate(filenames):
        session = parse_session(fname)
        label = labels[idx]
        key = (session, label)
        session_class_to_indices[key].append(idx)

    # Get unique sessions for each class
    target_sessions = sorted(
        set(s for (s, l) in session_class_to_indices.keys() if l == 1)
    )
    non_target_sessions = sorted(
        set(s for (s, l) in session_class_to_indices.keys() if l == 0)
    )

    # Limit n_splits by available sessions
    n_splits_target = min(n_splits, len(target_sessions))
    n_splits_non_target = min(n_splits, len(non_target_sessions))
    n_splits = min(n_splits_target, n_splits_non_target)

    if n_splits == 0:
        print("Warning: Not enough sessions for CV, using random split")
        indices = np.arange(len(filenames))
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        split_point = int(0.8 * len(indices))
        return [(indices[:split_point], indices[split_point:])]

    splits = []
    for fold in range(n_splits):
        val_sessions = set()

        # Select validation sessions in round-robin fashion
        val_sessions.add(target_sessions[fold % len(target_sessions)])
        val_sessions.add(non_target_sessions[fold % len(non_target_sessions)])

        train_idx = []
        val_idx = []

        for (session, label), indices in session_class_to_indices.items():
            if session in val_sessions:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)

        splits.append((np.array(train_idx), np.array(val_idx)))

    return splits
