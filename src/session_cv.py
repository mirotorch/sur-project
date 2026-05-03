"""
Session-based cross-validation strategies.
Ensures samples from the same session are not split across train/validation sets.

Implemented validators:
1. session_aware_kfold: K-fold where each fold respects session boundaries
2. session_loso: Leave-One-Session-Out cross-validation
"""

from collections import defaultdict

import numpy as np

from utils import parse_session


def k_fold(filenames, labels, n_splits=3, random_state=42):
    """
    K-fold cross-validation.

    Ensures all data from one session stays together in either train or validation set.
    Automatically determines n_splits based on available sessions.

    Args:
        filenames: List of filenames
        labels: Array of labels (typically 0=non-target, 1=target)
        n_splits: Requested number of folds (actual may be lower based on sessions)
        random_state: Random seed for reproducibility

    Returns:
        List of (train_idx, val_idx) tuples
    """
    rng = np.random.RandomState(random_state)

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

    # Determine actual number of splits
    n_splits_target = len(target_sessions)
    n_splits_non_target = len(non_target_sessions)
    actual_n_splits = min(n_splits, n_splits_target, n_splits_non_target)

    if actual_n_splits == 0:
        print("Warning: Not enough sessions for session-aware CV, using random split")
        indices = np.arange(len(filenames))
        rng.shuffle(indices)
        split_point = int(0.8 * len(indices))
        return [(indices[:split_point], indices[split_point:])]

    print(
        f"Session-Aware K-Fold: {actual_n_splits} folds "
        f"({n_splits_target} target sessions, {n_splits_non_target} non-target sessions)"
    )

    splits = []
    for fold in range(actual_n_splits):
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

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        # Shuffle indices within splits for better training
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        splits.append((train_idx, val_idx))

    return splits


def loso(filenames, labels, random_state=42):
    """
    Leave-One-Session-Out (LOSO) cross-validation.

    For each session in the dataset, uses that session as validation set and
    trains on all other sessions.

    Args:
        filenames: List of filenames
        labels: Array of labels (typically 0=non-target, 1=target)
        random_state: Random seed for reproducibility

    Returns:
        List of (train_idx, val_idx) tuples
    """
    rng = np.random.RandomState(random_state)

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

    n_target = len(target_sessions)
    n_non_target = len(non_target_sessions)

    print(
        f"Session LOSO: {n_target} target sessions × {n_non_target} non-target sessions = "
        f"{n_target * n_non_target} folds"
    )

    splits = []

    # Create a fold for each combination of target and non-target sessions
    for target_val_session in target_sessions:
        for non_target_val_session in non_target_sessions:
            val_sessions = {target_val_session, non_target_val_session}
            train_idx = []
            val_idx = []

            for (session, label), indices in session_class_to_indices.items():
                if session in val_sessions:
                    val_idx.extend(indices)
                else:
                    train_idx.extend(indices)

            train_idx = np.array(train_idx)
            val_idx = np.array(val_idx)

            # Shuffle indices within splits for better training
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)

            splits.append((train_idx, val_idx))

    return splits
