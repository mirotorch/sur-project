"""
Local Binary Patterns (LBP) feature extractor.
"""

import numpy as np


def lbp_uniform_mapping(neighbors=8):
    """
    Create mapping for uniform LBP patterns.
    Uniform patterns get IDs 0..n_uniform-1, non-uniform gets n_uniform.

    Returns:
        tuple: (mapping array, number of patterns)
    """
    n_patterns = 1 << neighbors
    mapping = np.zeros(n_patterns, dtype=np.int32)

    def is_uniform(pattern):
        binary = [(pattern >> i) & 1 for i in range(neighbors)]
        transitions = sum(
            1 for i in range(neighbors) if binary[i] != binary[(i + 1) % neighbors]
        )
        return transitions <= 2

    uniform_count = 0
    for pattern in range(n_patterns):
        if is_uniform(pattern):
            mapping[pattern] = uniform_count
            uniform_count += 1

    # Non-uniform patterns get the same label
    for pattern in range(n_patterns):
        if not is_uniform(pattern):
            mapping[pattern] = uniform_count

    return mapping, uniform_count + 1


def extract_lbp_vectorized(img, radius=1, neighbors=8):
    """
    Vectorized LBP extraction.

    Args:
        img: Grayscale image (2D numpy array)
        radius: LBP radius
        neighbors: Number of neighbor pixels
        use_uniform: Whether to use uniform LBP mapping

    Returns:
        LBP code image (2D array)
    """
    h, w = img.shape
    center = img[radius : h - radius, radius : w - radius]

    # Pre-compute neighbor coordinates
    angles = 2 * np.pi * np.arange(neighbors) / neighbors
    dx = radius * np.cos(angles)
    dy = radius * np.sin(angles)

    # Collect all neighbor values using numpy indexing
    neighbors_values = []
    for i in range(neighbors):
        # Sample coordinates (with floor)
        x_coord = np.arange(radius, w - radius) + dx[i]
        y_coord = np.arange(radius, h - radius)[:, np.newaxis] + dy[i]

        # Bilinear interpolation
        x0 = np.floor(x_coord).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(y_coord).astype(np.int32)
        y1 = np.minimum(y0 + 1, h - 1)

        # Weights
        wx = x_coord - x0
        wy = y_coord - y0

        # Interpolate (vectorized)
        x0 = x0[np.newaxis, :]
        x1 = x1[np.newaxis, :]
        y0 = y0[:, np.newaxis]
        y1 = y1[:, np.newaxis]

        sample = (
            (1 - wx) * (1 - wy) * img[y0, x0]
            + wx * (1 - wy) * img[y0, x1]
            + (1 - wx) * wy * img[y1, x0]
            + wx * wy * img[y1, x1]
        )

        neighbors_values.append(sample)

    # Stack and compare to center
    neighbors_stack = np.stack(neighbors_values, axis=-1)  # (h-2r, w-2r, neighbors)
    center_expanded = center[:, :, np.newaxis]

    # Binary pattern
    binary = (neighbors_stack >= center_expanded).astype(np.uint8)

    # Compute LBP codes (vectorized)
    powers = 1 << np.arange(neighbors)
    lbp_codes = np.sum(binary * powers, axis=-1).astype(np.uint8)

    return lbp_codes


def extract_lbp(
    img, radius=1, neighbors=8, use_uniform=True, grid_size=None, normalize=True
):
    """
    Extract LBP features from image.

    Args:
        img: Input image (grayscale, 2D numpy array) or RGB (3D)
        radius: LBP radius
        neighbors: Number of neighbor pixels
        use_uniform: Whether to use uniform LBP mapping
        grid_size: If provided, split image into grid and compute histogram per cell
        normalize: Whether to normalize histogram (L1 norm)

    Returns:
        Feature vector (histogram of LBP codes)
    """
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    img = img.astype(np.float32)
    h, w = img.shape

    if grid_size is None:
        # Compute LBP for entire image (vectorized)
        lbp_img = extract_lbp_vectorized(img, radius, neighbors)

        # Histogram
        if use_uniform and neighbors == 8:
            mapping, n_bins = lbp_uniform_mapping(neighbors)
            lbp_img = mapping[lbp_img]
            hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins))
        else:
            n_bins = 1 << neighbors
            hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins))

        if normalize:
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist = hist / (hist.sum() + 1e-10)

        return hist

    else:
        # Grid-based LBP (spatial pyramid)
        grid_h, grid_w = grid_size
        cell_h = h // grid_h
        cell_w = w // grid_w

        if use_uniform and neighbors == 8:
            mapping, n_bins = lbp_uniform_mapping(neighbors)
        else:
            n_bins = 1 << neighbors
            mapping = None

        histograms = []

        for i in range(grid_h):
            for j in range(grid_w):
                y0 = i * cell_h
                y1 = y0 + cell_h if i < grid_h - 1 else h
                x0 = j * cell_w
                x1 = x0 + cell_w if j < grid_w - 1 else w

                cell = img[y0:y1, x0:x1]
                lbp_cell = extract_lbp_vectorized(cell, radius, neighbors)

                # Histogram for this cell
                if mapping is not None:
                    lbp_cell = mapping[lbp_cell]
                    hist, _ = np.histogram(lbp_cell, bins=n_bins, range=(0, n_bins))
                else:
                    hist, _ = np.histogram(lbp_cell, bins=n_bins, range=(0, n_bins))

                if normalize:
                    hist = hist.astype(np.float32)
                    if hist.sum() > 0:
                        hist = hist / (hist.sum() + 1e-10)

                histograms.append(hist)

        return np.concatenate(histograms)


def extract_lbp_batch(
    images, radius=1, neighbors=8, use_uniform=True, grid_size=None, normalize=True
):
    """
    Extract LBP features from a batch of images.

    Args:
        images: Batch of images (n, h, w, c) or (n, h, w)
        radius, neighbors, use_uniform, grid_size, normalize: See extract_lbp

    Returns:
        Feature matrix (n_samples, n_features)
    """
    features = []
    for img in images:
        feat = extract_lbp(img, radius, neighbors, use_uniform, grid_size, normalize)
        features.append(feat)
    return np.array(features)
