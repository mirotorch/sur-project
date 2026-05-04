"""
Histogram of Oriented Gradients (HOG) feature extractor.
"""

import numpy as np


def compute_gradients(img):
    """
    Compute image gradients using central differences.

    Args:
        img: Grayscale image (2D numpy array)

    Returns:
        tuple: (magnitude, orientation) where
            magnitude: Gradient magnitude at each pixel
            orientation: Gradient orientation in degrees [0, 180)
    """
    # Central difference with replication padding
    img_padded = np.pad(img, ((1, 1), (1, 1)), mode="edge")

    # Gradient in x and y directions
    gy = (img_padded[2:, 1:-1] - img_padded[:-2, 1:-1]) / 2.0
    gx = (img_padded[1:-1, 2:] - img_padded[1:-1, :-2]) / 2.0

    # Magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180.0 / np.pi

    # Map to [0, 180)
    orientation = orientation % 180

    return magnitude, orientation


def get_block_histogram(mag_cell, ori_cell, orientations=9):
    """
    Compute histogram for a single cell.

    Args:
        mag_cell: Gradient magnitudes in the cell
        ori_cell: Gradient orientations in the cell
        orientations: Number of orientation bins

    Returns:
        Histogram of shape (orientations,)
    """
    hist = np.zeros(orientations)
    bin_width = 180.0 / orientations

    for m, o in zip(mag_cell.flatten(), ori_cell.flatten()):
        bin_idx = int(o / bin_width) % orientations
        hist[bin_idx] += m

    return hist


def normalize_block(block_hist, eps=1e-10):
    """
    Normalize block histogram using L2-norm.

    Args:
        block_hist: Block histogram (flattened)
        eps: Small constant to avoid division by zero

    Returns:
        Normalized block histogram
    """
    norm = np.sqrt(np.sum(block_hist**2) + eps)
    return block_hist / norm


def extract_hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    normalize=True,
    block_norm="L2",
    eps=1e-10,
):
    """
    Extract HOG features from image.

    Args:
        img: Input image (grayscale, 2D) or RGB (3D)
        orientations: Number of orientation bins
        pixels_per_cell: Cell size (height, width)
        cells_per_block: Block size (height, width)
        normalize: Whether to normalize block histograms
        block_norm: Normalization method ('L2' or 'L2-Hys')
        eps: Small constant for normalization

    Returns:
        Feature vector (flattened HOG descriptor)
    """
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    img = img.astype(np.float32)
    h, w = img.shape
    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block

    # Compute gradients
    magnitude, orientation = compute_gradients(img)

    # Number of cells
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    # Truncate to fit integer number of cells
    magnitude = magnitude[: n_cells_y * cell_h, : n_cells_x * cell_w]
    orientation = orientation[: n_cells_y * cell_h, : n_cells_x * cell_w]

    # Reshape into cells
    mag_cells = magnitude.reshape(n_cells_y, cell_h, n_cells_x, cell_w)
    ori_cells = orientation.reshape(n_cells_y, cell_h, n_cells_x, cell_w)

    # Compute histogram for each cell
    cell_hists = np.zeros((n_cells_y, n_cells_x, orientations))

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell_hists[cy, cx] = get_block_histogram(
                mag_cells[cy, :, cx, :], ori_cells[cy, :, cx, :], orientations
            )

    # Group cells into blocks and normalize
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1

    hog_features = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            # Extract block
            block = cell_hists[by : by + block_h, bx : bx + block_w, :]
            block_flat = block.flatten()

            if normalize:
                if block_norm == "L2":
                    block_flat = normalize_block(block_flat, eps)
                elif block_norm == "L2-Hys":
                    # L2 normalization followed by clipping and renormalization
                    block_flat = normalize_block(block_flat, eps)
                    block_flat = np.clip(block_flat, 0, 0.2)
                    block_flat = normalize_block(block_flat, eps)

            hog_features.append(block_flat)

    return np.concatenate(hog_features)


def extract_hog_batch(
    images,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    normalize=True,
    block_norm="L2",
    eps=1e-10,
):
    """
    Extract HOG features from a batch of images.

    Args:
        images: Batch of images (n, h, w, c) or (n, h, w)
        Other args: See extract_hog

    Returns:
        Feature matrix (n_samples, n_features)
    """
    features = []
    for img in images:
        feat = extract_hog(
            img,
            orientations,
            pixels_per_cell,
            cells_per_block,
            normalize,
            block_norm,
            eps,
        )
        features.append(feat)
    return np.array(features)


def get_hog_feature_size(
    img_size=(80, 80), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)
):
    """
    Calculate the size of HOG feature vector.

    Args:
        img_size: Image size (height, width)
        orientations, pixels_per_cell, cells_per_block: HOG parameters

    Returns:
        Feature vector size (int)
    """
    h, w = img_size
    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block

    n_cells_y = h // cell_h
    n_cells_x = w // cell_w
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1

    return n_blocks_y * n_blocks_x * block_h * block_w * orientations
