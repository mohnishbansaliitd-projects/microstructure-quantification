"""Otsu thresholding and marker-controlled watershed segmentation for cementite/carbide phases."""

import numpy as np
from typing import Tuple
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure


def segment_microstructure_otsu(
    image: np.ndarray,
    sigma: float = 1.0,
    min_size: int = 15
) -> np.ndarray:
    """Returns binary mask (1 = cementite/precipitate phase, 0 = matrix)."""
    smoothed = filters.gaussian(image, sigma=sigma)

    thresh = filters.threshold_otsu(smoothed)
    # cementite appears brighter than the matrix in secondary-electron SEM of UHCS
    binary = smoothed > thresh

    binary_clean = morphology.remove_small_objects(binary, min_size=min_size)
    binary_clean = morphology.remove_small_holes(binary_clean, area_threshold=min_size)
    
    return binary_clean.astype(np.uint8)


def segment_microstructure_watershed(
    image: np.ndarray,
    sigma: float = 1.0,
    min_distance: int = 5,
    min_size: int = 20,
    clear_border: bool = True
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Separates touching cementite particles via marker-controlled watershed on the distance transform.

    Returns (binary_mask, labeled_particles, num_particles).
    """
    binary = segment_microstructure_otsu(image, sigma=sigma, min_size=min_size)
    distance = ndi.distance_transform_edt(binary)

    coords = morphology.local_maxima(distance)
    markers, num_markers = ndi.label(coords)
    if num_markers == 0:
        markers, num_markers = ndi.label(binary)

    labels = segmentation.watershed(-distance, markers, mask=binary)

    if clear_border:
        # particles cut off at the image edge bias stereological area-fraction estimates
        labels = segmentation.clear_border(labels)

    num_particles = len(np.unique(labels)) - 1
    binary_mask = (labels > 0).astype(np.uint8)
    
    return binary_mask, labels, num_particles
