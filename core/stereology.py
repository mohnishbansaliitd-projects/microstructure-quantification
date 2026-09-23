"""ASTM E562 point-count stereology and particle morphology/dispersion measurements."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from skimage import measure
from typing import Dict, Any, Tuple, Optional


def simulate_astm_e562_point_count(
    binary_mask: np.ndarray,
    grid_spacing: int = 16,
    num_fields: int = 10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Simulates ASTM E562 systematic point counting; tests the Delesse principle V_V = A_A = P_P."""
    np.random.seed(random_seed)
    h, w = binary_mask.shape

    field_fractions = []

    for _ in range(num_fields):
        offset_y = np.random.randint(0, grid_spacing)
        offset_x = np.random.randint(0, grid_spacing)

        grid_y = np.arange(offset_y, h, grid_spacing)
        grid_x = np.arange(offset_x, w, grid_spacing)

        yy, xx = np.meshgrid(grid_y, grid_x)
        sampled_points = binary_mask[yy, xx]

        p_p = np.mean(sampled_points > 0)  # points on phase / total test points
        field_fractions.append(p_p)

    field_fractions = np.array(field_fractions)
    p_mean = float(np.mean(field_fractions))
    p_std = float(np.std(field_fractions, ddof=1)) if num_fields > 1 else 0.0

    t_val = stats.t.ppf(0.975, df=num_fields - 1) if num_fields > 1 else 1.96
    ci_95 = float(t_val * p_std / np.sqrt(num_fields))
    rel_accuracy_pct = float((ci_95 / (p_mean + 1e-12)) * 100.0)
    
    true_a_a = float(np.mean(binary_mask > 0))
    
    return {
        "estimated_volume_fraction_VV": p_mean,
        "std_deviation": p_std,
        "ci_95_half_width": ci_95,
        "relative_accuracy_pct": rel_accuracy_pct,
        "true_area_fraction_AA": true_a_a,
        "delesse_absolute_error": abs(p_mean - true_a_a),
        "num_fields_evaluated": num_fields
    }


def compute_particle_morphology_and_distribution(
    labeled_mask: np.ndarray,
    pixel_scale_um: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Measures per-particle size, aspect ratio, circularity, and nearest-neighbor spacing."""
    props = measure.regionprops(labeled_mask)
    if len(props) == 0:
        return pd.DataFrame(), {}
        
    records = []
    centroids = []
    
    for p in props:
        area_px = p.area
        area_um2 = area_px * (pixel_scale_um ** 2)
        eq_diam_um = 2.0 * np.sqrt(area_um2 / np.pi)
        
        perimeter_um = p.perimeter * pixel_scale_um
        circularity = (4.0 * np.pi * area_um2) / (perimeter_um ** 2 + 1e-12) if perimeter_um > 0 else 0.0
        
        major_axis = p.major_axis_length * pixel_scale_um
        minor_axis = p.minor_axis_length * pixel_scale_um
        aspect_ratio = major_axis / (minor_axis + 1e-12)
        
        records.append({
            "particle_id": p.label,
            "area_um2": area_um2,
            "equivalent_diameter_um": eq_diam_um,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "centroid_y": p.centroid[0],
            "centroid_x": p.centroid[1]
        })
        centroids.append(p.centroid)
        
    df_particles = pd.DataFrame(records)

    if len(centroids) > 1:
        kdtree = cKDTree(np.array(centroids) * pixel_scale_um)
        dists, _ = kdtree.query(np.array(centroids) * pixel_scale_um, k=2)  # k=2: nearest is the point itself
        nn_dists_um = dists[:, 1]
        df_particles["nearest_neighbor_distance_um"] = nn_dists_um
        mean_nn_dist = float(np.mean(nn_dists_um))
    else:
        df_particles["nearest_neighbor_distance_um"] = 0.0
        mean_nn_dist = 0.0
        
    summary_stats = {
        "particle_count": len(df_particles),
        "mean_diameter_um": float(df_particles["equivalent_diameter_um"].mean()),
        "std_diameter_um": float(df_particles["equivalent_diameter_um"].std()),
        "median_diameter_um": float(df_particles["equivalent_diameter_um"].median()),
        "mean_aspect_ratio": float(df_particles["aspect_ratio"].mean()),
        "mean_circularity": float(df_particles["circularity"].mean()),
        "mean_nearest_neighbor_um": mean_nn_dist
    }
    
    return df_particles, summary_stats
