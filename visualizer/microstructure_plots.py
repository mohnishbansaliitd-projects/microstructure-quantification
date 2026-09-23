"""Figure generation for segmentation comparison, particle size distribution, and stereology benchmark."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict


def plot_segmentation_comparison(
    raw_img: np.ndarray,
    true_mask: np.ndarray,
    ws_mask: np.ndarray,
    unet_mask: np.ndarray,
    output_path: str = "outputs/figures/segmentation_4panel_comparison.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    
    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title("(a) Raw SEM Micrograph\n(NIST UHCS)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(true_mask, cmap="inferno")
    axes[1].set_title(f"(b) Ground Truth Mask\n($V_V = {true_mask.mean():.3f}$)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(ws_mask, cmap="viridis")
    axes[2].set_title(f"(c) Classical Watershed\n($V_V = {ws_mask.mean():.3f}$)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(unet_mask, cmap="magma")
    axes[3].set_title(f"(d) PyTorch U-Net Prediction\n($V_V = {unet_mask.mean():.3f}$)", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_multiclass_segmentation_comparison(
    raw_img: np.ndarray,
    true_mask: np.ndarray,
    ws_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_names: Dict[int, str],
    output_path: str = "outputs/figures/uhcs_multiclass_4panel_comparison.png"
):
    """4-class ground truth vs U-Net prediction (raw integer labels), classical watershed shown
    binary since it has no notion of the 4-phase legend."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_classes = len(class_names)
    cmap = plt.get_cmap("viridis", num_classes)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)

    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title("(a) Raw SEM Micrograph\n(NIST UHCS)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    true_viz = np.where(true_mask == -1, 0, true_mask)  # -1 (scale-bar/metadata) shown as matrix color, not a real class
    axes[1].imshow(true_viz, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[1].set_title("(b) Ground Truth\n(4-class legend)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(ws_mask, cmap="magma")
    axes[2].set_title(f"(c) Classical Watershed\n(binary, $V_V$={ws_mask.mean():.3f})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[3].set_title("(d) U-Net Prediction\n(ResNet18/ImageNet, held-out fold)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(num_classes)]
    fig.legend(handles, list(class_names.values()), loc="lower center", ncol=num_classes, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_iou(
    per_class_iou_mean: Dict[str, float],
    per_class_iou_std: Dict[str, float],
    output_path: str = "outputs/figures/per_class_iou_uhcs.png"
):
    """Per-class IoU (mean +/- std across K-fold CV folds) for the uhcs_general 4-class legend."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    names = list(per_class_iou_mean.keys())
    means = [per_class_iou_mean[n] for n in names]
    stds = [per_class_iou_std[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="#a855f7", edgecolor="#581c87")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in names], rotation=20, ha="right")
    ax.set_ylabel("IoU (mean $\\pm$ std across folds)", fontsize=11)
    ax.set_title("Per-Class IoU: U-Net (ResNet18/ImageNet) on UHCS 4-Class Legend", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_kfold_variance(
    df_folds_uhcs: pd.DataFrame,
    df_folds_particles: pd.DataFrame,
    output_path: str = "outputs/figures/kfold_iou_variance.png"
):
    """Held-out U-Net IoU per fold for both subsets, showing fold-to-fold variance."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    plot_specs = [
        (axes[0], df_folds_uhcs, "UHCS (4-class mean IoU)", "unet_mean_iou"),
        (axes[1], df_folds_particles, "Particles (binary IoU)", "unet_iou")
    ]
    for ax, df, title, col in plot_specs:
        x = df["fold"]
        mean_val = df[col].mean()
        std_val = df[col].std(ddof=1) if len(df) > 1 else 0.0

        ax.plot(x, df[col], "o-", color="#a855f7", label="U-Net (held-out fold)")
        ax.axhline(mean_val, color="#0f172a", linestyle="--", label=f"Mean: {mean_val:.3f}")
        ax.fill_between(x, mean_val - std_val, mean_val + std_val, color="#a855f7", alpha=0.15, label="$\\pm$1 std")

        ax.set_xlabel("Fold", fontsize=10)
        ax.set_ylabel("IoU", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("K-Fold Cross-Validation: Held-Out IoU per Fold", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_particle_size_distribution(
    df_particles: pd.DataFrame,
    output_path: str = "outputs/figures/particle_size_distribution.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    if len(df_particles) > 0 and "equivalent_diameter_um" in df_particles:
        diams = df_particles["equivalent_diameter_um"].dropna()
        ax.hist(diams, bins=25, density=True, color="#38bdf8", edgecolor="#0284c7", alpha=0.75, label="Measured Particles")
        
        # Add summary lines
        mean_d = diams.mean()
        median_d = diams.median()
        ax.axvline(mean_d, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_d:.2f} $\mu$m")
        ax.axvline(median_d, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_d:.2f} $\mu$m")
        
    ax.set_xlabel(r"Equivalent Circular Diameter $d_{\mathrm{eq}}$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("Cementite / Carbide Particle Size Distribution (NIST UHCS)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_stereology_benchmark(
    df_results: pd.DataFrame,
    output_path: str = "outputs/figures/stereology_volume_fraction_benchmark.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    n_samples = min(len(df_results), 8)
    sub_df = df_results.head(n_samples)
    x = np.arange(n_samples)
    width = 0.20
    
    ax.bar(x - 1.5 * width, sub_df["true_vv"] * 100, width, label="Ground Truth ($A_A$)", color="#0f172a", edgecolor="k")
    ax.bar(x - 0.5 * width, sub_df["watershed_vv"] * 100, width, label="Classical Watershed", color="#0284c7")
    ax.bar(x + 0.5 * width, sub_df["unet_vv"] * 100, width, label="PyTorch U-Net", color="#a855f7")
    ax.bar(x + 1.5 * width, sub_df["astm_e562_vv"] * 100, width, yerr=sub_df["astm_ci95"] * 100, capsize=3, label="ASTM E562 Point Count ($\pm 95\%$ CI)", color="#34d399")
    
    ax.set_xlabel("Micrograph Sample", fontsize=11)
    ax.set_ylabel("Phase Volume Fraction $V_V$ [%]", fontsize=11)
    ax.set_title("Stereology Benchmark: True Phase Fraction vs Segmentation Models vs ASTM E562", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[:8] for m in sub_df["micrograph"]], rotation=30)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
