"""Runs the microstructure segmentation pipeline: self-supervised encoder pretraining on the
full 961-image unlabeled UHCS corpus, then K-fold cross-validated U-Net (segmentation-models-
pytorch, ResNet18 encoder, SSL-pretrained + ImageNet-initialized) fine-tuning vs classical
watershed, per-class IoU on the real uhcs_general 4-class legend, binary IoU on the
particles_spheroidite subset, ASTM E562 stereology comparison, particle morphology, and figure
generation."""

import os
import sys
import time
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from core.classical_segmentation import segment_microstructure_watershed
from core.stereology import compute_particle_morphology_and_distribution
from core.ssl_pretrain import pretrain_ssl_encoder, save_pretrained_encoder
from data.dataset_loader import UHCS_MULTICLASS_NAMES
from benchmark.train_and_evaluate import run_uhcs_kfold_benchmark, run_particles_kfold_benchmark
from visualizer.microstructure_plots import (
    plot_segmentation_comparison,
    plot_multiclass_segmentation_comparison,
    plot_particle_size_distribution,
    plot_stereology_benchmark,
    plot_per_class_iou,
    plot_kfold_variance
)

K_FOLDS = 4                # 24 images/subset -> 6 held-out per fold, a reasonable split at this sample size
NUM_EPOCHS = 8              # modest epoch count so K x smp.Unet(resnet18) CPU training stays under a few minutes
TARGET_SIZE = (160, 160)    # downsized from the original 256x256 for the same CPU-time reason

# Full unlabeled UHCS corpus (961 images, Kaggle mirror of DeCost et al./NIST) used for
# self-supervised pretraining -- see core/ssl_pretrain.py.
UHCS_FULL_CORPUS_DIR = os.path.abspath(os.path.join(
    PROJECT_DIR, "..", "data_check", "uhcs_full", "extracted", "For Training", "Cropped"
))
# Subsample/epoch count picked from timed pilots (see core/ssl_pretrain.py docstring): a 20-image
# x 2-epoch pilot extrapolated to ~3.5 min for 300x6, but a full timed run at 300x6 actually took
# 5.4 min (322.8s train + 3.0s load) on this CPU -- likely batch-size and thermal-throttling
# effects the short pilot didn't capture. 300x6 alone would eat the whole "few minutes" budget
# and leave too little headroom for the K-fold fine-tuning + figure-generation steps that follow
# in the same main.py run, so this was scaled down to 200x5 (1000 image-epochs, ~40% of the
# 1800 image-epochs measured at 300x6, i.e. ~expected ~130-180s train time) after the real
# timing was known.
SSL_NUM_IMAGES = 200        # seeded subsample of the 961; see ssl_pretrain docstring for why
SSL_NUM_EPOCHS = 5          # full passes over the 200-image subsample
PRETRAINED_ENCODER_PATH = os.path.join(PROJECT_DIR, "outputs", "pretrained_encoder.pth")


def run_full_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    t0 = time.time()

    print(f"\n[0] Self-Supervised Pretraining on Full UHCS Corpus ({SSL_NUM_IMAGES} of 961 images, "
          f"{SSL_NUM_EPOCHS} epochs, masked-patch denoising autoencoder)...")
    t_ssl0 = time.time()
    pretrain_result = pretrain_ssl_encoder(
        UHCS_FULL_CORPUS_DIR, num_images=SSL_NUM_IMAGES, num_epochs=SSL_NUM_EPOCHS, device=device
    )
    save_pretrained_encoder(pretrain_result, PRETRAINED_ENCODER_PATH)
    print(f"    Corpus available: {pretrain_result['corpus_size']} images; "
          f"used {pretrain_result['num_images']} (seeded random subsample) x {pretrain_result['num_epochs']} epochs")
    print(f"    Reconstruction MSE loss per epoch: " +
          ", ".join(f"{l:.4f}" for l in pretrain_result["loss_history"]))
    print(f"    Final reconstruction loss: {pretrain_result['final_loss']:.4f}")
    print(f"    Pretraining time: load {pretrain_result['load_time_s']:.1f}s + "
          f"train {pretrain_result['train_time_s']:.1f}s = {time.time() - t_ssl0:.1f}s")
    print(f"    Saved pretrained encoder to {PRETRAINED_ENCODER_PATH}")

    print(f"\nRunning {K_FOLDS}-fold CV on uhcs_general (4-class: matrix/carbide/spheroidite/Widmanstatten)...")
    uhcs = run_uhcs_kfold_benchmark(
        k=K_FOLDS, num_epochs=NUM_EPOCHS, device=device, target_size=TARGET_SIZE,
        pretrained_encoder_path=PRETRAINED_ENCODER_PATH
    )

    print(f"Running {K_FOLDS}-fold CV on particles_spheroidite (binary: matrix/spheroidite)...")
    particles = run_particles_kfold_benchmark(
        k=K_FOLDS, num_epochs=NUM_EPOCHS, device=device, target_size=TARGET_SIZE,
        pretrained_encoder_path=PRETRAINED_ENCODER_PATH
    )

    print("\n=== UHCS subset (4-class), held-out per-fold results ===")
    print(f"U-Net mean IoU:         {uhcs['unet_mean_iou_mean']:.3f} +/- {uhcs['unet_mean_iou_std']:.3f} (std across {K_FOLDS} folds)")
    print(f"Watershed (binary) IoU: {uhcs['watershed_binary_iou_mean']:.3f} +/- {uhcs['watershed_binary_iou_std']:.3f}")
    print("Per-class IoU (mean +/- std across folds):")
    for cname in UHCS_MULTICLASS_NAMES.values():
        print(f"    {cname}: {uhcs['per_class_iou_mean'][cname]:.3f} +/- {uhcs['per_class_iou_std'][cname]:.3f}")

    print("\n=== Particles subset (binary), held-out per-fold results ===")
    print(f"U-Net IoU:     {particles['unet_iou_mean']:.3f} +/- {particles['unet_iou_std']:.3f} (std across {K_FOLDS} folds)")
    print(f"U-Net Dice:    {particles['unet_dice_mean']:.3f} +/- {particles['unet_dice_std']:.3f}")
    print(f"Watershed IoU: {particles['watershed_iou_mean']:.3f} +/- {particles['watershed_iou_std']:.3f}")

    figures_dir = os.path.join(PROJECT_DIR, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    sv_uhcs = uhcs["sample_viz"]
    plot_multiclass_segmentation_comparison(
        sv_uhcs["img"], sv_uhcs["true_mask"], sv_uhcs["ws_mask"], sv_uhcs["pred_mask"], UHCS_MULTICLASS_NAMES,
        output_path=os.path.join(figures_dir, "uhcs_multiclass_4panel_comparison.png")
    )

    sv_part = particles["sample_viz"]
    plot_segmentation_comparison(
        sv_part["img"], sv_part["true_mask"], sv_part["ws_mask"], sv_part["pred_mask"],
        output_path=os.path.join(figures_dir, "segmentation_4panel_comparison.png")
    )

    plot_per_class_iou(
        uhcs["per_class_iou_mean"], uhcs["per_class_iou_std"],
        output_path=os.path.join(figures_dir, "per_class_iou_uhcs.png")
    )

    plot_kfold_variance(
        uhcs["df_folds"], particles["df_folds"],
        output_path=os.path.join(figures_dir, "kfold_iou_variance.png")
    )

    _, ws_labels, _ = segment_microstructure_watershed(sv_part["img"])
    df_part, part_stats = compute_particle_morphology_and_distribution(ws_labels, pixel_scale_um=0.05)
    plot_particle_size_distribution(df_part, output_path=os.path.join(figures_dir, "particle_size_distribution.png"))

    plot_stereology_benchmark(
        uhcs["df_per_image"], output_path=os.path.join(figures_dir, "stereology_volume_fraction_benchmark_uhcs.png")
    )
    plot_stereology_benchmark(
        particles["df_per_image"], output_path=os.path.join(figures_dir, "stereology_volume_fraction_benchmark_particles.png")
    )

    print(f"\nSaved figures to {figures_dir}")

    print("\nParticle morphology (particles-subset sample micrograph):")
    for key, v in part_stats.items():
        print(f"    {key}: {v:.3f}" if isinstance(v, float) else f"    {key}: {v}")

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_full_pipeline()
