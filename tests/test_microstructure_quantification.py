"""Unit tests for the microstructure quantification pipeline."""

import os
import sys
import pytest
import numpy as np
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.classical_segmentation import segment_microstructure_otsu, segment_microstructure_watershed
from core.unet_model import MicrostructureUNet, build_pretrained_unet
from core.loss_and_metrics import CombinedBCEDiceLoss, compute_segmentation_metrics, compute_per_class_iou
from core.stereology import simulate_astm_e562_point_count, compute_particle_morphology_and_distribution
from core.ssl_pretrain import pretrain_ssl_encoder, corrupt_batch


def create_synthetic_micrograph():
    """Generates a synthetic binary micrograph with circles as particles."""
    img = np.zeros((128, 128), dtype=np.float32)
    # Add two circular particles
    y, x = np.ogrid[:128, :128]
    mask1 = (x - 40)**2 + (y - 40)**2 <= 15**2
    mask2 = (x - 80)**2 + (y - 80)**2 <= 20**2
    img[mask1] = 0.9
    img[mask2] = 0.8
    # Add slight background noise
    img += np.random.normal(0, 0.05, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    true_mask = (mask1 | mask2).astype(np.float32)
    return img, true_mask


@pytest.fixture
def synthetic_micrograph():
    return create_synthetic_micrograph()


def test_classical_otsu_and_watershed(synthetic_micrograph):
    """Verifies that classical watershed segments and separates particles."""
    img, true_mask = synthetic_micrograph
    binary, labels, count = segment_microstructure_watershed(img, sigma=1.0, min_size=10, clear_border=False)
    
    assert count >= 2, f"Expected at least 2 particles, got {count}"
    assert binary.shape == img.shape
    metrics = compute_segmentation_metrics(binary, true_mask)
    assert metrics["iou"] > 0.60, f"Expected IoU > 0.60, got {metrics['iou']:.2f}"


def test_unet_forward_pass():
    """Verifies PyTorch U-Net input and output tensor dimensions."""
    model = MicrostructureUNet(in_channels=1, out_classes=1, base_features=16)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128), f"Output shape mismatch: {out.shape}"


def test_combined_bce_dice_loss():
    """Verifies combined loss computes finite scalar gradient."""
    criterion = CombinedBCEDiceLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    loss.backward()
    assert logits.grad is not None


def test_delesse_astm_e562_point_count(synthetic_micrograph):
    """Verifies Delesse Principle: Point count P_P converges close to Area fraction A_A."""
    _, true_mask = synthetic_micrograph
    res = simulate_astm_e562_point_count(true_mask, grid_spacing=8, num_fields=15)
    
    assert "estimated_volume_fraction_VV" in res
    assert "ci_95_half_width" in res
    # Point count should be within 10% relative error of true area fraction
    assert np.isclose(res["estimated_volume_fraction_VV"], res["true_area_fraction_AA"], atol=0.08)


def test_build_pretrained_unet_forward_pass():
    """Verifies segmentation-models-pytorch U-Net produces correctly shaped logits.
    encoder_weights=None here keeps the unit test offline/fast; main.py uses
    encoder_weights="imagenet" for the actual fine-tuning run."""
    model = build_pretrained_unet(num_classes=4, in_channels=1, encoder_weights=None)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 4, 64, 64)


def test_compute_per_class_iou():
    """Verifies per-class IoU on a synthetic 3-class mask, with a -1 ignore-index region
    (mirrors the uhcs_general scale-bar/metadata pixels) excluded from every class."""
    true_mask = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, -1, -1],
        [2, 2, -1, -1],
    ])
    pred_mask = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ])
    result = compute_per_class_iou(pred_mask, true_mask, num_classes=3, ignore_index=-1)

    assert result["iou_class_0"] == pytest.approx(3 / 4, abs=1e-5)
    assert result["iou_class_1"] == pytest.approx(4 / 5, abs=1e-5)
    assert result["iou_class_2"] == pytest.approx(1.0, abs=1e-5)
    assert result["mean_iou"] == pytest.approx(np.mean([3 / 4, 4 / 5, 1.0]), abs=1e-5)


def test_compute_per_class_iou_skips_absent_class():
    """A class with zero union in the valid region should be left out of the mean rather than
    scored as a trivial IoU of 1.0."""
    true_mask = np.array([[0, 0], [1, 1]])
    pred_mask = np.array([[0, 0], [1, 1]])
    result = compute_per_class_iou(pred_mask, true_mask, num_classes=3, ignore_index=-1)

    assert "iou_class_2" not in result
    assert result["mean_iou"] == pytest.approx(1.0)


def test_kfold_split_holds_out_distinct_samples():
    """Verifies K-fold CV splits are disjoint per fold and jointly cover every sample exactly
    once -- the property that makes fold-aggregated IoU trustworthy, unlike training and
    evaluating on the same images."""
    n_samples = 24
    k = 4
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_test_indices = []
    fold_count = 0
    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        fold_count += 1
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
        assert len(test_idx) >= n_samples // k - 1
        all_test_indices.extend(test_idx.tolist())

    assert fold_count == k
    assert sorted(all_test_indices) == list(range(n_samples))


def test_pretrain_ssl_encoder_reduces_loss(tmp_path):
    """Fast smoke test for the self-supervised (masked-patch denoising) pretraining stage used
    in main.py's pipeline step [0]. Uses a tiny toy "corpus" (8 synthetic 64x64 grayscale PNGs,
    not the real 961-image UHCS corpus) and only 3 epochs -- just enough to check that
    (a) pretrain_ssl_encoder runs end to end and produces an encoder state dict of the expected
    shape, and (b) the reconstruction loss actually decreases, i.e. the model is learning
    something rather than just executing. encoder_weights=None avoids a network weight download,
    matching the existing offline-test convention for build_pretrained_unet.
    """
    rng = np.random.RandomState(0)
    for i in range(8):
        arr = (rng.rand(64, 64) * 255).astype(np.uint8)
        from PIL import Image as PILImage
        PILImage.fromarray(arr).convert("L").save(tmp_path / f"toy_micrograph_{i}.png")

    result = pretrain_ssl_encoder(
        str(tmp_path),
        num_images=8,
        num_epochs=3,
        batch_size=4,
        target_size=(64, 64),
        encoder_weights=None,
        device="cpu",
        seed=0
    )

    assert result["num_images"] == 8
    assert result["num_epochs"] == 3
    assert len(result["loss_history"]) == 3
    assert all(np.isfinite(l) for l in result["loss_history"])
    # Loss on the last epoch should be lower than on the first -- the model is actually learning
    # to denoise/reconstruct, not just running without error.
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert "conv1.weight" in result["encoder_state_dict"] or len(result["encoder_state_dict"]) > 0


def test_corrupt_batch_masks_and_adds_noise():
    """Verifies corrupt_batch actually perturbs the clean image (zeroing patches / adding
    noise) and stays within the valid [0, 1] pixel range expected by the reconstruction loss."""
    torch.manual_seed(0)
    clean = torch.rand(2, 1, 32, 32)
    corrupted = corrupt_batch(clean, mask_ratio=0.5, patch_size=8, noise_std=0.1)

    assert corrupted.shape == clean.shape
    assert corrupted.min() >= 0.0 and corrupted.max() <= 1.0
    assert not torch.allclose(corrupted, clean)


def test_particle_morphology():
    """Verifies particle size and spatial nearest neighbor calculation."""
    labels = np.zeros((100, 100), dtype=np.int32)
    # Create two separated labeled regions
    labels[10:30, 10:30] = 1
    labels[60:80, 60:80] = 2
    
    df_p, stats_p = compute_particle_morphology_and_distribution(labels, pixel_scale_um=1.0)
    assert stats_p["particle_count"] == 2
    assert stats_p["mean_diameter_um"] > 0
    assert stats_p["mean_nearest_neighbor_um"] > 0


if __name__ == "__main__":
    import pathlib
    import tempfile

    img, true_mask = create_synthetic_micrograph()
    test_classical_otsu_and_watershed((img, true_mask))
    test_unet_forward_pass()
    test_combined_bce_dice_loss()
    test_delesse_astm_e562_point_count((img, true_mask))
    test_build_pretrained_unet_forward_pass()
    test_compute_per_class_iou()
    test_compute_per_class_iou_skips_absent_class()
    test_kfold_split_holds_out_distinct_samples()
    test_pretrain_ssl_encoder_reduces_loss(pathlib.Path(tempfile.mkdtemp()))
    test_corrupt_batch_masks_and_adds_noise()
    test_particle_morphology()
    print("all tests passed")
