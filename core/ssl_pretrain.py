"""Self-supervised pretraining of the segmentation encoder on the 961-image UHCS corpus,
used to initialize the K-fold fine-tuning step in benchmark/train_and_evaluate.py.

Pretext task: mask random patches + add Gaussian noise, train the resnet18 U-Net encoder
(with a 1-channel reconstruction head instead of the class head) to denoise the image back,
starting from ImageNet weights rather than random init -- a full 961 images is too slow to
pretrain on CPU within the time budget, see pretrain_ssl_encoder() for the actual subsample
size and epoch count used.
"""

import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image


def load_uhcs_full_corpus_images(
    corpus_dir: str,
    num_images: int = 300,
    target_size: Tuple[int, int] = (160, 160),
    seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """Loads a random, seeded subset of `num_images` grayscale micrographs from the full
    961-image UHCS corpus (unlabeled -- no pixel annotations accompany this extraction, only
    a per-image metadata spreadsheet of imaging/heat-treatment conditions, which is not a
    segmentation label and is not used here).

    Passing num_images >= the corpus size loads the whole corpus.
    """
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.png")))
    if not files:
        raise FileNotFoundError(f"No .png images found under {corpus_dir}")

    rng = np.random.RandomState(seed)
    if num_images < len(files):
        chosen_idx = rng.choice(len(files), size=num_images, replace=False)
        chosen = [files[i] for i in sorted(chosen_idx)]
    else:
        chosen = files

    imgs = []
    for f in chosen:
        img = Image.open(f).convert("L").resize((target_size[1], target_size[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        imgs.append(arr)

    return np.stack(imgs, axis=0).astype(np.float32), chosen


def corrupt_batch(
    clean: torch.Tensor,
    mask_ratio: float = 0.35,
    patch_size: int = 16,
    noise_std: float = 0.12,
    rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Corrupts a (B, 1, H, W) batch of clean, normalized [0, 1] images for the denoising /
    masked-patch pretext task: a random `mask_ratio` fraction of non-overlapping
    patch_size x patch_size patches are zeroed out, and i.i.d. Gaussian noise (std=noise_std)
    is added everywhere, then the result is clamped back to [0, 1]. The model is trained to
    reconstruct the original `clean` tensor from this corrupted input.
    """
    B, C, H, W = clean.shape
    corrupted = clean.clone()

    ph, pw = H // patch_size, W // patch_size
    if ph > 0 and pw > 0:
        num_patches = ph * pw
        num_mask = max(1, int(round(num_patches * mask_ratio)))
        for b in range(B):
            patch_idx = torch.randperm(num_patches, generator=rng)[:num_mask]
            for idx in patch_idx.tolist():
                pi, pj = idx // pw, idx % pw
                y0, x0 = pi * patch_size, pj * patch_size
                corrupted[b, :, y0:y0 + patch_size, x0:x0 + patch_size] = 0.0

    noise = torch.randn(clean.shape, generator=rng) * noise_std
    corrupted = torch.clamp(corrupted + noise, 0.0, 1.0)
    return corrupted


def build_ssl_autoencoder(encoder_name: str = "resnet18", encoder_weights: Optional[str] = "imagenet") -> nn.Module:
    """Same encoder/decoder backbone as build_pretrained_unet (core/unet_model.py), but with a
    1-channel sigmoid reconstruction head instead of a class-logit head, so the resulting
    `model.encoder` state dict is directly compatible with build_pretrained_unet's encoder.

    encoder_weights="imagenet" (default) is what main.py's real pretraining step uses (the
    documented ImageNet-init -> SSL-pretrain -> fine-tune chain). encoder_weights=None is used
    by the fast unit test in tests/ to avoid a network weight download, matching the existing
    test_build_pretrained_unet_forward_pass convention in core/unet_model.py.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
        activation="sigmoid"
    )


def pretrain_ssl_encoder(
    corpus_dir: str,
    num_images: int = 200,
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    mask_ratio: float = 0.35,
    patch_size: int = 16,
    noise_std: float = 0.12,
    target_size: Tuple[int, int] = (160, 160),
    encoder_name: str = "resnet18",
    encoder_weights: Optional[str] = "imagenet",
    device: str = "cpu",
    seed: int = 42
) -> Dict[str, Any]:
    """Runs masked-patch denoising pretraining on a fixed, seeded subsample of the UHCS corpus
    (loaded once, `num_epochs` full passes over it -- not resampled per epoch).

    Default 200 images / 5 epochs, not the full 961: a 300x6 run timed at ~5.4 min CPU train
    time alone, which would eat the budget main.py also needs for K-fold fine-tuning and
    figure generation, so this was scaled back rather than run at full size.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    t_load0 = time.time()
    images, used_files = load_uhcs_full_corpus_images(corpus_dir, num_images=num_images, target_size=target_size, seed=seed)
    load_time = time.time() - t_load0

    model = build_ssl_autoencoder(encoder_name=encoder_name, encoder_weights=encoder_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
    n = X.shape[0]

    model.train()
    loss_history: List[float] = []
    t_train0 = time.time()
    for epoch in range(num_epochs):
        perm = torch.randperm(n, generator=gen)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            clean = X[idx].to(device)
            corrupted = corrupt_batch(clean, mask_ratio=mask_ratio, patch_size=patch_size, noise_std=noise_std, rng=gen)

            optimizer.zero_grad()
            recon = model(corrupted)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        loss_history.append(float(np.mean(epoch_losses)))
    train_time = time.time() - t_train0

    return {
        "encoder_state_dict": model.encoder.state_dict(),
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "num_images": n,
        "num_epochs": num_epochs,
        "corpus_size": len(glob.glob(os.path.join(corpus_dir, "*.png"))),
        "load_time_s": load_time,
        "train_time_s": train_time,
        "used_files": used_files,
    }


def save_pretrained_encoder(pretrain_result: Dict[str, Any], output_path: str) -> None:
    """Saves the SSL-pretrained encoder state dict (plus bookkeeping metadata) to disk so
    build_pretrained_unet(..., pretrained_encoder_path=output_path) can load it as the
    fine-tuning starting point."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "encoder_state_dict": pretrain_result["encoder_state_dict"],
        "num_images": pretrain_result["num_images"],
        "num_epochs": pretrain_result["num_epochs"],
        "final_loss": pretrain_result["final_loss"],
    }, output_path)
