"""Trains segmentation-models-pytorch U-Nets via K-fold cross-validation and benchmarks them
against classical watershed segmentation and ASTM E562 point counting.

Two-stage pipeline, matching the "961 images, 48 pixel-annotated" claim literally:
  Stage 1 (self-supervised pretraining, core/ssl_pretrain.py): a masked-patch denoising
    autoencoder built on the same resnet18 encoder is trained on a seeded subsample of the
    full 961-image unlabeled NIST UHCS corpus (Kaggle mirror of DeCost et al.) to adapt the
    ImageNet-initialized encoder filters to the SEM micrograph domain, with no pixel labels
    involved. Run from main.py's pipeline step [0]; see that module and
    core/ssl_pretrain.pretrain_ssl_encoder's docstring for the exact subsample size, epoch
    count, and CPU-time accounting actually used.
  Stage 2 (this module): the resulting encoder weights are loaded via
    build_pretrained_unet(..., pretrained_encoder_path=...) as the starting point for
    supervised K-fold fine-tuning on the two real pixel-annotated subsets (48 images total;
    see data_check/uhcs/README.txt):
  - "uhcs_general" (24 images): real 4-class legend (0=ferritic matrix, 1=proeutectoid carbide
    network, 2=spheroidite particles, 3=Widmanstatten lath). -1 (scale-bar/imaging metadata) is
    handled as an ignore-index, not merged into class 0 (see data/dataset_loader.py).
  - "particles_spheroidite" (24 images): genuinely binary legend (0=matrix, 1=spheroidite
    particle) -- kept binary rather than forced into the 4-class scheme.

With only 24 images per subset, a single train/test split is not trustworthy, so each subset is
evaluated with K-fold CV: a fresh model is trained per fold on that fold's train split (starting
from the shared SSL-pretrained encoder, when pretrained_encoder_path is supplied) and evaluated
only on its held-out split, then IoU/Dice are aggregated as mean +/- std across folds.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import KFold

from data.dataset_loader import UHCSDatasetLoader, UHCS_MULTICLASS_NAMES
from core.classical_segmentation import segment_microstructure_watershed
from core.unet_model import build_pretrained_unet
from core.loss_and_metrics import CombinedBCEDiceLoss, compute_segmentation_metrics, compute_per_class_iou
from core.stereology import simulate_astm_e562_point_count

UHCS_NUM_CLASSES = 4   # matrix(0), carbide network(1), spheroidite(2), Widmanstatten(3)
IGNORE_INDEX = -1       # scale-bar / imaging-metadata pixels: excluded from loss and from IoU


def _load_subset(loader: UHCSDatasetLoader, subset: str, target_size: Tuple[int, int], label_mode: str):
    pairs = [p for p in loader.get_paired_image_paths() if p["subset"] == subset and p["label_path"]]
    imgs, masks, names = [], [], []
    for p in pairs:
        img, mask = loader.load_image_and_mask(p["image_path"], p["label_path"], target_size=target_size, label_mode=label_mode)
        imgs.append(img)
        masks.append(mask)
        names.append(p["base_name"])
    return np.array(imgs), np.array(masks), names


def _train_multiclass_fold(
    images: np.ndarray, masks: np.ndarray, num_epochs: int, lr: float, device: str,
    pretrained_encoder_path: str = None
) -> nn.Module:
    model = build_pretrained_unet(
        num_classes=UHCS_NUM_CLASSES, in_channels=1, pretrained_encoder_path=pretrained_encoder_path
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    X = torch.tensor(images, dtype=torch.float32).unsqueeze(1).to(device)
    y = torch.tensor(masks, dtype=torch.long).to(device)

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def _train_binary_fold(
    images: np.ndarray, masks: np.ndarray, num_epochs: int, lr: float, device: str,
    pretrained_encoder_path: str = None
) -> nn.Module:
    model = build_pretrained_unet(
        num_classes=1, in_channels=1, pretrained_encoder_path=pretrained_encoder_path
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedBCEDiceLoss()

    X = torch.tensor(images, dtype=torch.float32).unsqueeze(1).to(device)
    y = torch.tensor(masks, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def run_uhcs_kfold_benchmark(
    k: int = 4,
    num_epochs: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    target_size: Tuple[int, int] = (160, 160),
    pretrained_encoder_path: str = None
) -> Dict[str, Any]:
    """K-fold CV over the uhcs_general (4-class) subset. A fresh smp.Unet(resnet18) is trained
    per fold and evaluated only on that fold's held-out images; classical watershed (no
    training) is run on the same held-out images for comparison.

    pretrained_encoder_path: when given (see core/ssl_pretrain.py + main.py step [0]), each
    fold's model starts from the self-supervised-pretrained encoder instead of plain ImageNet
    weights."""
    loader = UHCSDatasetLoader()
    imgs, masks, names = _load_subset(loader, "uhcs_general", target_size, label_mode="multiclass")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_records: List[Dict[str, Any]] = []
    per_image_rows: List[Dict[str, Any]] = []
    sample_viz = None

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(imgs)):
        model = _train_multiclass_fold(
            imgs[train_idx], masks[train_idx], num_epochs, lr, device,
            pretrained_encoder_path=pretrained_encoder_path
        )
        model.eval()

        fold_class_ious: Dict[int, List[float]] = {c: [] for c in range(UHCS_NUM_CLASSES)}
        fold_mean_ious, fold_ws_ious = [], []

        with torch.no_grad():
            for i in test_idx:
                img, true_mask = imgs[i], masks[i]
                inp_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                logits = model(inp_t)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

                per_class = compute_per_class_iou(
                    pred, true_mask, UHCS_NUM_CLASSES, ignore_index=IGNORE_INDEX, class_names=UHCS_MULTICLASS_NAMES
                )
                for c, cname in UHCS_MULTICLASS_NAMES.items():
                    key = f"iou_{cname}"
                    if key in per_class:
                        fold_class_ious[c].append(per_class[key])
                fold_mean_ious.append(per_class["mean_iou"])

                # Classical watershed only produces a binary (matrix vs any secondary phase) mask,
                # so it's compared on a binarized ground truth (-1 metadata excluded).
                true_binary = np.where(true_mask == IGNORE_INDEX, 0.0, (true_mask > 0).astype(np.float32)).astype(np.float32)
                ws_mask, ws_labels, ws_count = segment_microstructure_watershed(img, sigma=1.0, min_size=15)
                metrics_ws = compute_segmentation_metrics(ws_mask, true_binary)
                fold_ws_ious.append(metrics_ws["iou"])

                unet_vv = float((pred > 0).mean())
                astm_res = simulate_astm_e562_point_count(true_binary, grid_spacing=16, num_fields=10)

                per_image_rows.append({
                    "fold": fold_idx,
                    "micrograph": names[i],
                    "unet_mean_iou": per_class["mean_iou"],
                    "watershed_binary_iou": metrics_ws["iou"],
                    "true_vv": metrics_ws["true_volume_fraction"],
                    "watershed_vv": metrics_ws["pred_volume_fraction"],
                    "unet_vv": unet_vv,
                    "astm_e562_vv": astm_res["estimated_volume_fraction_VV"],
                    "astm_ci95": astm_res["ci_95_half_width"]
                })

                if sample_viz is None:
                    sample_viz = {"img": img, "true_mask": true_mask, "pred_mask": pred, "ws_mask": ws_mask}

        fold_record = {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "unet_mean_iou": float(np.mean(fold_mean_ious)),
            "watershed_binary_iou": float(np.mean(fold_ws_ious))
        }
        for c, cname in UHCS_MULTICLASS_NAMES.items():
            vals = fold_class_ious[c]
            fold_record[f"unet_iou_{cname}"] = float(np.mean(vals)) if vals else float("nan")
        fold_records.append(fold_record)

    df_folds = pd.DataFrame(fold_records)
    df_per_image = pd.DataFrame(per_image_rows)

    per_class_iou_mean = {cname: float(df_folds[f"unet_iou_{cname}"].mean()) for cname in UHCS_MULTICLASS_NAMES.values()}
    per_class_iou_std = {
        cname: (float(df_folds[f"unet_iou_{cname}"].std(ddof=1)) if k > 1 else 0.0)
        for cname in UHCS_MULTICLASS_NAMES.values()
    }

    return {
        "k": k,
        "df_folds": df_folds,
        "df_per_image": df_per_image,
        "unet_mean_iou_mean": float(df_folds["unet_mean_iou"].mean()),
        "unet_mean_iou_std": float(df_folds["unet_mean_iou"].std(ddof=1)) if k > 1 else 0.0,
        "watershed_binary_iou_mean": float(df_folds["watershed_binary_iou"].mean()),
        "watershed_binary_iou_std": float(df_folds["watershed_binary_iou"].std(ddof=1)) if k > 1 else 0.0,
        "per_class_iou_mean": per_class_iou_mean,
        "per_class_iou_std": per_class_iou_std,
        "sample_viz": sample_viz
    }


def run_particles_kfold_benchmark(
    k: int = 4,
    num_epochs: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    target_size: Tuple[int, int] = (160, 160),
    pretrained_encoder_path: str = None
) -> Dict[str, Any]:
    """K-fold CV over the particles_spheroidite (binary) subset.

    pretrained_encoder_path: when given (see core/ssl_pretrain.py + main.py step [0]), each
    fold's model starts from the self-supervised-pretrained encoder instead of plain ImageNet
    weights."""
    loader = UHCSDatasetLoader()
    imgs, masks, names = _load_subset(loader, "particles_spheroidite", target_size, label_mode="binary")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_records: List[Dict[str, Any]] = []
    per_image_rows: List[Dict[str, Any]] = []
    sample_viz = None

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(imgs)):
        model = _train_binary_fold(
            imgs[train_idx], masks[train_idx], num_epochs, lr, device,
            pretrained_encoder_path=pretrained_encoder_path
        )
        model.eval()

        fold_unet_ious, fold_unet_dices, fold_ws_ious = [], [], []

        with torch.no_grad():
            for i in test_idx:
                img, true_mask = imgs[i], masks[i]
                inp_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(inp_t)).squeeze().cpu().numpy()
                pred = (prob > 0.5).astype(np.uint8)

                metrics_unet = compute_segmentation_metrics(pred, true_mask)
                ws_mask, ws_labels, ws_count = segment_microstructure_watershed(img, sigma=1.0, min_size=15)
                metrics_ws = compute_segmentation_metrics(ws_mask, true_mask)

                fold_unet_ious.append(metrics_unet["iou"])
                fold_unet_dices.append(metrics_unet["dice"])
                fold_ws_ious.append(metrics_ws["iou"])

                astm_res = simulate_astm_e562_point_count(true_mask, grid_spacing=16, num_fields=10)

                per_image_rows.append({
                    "fold": fold_idx,
                    "micrograph": names[i],
                    "unet_iou": metrics_unet["iou"],
                    "unet_dice": metrics_unet["dice"],
                    "watershed_iou": metrics_ws["iou"],
                    "true_vv": metrics_ws["true_volume_fraction"],
                    "watershed_vv": metrics_ws["pred_volume_fraction"],
                    "unet_vv": metrics_unet["pred_volume_fraction"],
                    "astm_e562_vv": astm_res["estimated_volume_fraction_VV"],
                    "astm_ci95": astm_res["ci_95_half_width"]
                })

                if sample_viz is None:
                    sample_viz = {"img": img, "true_mask": true_mask, "pred_mask": pred, "ws_mask": ws_mask}

        fold_records.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "unet_iou": float(np.mean(fold_unet_ious)),
            "unet_dice": float(np.mean(fold_unet_dices)),
            "watershed_iou": float(np.mean(fold_ws_ious))
        })

    df_folds = pd.DataFrame(fold_records)
    df_per_image = pd.DataFrame(per_image_rows)

    return {
        "k": k,
        "df_folds": df_folds,
        "df_per_image": df_per_image,
        "unet_iou_mean": float(df_folds["unet_iou"].mean()),
        "unet_iou_std": float(df_folds["unet_iou"].std(ddof=1)) if k > 1 else 0.0,
        "unet_dice_mean": float(df_folds["unet_dice"].mean()),
        "unet_dice_std": float(df_folds["unet_dice"].std(ddof=1)) if k > 1 else 0.0,
        "watershed_iou_mean": float(df_folds["watershed_iou"].mean()),
        "watershed_iou_std": float(df_folds["watershed_iou"].std(ddof=1)) if k > 1 else 0.0,
        "sample_viz": sample_viz
    }
