"""Loss and evaluation metrics for microstructure segmentation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class CombinedBCEDiceLoss(nn.Module):
    """BCE + soft Dice loss, weighted by bce_weight, to counter class imbalance in the phase mask."""

    def __init__(self, smooth: float = 1e-6, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-7
) -> Dict[str, float]:
    """Compares a predicted binary mask against ground-truth annotations."""
    p = (pred_mask > 0.5).astype(bool)
    t = (true_mask > 0.5).astype(bool)
    
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (p.sum() + t.sum() + eps)
    
    precision = (intersection + eps) / (p.sum() + eps)
    recall = (intersection + eps) / (t.sum() + eps)
    accuracy = (p == t).mean()

    pred_area_frac = float(p.mean())
    true_area_frac = float(t.mean())
    abs_area_err = abs(pred_area_frac - true_area_frac)
    rel_area_err = abs_area_err / (true_area_frac + eps) * 100.0
    
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(accuracy),
        "pred_volume_fraction": pred_area_frac,
        "true_volume_fraction": true_area_frac,
        "relative_area_error_pct": float(rel_area_err)
    }


def compute_per_class_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = -1,
    class_names: Optional[Dict[int, str]] = None,
    eps: float = 1e-7
) -> Dict[str, float]:
    """Per-class IoU for a multi-class integer-label mask (e.g. the uhcs_general 4-phase legend).

    Pixels equal to `ignore_index` in true_mask are excluded from every class's intersection and
    union (used here for the -1 scale-bar/metadata pixels, which are not a real microstructural
    phase). A class with zero union in the valid region (absent from both prediction and ground
    truth) is left out of both the per-class dict and the mean, rather than being counted as a
    trivial IoU of 1.0.
    """
    valid = np.ones_like(true_mask, dtype=bool) if ignore_index is None else (true_mask != ignore_index)

    per_class: Dict[str, float] = {}
    ious = []
    for c in range(num_classes):
        name = class_names[c] if class_names is not None else f"class_{c}"
        p_c = (pred_mask == c) & valid
        t_c = (true_mask == c) & valid

        union = np.logical_or(p_c, t_c).sum()
        if union == 0:
            continue

        intersection = np.logical_and(p_c, t_c).sum()
        iou = (intersection + eps) / (union + eps)
        per_class[f"iou_{name}"] = float(iou)
        ious.append(float(iou))

    per_class["mean_iou"] = float(np.mean(ious)) if ious else 0.0
    return per_class
