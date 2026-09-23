"""Loads and pairs NIST UHCS micrographs with their ground-truth label masks."""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict, Optional

# Real 4-class legend for the "uhcs_general" subset (see data_check/uhcs/README.txt).
# -1 (scale-bar/imaging metadata) is deliberately NOT included here: it is handled as an
# ignore-index in the loss and in compute_per_class_iou rather than merged into class 0,
# since scale-bar/text-overlay pixels are not actually ferritic matrix.
UHCS_MULTICLASS_NAMES = {
    0: "ferritic_matrix",
    1: "proeutectoid_carbide_network",
    2: "spheroidite_particles",
    3: "widmanstatten_lath",
}


class UHCSDatasetLoader:
    def __init__(self, data_root: Optional[str] = None):
        if data_root is None:
            # default: sibling data_check/uhcs directory in the workspace
            default_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data_check", "uhcs")
            )
            self.data_root = default_path
        else:
            self.data_root = data_root

    def get_paired_image_paths(self) -> List[Dict[str, str]]:
        pairs = []

        img_dir_uhcs = os.path.join(self.data_root, "uhcs", "images")
        lbl_dir_uhcs = os.path.join(self.data_root, "uhcs", "labels")

        img_dir_part = os.path.join(self.data_root, "particles", "images")
        lbl_dir_part = os.path.join(self.data_root, "particles", "labels")
        
        search_dirs = [
            (img_dir_uhcs, lbl_dir_uhcs, "uhcs_general"),
            (img_dir_part, lbl_dir_part, "particles_spheroidite")
        ]
        
        for img_dir, lbl_dir, subset_name in search_dirs:
            if not os.path.exists(img_dir):
                continue
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")) + glob.glob(os.path.join(img_dir, "*.png")))
            for img_p in img_files:
                base_name = os.path.splitext(os.path.basename(img_p))[0]
                lbl_candidates = [
                    os.path.join(lbl_dir, f"{base_name}.tif"),
                    os.path.join(lbl_dir, f"{base_name}.png"),
                    os.path.join(lbl_dir, f"{base_name}_label.tif"),
                    os.path.join(lbl_dir, f"{base_name}_mask.png")
                ]
                lbl_p = None
                for c in lbl_candidates:
                    if os.path.exists(c):
                        lbl_p = c
                        break
                
                # Extract specimen ID (e.g. 'uhcs0006' -> '0006' or prefix)
                sample_id = base_name[:8]
                pairs.append({
                    "sample_id": sample_id,
                    "image_path": img_p,
                    "label_path": lbl_p,
                    "subset": subset_name,
                    "base_name": base_name
                })
                
        return pairs

    def load_image_and_mask(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = (256, 256),
        label_mode: str = "binary"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """label_mode="binary" (default) collapses the label to foreground/background, which is
        the correct representation for the particles_spheroidite subset (its real legend is
        already binary: 0=matrix, 1=spheroidite particle).

        label_mode="multiclass" returns the raw integer label array (-1, 0, 1, 2, 3) for the
        uhcs_general subset instead of collapsing it, so the real 4-phase legend isn't thrown
        away. -1 pixels are scale-bar/imaging metadata, not matrix; callers should treat -1 as
        an ignore-index (e.g. nn.CrossEntropyLoss(ignore_index=-1)) rather than merge it into
        class 0.
        """
        import skimage.io
        import skimage.transform

        try:
            img = skimage.io.imread(image_path)
            if img.ndim == 3:
                img = img[:, :, 0]
        except Exception:
            img_pil = Image.open(image_path).convert("L")
            img = np.array(img_pil)
            
        if target_size is not None:
            img = skimage.transform.resize(img, target_size, preserve_range=True, order=1)

        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_arr = (img - img_min) / (img_max - img_min)
        else:
            img_arr = np.zeros_like(img, dtype=np.float32)
        img_arr = img_arr.astype(np.float32)

        mask_arr = None
        if label_path is not None and os.path.exists(label_path):
            try:
                lbl = skimage.io.imread(label_path)
                if lbl.ndim == 3:
                    lbl = lbl[:, :, 0]
            except Exception:
                lbl_pil = Image.open(label_path)
                lbl = np.array(lbl_pil)
                
            if target_size is not None:
                lbl = skimage.transform.resize(lbl, target_size, preserve_range=True, order=0)

            if label_mode == "multiclass":
                # Raw multi-class labels: -1=metadata, 0=matrix, 1=carbide network,
                # 2=spheroidite, 3=Widmanstatten lath. Kept as-is (not binarized, -1 not merged).
                mask_arr = np.round(lbl).astype(np.int64)
            else:
                # Foreground phases: values > 0 (carbide network=1, spheroidite=2, Widmanstatten=3)
                # -1 is metadata background, 0 is ferritic matrix
                mask_arr = (lbl > 0).astype(np.float32)
            
        return img_arr, mask_arr
