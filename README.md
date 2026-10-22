# Microstructure Quantification

A pipeline that segments scanning-electron micrographs of ultrahigh carbon steel (UHCS), compares a U-Net with a classical Otsu/watershed baseline, and derives quantitative metrics from the masks: phase fraction (including a simulated ASTM E562 point count) and particle size and shape. Models are evaluated with K-fold cross-validation on the small annotated subsets, so every reported score is measured on held-out images.

## Motivation

Phase fraction and particle size are standard quantities in microstructure characterisation, but they depend on how the image is segmented. This project measures how much the choice of segmentation method changes those quantities, and how well a deep model trained on only a few dozen annotated images can do compared with a threshold-based method.

## Method

1. Self-supervised pretraining: a ResNet18 U-Net encoder (ImageNet initialisation) is trained to reconstruct masked, noisy patches of unlabeled UHCS images (`core/ssl_pretrain.py`).
2. Fine-tuning: a `segmentation_models_pytorch` U-Net is trained with 4-fold cross-validation on two annotated subsets, `uhcs_general` (4 classes) and `particles_spheroidite` (binary) (`benchmark/train_and_evaluate.py`).
3. Baseline: Gaussian smoothing, Otsu threshold and marker-controlled watershed (`core/classical_segmentation.py`).
4. Metrics: IoU, Dice and per-class IoU with an ignore index for scale-bar pixels (`core/loss_and_metrics.py`).
5. Stereology: simulated ASTM E562 point count with a 95% confidence interval, and equivalent-diameter and shape statistics for particles (`core/stereology.py`).

## Dataset

NIST Ultrahigh Carbon Steel micrograph database (DeCost et al.). The code reads the annotated subsets from `../data_check/uhcs/` and the unlabeled corpus (961 images, as described in `main.py`) from `../data_check/uhcs_full/`. The data are not included in this repository and the code gives no download URL; obtain them from the NIST UHCS database and place them at those paths.

## Results

Source: the labels printed in the figures in `outputs/figures/`. The repository contains no saved numeric result files or logs, so only values printed on the figures are quoted.

- `kfold_iou_variance.png`: held-out mean IoU of the U-Net across the 4 folds is 0.224 on the 4-class UHCS subset and 0.198 on the binary particles subset (mean line in each panel; fold values vary visibly, from roughly 0.1 to 0.3).
- `particle_size_distribution.png`: for the sample micrograph, the legend gives mean and median equivalent circular diameter of 0.23 um (watershed particles, pixel scale 0.05 um per pixel as set in `main.py`).
- `per_class_iou_uhcs.png`, `stereology_volume_fraction_benchmark_*.png` and the 4-panel comparisons are shown as plots without printed values. Run `python main.py` to reproduce and print the full numbers, including the watershed baseline IoU.

## Repository structure

```
main.py                      full pipeline
core/                        U-Net, SSL pretraining, loss/metrics, watershed, stereology
data/dataset_loader.py       image/mask pairing and loading
benchmark/                   K-fold training and evaluation
visualizer/                  figure generation
tests/                       unit tests
outputs/figures/             saved figures
```

## How to run

Dependencies are listed in `requirements.txt`.

```
pip install -r requirements.txt
python -m pytest tests
python main.py
```

`main.py` needs the dataset at the paths above. ImageNet encoder weights are downloaded on first run. It was configured for CPU (160x160 images, 8 epochs, 200 images x 5 epochs for pretraining).

## Limitations and next steps

- Each annotated subset has about 24 images, so fold-to-fold variance is large and the scores are low.
- Images are downsized to 160x160 and training is short; the pretraining uses a 200-image subsample of the 961-image corpus.
- Whether self-supervised pretraining helps was not tested against an ImageNet-only control.
- The watershed baseline is binary, so it is compared with the U-Net only on foreground versus background.
- Next steps: no saved metrics table yet; add one, and train longer at full resolution.

Mohnish Bansal, B.Tech Materials Engineering, IIT Delhi
