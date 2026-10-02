# Experiment plan

## Baselines

1. Six-channel U-Net: concatenate RGB pre and post images at the input.
2. Siamese ReliefLensNet: shared ResNet-18 encoders + per-scale `pre`, `post`, `abs(diff)` fusion.

## Ablations

- `pre + post` vs `pre + post + abs(diff)` fusion
- CE only vs CE + Dice/Focal
- no calibration vs temperature scaling
- random tile split vs event split (report only to demonstrate leakage risk; use event split for the main result)

## Failure-analysis buckets

Save examples for:

- floods / large water changes
- wildfire burn scars
- dense urban blocks
- shadows
- seasonal/illumination changes
- imagery misregistration
- very small buildings
- building clusters touching each other

## Model selection

Select checkpoints using validation damage macro-F1. Never use test metrics to select checkpoints or temperature.
