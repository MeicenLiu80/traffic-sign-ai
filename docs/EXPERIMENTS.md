# Experiment record

This file summarizes the recorded experiments and links to the raw artifacts preserved in this repository. The metrics are validation results, not independent test results.

## CNN classification

Source: `results/PROJECT_STATUS.txt`, `results/cnn/comparison/Model_Comparison_Results.csv`, and the recovered training notebook `notebooks/02_CNN_Training_and_Experiments.ipynb`.

| Model | Best validation accuracy | Best epoch | Training duration in project status |
|---|---:|---:|---:|
| Baseline CNN | 90.66% | 19 | 21.04 minutes |
| CNN with data augmentation | 92.91% | 17 | 46.41 minutes |

The project status reports 60,162 training crops and 8,736 validation crops across 326 selected classes. The recovered IPython history includes the training loop, model architecture, preprocessing, augmentation, training settings, and plotting code. The dataset images themselves are not bundled.

## YOLO detection

Source: `results/yolo/results.csv` and `results/yolo/args.yaml`.

The final recorded validation row is epoch 29, with precision 0.73038, recall 0.56097, mAP@50 0.60818, and mAP@50–95 0.40564. The run configuration records a 50-epoch target, 640-pixel input, batch size 8, and a single `traffic_sign` detection class. The saved results include repeated epoch entries because training was resumed; consult the raw CSV and YAML when describing the training run.

## Result files

- `results/cnn/baseline/` — baseline history and training plots
- `results/cnn/augmentation/` — augmented CNN history
- `results/cnn/comparison/` — comparison table and validation plots
- `results/dataset/` — dataset class statistics and quality plots
- `results/yolo/` — YOLO metrics, arguments, training visualizations, and 20-image prediction CSV

## Test split

The local test image folder contains 10,544 images matching the official MTSD test split. MTSD does not release test annotations to preserve a fair benchmark. Therefore, the project has no test ground truth for independent metrics. Do not report validation scores above as test results. See the [MTSD paper](https://arxiv.org/abs/1909.04422).
