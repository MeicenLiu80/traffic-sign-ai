# Dataset setup

The project uses the Mapillary Traffic Sign Dataset (MTSD). The full image dataset is not included in this repository. Obtain it from the [official Mapillary MTSD page](https://www.mapillary.com/dataset/trafficsign), review the license and attribution files included with the download, and cite the [MTSD paper](https://arxiv.org/abs/1909.04422). The local source dataset README states CC BY-NC-SA 4.0 terms. This repository links to the source and does not redistribute the complete images or annotations.

## Local files represented in this repository

- `manifests/` contains training, validation, and test manifests and processing reports.
- The training notebook reads the original MTSD annotation JSON, split lists, and scene images.
- CNN training uses selected traffic-sign crops and the 326-class project mapping.
- YOLO training uses full road scenes with one detection class: `traffic_sign`.

The copied training notebook uses repository-relative paths and should be started from the repository root. Place the original annotation files under `data/raw/mtsd_v2_fully_annotated/` and the scene images under `data/raw_images/train_part_1/images/`, `data/raw_images/train_part_2/images/`, `data/raw_images/train_part_3/images/`, and `data/raw_images/val/images/`. Keep train, validation, and test images separate and preserve the official split IDs.

The standalone scripts default to a complete original working-folder layout at `data/local_mtsd/`, or use the `MTSD_PROJECT_ROOT` environment variable to point to your local copy. Run the notebook preparation cells first to create the working crop and YOLO directories.

Do not add the full dataset images to Git. The official MTSD paper states that test annotations are withheld, so use the validation set for scored experiments and label those results as validation metrics.
