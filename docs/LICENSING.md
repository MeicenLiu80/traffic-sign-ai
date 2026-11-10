# Licensing and attribution

## Project code and Ultralytics YOLO

This repository is released under the GNU Affero General Public License, version 3 (AGPL-3.0). See the root `LICENSE` file. The project uses the Ultralytics YOLO11n package and model; Ultralytics publishes its code and trained models under AGPL-3.0 by default and states that projects using them must provide their complete corresponding source under AGPL-3.0. This repository includes the Flask app, frontend, training and inference notebooks/scripts, configuration, and fine-tuned project checkpoints to support that release. See [Ultralytics' AGPL compliance guidance](https://docs.ultralytics.com/help/contributing/) and [license FAQ](https://www.ultralytics.com/license).

This is an educational, non-commercial course project. No Enterprise license is claimed.

## Dataset source and use

The project uses the Mapillary Traffic Sign Dataset (MTSD). The full images and annotations are not redistributed in this repository. Obtain the dataset from the [official Mapillary Traffic Sign Dataset page](https://www.mapillary.com/dataset/trafficsign), and read the terms provided with the downloaded files before use. The dataset paper describes the splits and notes that test annotations are withheld: [Ertler et al., ECCV 2020](https://arxiv.org/abs/1909.04422).

The local dataset README supplied with the source files states CC BY-NC-SA 4.0. Preserve the dataset's attribution and share-alike notices when using or adapting those materials. This project provides a link rather than redistributing the dataset. The 326-class mapping is a project-curated mapping and should not be represented as the unchanged official 313-class taxonomy.

## Checkpoints and demo assets

The YOLO checkpoint was fine-tuned with Ultralytics YOLO and is included in this AGPL-3.0 repository. The CNN checkpoints were trained using MTSD-derived crops; do not separate them from the dataset attribution and non-commercial terms when redistributing them. The project owner has approved inclusion of the website's current demo images and composite graphic in this project repository. Do not claim those visuals are MTSD images unless their provenance is independently confirmed.

## Third-party packages

Python dependencies remain under their own licenses. See package metadata and the upstream project pages for the current terms of Flask, PyTorch, torchvision, Pillow, OpenCV, pandas, matplotlib, and Ultralytics.

