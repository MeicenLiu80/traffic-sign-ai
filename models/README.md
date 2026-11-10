# Model files

This repository includes the trained checkpoints used by the project:

- `yolo_best.pt` — YOLO11n traffic-sign detector
- `cnn_augmentation_best.pth` — augmented CNN classifier used by the Flask application
- `baseline_cnn_best.pth` — baseline CNN retained for experiment comparison
- `final_class_to_index.json` — class-name to output-index mapping used by the CNN

The checkpoint files are tracked by Git and are covered by the project license and attribution notes in the repository root and `docs/LICENSING.md`. The CNN weights were trained using MTSD-derived crops; consult the MTSD source terms before reuse or redistribution. The website requires the exact class mapping associated with the classifier output indexes.
