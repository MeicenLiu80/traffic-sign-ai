# Traffic Sign AI

A Flask research website that detects traffic signs in road images and classifies each detected crop. The project combines a YOLO11n detector with a separate 326-class CNN classifier.

## Project goal and pipeline

1. **Detect:** YOLO11n finds sign locations in a full road scene and predicts bounding boxes.
2. **Crop:** The app extracts each detected region from the original image.
3. **Classify:** The augmented CNN predicts one of the project’s 326 selected classes for each crop.
4. **Present:** Flask returns an annotated image, predicted class names, and both model confidence scores.

The detector and classifier have different roles; they are not two competing models for the same task. The experiment compares a baseline CNN with a CNN trained using data augmentation. The website uses YOLO11n and the augmented CNN together.

> This is a research prototype. It is not validated for driving or safety decisions.

## Main features

- Upload JPG, JPEG, PNG, WEBP, or BMP images up to 10 MB.
- Use two bundled example images.
- View sign bounding boxes, predicted classes, and YOLO/CNN confidence values.
- Run inference on CPU or CUDA when the installed PyTorch build supports it.

A confidence value is not an accuracy score.

## Repository structure

```text
app.py                         Flask app and inference pipeline
models/                        Class mapping and model setup notes
notebooks/
  MTSD_Object_Detection.ipynb   Original YOLO data preparation/training/inference notebook
  02_CNN_Training_and_Experiments.ipynb
                                 CNN training code recovered from local IPython history
scripts/                       Original standalone inference scripts
templates/                      Website HTML
static/                         CSS, JavaScript, and demo visuals
data/manifests/                 Dataset manifests and processing metadata
results/                        Training histories, metrics, plots, and YOLO run files
docs/                           Experiment, data, recovery, and licensing notes
```

The full image dataset is not included. See [`data/README.md`](data/README.md) before running a training notebook.

## Quick start: website

Create a Python environment and install the project dependencies. PyTorch and torchvision must be compatible with one another and with the machine’s CPU/CUDA setup; use the official PyTorch selector if a platform-specific install is needed.

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser. The app binds to localhost and uses Flask’s development server, which is intended for a local demo only.

The app expects the following files in `models/`:

- `yolo_best.pt`
- `cnn_augmentation_best.pth`
- `final_class_to_index.json`

The trained checkpoints are included in `models/` and are tracked by Git. The dataset itself is not included; see `data/README.md` for its source and terms. To reproduce the models, follow the data preparation and training notebooks, then save the best checkpoints using the filenames listed above.

## Reproduce the experiments

1. Obtain the MTSD dataset from its official source and review its terms. Do not upload the full image collection to this repository.
2. Follow `data/README.md` to place dataset files in the expected local folders. The notebooks preserve the original project steps; update their project-folder paths for your clone and run them from top to bottom.
3. Run `notebooks/MTSD_Object_Detection.ipynb` to inspect MTSD annotations, prepare the one-class YOLO detection set, and train/evaluate the detector.
4. Run `notebooks/02_CNN_Training_and_Experiments.ipynb` to create the selected-class manifests, train the baseline and augmented CNNs, and save training histories and plots.
5. Copy the best detector and augmented classifier checkpoints to `models/` as listed above to run the website.

The recovered CNN experiment records seed 42, 64 × 64 inputs, batch size 64, 20 epochs, Adam with learning rate 0.001, and cross-entropy loss. Its augmentation pipeline uses 10-degree random rotation, 5% translation, scale 0.95–1.05, and color jitter. See [`docs/SOURCE_RECOVERY.md`](docs/SOURCE_RECOVERY.md) for provenance and the full experiment notes.

The original environment record lists Python 3.14.6, PyTorch 2.14.1+cu126, Ultralytics 8.4.190, and an NVIDIA RTX 3060 Laptop GPU. These records are not a complete lockfile; exact dependency versions may need adjustment for another machine.

## Recorded results

All scores below are from validation data. They are not independent test-set scores.

| Experiment | Validation result | Source |
|---|---:|---|
| Baseline CNN | 90.66% best validation accuracy at epoch 19 | `results/PROJECT_STATUS.txt` |
| CNN with augmentation | 92.91% best validation accuracy at epoch 17 | `results/PROJECT_STATUS.txt` |
| YOLO11n detector | Precision 0.73038; recall 0.56097; mAP@50 0.60818; mAP@50–95 0.40564 | `results/yolo/results.csv` |

The CNN experiment reports 60,162 training crops, 8,736 validation crops, and 326 selected classes. The dataset preparation summary and raw training histories are in `results/` and `data/manifests/`.

The project uses a custom 326-class mapping. The MTSD paper describes a 313-class benchmark taxonomy, so the project’s selected mapping should not be described as the unchanged official taxonomy. Obtain MTSD from the [official Mapillary dataset page](https://www.mapillary.com/dataset/trafficsign); the full images are not redistributed here. Obtain MTSD from the [official Mapillary dataset page](https://www.mapillary.com/dataset/trafficsign); the full images are not redistributed here.

## Dataset and evaluation limitations

The local MTSD test split has 10,544 images, but MTSD test annotations are intentionally withheld to ensure fair evaluation. This repository therefore reports validation metrics only; it does not claim independent test accuracy or detection metrics. See the [MTSD paper](https://arxiv.org/abs/1909.04422).

The local dataset notice states CC BY-NC-SA 4.0 terms. The full dataset is not included; obtain it from the [official Mapillary source](https://www.mapillary.com/dataset/trafficsign) and follow its attribution and non-commercial terms. This repository uses AGPL-3.0; see [`LICENSE`](LICENSE) and [`docs/LICENSING.md`](docs/LICENSING.md).

## Project notes

- The CNN training code was recovered from the local IPython history and is preserved as a notebook. It has not been rewritten as a new training implementation.
- The notebooks and standalone helper scripts contain Windows paths from the original workstation. Update the path variables for a different clone location before running them.
- The model checkpoints are included in this repository. The full MTSD images and annotations must be obtained separately from their official source.
- The YOLO training target was 50 epochs, but the saved run history shows a resumed run ending at recorded epoch 29.
- The app displays confidence values for an unvalidated prototype. It should not be described as a safety-critical system.


## License

This project is licensed under AGPL-3.0. See [LICENSE](LICENSE) and [docs/LICENSING.md](docs/LICENSING.md) for dataset and attribution notes.

