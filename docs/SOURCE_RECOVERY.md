# Source recovery notes

## Confirmed traffic sign project training history

The original workstation files are preserved. This document records where the submission copy's training code came from and how the experiments progressed.

### Anaconda / Jupyter history

The user's IPython history database is located at `C:\Users\ASUS\.ipython\profile_default\history.sqlite`. It contains relevant traffic sign project work across sessions 39–49 (October 25–28, 2026). Notebook inputs are stored, but the history database does not retain cell outputs for these sessions.

The recovered code is preserved in `notebooks/MTSD_Object_Detection.ipynb` and `notebooks/02_CNN_Training_and_Experiments.ipynb`.

- **Session 39 — CNN dataset and training (Oct. 25):** crop preparation, selected class manifests and quality checks, CNN definition, baseline training, augmented training, and plots. The ordered inputs are in the CNN notebook.
- **Session 40 — test split investigation (Oct. 26):** checks for official test images/annotations and a cell that attempted to make labeled test crops. This attempt depended on local annotations and images in several Desktop folders. It is not evidence of a valid test evaluation; test annotations were not available in the final project files, so this test-crop attempt is documented but not treated as a result.
- **Session 41 — YOLO data and training (Oct. 26):** annotation inspection, creation of a one-class YOLO detection dataset, visual checks, environment checks, an initial 50-epoch YOLO11n run, then a retry configuration with `workers=0` after a Windows multiprocessing issue. The retry used image size 640, batch size 8, patience 10, CUDA device 0, and a fresh run name `yolo11n_traffic_sign_retry`.
- **Sessions 42–45 — interrupted YOLO run:** repeated checkpoint checks and attempts to resume from `last.pt`. These show that training was interrupted and resumed; they do not establish that all 50 epochs completed. The saved run artifacts and recorded final epoch must be used to describe the achieved result.
- **Sessions 46–48 — evaluation and inference:** full validation-set metric evaluation, CNN checkpoint/architecture checks, integrated YOLO + CNN inference on validation images, and single-image YOLO checks after kernel restarts.
- **Session 49 — kernel recovery check:** only checks that the notebook kernel restarted.

### What the recovered training code says

The YOLO code proceeds through dataset creation, an initial attempt, a retry using `workers=0`, and checkpoint resumption. The selected published metric row is in `results/yolo/results.csv`; run parameters are in `results/yolo/args.yaml`. The recorded best validation row is epoch 29, so describe the experiment as a resumed/interrupted run with 29 recorded epochs, not as a completed 50-epoch training run.

CNN settings and training-history files are summarized in `docs/EXPERIMENTS.md` and `results/cnn/`. Since IPython inputs have no captured outputs, use the saved CSV histories and plots as the metric evidence rather than implying the history database itself contains output.

## Other local notebooks reviewed and excluded

- `C:\Users\ASUS\Downloads\CMPE401_Project1*.ipynb` describe a separate YOLOv11n project on VisDrone2019-DET, with its own `D:\CMPE401_Project1` paths and experiments. They are unrelated to the MTSD traffic sign website and were not copied into this submission.
- `C:\Users\ASUS\Downloads\visionPhase1.ipynb` and `visionPhase2.ipynb` describe a separate phase-based project with a Kaggle MTSD source, 96 × 96 inputs, a different curation/split, HOG/SVM and EfficientNet-B0 experiments, and Apple MPS hardware. Those settings do not match this Windows website project (326 classes, 64 × 64 CNN inputs, Flask + YOLO/CNN pipeline), so they were not merged.
- Other notebook matches in Downloads were dataset/tutorial references, not additional runs for this website.

## Terminal / console history

The persisted Windows PowerShell PSReadLine history contains only a command that changes into the website folder; it does not preserve the training commands. Windows Command Prompt does not normally keep a durable command history across closed sessions. The strongest available console-side evidence is therefore the saved YOLO run output (`results.csv`, `args.yaml`, plots/checkpoints) plus the notebook's resume cells. No separate terminal transcript or console training log was found in the inspected locations.
