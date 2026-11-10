# Standalone scripts

These files are the original project’s standalone inference utilities. In this draft, their project-root variable is configurable so it does not point to the original author's Windows account.

By default, the scripts look for the full local working dataset under `data/local_mtsd/`. Set the `MTSD_PROJECT_ROOT` environment variable if your local `MTSD_Cropped` working folder is elsewhere. These scripts expect the original working-folder structure, including the trained weights, class mapping, validation images, and YOLO output directories. The full image dataset is not included in the repository.

The website is started with `python app.py`; it reads its required model files directly from the repository's `models/` folder.
