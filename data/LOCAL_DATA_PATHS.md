# Local dataset paths

The full Mapillary Traffic Sign Dataset and historical training files are stored locally outside this Git repository, under the sibling desktop folder `道路标志项目归档\数据与训练`.

On the original computer, local directory junctions inside `data/` point to the archived annotations, train/validation/test images, and older `MTSD_Cropped` experiments. These junctions are ignored by Git and are not uploaded. The links let the notebooks and local training scripts use their documented repository-relative paths without copying dataset files into the repository.

On another computer, follow `data/README.md` to obtain the dataset from its official source, review the license, and place the files in the documented directories. Do not upload or redistribute the full dataset through this repository.
