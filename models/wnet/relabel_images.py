import sys

from pathlib import Path
sys.path.append(str(Path(__file__) / "../../../"))
repo_path = Path(__file__).resolve().parents[2]
import label.relabel as lab


path_true_labels_2=str(
        repo_path
        /"dataset_clean/cropped_visual/val/labels/crop_lab_val2.tif")
path_image_2=str(
        repo_path
        /"dataset_clean/cropped_visual/val/volumes/crop_vol_val2.tif")

lab.relabel(path_image_2,path_true_labels_2)
