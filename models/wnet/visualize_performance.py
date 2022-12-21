import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__) / "../../../"))
repo_path = Path(__file__).resolve().parents[2]
import evaluate_model_performance as perf
import tifffile as tiff

models_label_2=tiff.imread(str(
        repo_path
        /"results/self supervised/original_2_instance.tif"
    ))
path_true_labels_2=str(
        repo_path
        /"dataset_clean/cropped_visual/val/labels/crop_lab_val2_new_label.tif")


perf.evaluate_model_performance(tiff.imread(path_true_labels_2),models_label_2, visualize=True)