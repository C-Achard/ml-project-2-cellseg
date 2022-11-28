import logging

import napari
from tifffile import imread
import matplotlib.pyplot as plt
from pathlib import Path

from config import InferenceWorkerConfig
from example import Inference

# file to run model on a folder and evaluate the results

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    repo_path = Path(__file__).resolve().parents[1]
    print(f"REPO PATH : {repo_path}")

    pred_conf = InferenceWorkerConfig()
    pred_conf.model_info.name = "TRAILMAP_MS"
    pred_conf.weights_config.path = str(
        repo_path / "models/pretrained/TRAILMAP_PyTorch.pth"
    )

    pred_conf.model_info.model_input_size = 128
    pred_conf.post_process_config.thresholding.threshold_value = 0.8

    pred_conf.results_path = str(repo_path / "test")
    Path(pred_conf.results_path).mkdir(exist_ok=True)

    pred_conf.image = imread(
        str(repo_path / "dataset/axons/volume-0.tiff")
    )

    worker = Inference(config=pred_conf)
    worker.log_parameters()
    worker.inference()

    ground_truth = imread(str(repo_path / "dataset/cropped_visual/val/lab/crop_lab_val.tif"))
    result = imread(str(repo_path / "test/semantic_labels/Semantic_labels_0_.tif"))

    from utils import dice_metric, normalize
    logger.info(f"DICE METRIC : {dice_metric(ground_truth, result)}")

    from napari.viewer import Viewer
    viewer = Viewer()
    viewer.add_labels(ground_truth, name="ground truth")

    prob_gradient = imread(str(repo_path / "test/prediction/prediction_.tif"))
    viewer.add_image(normalize(prob_gradient), name="result")
    viewer.add_image(prob_gradient, name="prediction", colormap="hsv")
    viewer.add_image(pred_conf.image, name="image", colormap="inferno")
    napari.run()

