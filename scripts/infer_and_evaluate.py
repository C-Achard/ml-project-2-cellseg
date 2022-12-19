import logging

import napari
from tifffile import imread
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__) / "../../"))
from config import InferenceWorkerConfig
from example import Inference
from utils import dice_metric, normalize
from napari.viewer import Viewer

# file to run model on a folder and evaluate the results

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    repo_path = Path(__file__).resolve().parents[1]
    print(f"REPO PATH : {repo_path}")

    pred_conf = InferenceWorkerConfig()
    pred_conf.model_info.name = "SwinUNetR"
    pred_conf.weights_config.path = str(
        repo_path
        / "results"
        # / f"results_augmented_lr2/{pred_conf.model_info.name}_best_metric.pth"
        / f"results_augmented_lr2/{pred_conf.model_info.name}_checkpoint.pth"
        # repo_path / f"models/pretrained/Swin64_best_metric.pth"
    )
    pred_conf.model_info.out_channels = 3

    pred_conf.model_info.model_input_size = 64
    pred_conf.post_process_config.thresholding.enabled = True
    pred_conf.post_process_config.thresholding.threshold_value = 0.99

    pred_conf.sliding_window_config.window_size = 128
    pred_conf.sliding_window_config.window_overlap = 0.1

    pred_conf.results_path = str(repo_path / "test")
    Path(pred_conf.results_path).mkdir(exist_ok=True)

    # pred_conf.image = imread(str(repo_path / "dataset/visual_tif/volumes/images.tif"))
    pred_conf.image = imread(
        str(
            repo_path
            # / "dataset/somatomotor/augmented/c1images_with_artefact.tif"
            / "dataset/visual_tif/volumes/images.tif"
            # / "dataset/axons/validation/validation-set/volumes/volume-0.tiff"
            # / "dataset/images_with_artefacts/cropped_crop_12022_12_16_14_17_56.tif"
        )
    )

    worker = Inference(config=pred_conf)
    worker.log_parameters()
    worker.inference()

    # ground_truth = imread(str(repo_path / "dataset/cropped_visual/train/labels/artifact_test.tif"))
    ground_truth = None
    result = imread(str(repo_path / "test/semantic_labels/Semantic_labels_0_.tif"))

    viewer = Viewer()
    logger.debug(f"Result shape : {result.shape}")

    if ground_truth is not None:
        if pred_conf.model_info.out_channels > 1:
            logger.info(f"DICE METRIC : {dice_metric(ground_truth, result[1])}")
        else:
            logger.info(f"DICE METRIC : {dice_metric(ground_truth, result)}")
        viewer.add_labels(ground_truth, name="ground truth")

    prob_gradient = imread(str(repo_path / "test/prediction/prediction_.tif"))
    viewer.add_image(normalize(prob_gradient), name="result")
    viewer.add_image(prob_gradient, name="prediction", colormap="hsv")
    viewer.add_image(pred_conf.image, name="image", colormap="inferno")

    # from monai.transforms import Activations
    # viewer.add_image(Activations(softmax=True)(result).numpy(), name="softmax", colormap="inferno")

    napari.run()

    # TODO(cyril) :
    # - test generalized Dice
    # - test new artifacts
    # - test much longer training
    # - test larger LR : mediocre results

