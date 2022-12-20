import logging
import numpy as np
import napari
from monai.transforms import AsDiscrete
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__) / "../../"))
from config import InferenceWorkerConfig
from example import Inference
from utils import dice_metric, normalize
from evaluate_model_performance import evaluate_model_performance, save_as_csv
from post_processing import binary_watershed
from napari.viewer import Viewer
"""
Previous code by Cyril Achard and Maxime Vidal
Adapted by Cyril
Runs inference on a single image and evaluate the results
"""

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO  )
    repo_path = Path(__file__).resolve().parents[1]
    print(f"REPO PATH : {repo_path}")

    pred_conf = InferenceWorkerConfig()
    pred_conf.model_info.name = "SwinUNetR"
    pred_conf.weights_config.path = str(
        repo_path
        / "results"
        # / f"results_DiceCE_monochannel_aug/{pred_conf.model_info.name}_best_metric.pth"
        / f"results_DiceCE_monochannel_aug/{pred_conf.model_info.name}_checkpoint.pth"
        # repo_path / f"models/pretrained/Swin64_best_metric.pth"
    )
    pred_conf.model_info.out_channels = 1

    pred_conf.model_info.model_input_size = 64
    pred_conf.post_process_config.thresholding.enabled = True
    pred_conf.post_process_config.thresholding.threshold_value = 0.99

    pred_conf.sliding_window_config.window_size = 128
    pred_conf.sliding_window_config.window_overlap = 0

    pred_conf.results_path = str(repo_path / "test")
    Path(pred_conf.results_path).mkdir(exist_ok=True)

    # pred_conf.image = imread(str(repo_path / "dataset/visual_tif/volumes/images.tif"))
    pred_conf.image = imread(
        str(
            repo_path
            # / "dataset/axons/training/custom_training/volumes_augmented/c1images_with_artefact.tif"
            / "dataset/visual_tif/volumes/0-visual.tif"
            # / "dataset/axons/validation/validation-set/volumes/volume-0.tiff"
            # / "dataset/axons/training/training-set/volumes/volume-0.tiff"
            # / "dataset/images_with_artefacts/cropped_crop_12022_12_16_14_17_56.tif"
        )
    )

    worker = Inference(config=pred_conf)
    worker.log_parameters()
    # worker.inference()

    ground_truth = imread(str(repo_path / "dataset/visual_tif/labels/testing_im_new_label.tif"))
    # ground_truth = None
    result = imread(str(repo_path / "test/semantic_labels/Semantic_labels_0_.tif"))
    if pred_conf.model_info.out_channels > 1:
        pre_instance = AsDiscrete(argmax=True, to_onehot=True)(result)
    else:
        pre_instance = AsDiscrete(threshold=0.8)(result)
    instance = binary_watershed(pre_instance, thres_seeding=0.95, thres_objects=0.4, thres_small=30)
    imwrite(str(repo_path / "test/Instance_labels.tif"), instance)

    view = Viewer()
    logger.debug(f"Result shape : {result.shape}")

    do_visualize = False
    if ground_truth is not None:
        if pred_conf.model_info.out_channels > 1:
            logger.info(f"DICE METRIC : {dice_metric(ground_truth, result[1])}")
            evaluation_stats = evaluate_model_performance(ground_truth, instance[1], visualize=do_visualize)
            # logger.info(f"MODEL PERFORMANCE : {}")
        else:
            logger.info(f"DICE METRIC : {dice_metric(ground_truth, result)}")
            evaluation_stats = evaluate_model_performance(ground_truth, instance, visualize=do_visualize)
            # logger.info(f"MODEL PERFORMANCE : {evaluate_model_performance(ground_truth, instance, visualize=do_visualize)}")
        view.add_labels(ground_truth, name="ground truth")
        save_as_csv(evaluation_stats, str(repo_path / "test/evaluation_stats.csv"))

    prob_gradient = imread(str(repo_path / "test/prediction/prediction_.tif"))
    view.add_image(normalize(prob_gradient), name="result")
    view.add_image(prob_gradient, name="prediction", colormap="hsv")
    view.add_image(pred_conf.image, name="image", colormap="inferno")
    view.add_labels(instance, name="instance")

    # from monai.transforms import Activations
    # view.add_image(Activations(softmax=True)(result).numpy(), name="softmax", colormap="inferno")

    napari.run()

    # TODO(cyril) :
    # - train baseline
    # - test new artifacts
    # - test much longer training
    # - test LR tuning
    # - test GradScaler


