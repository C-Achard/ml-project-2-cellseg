from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json
import logging
from typing import List
from typing import Optional
from pathlib import Path
import warnings

import numpy as np

from models import model_SegResNet as SegResNet
from models import model_Swin as SwinUNetR
from models import model_VNet as VNet
from models import model_TRAILMAP as TRAILMAP_MS

from post_processing import binary_connected
from post_processing import binary_watershed

MODEL_LIST = {
    "VNet": VNet,
    "SegResNet": SegResNet,
    "TRAILMAP_MS": TRAILMAP_MS,
    "SwinUNetR": SwinUNetR,
}

INSTANCE_SEGMENTATION_METHOD_LIST = {
    "Watershed": binary_watershed,
    "Connected components": binary_connected,
}

FILE_STORAGE = Path.home() / Path("Desktop/Code/test")
WEIGHTS_PATH = Path(__file__).parent.resolve() / Path("models/pretrained")

logger = logging.getLogger(__name__)


def load_json_config(path):
    logger.info(f"Loading json config: {path}")
    f = open(path)
    data = json.load(f)
    return data


@dataclass_json
@dataclass
class ModelInfo:
    """Dataclass recording model info :
    - name (str): name of the model"""

    name: str
    model_input_size: Optional[int] = 64
    out_channels: Optional[int] = 1

    def get_model(self):
        try:
            return MODEL_LIST[self.name]
        except KeyError as e:
            warnings.warn(f"Model {self.name} is not defined")
            raise KeyError(e)

    @classmethod
    def load_from_json(cls, path):
        data = load_json_config(path)
        return cls.from_json(data)

    @staticmethod
    def get_model_name_list():
        logger.info(
            f"Model list :\n" + str([f"{name}\n" for name in MODEL_LIST.keys()])
        )
        return MODEL_LIST.keys()


@dataclass
class WeightsInfo:
    path: str = None
    custom: bool = False
    use_pretrained: Optional[bool] = False


################
# Post processing & instance segmentation


@dataclass
class Thresholding:
    enabled: bool = True
    threshold_value: float = 0.8


@dataclass
class Zoom:
    enabled: bool = False
    zoom_values: Optional[List[float]] = None


@dataclass
class InstanceSegConfig:
    enabled: bool = False
    method: str = "Watershed"
    threshold: Thresholding = Thresholding(threshold_value=0.9)
    small_object_removal_threshold: Thresholding = Thresholding(threshold_value=3)


@dataclass
class PostProcessConfig:
    zoom: Zoom = Zoom()
    thresholding: Thresholding = Thresholding()
    instance: InstanceSegConfig = InstanceSegConfig()


################
# Inference configs


@dataclass
class SlidingWindowConfig:
    window_size: int = 64
    window_overlap: float = 0.45


@dataclass
class InfererConfig:
    """Class to record params for Inferer plugin"""

    model_info: ModelInfo = None
    show_results: bool = False
    show_results_count: int = 5
    show_original: bool = True
    anisotropy_resolution: List[int] = None


@dataclass_json
@dataclass
class InferenceWorkerConfig:
    """Class to record configuration for Inference job"""

    device: str = "cpu"
    model_info: ModelInfo = ModelInfo("TRAILMAP_MS")
    weights_config: WeightsInfo = WeightsInfo()
    results_path: str = str(FILE_STORAGE)
    filetype: str = ".tif"
    keep_on_cpu: bool = False
    compute_stats: bool = False
    post_process_config: PostProcessConfig = PostProcessConfig()
    sliding_window_config: SlidingWindowConfig = SlidingWindowConfig()

    run_semantic_evaluation: bool = False
    run_instance_evaluation: bool = False
    compute_instance_boundaries: bool = False
    keep_boundary_predictions: bool = False
    # images_filepaths: str = None
    # layer: napari.layers.Layer = None
    image: Optional[np.array] = None

    @classmethod
    def load_from_json(cls, path=None):
        data = load_json_config(path)
        logger.info(f"Config loaded from json for inference worker")
        return cls.from_dict(data)

####
# Training configs
class TrainerConfig:
    def __init__(self, **kwargs):
        self.model_info = ModelInfo("TRAILMAP_MS")
        self.model_name = self.model_info.name
        self.weights_path = None
        self.validation_percent = None  # 0.8
        self.train_volume_directory = ()
        self.train_label_directory = ()
        self.validation_volume_directory = ()
        self.validation_label_directory = ()

        self.max_epochs = 50
        self.learning_rate = 3e-4
        self.val_interval = 1
        self.batch_size = 16
        self.results_path = ()
        self.weights_dir = ()
        self.sampling = True
        self.num_samples = 160
        self.sample_size = [64, 64, 64]
        self.do_augmentation = True
        self.deterministic = True
        self.grad_norm_clip = 1.0
        self.weight_decay = 0.00001
        self.compute_instance_boundaries = (
            False  # Change class loss weights in utils.get_loss TODO: choose in config
        )
        self.loss_function_name = "Dice loss"  # DiceCELoss
        self.plot_training_inputs = False

        for k, v in kwargs.items():
            # this will generate new attributes based on the supplementary arguments
            setattr(self, k, v)

@dataclass
class ImageStats:
    volume: List[float]
    centroid_x: List[float]
    centroid_y: List[float]
    centroid_z: List[float]
    sphericity_ax: List[float]
    image_size: List[int]
    total_image_volume: int
    total_filled_volume: int
    filling_ratio: float
    number_objects: int

    def get_dict(self):
        return {
            "Volume": self.volume,
            "Centroid x": self.centroid_x,
            "Centroid y": self.centroid_y,
            "Centroid z": self.centroid_z,
            # "Sphericity (volume/area)": sphericity_va,
            "Sphericity (axes)": self.sphericity_ax,
            "Image size": self.image_size,
            "Total image volume": self.total_image_volume,
            "Total object volume (pixels)": self.total_filled_volume,
            "Filling ratio": self.filling_ratio,
            "Number objects": self.number_objects,
        }
