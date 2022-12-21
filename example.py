import logging

import matplotlib.pyplot as plt

# MONAI
from monai.data import (
    CacheDataset,
    DataLoader,
    PatchDataset,
    decollate_batch,
    pad_list_data_collate,
)
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    SpatialPadd,
    ScaleIntensityRanged,
)
from monai.utils import set_determinism
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import get_loss, create_dataset_dict, get_padding_dim
from tifffile import imwrite

from config import InferenceWorkerConfig, TrainerConfig
from predict import Inference

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

"""
Demo of training/inference by Cyril Achard
Adapted from previous code by Maxime Vidal and Cyril Achard
"""


class Trainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        if self.config.validation_percent is not None:
            self.data_dicts = create_dataset_dict(
                volume_directory=self.config.train_volume_directory,
                label_directory=self.config.train_label_directory,
            )
        else:
            self.train_data_dict = create_dataset_dict(
                volume_directory=self.config.train_volume_directory,
                label_directory=self.config.train_label_directory,
            )

            self.val_data_dict = create_dataset_dict(
                volume_directory=self.config.validation_volume_directory,
                label_directory=self.config.validation_label_directory,
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {self.device} device")
        logger.info(f"Using torch : {torch.__version__}")
        self.model_class = self.config.model_info.get_model()
        self.weights_path = self.config.weights_path
        self.validation_percent = self.config.validation_percent
        self.max_epochs = self.config.max_epochs
        self.learning_rate = self.config.learning_rate
        self.val_interval = self.config.val_interval
        self.batch_size = self.config.batch_size
        self.results_path = self.config.results_path
        self.num_samples = self.config.num_samples
        self.sampling = self.config.sampling
        self.sample_size = self.config.sample_size
        self.do_augment = self.config.do_augmentation
        self.seed_dict = self.config.deterministic
        self.weight_decay = self.config.weight_decay
        self.plot_training_inputs = self.config.plot_training_inputs
        self.train_files = []
        self.val_files = []
        self.compute_instance_boundaries = self.config.compute_instance_boundaries
        self.deterministic = self.config.deterministic
        self.loss_function = get_loss(self.config.loss_function_name, self.device)

    def log_parameters(self):

        logger.info("Parameters summary: ")
        logger.info("-" * 20)

        if self.validation_percent is not None:
            logger.info(
                f"Percentage of dataset used for validation : {self.validation_percent * 100}%"
            )

        if self.deterministic:
            logger.info(f"Deterministic training is enabled")
            logger.info(f"Seed is 42")

        logger.info(f"Training for {self.max_epochs} epochs")
        logger.info(f"Loss function is : {str(self.loss_function)}")
        logger.info(f"Validation is performed every {self.val_interval} epochs")
        logger.info(f"Batch size is {self.batch_size}")
        logger.info(f"Learning rate is {self.learning_rate}")

        if self.sampling:
            logger.info(
                f"Extracting {self.num_samples} patches of size {self.sample_size}"
            )
        else:
            logger.info("Using whole images as dataset")

        if self.do_augment:
            logger.info("Data augmentation is enabled")

        if self.weights_path is not None:
            logger.info(f"Using weights from : {self.weights_path}")

        if self.compute_instance_boundaries:
            logger.info(f"Computing instance boundaries")

        logger.info("-" * 20)

    def train(self):
        if self.config.deterministic:
            set_determinism(seed=42)

        if not self.sampling:
            first_volume = LoadImaged(keys=["image"])(self.train_data_dict[0])
            first_volume_shape = first_volume["image"].shape

        if self.config.model_info.name == "SegResNet":
            if self.sampling:
                size = self.sample_size
            else:
                size = first_volume_shape
            logger.info(f"The volume size is {size}")
            model = self.model_class.get_net(
                input_image_size=get_padding_dim(size),
                out_channels=1,
                dropout_prob=0.3,
            )
        else:
            model = self.model_class.get_net()

        model = torch.nn.DataParallel(model).to(self.device)

        epoch_loss_values = []
        val_metric_values = []

        if self.validation_percent is not None:
            self.train_files, self.val_files = (
                self.data_dicts[
                    0 : int(len(self.data_dicts) * self.validation_percent)
                ],
                self.data_dicts[int(len(self.data_dicts) * self.validation_percent) :],
            )
        else:
            self.train_files, self.val_files = self.train_data_dict, self.val_data_dict

        if self.sampling:
            sample_loader = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    RandSpatialCropSamplesd(
                        keys=["image", "label"],
                        roi_size=(self.sample_size),
                        random_size=False,
                        num_samples=self.num_samples,
                    ),
                    Orientationd(keys=["image", "label"], axcodes="PLI"),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=(get_padding_dim(self.sample_size)),
                    ),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )

        if self.do_augment:  # TODO : investigate more augmentations
            train_transforms = Compose(
                [
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=0,
                        a_max=2000,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    # Zoomd(keys=["image", "label"], zoom=[1, 1, 5], keep_size=True, ),
                    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
                    RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )
            val_transforms = Compose(
                [
                    EnsureTyped(keys=["image", "label"]),
                    # Zoomd(keys=["image", "label"], zoom=[1, 1, 5], keep_size=True, ),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=0,
                        a_max=2000,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                ]
            )
        else:
            train_transforms = EnsureTyped(keys=["image", "label"])
            val_transforms = Compose(
                [
                    EnsureTyped(keys=["image", "label"]),
                ]
            )

        if self.sampling:
            logger.info("Create patches for train dataset")
            train_ds = PatchDataset(
                data=self.train_files,
                transform=train_transforms,
                patch_func=sample_loader,
                samples_per_image=self.num_samples,
            )
            logger.info("Create patches for val dataset")
            val_ds = PatchDataset(
                data=self.val_files,
                transform=val_transforms,
                patch_func=sample_loader,
                samples_per_image=self.num_samples,
            )

        else:
            load_single_images = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="PLI"),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=(get_padding_dim(first_volume_shape)),
                    ),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )
            logger.info("Cache dataset : train")
            train_ds = CacheDataset(
                data=self.train_files,
                transform=Compose(load_single_images, train_transforms),
            )
            logger.info("Cache dataset : val")
            val_ds = CacheDataset(data=self.val_files, transform=load_single_images)

        logger.info("Initializing DataLoaders")
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=pad_list_data_collate,
        )
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=4)

        if self.plot_training_inputs:
            logger.info("Plotting dataset")
            for check_data in train_loader:
                image, label = (check_data["image"][0][0], check_data["label"][0][0])
                print(f"image shape: {image.shape}, label shape: {label.shape}")
                plt.figure("check", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("image")
                plt.imshow(image[:, :, 40], cmap="gray")
                plt.subplot(1, 2, 2)
                plt.title("label")
                plt.imshow(label[:, :, 40])
                # plt.savefig('/home/maximevidal/Documents/cell-segmentation-models/results/imageinput.png')
                plt.show()

        logger.info("Creating optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, "max", patience=10, factor=0.5, verbose=True
        )

        dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        if self.compute_instance_boundaries:
            dice_metric_only_cells = DiceMetric(
                include_background=True, reduction="mean", get_not_nans=False
            )

        best_metric = -1
        best_metric_epoch = -1

        if self.weights_path is not None:
            logger.info("Loading weights")
            if self.weights_path == "use_pretrained":
                weights_file = self.model_class.get_weights_file()
                weights = Path(self.config.weights_dir) / weights_file
                self.weights_path = weights
            else:
                weights = Path(self.weights_path)

            # # original saved file with DataParallel
            # state_dict = torch.load(weights, map_location=self.device)
            # # create new OrderedDict that does not contain `module.`
            # from collections import OrderedDict
            #
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = "module." + k  # add `module.`
            #     new_state_dict[name] = v
            # # load params
            # model.load_state_dict(new_state_dict)
            # TODO(maxime) for DataParallel
            model.load_state_dict(
                torch.load(
                    weights,
                    map_location=self.device,
                )
            )

        if self.device.type == "cuda":
            logger.info(f"Using GPU : {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU")

        for epoch in range(self.max_epochs):
            # logger.info("\n")
            logger.info("-" * 10)
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")
            if self.device.type == "cuda":
                logger.info("Memory Usage:")
                alloc_mem = round(torch.cuda.memory_allocated(0) / 1024**3, 1)
                reserved_mem = round(torch.cuda.memory_reserved(0) / 1024**3, 1)
                logger.info(f"Allocated: {alloc_mem}GB")
                logger.info(f"Cached: {reserved_mem}GB")

            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )
                optimizer.zero_grad()
                outputs = self.model_class.get_output(model, inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                logger.info(
                    f"* {step - 1}/{len(train_ds) // train_loader.batch_size}, "
                    f"Train loss: {loss.detach().item():.4f}"
                )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            logger.info(f"Epoch: {epoch + 1}, Average loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(self.device),
                            val_data["label"].to(self.device),
                        )

                        val_outputs = self.model_class.get_validation(model, val_inputs)

                        val_loss = self.loss_function(val_outputs, val_labels)
                        # wandb.log({"validation loss": val_loss.detach().item()})
                        logger.info(f"Validation loss: {val_loss.detach().item():.4f}")

                        pred = decollate_batch(val_outputs)

                        labs = decollate_batch(val_labels)

                        if self.compute_instance_boundaries:
                            post_pred = AsDiscrete(argmax=True, to_onehot=3)
                            post_label = AsDiscrete(to_onehot=3)
                        else:
                            post_pred = Compose(AsDiscrete(threshold=0.6), EnsureType())
                            post_label = EnsureType()

                        val_outputs = [post_pred(res_tensor) for res_tensor in pred]
                        val_labels = [post_label(res_tensor) for res_tensor in labs]

                        dice_metric(y_pred=val_outputs, y=val_labels)

                        if self.compute_instance_boundaries:
                            val_labels = [
                                val_label[1, :, :, :] for val_label in val_labels
                            ]
                            val_outputs = [
                                val_output[1, :, :, :] for val_output in val_outputs
                            ]
                            dice_metric_only_cells(y_pred=val_outputs, y=val_labels)

                    metric = dice_metric.aggregate().detach().item()
                    scheduler.step(metric)
                    # wandb.log({"dice metric validation": metric})
                    dice_metric.reset()

                    if self.compute_instance_boundaries:
                        metric_cells = (
                            dice_metric_only_cells.aggregate().detach().item()
                        )
                        scheduler.step(metric_cells)
                        # wandb.log({"dice metric only cells validation": metric_cells})
                        dice_metric_only_cells.reset()

                    val_metric_values.append(metric)

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        logger.info("Saving best metric model")

                        weights_filename = f"{self.config.model_info.name}64_onechannel_best_metric.pth"  # f"_epoch_{epoch + 1}

                        # DataParallel wrappers keep raw model object in .module attribute
                        raw_model = model.module if hasattr(model, "module") else model
                        torch.save(
                            raw_model.state_dict(),
                            Path(self.results_path) / weights_filename,
                        )
                        logger.info("Saving complete")

                    logger.info(
                        f"Current epoch: {epoch + 1}, Current mean dice: {metric:.4f}"
                        f"\nBest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
        logger.info("=" * 10)
        logger.info(
            f"Train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )


import logging
from pathlib import Path
from pathlib import PurePath

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChannel,
    Compose,
    EnsureChannelFirst,
    EnsureType,
    ToTensor,
    Zoom,
    ScaleIntensityRange,
)

from post_processing import binary_watershed, binary_connected

SWIN = "SwinUNetR"


class Inference:
    def __init__(
        self, config: InferenceWorkerConfig = InferenceWorkerConfig(), logger=None
    ):
        self.config = config
        # print(self.config)
        # logger.debug(CONFIG_PATH)
        if logger is None:
            logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.DEBUG)

        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {self.config.device} device")
        logger.info("Using torch :")
        logger.info(torch.__version__)

    def log(self, message):
        logger.info(message)

    @staticmethod
    def create_inference_dict(images_filepaths):
        data_dicts = [{"image": image_name} for image_name in images_filepaths]
        return data_dicts

    def log_parameters(self):

        config = self.config

        self.log("\nParameters summary :")

        self.log(f"Model is : {config.model_info.name}")
        if config.post_process_config.thresholding.enabled:
            self.log(
                f"Thresholding is enabled at {config.post_process_config.thresholding.threshold_value}"
            )

        self.log(f"Window size is {config.sliding_window_config.window_size}")
        self.log(f"Window overlap is {config.sliding_window_config.window_overlap}\n")

        if config.keep_on_cpu:
            self.log(f"Dataset loaded to CPU")
        else:
            self.log(f"Dataset loaded on {config.device}")

        if config.post_process_config.zoom.enabled:
            self.log(
                f"Scaling factor : {config.post_process_config.zoom.zoom_values} (x,y,z)"
            )

        instance_config = config.post_process_config.instance
        if instance_config.enabled:
            self.log(
                f"Instance segmentation enabled, method : {instance_config.method}\n"
                f"Probability threshold is {instance_config.threshold.threshold_value:.2f}\n"
                f"Objects smaller than {instance_config.small_object_removal_threshold.threshold_value} pixels will be removed\n"
            )

    def instance_seg(self, to_instance, image_id: int = 0):

        if image_id is not None:
            self.log(f"\nRunning instance segmentation for image n°{image_id}")

        threshold = self.config.post_process_config.instance.threshold.threshold_value
        size_small = (
            self.config.post_process_config.instance.small_object_removal_threshold.threshold_value
        )
        method_name = self.config.post_process_config.instance.method

        if method_name == "Watershed":

            def method(image):
                return binary_watershed(image, threshold, size_small)

        elif method_name == "Connected components":

            def method(image):
                return binary_connected(image, threshold, size_small)

        else:
            raise NotImplementedError(
                "Selected instance segmentation method is not defined"
            )

        instance_labels = method(to_instance)

        instance_filepath = self.save_image(
            name=f"Instance_labels_{image_id}",
            image=instance_labels,
            folder="instance_labels",
        )

        self.log(
            f"Instance segmentation results for image n°{image_id} have been saved as:"
        )
        self.log(PurePath(instance_filepath).name)
        return instance_filepath

    def load_layer(self, volume):

        # data = np.squeeze(self.config.layer.data)

        volume = np.array(volume, dtype=np.int16)

        volume_dims = len(volume.shape)
        if volume_dims != 3:
            raise ValueError(
                f"Data array is not 3-dimensional but {volume_dims}-dimensional,"
                f" please check for extra channel/batch dimensions"
            )

        print("Loading layer\n")

        load_transforms = Compose(
            [
                ToTensor(),
                # anisotropic_transform,
                AddChannel(),
                # SpatialPad(spatial_size=pad),
                AddChannel(),
                # ScaleIntensityRange(
                #     a_min=0,
                #     a_max=2000,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=True,
                # ),
                # EnsureType(),
            ],
            # map_items=False,
            # log_stats=True,
        )

        self.log("\nLoading dataset...")
        input_image = load_transforms(volume)
        self.log("Done")
        return input_image

    def save_image(self, name, image, folder: str = None):
        # time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

        result_folder = ""
        if folder is not None:
            result_folder = folder

        folder_path = Path(self.config.results_path) / Path(result_folder)

        folder_path.mkdir(exist_ok=True)

        file_path = folder_path / Path(
            f"{name}_"
            +
            # f"{time}_" +
            self.config.filetype
        )
        imwrite(file_path, image)
        filename = PurePath(file_path).name

        self.log(f"\nPrediction saved as : {filename}")
        return file_path

    def aniso_transform(self, image):
        zoom = self.config.post_process_config.zoom.zoom_values
        if zoom is None:
            zoom = [1, 1, 1]
        anisotropic_transform = Zoom(
            zoom=zoom,
            keep_size=False,
            padding_mode="empty",
        )
        return anisotropic_transform(image[0])

    def model_output(
        self,
        inputs,
        model,
        post_process_transforms,
        softmax=False,
        post_process=True,
        aniso_transform=None,
    ):

        inputs = inputs.to("cpu")
        # print(f"Input size: {inputs.shape}")
        model_output = lambda inputs: post_process_transforms(
            self.config.model_info.get_model().get_output(model, inputs)
        )

        if self.config.keep_on_cpu:
            dataset_device = "cpu"
        else:
            dataset_device = self.device

        window_size = self.config.sliding_window_config.window_size
        window_overlap = self.config.sliding_window_config.window_overlap

        outputs = sliding_window_inference(
            inputs,
            roi_size=window_size,
            sw_batch_size=1,
            predictor=model_output,
            sw_device=self.device,
            device=dataset_device,
            overlap=window_overlap,
            progress=True,
        )

        out = outputs.detach().cpu()

        # if softmax:
        #     out = F.softmax(out, dim=1)
        # else:
        #     out = F.sigmoid(out)

        if aniso_transform is not None:
            out = aniso_transform(out)

        if post_process:
            out = np.array(out).astype(np.float32)
            out = np.squeeze(out)
            return out
        else:
            return out

    def inference(self, image_id: int = 0):

        try:
            dims = self.config.model_info.model_input_size
            # self.log(f"MODEL DIMS : {dims}")
            model_name = self.config.model_info.name
            model_class = self.config.model_info.get_model()
            self.log(model_name)

            weights_config = self.config.weights_config
            post_process_config = self.config.post_process_config

            if model_name == "SegResNet":
                model = model_class.get_net(
                    input_image_size=[
                        dims,
                        dims,
                        dims,
                    ],
                    out_channels=self.config.model_info.out_channels,
                )
            elif model_name == "SwinUNetR":

                # out_channels = 1
                # if self.config.compute_instance_boundaries:
                #     out_channels = 3
                model = model_class.get_net(
                    img_size=[dims, dims, dims],
                    use_checkpoint=False,
                    out_channels=self.config.model_info.out_channels,
                )
            else:
                model = model_class.get_net(
                    out_channels=self.config.model_info.out_channels
                )
            model = model.to(self.device)

            self.log_parameters()

            model.to(self.device)

            if not post_process_config.thresholding.enabled:
                post_process_transforms = EnsureType()
            elif (
                post_process_config.thresholding.enabled
                and self.config.model_info.out_channels > 1
            ):
                self.log("Using argmax and one-hot format")
                t = post_process_config.thresholding.threshold_value
                post_process_transforms = Compose(
                    [
                        AsDiscrete(argmax=True, to_onehot=3, threshold=t),
                        # Activations(softmax=True),
                        # EnsureType()
                    ]
                )
            else:
                t = post_process_config.thresholding.threshold_value
                post_process_transforms = Compose(
                    [AsDiscrete(threshold=t), EnsureType()]
                )

            self.log("\nLoading weights...")
            if weights_config.path is not None:
                weights_path = weights_config.path
            else:
                raise ValueError("No weights path provided")
                # downloader = WeightsDownloader()
                # downloader.download_weights(model_name, model_class.get_weights_file())
                # weights_path = str(Path(WEIGHTS_PATH) / model_class.get_weights_file())
            logger.info(f"Trying to load weights : {weights_path}")
            model.load_state_dict(
                torch.load(
                    weights_path,
                    map_location=self.device,
                )
            )
            self.log("Done")

            input_image = self.load_layer(self.config.image)
            model.eval()
            with torch.no_grad():
                self.log(f"Inference started on layer...")

                image = input_image.type(torch.FloatTensor)
                self.log("Saving original image...")
                logger.debug(f"Image shape : {image.shape}")
                self.save_image(
                    "original_volume", input_image.numpy(), self.config.results_path
                )

                self.log("Predicting...")

                use_softmax = self.config.model_info.out_channels > 1

                out = self.model_output(
                    image,
                    model,
                    EnsureType(),
                    softmax=use_softmax,
                    post_process=True,
                    aniso_transform=self.aniso_transform,
                )

                if self.config.compute_instance_boundaries:
                    out = F.softmax(out, dim=1)
                    out = np.array(out)  # .astype(np.float32)

                    if self.config.keep_boundary_predictions:
                        out = out[:, 1:, :, :, :]
                    else:
                        out = out[:, 1, :, :, :]
                    if self.config.post_process_config.thresholding.enabled:
                        out = (
                            out
                            > self.config.post_process_config.thresholding.threshold_value
                        )
                    logger.info(f" Output shape: {out.shape}")
                    out = np.squeeze(out)
                    logger.info(f" Output shape: {out.shape}")
                    if self.config.keep_boundary_predictions:
                        out = np.transpose(out, (0, 3, 2, 1))
                    else:
                        out = np.transpose(out, (2, 1, 0))
                else:
                    out = np.array(out)
                    logger.info(
                        f" Output max {out.max()}, output min {out.min()},"
                        f" output mean {out.mean()}, output median {np.median(out)}"
                    )

                self.log("Saving prediction...")
                self.save_image("prediction", out, "prediction")
                self.log("Done")

                # if self.transforms["thresh"][0]:
                #     out = out > self.transforms["thresh"][1]
                # logger.info(f" Output shape: {out.shape}")
                # out = np.squeeze(out)
                # logger.info(f" Output shape: {out.shape}")
                # if self.config.model_info.out_channels >1 :
                #     out = np.transpose(out, (0, 3, 2, 1))
                # else:
                #     out =np.transpose(out, (0, 2, 1))
                #
                # if self.config.run_semantic_evaluation:
                #     from evaluate_semantic import run_evaluation
                #
                #     run_evaluation(out)

                model.to("cpu")

                self.log("Saving semantic labels...")
                out = post_process_transforms(out)
                logger.info(
                    f" Output max {out.max()}, output min {out.min()},"
                    f" output mean {out.mean()}, output median {np.median(out)}"
                )
                logger.info(f" Output shape: {out.shape}")
                file_path = self.save_image(
                    name=f"Semantic_labels_{image_id}",
                    image=np.array(out).astype(np.float32),
                    folder="semantic_labels",
                )

                return file_path
        except Exception as e:
            raise e


if __name__ == "__main__":
    from tifffile import imread

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Starting training")

    config = TrainerConfig()
    config.validation_percent = 0.8
    config.train_volume_directory = str(Path("dataset/cropped_visual/train/vol"))
    config.train_label_directory = str(Path("dataset/cropped_visual/train/lab_sem"))

    # use these if not using validation_percent
    # config.val_volume_directory = str(Path.home() / "Desktop/Proj_bachelor/data/cropped_visual/val/vol")
    # config.val_label_directory = str(Path.home() / "Desktop/Proj_bachelor/data/cropped_visual/val/lab_sem")

    config.results_path = str("../results")

    config.num_samples = 1
    config.max_epochs = 10

    trainer = Trainer(config)
    # trainer.log_parameters()
    # trainer.train()

    pred_conf = InferenceWorkerConfig()
    # pred_conf.model_info.name = "SegResNet"
    pred_conf.model_info.name = "SwinUNetR"
    pred_conf.model_info.model_input_size = 128

    pred_conf.results_path = str("../test")
    pred_conf.weights_config.path = str()
    pred_conf.image = imread(str())

    worker = Inference(config=pred_conf)
    worker.log_parameters()
    worker.inference()
