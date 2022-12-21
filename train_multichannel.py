import logging
import warnings
from functools import partial
import pandas as pd
from pathlib import Path
import torch
import numpy as np

# MONAI
import napari
import tifffile
from monai.data import (
    CacheDataset,
    DataLoader,
    PatchDataset,
    # decollate_batch,
    pad_list_data_collate,
)

from monai.losses import DiceLoss, DiceCELoss
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
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

from config import TrainerConfig
from utils import (
    fill_list_in_between,
    create_dataset_dict,
    get_padding_dim,
    plot_tensor,
)

from os import environ

environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic training

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
"""
Adapted from code by Cyril Achard and Maxime Vidal, originally from MONAI tutorials
Author : Cyril Achard
Trains a model on several classes : can be artifacts or axons with our datasets
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
        self.out_channels = self.config.model_info.out_channels
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
        # self.compute_instance_boundaries = self.config.compute_instance_boundaries
        self.deterministic = self.config.deterministic

        self.loss_values = []
        self.validation_values = []
        self.validation_loss_values = []
        self.confusion = []
        self.df = None

        ######################
        # DEBUG
        self.testing_interval = 20
        self.plot_train = False
        self.show_grad = False
        self.plot_val = False

        if self.config.model_info.out_channels > 1:
            logger.info("Using SOFTMAX loss")
            # self.loss_function = DiceLoss( # NOTE : DiceLoss has not been updated to support multichannel
            #     softmax=True,
            #     to_onehot_y=True  # removed here, done at model level to account for possible error with images with single label
            # )
            self.loss_function = DiceCELoss(
                softmax=True,
                # lambda_dice=0.5,
                # to_onehot_y=True
                # removed here, done at model level to account for possible error with images with single label
            )
            # self.loss_function = DiceLoss(to_onehot_y=True)
        else:
            logger.info("Using SIGMOID loss")
            self.loss_function = DiceLoss(sigmoid=True)
        # self.loss_function = get_loss(self.config.loss_function_name, self.device)

    def make_train_csv(self):
        """Records the training stats in a csv file"""
        size_column = range(1, len(self.loss_values) + 1)

        if len(self.loss_values) == 0 or self.loss_values is None:
            warnings.warn("No loss values to add to csv !")
            return

        def prepare_validations(val):
            """Fills the gap due to val_interval with blank"""
            return fill_list_in_between(val, self.val_interval - 1, "")[
                : len(size_column)
            ]

        if len(self.confusion) == 1:
            confusion = self.confusion[0]
            sensi = [confusion[0]]
            spec = [confusion[1]]
            fall_out = [confusion[2]]
            miss = [confusion[3]]
        else:
            confusion = np.array(self.confusion)
            sensi = confusion[:, 0]
            spec = confusion[:, 1]
            fall_out = confusion[:, 2]
            miss = confusion[:, 3]

        self.df = pd.DataFrame(
            {
                "epoch": size_column,
                "mean loss": self.loss_values,
                "mean_validation": prepare_validations(self.validation_values),
                "mean_validation_loss": prepare_validations(
                    self.validation_loss_values
                ),
                "mean sensitivity": prepare_validations(sensi),
                "mean specificity": prepare_validations(spec),
                "mean fall out": prepare_validations(fall_out),
                "mean miss": prepare_validations(miss),
            }
        )
        path = str(
            Path(self.results_path)
            / f"{self.config.model_info.name}_{self.max_epochs}e_training.csv"
        )
        self.df.to_csv(path, index=False)

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
        logger.info(f"Number of output channels : {self.out_channels}")
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

        logger.info("-" * 20)

    def train(self):
        if self.config.deterministic:
            set_determinism(seed=42, use_deterministic_algorithms=True)

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
                out_channels=self.out_channels,
                dropout_prob=0.3,
            )
        elif self.config.model_info.name == "SwinUNetR":

            if self.sampling:
                size = self.sample_size
            else:
                size = first_volume_shape
            model = self.model_class.get_net(
                img_size=get_padding_dim(size),
                use_checkpoint=False,
                out_channels=self.out_channels,
            )
        else:
            model = self.model_class.get_net(out_channels=self.out_channels)

        model = torch.nn.DataParallel(model).to(self.device)
        # model = model.to(self.device)

        epoch_loss_values = []
        val_epoch_loss_values = []
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
                    LoadImaged(
                        keys=["image", "label"],
                        # ensure_channel_first=True, image_only=True, simple_keys=True
                    ),
                    # ToTensord(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    RandSpatialCropSamplesd(
                        keys=["image", "label"],
                        roi_size=(self.sample_size),
                        random_size=False,
                        num_samples=self.num_samples,
                    ),
                    Orientationd(keys=["image", "label"], axcodes="PLI"),
                    # SpatialPadd(
                    #     keys=["image", "label"],
                    #     spatial_size=(get_padding_dim(self.sample_size)),
                    # ),
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
                    # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
                    RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )
            val_transforms = Compose(
                [
                    # EnsureTyped(keys=["image", "label"]),
                    # Zoomd(keys=["image", "label"], zoom=[1, 1, 5], keep_size=True, ),
                    # ScaleIntensityRanged(
                    #     keys=["image"],
                    #     a_min=0,
                    #     a_max=2000,
                    #     b_min=0.0,
                    #     b_max=1.0,
                    #     clip=True,
                    # ),
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
                    LoadImaged(
                        keys=["image", "label"],
                        reader=tifffile.imread,
                        image_only=True,
                        simple_keys=True,
                    ),
                    EnsureChannelFirstd(
                        keys=["image", "label"], channel_dim=config.out_channels
                    ),
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
            view = napari.viewer.Viewer()
            for check_data in train_loader:
                logger.info(check_data.keys())
                image, label = (check_data["image"], check_data["label"])

                view.add_image(image.numpy())
                view.add_labels(label.numpy().astype(np.int8))
            napari.run()
            # image, label = (check_data["image"][0][0], check_data["label"][0][0])
            # logger.info(f"image shape: {image.shape}, label shape: {label.shape}")
            # plt.figure("check", (12, 6))
            # plt.subplot(1, 2, 1)
            # plt.title("image")6
            # plt.imshow(image[:, :, 40], cmap="gray")
            # plt.subplot(1, 2, 2)
            # plt.title("label")
            # plt.imshow(label[:, :, 40])
            # # plt.savefig('/home/maximevidal/Documents/cell-segmentation-models/results/imageinput.png')
            # plt.show()

        logger.info("Creating optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        logger.info("Creating scheduler")
        scheduler = ReduceLROnPlateau(
            optimizer, "max", patience=10, factor=0.5, verbose=True
        )
        logger.info("Creating scaler")
        scaler = torch.cuda.amp.GradScaler()

        dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        # dice_metric = GeneralizedDiceScore(include_background=False)
        confusion_matrix = ConfusionMatrixMetric(
            include_background=False,
            metric_name=["sensitivity", "specificity", "fall out", "miss rate"],
            reduction="mean",
            get_not_nans=False,
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
            val_epoch_loss = 0
            step = 0

            for batch_data in train_loader:

                step += 1
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )

                # optimizer.zero_grad()
                # logits = self.model_class.get_output(model, inputs)
                # if self.out_channels > 1:
                #     ohe_labels = one_hot(
                #         labels, num_classes=self.config.model_info.out_channels
                #     )
                # else:
                #     ohe_labels = labels

                # loss = self.loss_function(  # softmax is done by DiceLoss
                #     logits,
                #     ohe_labels,
                # )
                # loss.backward()
                # optimizer.step()
                # epoch_loss += loss.detach().item()

                with torch.cuda.amp.autocast():
                    if self.out_channels > 1:
                        ohe_labels = one_hot(
                            labels, num_classes=self.config.model_info.out_channels
                        )
                    else:
                        ohe_labels = labels
                    logits = self.model_class.get_output(model, inputs)
                    loss = self.loss_function(  # softmax is done by DiceLoss
                        logits,
                        ohe_labels,
                    )

                if self.plot_train:
                    test_logits = logits.detach().cpu().numpy()
                    test_logits = Activations(softmax=True)(test_logits)
                    test_logits = np.where(test_logits > 0.9, 1, 0)

                    if (epoch + 1) % self.testing_interval == 0 and step < 4:
                        logger.info("Plotting training")
                        # logger.info(f"Logits shape {test_logits.shape}")
                        # logger.info(f"Labels shape {ohe_labels.shape}")
                        logger.info("-----------------")

                        # view = napari.viewer.Viewer()
                        # view.add_image(test_logits[0].cpu().numpy(), name="test_logits")
                        # view.add_labels(ohe_labels[0].cpu().numpy().astype(np.int8), name="label")
                        # napari.run()
                        for j in range(self.config.batch_size):
                            # logger.info(f"Logits min {test_logits[j].min()}")
                            # logger.info(f"Logits max {test_logits[j].max()}")
                            # logger.info(f"Labels min {ohe_labels[j].min()}")
                            # logger.info(f"Labels max {ohe_labels[j].max()}")
                            for i in range(self.out_channels):
                                if i == 0 and self.out_channels > 1:
                                    continue
                                log = test_logits[j][i]
                                logger.debug(f"Train : Logits shape {log.shape}")
                                plot_tensor(log, f"Train : batch {j} logits", i)

                                labels_test = ohe_labels[j][i].detach().cpu().numpy()
                                logger.debug(f"Labels test shape {labels_test.shape}")
                                plot_tensor(labels_test, f"Train : batch {j} labels", i)

                # loss = self.loss_function(  # softmax is done by DiceLoss
                #     logits,
                #     ohe_labels,
                # )
                # loss.backward()

                if self.show_grad and (epoch + 1) % self.testing_interval == 0:
                    grad_model = model.module if hasattr(model, "module") else model
                    # logger.info(f"Out channels grad {model.out.conv[0].weight.grad}")
                    logger.info(
                        f"Out channels shape {grad_model.out.conv[0].weight.grad.shape}"
                    )
                    logger.info(f"NOTE : Scaler might show gradients as 0")
                    for i in range(self.out_channels):
                        logger.info(f"CHANNEL {i}")
                        grad = torch.abs(grad_model.out.conv[0].weight.grad)
                        logger.info(f"Out channel {i} shape {grad[i].shape}")
                        logger.info(
                            f"Out channel {i} mean {grad[i].view(grad[i].size(0), -1).mean(1)}"
                        )
                        # logger.info(f"Out channel {i} min {grad[i].min()}")
                        # logger.info(f"Out channel {i} max {grad[i].max()}")

                # optimizer.step()
                # epoch_loss += loss.detach().item()

                scaler.scale(loss).backward()
                epoch_loss += loss.detach().item()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                logger.info(
                    f"* {step - 1}/{len(train_ds) // train_loader.batch_size}, "
                    f"Train loss: {loss.detach().item():.4f}"
                )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            logger.info(f"Epoch: {epoch + 1}, Average loss: {epoch_loss:.4f}")
            self.loss_values.append(epoch_loss)

            if (epoch + 1) % self.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_step = 0
                    for val_data in val_loader:
                        val_step += 1
                        val_inputs, val_labels = (
                            val_data["image"].to(self.device),
                            val_data["label"].to(self.device),
                        )

                        val_outputs = self.model_class.get_validation(model, val_inputs)

                        if self.out_channels > 1:
                            ohe_val_labels = one_hot(
                                val_labels,
                                num_classes=self.config.model_info.out_channels,
                            )
                        else:
                            ohe_val_labels = val_labels
                        val_loss = self.loss_function(val_outputs, ohe_val_labels)
                        # wandb.log({"validation loss": val_loss.detach().item()})
                        logger.info(f"Validation loss: {val_loss.detach().item():.4f}")
                        val_epoch_loss += val_loss.detach().item()

                        # pred = decollate_batch(val_outputs)
                        # labs = decollate_batch(val_labels)

                        # logger.info(f"VAL LABELS SHAPE {val_labels.shape}")
                        # logger.info(f"LABS SHAPE {len(labs)}")
                        # logger.info(f"LABS 0 SHAPE {labs[0].shape}")

                        if self.out_channels > 1:
                            post_pred = Compose(
                                [
                                    Activations(softmax=True),
                                    AsDiscrete(
                                        argmax=True,
                                    ),  # , n_classes=2
                                    partial(
                                        one_hot, num_classes=self.out_channels, dim=0
                                    ),
                                ]
                            )

                            post_label = Compose(
                                [
                                    # partial(one_hot, num_classes=self.config.model_info.out_channels, dim=0)
                                ]
                            )

                            # post_label = AsDiscrete(
                            #     to_onehot=self.out_channels
                            # )  # , n_classes=2)

                        else:
                            post_pred = Compose(AsDiscrete(threshold=0.6), EnsureType())
                            post_label = EnsureType()

                        # [logger.info(f"Pred shape {p.shape}") for p in pred]
                        # [logger.info(f"lab shape {lab.shape}") for lab in labs]
                        # logger.info(f"VAL LABELS SHAPE {ohe_val_labels.shape}")
                        # for raw_label in ohe_val_labels:
                        #     logger.info(f"RAW LABEL SHAPE {raw_label[0].shape}")
                        #     plot_tensor(raw_label[0], "raw_label", 0)

                        post_outputs = [
                            post_pred(res_tensor) for res_tensor in val_outputs
                        ]
                        post_labels = [
                            # post_label(res_tensor) for res_tensor in ohe_val_labels
                            res_tensor
                            for res_tensor in ohe_val_labels
                        ]

                        if (
                            (epoch + 1) % self.testing_interval == 0
                            and self.plot_val
                            and val_step < 10
                        ):
                            logger.info("Plotting validation")
                            # logger.info(f"Val inputs shape {val_outputs[0].shape}")
                            # logger.info(f"Val labels shape {val_labels[0].shape}")
                            logger.info("-----------------")
                            # logger.info(f"Val inputs min {post_outputs[0].min()}")
                            # logger.info(f"Val inputs max {post_outputs[0].max()}")
                            # logger.info(f"Val labels min {post_labels[0].min()}")
                            # logger.info(f"Val labels max {post_labels[0].max()}")

                            for i in range(self.out_channels):
                                if i == 0 and self.out_channels > 1:
                                    continue
                                pred = post_outputs[0].cpu().numpy()
                                # logger.info(f"Pred shape {pred.shape}")
                                plot_tensor(pred[i], "Validation : Prediction", i)
                                lab = post_labels[0].cpu().numpy()
                                # logger.info(f"Lab shape {lab.shape}")
                                plot_tensor(lab[i], "Validation : Labels", i)

                        dice_metric(y_pred=post_outputs, y=post_labels)
                        confusion_matrix(y_pred=post_outputs, y=post_labels)

                    metric = dice_metric.aggregate().detach().item()
                    confusion_values = confusion_matrix.aggregate()  # .detach().item()
                    val_epoch_loss /= step
                    val_epoch_loss_values.append(val_epoch_loss)
                    self.validation_loss_values.append(val_epoch_loss)
                    self.validation_values.append(metric)
                    self.confusion.append(
                        [res.detach().item() for res in confusion_values]
                    )

                    if self.config.use_val_loss_for_validation:
                        metric += val_epoch_loss
                    scheduler.step(metric)
                    dice_metric.reset()

                    val_metric_values.append(metric)

                    # try:
                    self.make_train_csv()
                    # except Exception as e:
                    #     logger.info(e)

                    if metric >= best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        logger.info("Saving best metric model")

                        weights_filename = f"{self.config.model_info.name}_best_metric.pth"  # f"_epoch_{epoch + 1}

                        # DataParallel wrappers keep raw model object in .module attribute
                        raw_model = model.module if hasattr(model, "module") else model
                        torch.save(
                            raw_model.state_dict(),
                            Path(self.results_path) / weights_filename,
                        )
                        logger.info("Saving complete")

                    logger.info("Saving checkpoint model")

                    weights_filename = f"{self.config.model_info.name}_checkpoint.pth"

                    # DataParallel wrappers keep raw model object in .module attribute
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(
                        raw_model.state_dict(),
                        Path(self.results_path) / weights_filename,
                    )
                    logger.info("Saving complete")

                    logger.info(
                        f"Current epoch: {epoch + 1}, Current mean dice: {metric:.4f}"
                        f"\nConfusion matrix: {[res.detach().item() for res in confusion_values]}"
                        f"\nBest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

        logger.info("=" * 10)
        logger.info(
            f"Train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )


def initialize_config():
    config = TrainerConfig()
    config.model_info.name = "SwinUNetR"
    # config.validation_percent = 0.8 # None if commented -> use train/val folders instead

    config.val_interval = 2

    config.learning_rate = 1e-4
    config.use_val_loss_for_validation = False
    # config.plot_training_inputs = True
    config.sampling = True
    config.do_augmentation = False # disabled to ensure there were no issues with it
    config.num_samples = 15
    config.max_epochs = 100

    repo_path = Path(__file__).resolve().parents[0]
    logger.info(f"REPO PATH : {repo_path}")

    return config, repo_path


def start_train(config):
    logger.info(f"Saving to {config.results_path}")
    trainer = Trainer(config)
    trainer.log_parameters()

    #############
    # DEBUG
    trainer.testing_interval = 20
    trainer.plot_train = False
    trainer.show_grad = False
    trainer.plot_val = False
    #############

    trainer.train()


if __name__ == "__main__":
    """
    Trains the mono and multichannel models
    To reproduce the results of monochannel, use the indicated patths and channel numbers
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Starting training")
    config, repo_path = initialize_config()
    ###################################################
    ###################################################
    ###################################################
    ###################################################

    # CHANGE ONLY THE PARAMETERS BELOW for reproducibility

    # BATCH SIZE
    config.batch_size = 10  # change if memory issues

    # PATHS
    config.train_volume_directory = str(
        repo_path
        /
         "dataset_clean/somatomotor/volumes" # USE FOR : monochannel
        # "dataset_clean/somatomotor/augmented_volumes"  # USE FOR : monochannel_aug
        #  "dataset_clean/axons/training/custom-training/volumes" # USE FOR : multichannel
        # "dataset_clean/axons/training/custom-training/volumes_augmented" # USE FOR : multichannel_aug
    )
    config.train_label_directory = str(
        repo_path
        / "dataset_clean/somatomotor/lab_sem"  # USE FOR : monochannel and monochannel_aug
        # / "dataset_clean/axons/training/custom-training/labels" # USE FOR : multichannel and multichannel_aug
    )

    # use these if not using validation_percent
    config.validation_volume_directory = str(
        repo_path
        / "dataset_clean/visual_tif/volumes"  # USE FOR : monochannel and monochannel_aug
        # / "dataset_clean/axons/validation/custom-validation/volumes" # USE FOR : multichannel and multichannel_aug
    )
    config.validation_label_directory = str(
        repo_path
        / "dataset_clean/visual_tif/labels_sem"  # USE FOR : monochannel and monochannel_aug
        # / "dataset_clean/axons/validation/custom-validation/labels" # USE FOR : multichannel and multichannel_aug
    )

    # CHANGE CHANNELS FOR MONO/MULTI
    config.model_info.out_channels = 1  # USE : 1 for monochannel
    # config.model_info.out_channels = 3  # USE : 3 for multichannel

    save_folder = "results/TESTTEST-training_output"
    ###################################################
    ###################################################
    ###################################################
    ###################################################
    config.results_path = str(repo_path / save_folder)
    (repo_path / save_folder).mkdir(exist_ok=True)
    start_train(config)
