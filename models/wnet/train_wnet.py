"""
This file contains the code to train the WNet model.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import tifffile as tiff
import pickle
import wandb

from monai.data import CacheDataset, pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    SpatialPadd,
    EnsureTyped,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandZoomd,
    RandRotate90d,
)

from model import WNet
from config import Config, WANDB_CONFIG, WANDB_MODE
from soft_Ncuts import SoftNCutsLoss

sys.path.append("../..")
from utils import create_dataset_dict_no_labs, get_padding_dim

__author__ = "Yves Paychère, Colin Hofmann, Cyril Achard"


def train(weights_path = None):
    wandb.init(config=WANDB_CONFIG, project="WNet-ml-project", mode=WANDB_MODE)
    config = Config()
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if CUDA else "cpu")

    print("Config:")
    [print(a) for a in config.__dict__.items()]

    print("Initializing training...")
    ###################################################
    #               Getting the data                  #
    ###################################################
    print("Getting the data")

    (data_shape, dataset) = get_dataset(config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=pad_list_data_collate,
    )

    # import napari
    # for data in dataloader:
        # print(f"data test : {data.shape}")
        # images = data.detach().cpu().numpy()
        # images = data.detach().cpu().numpy()
        # labels = data["label"][0].numpy()
        # view = napari.Viewer()
        # view.add_image(images)
        # view.add_image(labels)
        # napari.run()
        # break

    ###################################################
    #               Training the model                #
    ###################################################
    print("Initializing the model:")

    print("- getting the model")
    # Initialize the model
    model = WNet(
        device=device,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
    )
    model = nn.DataParallel(model).cuda() if CUDA and config.parallel else model

    if weights_path is not None:
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=device
            )
        )

    wandb.watch(model, log_freq=200)

    print("- getting the optimizers")
    # Initialize the optimizers
    optimizerW = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizerE = torch.optim.Adam(model.encoder.parameters(), lr=config.lr)

    print("- getting the loss functions")
    # Initialize the Ncuts loss function
    criterionE = SoftNCutsLoss(
        data_shape=data_shape,
        device=device,
        o_i=config.o_i,
        o_x=config.o_x,
        radius=config.radius,
    )

    criterionW = nn.MSELoss()

    print("- getting the learning rate schedulers")
    # Initialize the learning rate schedulers
    schedulerW = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerW, mode="min", factor=0.5, patience=10, verbose=True
    )
    schedulerE = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerE, mode="min", factor=0.5, patience=10, verbose=True
    )

    model.train()

    print("Ready")
    print("Training the model")
    print("*" * 50)

    startTime = time.time()
    ncuts_losses = []
    rec_losses = []

    # Train the model
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1} of {config.num_epochs}")

        epoch_ncuts_loss = 0
        epoch_rec_loss = 0

        for batch in dataloader:
            image = batch.to(device)

            if config.batch_size == 1:
                image = image.unsqueeze(0)

            # Forward pass
            print(f"Input shape : {image.shape}")
            enc = model.forward_encoder(image)

            # Compute the Ncuts loss
            Ncuts = criterionE(enc, image)

            # Backward pass for the Ncuts loss
            optimizerW.zero_grad()

            Ncuts.backward()

            optimizerE.step()

            epoch_ncuts_loss += Ncuts.item()
            wandb.log({"NCuts loss":Ncuts.item()})

            # Forward pass
            enc, dec = model(image)

            # Compute the reconstruction loss
            reconstruction_loss = criterionW(dec, image)

            # Backward pass for the reconstruction loss
            optimizerE.zero_grad()

            reconstruction_loss.backward()

            optimizerW.step()

            epoch_rec_loss += reconstruction_loss.item()
            wandb.log({"Reconstruction loss": reconstruction_loss.item()})

        ncuts_losses.append(epoch_ncuts_loss / len(dataloader))
        wandb.log({"Ncuts loss_epoch": ncuts_losses[-1]})
        rec_losses.append(epoch_rec_loss / len(dataloader))
        wandb.log({"Reconstruction loss_epoch": rec_losses[-1]})
        print("Ncuts loss: ", ncuts_losses[-1])
        wandb.log({"learning_rate model": optimizerW.param_groups[0]["lr"]})
        wandb.log({"learning_rate encoder": optimizerE.param_groups[0]["lr"]})

        if epoch > 0:
            print(
                "Ncuts loss difference: ",
                ncuts_losses[-1] - ncuts_losses[-2],
            )
        print("Reconstruction loss: ", rec_losses[-1])
        if epoch > 0:
            print(
                "Reconstruction loss difference: ",
                rec_losses[-1] - rec_losses[-2],
            )

        # Update the learning rate
        schedulerE.step(epoch_ncuts_loss)
        schedulerW.step(epoch_rec_loss)

        print(
            "ETA: ",
            (time.time() - startTime) * (config.num_epochs / (epoch + 1) - 1) / 60,
            "minutes",
        )
        print("-" * 20)

        # Save the model
        if config.save_model and epoch % config.save_every == 0:
            torch.save(model.state_dict(), config.save_model_path)
            with open(config.save_losses_path, "wb") as f:
                pickle.dump((ncuts_losses, rec_losses), f)

    print("Training finished")
    print("*" * 50)

    # Save the model
    if config.save_model:
        print("Saving the model")
        torch.save(model.state_dict(), config.save_model_path)
        with open(config.save_losses_path, "wb") as f:
            pickle.dump((ncuts_losses, rec_losses), f)

    model_artifact = wandb.Artifact(
        "WNet-ml-project",
        type="model",
        description="WNet-ml-project",
        metadata=dict(WANDB_CONFIG),
    )
    model_artifact.add_file(config.save_model_path)
    wandb.log_artifact(model_artifact)

    return ncuts_losses, rec_losses, model


def get_dataset(config):
    """Creates a Dataset from the original data using the tifffile library

    Args:
        config (Config): The configuration object

    Returns:
        (tuple): A tuple containing the shape of the data and the dataset
    """

    train_files = create_dataset_dict_no_labs(
        volume_directory=config.train_volume_directory
    )
    train_files = [d.get("image") for d in train_files]
    volumes = tiff.imread(train_files).astype(np.float32)
    volume_shape = volumes.shape

    dataset = CacheDataset(data=volumes)

    return (volume_shape, dataset)


def get_dataset_monai(config):
    """Creates a Dataset applying some transforms/augmentation on the data using the MONAI library

    Args:
        config (Config): The configuration object

    Returns:
        (tuple): A tuple containing the shape of the data and the dataset
    """
    train_files = create_dataset_dict_no_labs(
        volume_directory=config.train_volume_directory
    )
    print(train_files)
    print(len(train_files))
    print(train_files[0])
    first_volume = LoadImaged(keys=["image"])(train_files[0])
    first_volume_shape = first_volume["image"].shape

    # Transforms to be applied to each volume
    load_single_images = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="PLI"),
            SpatialPadd(
                keys=["image"],
                spatial_size=(get_padding_dim(first_volume_shape)),
            ),
            EnsureTyped(keys=["image"]),
        ]
    )

    if config.do_augmentation:
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
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
                RandRotate90d(keys=["image"], prob=0.1, max_k=3),
                EnsureTyped(keys=["image"]),
            ]
        )
    else:
        train_transforms = EnsureTyped(keys=["image"])

    # Create the dataset
    dataset = CacheDataset(
        data=train_files, transform=Compose(load_single_images, train_transforms)
    )

    return first_volume_shape, dataset


if __name__ == "__main__":

    from pathlib import Path
    # weights_location = str(Path(__file__).resolve().parent / "2_class/test_wnet_2class.pth")
    train(
        # weights_location
    )
