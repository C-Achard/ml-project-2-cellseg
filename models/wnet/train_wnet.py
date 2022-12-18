import torch
import torch.nn as nn

from models.wnet.model import WNet
import models.wnet.crf as crf
from models.wnet.config import Config
from models.wnet.soft_Ncuts import SoftNCutsLoss

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

from utils import create_dataset_dict, get_padding_dim


def main():
    config = Config()

    ###################################################
    #               Getting the data                  #
    ###################################################

    (data_shape, dataset) = get_dataset(config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=pad_list_data_collate,
    )

    ###################################################
    #               Training the model                #
    ###################################################

    CUDA = torch.cuda.is_available()

    # Initialize the model
    model = WNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
    )
    model = nn.DataParallel(model).cuda() if CUDA else model

    # Initialize the optimizers
    optimizerW = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizerE = torch.optim.Adam(model.encoder.parameters(), lr=config.lr)

    # Initialize the Ncuts loss function
    criterionE = SoftNCutsLoss(
        data_shape=data_shape, o_i=config.o_i, o_x=config.o_x, radius=config.radius
    )

    criterionW = nn.MSELoss()

    # Initialize the learning rate schedulers
    schedulerW = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerW, mode="min", factor=0.5, patience=10, verbose=True
    )
    schedulerE = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerE, mode="min", factor=0.5, patience=10, verbose=True
    )

    model.train()

    # Train the model
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1} of {config.num_epochs}")

        epoch_ncuts_loss = 0
        epoch_rec_loss = 0

        for batch in dataloader:
            image = batch["image"].cuda() if CUDA else batch["image"]

            # Forward pass
            enc, dec = model(image)

            # Compute the Ncuts loss
            Ncuts = criterionE(enc, image)

            # Backward pass for the Ncuts loss
            optimizerW.zero_grad()

            Ncuts.backward()

            optimizerE.step()

            epoch_ncuts_loss += Ncuts.item()

            # Forward pass
            enc, dec = model(image)

            # Compute the reconstruction loss
            reconstruction_loss = criterionW(dec, image)

            # Backward pass for the reconstruction loss
            optimizerE.zero_grad()

            reconstruction_loss.backward()

            optimizerW.step()

            epoch_rec_loss += reconstruction_loss.item()

        # Update the learning rate
        schedulerE.step(epoch_ncuts_loss)
        schedulerW.step(epoch_rec_loss)

    # Save the model
    torch.save(model.state_dict(), "models/wnet/wnet.pth")


def get_dataset(config):
    train_files = create_dataset_dict(volume_directory=config.train_volume_directory)

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
    main()
