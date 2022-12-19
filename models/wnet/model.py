import torch
import torch.nn as nn


class WNet(nn.Module):
    """Implementation of a 3D W-Net model, based on the 2D version from https://arxiv.org/abs/1711.08506.
    The model performs unsupervised segmentation of 3D images.
    It first encodes the input image into a latent space using the U-Net UEncoder, then decodes it back to the original image using the U-Net UDecoder.
    """

    def __init__(self, in_channels=1, out_channels=1, num_classes=2):
        super(WNet, self).__init__()
        self.encoder = UNet(in_channels, num_classes, encoder=True)
        self.decoder = UNet(num_classes, out_channels, encoder=False)

    def forward(self, x):
        """Forward pass of the W-Net model."""
        enc = self.forward_encoder(x)
        dec = self.forward_decoder(enc)
        return enc, dec

    def forward_encoder(self, x):
        """Forward pass of the encoder part of the W-Net model."""
        enc = self.encoder(x)
        return enc

    def forward_decoder(self, enc):
        """Forward pass of the decoder part of the W-Net model."""
        dec = self.decoder(enc)
        return dec


class UNet(nn.Module):
    """Half of the W-Net model, based on the U-Net architecture."""

    def __init__(self, in_channels, out_channels, encoder=True):
        super(UNet, self).__init__()
        self.in_b = InBlock(in_channels, 64)
        self.conv1 = Block(64, 128)
        self.conv2 = Block(128, 256)
        self.conv3 = Block(256, 512)
        self.bot = Block(512, 1024)
        self.deconv1 = Block(1024, 512)
        self.deconv2 = Block(512, 256)
        self.deconv3 = Block(256, 128)
        self.out_b = OutBlock(128, out_channels)

        self.sm = nn.Softmax(dim=1)
        self.encoder = encoder

    def forward(self, x):
        """Forward pass of the U-Net model."""
        in_b = self.in_b(x)
        c1 = self.conv1(nn.MaxPool3d(2)(in_b))
        c2 = self.conv2(nn.MaxPool3d(2)(c1))
        c3 = self.conv3(nn.MaxPool3d(2)(c2))
        x = self.bot(nn.MaxPool3d(2)(c3))
        x = self.deconv1(
            torch.cat([c3, nn.ConvTranspose3d(1024, 512, 2, stride=2)(x)], dim=1)
        )
        x = self.deconv2(
            torch.cat([c2, nn.ConvTranspose3d(512, 256, 2, stride=2)(x)], dim=1)
        )
        x = self.deconv3(
            torch.cat([c1, nn.ConvTranspose3d(256, 128, 2, stride=2)(x)], dim=1)
        )
        x = self.out_b(
            torch.cat([in_b, nn.ConvTranspose3d(128, 64, 2, stride=2)(x)], dim=1)
        )
        if self.encoder:
            x = self.sm(x)
        return x


class InBlock(nn.Module):
    """Input block of the U-Net architecture."""

    def __init__(self, in_channels, out_channels):
        super(InBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.65),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.65),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        """Forward pass of the input block."""
        return self.module(x)


class Block(nn.Module):
    """Basic block of the U-Net architecture."""

    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(p=0.65),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.Conv3d(out_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(p=0.65),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        """Forward pass of the basic block."""
        return self.module(x)


class OutBlock(nn.Module):
    """Output block of the U-Net architecture."""

    def __init__(self, in_channels, out_channels):
        super(OutBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.65),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.65),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, out_channels, 1),
        )

    def forward(self, x):
        """Forward pass of the output block."""
        return self.module(x)
