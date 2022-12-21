"""
Predicts the segmentation of an image using a trained WNet model
"""
import torch
import torch.nn as nn

import numpy as np

from model import WNet
from train_wnet import train
from crf import crf, crf_batch

__author__ = "Yves Paychère, Colin Hofmann, Cyril Achard"


class PredictWNet:
    def __init__(self, trained_model_path, config, crf=False):
        self.config = config
        self.crf = crf
        CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if CUDA else "cpu")

        if trained_model_path is None:
            print("No trained model found. Training a new model...")
            self.model = train()[2]
        else:
            self.model = self.load_model(self.device, trained_model_path)

        self.model.to(self.device)
        self.model.eval()

    def load_model(self, device, trained_model_path):
        model = WNet(
            device=device,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            num_classes=self.config.num_classes,
        )
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        return model

    def predict(self, image: np.ndarray):
        """Predicts the segmentation of a single image

        Args:
            image (np.ndarray): The image to predict the segmentation of

        Returns:
            np.ndarray: The predicted segmentation
        """
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model.forward_encoder(image)
            output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()

        if self.crf:
            output = crf(
                image,
                output,
                self.config.sa,
                self.config.sb,
                self.config.sg,
                self.config.w1,
                self.config.w2,
                self.config.n_iter,
            )
        return output

    def predict_batch(self, images: list):
        """Predicts the segmentation of a batch of images

        Args:
            images (list): A list of images to predict the segmentation of

        Returns:
            np.ndarray: The predicted segmentations
        """
        images = torch.from_numpy(images).unsqueeze(1).float()
        images = images.to(self.device)
        with torch.no_grad():
            output = self.model.forward_encoder(images)
            output = output.cpu().numpy()
            
        if self.crf:
            output = crf_batch(
                images.cpu().numpy(),
                output,
                self.config.sa,
                self.config.sb,
                self.config.sg,
                self.config.w1,
                self.config.w2,
                self.config.n_iter,
            )
        return output
