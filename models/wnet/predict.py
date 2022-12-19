import torch
import torch.nn as nn

import numpy as np

from models.wnet.model import WNet
from models.wnet.train_wnet import train
from models.wnet.crf import crf, crf_batch


class PredictWNet:
    def __init__(self, trained_model_path, config, crf=True):
        self.config = config
        self.crf = crf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if trained_model_path is None:
            train()
            self.model = self.load_model(self.config.save_model_path)
        else:
            self.model = self.load_model(trained_model_path)

        self.model.to(self.device)
        self.model.eval()

    def load_model(self, trained_model_path):
        model = WNet(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            num_classes=self.config.num_classes,
        )
        model.load_state_dict(torch.load(trained_model_path))
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
            list: A list of predicted segmentations
        """
        images = torch.from_numpy(images).unsqueeze(1).float()
        images = images.to(self.device)
        with torch.no_grad():
            output = self.model.forward_encoder(images)
            output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()

        if self.crf:
            output = crf_batch(
                images,
                output,
                self.config.sa,
                self.config.sb,
                self.config.sg,
                self.config.w1,
                self.config.w2,
                self.config.n_iter,
            )
        return output
