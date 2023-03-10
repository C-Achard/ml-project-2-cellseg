"""
Predicts the segmentation of an image using a trained WNet model
"""
import napari
import torch
import torch.nn as nn

import numpy as np

from model import WNet
from train_wnet import train


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
            from crf import crf, crf_batch
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
            from crf import crf, crf_batch
            output = crf_batch(
                images.cpu().numpy(),
                output  ,
                self.config.sa,
                self.config.sb,
                self.config.sg,
                self.config.w1,
                self.config.w2,
                self.config.n_iter,
            )
        return output

def monai_window_inference(config, trained_model_path, crf = True):
    from utils import create_dataset_dict_no_labs
    from monai.inferers import sliding_window_inference
    import tifffile as tiff


    CUDA = torch.cuda.is_available()
    device = torch.device("cuda:1" if CUDA else "cpu")
    print(f"Using {device}")

    model = WNet(
        device=device,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
    )
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    print(f"Loading from {config.train_volume_directory}")
    train_files = create_dataset_dict_no_labs(
        volume_directory=config.train_volume_directory
    )
    train_files = [d.get("image") for d in train_files]
    volumes = tiff.imread(train_files[0]).astype(np.float32)

    print(f"Vol shape {volumes.shape}")
    volumes=np.expand_dims(volumes,0)
    volumes=np.expand_dims(volumes,0)

    images = torch.from_numpy(volumes).float()
    images = images.to(device)
    print(images.shape)

    res = sliding_window_inference(
        inputs=images,
        roi_size=[64,64,64],
        sw_batch_size=1,
        predictor=model.forward_encoder,
        overlap=0,
        progress=True,
    )
    res = res.detach().cpu().numpy()[0]

    # if crf:
    from crf import crf

    I =  images.detach().cpu().numpy()[0]
    P = res

    print(f"I shape : {I.shape}")
    print(f"P shape : {P.shape}")

    crf_output = crf(
        I,
        P,
        config.sa,
        config.sb,
        config.sg,
        config.w1,
        config.w2,
        config.n_iter,
    )
    return np.array([res, crf_output])

if __name__ == "__main__":
    from config import Config
    config = Config()
    # config.train_volume_directory = r"../../dataset/cropped_visual/val/volumes"
    # config.train_volume_directory = r"../../dataset/somatomotor/volumes"
    config.train_volume_directory = r"C:/Users/Cyril/Desktop/test/test"
    trained_model_path = r"./chkpt_res/test_wnet_checkpoint_4500e.pth"
    [print(a) for a in config.__dict__.items()]
    result = monai_window_inference(config, trained_model_path, crf=True)
    print(result.shape)

    import napari

    view = napari.Viewer()
    view.add_image(result[0], name="no_crf")
    view.add_image(result[1], name="crf")
    napari.run()
